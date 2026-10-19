const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const config = @import("config");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const Sm80FwdTiling = configurations.Sm80FwdTiling;
const ForwardParams = @import("ForwardParams.zig");
const m_block_work = @import("m_block_work.zig");
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const applyMask = @import("mask.zig").applyMask;
const ColLimits = @import("BlockInfo.zig").ColLimits;
const smem = @import("smem.zig");
const elect = @import("elect.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");
const inter_sequence = @import("inter_sequence.zig");
const inter_sequence_forward = @import("inter_sequence_forward.zig");
const profile = @import("profile.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const chunk_elements = Sm80FwdTiling.chunk_elements;

pub const Zone = enum(u7) { tile, load_q, scores, softmax, output, store, qk_mma, pv_mma };

const Span = struct {
    first: u32,
    end: u32,
};

pub fn SharedStorage(
    comptime tiling: Sm80FwdTiling,
    comptime specialization: configurations.Specialization,
) type {
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const threads = tiling.threads();
    const warps = tiling.warps();
    const query_in_stage = tiling.shareQvSmem();
    const Rows = smem.Layout(row_chunks);
    const query_chunks = tiling.queryChunks();
    const QueryRows = smem.Layout(query_chunks);
    const PackGqaRows = if (specialization.pack_gqa) m_block_work.PackGqa else void;
    // With head dimension 256, one CTA of four warps fills the shared memory of an SM, so each SM
    // sub-partition runs a single warp, and no other warp hides address arithmetic or branches.
    const single_warp = tiling.head_dim == 256;

    return struct {
        pub const Layout = Rows;
        pub const one_warp_per_scheduler = single_warp;
        pub const prologue_parts = if (single_warp) 2 else 1;
        pub const QueryLayout = QueryRows;
        pub const shared_bytes = tiling.sharedBytes();
        pub const Args = struct {};

        const Stage = struct {
            k: Rows.Tile(block_n),
            v: Rows.Tile(block_n),
        };
        const Tiles = struct {
            stages: [stages]Stage,
            q: if (query_in_stage) void else QueryRows.Tile(block_m),
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
            if (tiling.separate_v_empty != (stages == 1)) @compileError(
                "one stage loads keys and values separately, and more stages load whole tiles",
            );
            if (stages > 1 and (tiling.prefetch_stages == 0 or tiling.prefetch_stages >= stages))
                @compileError("a refill must not wait for the release of its own stage");
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        // With several stages, each warp waits only for the tiles that it reads, and a refill
        // waits until every warp has released the stage.
        var k_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var v_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        /// Locates the chunks that one thread copies from the rows of a matrix in global memory.
        /// Each group of `swizzle_columns` threads copies whole rows. The swizzle repeats every 8
        /// rows, so the shared memory offsets of a thread differ from pass to pass by constants.
        const RowCopy = struct {
            first: u64,
            row_bytes: u64,
            rows_left: i32,

            const group = Rows.swizzle_columns;
            const pass_rows = threads / group;

            /// Returns whether a `RowCopy` can copy tiles of `count` rows.
            fn fits(count: u32) bool {
                if (group == 1 or threads % group != 0 or row_chunks % group != 0) return false;
                return pass_rows % 8 == 0 and count % pass_rows == 0;
            }

            fn init(source: [*]addrspace(.global) const Chunk, stride: u64, limit: u32) RowCopy {
                const thread_row = thread.id() / group;
                const first = source + thread_row * stride + thread.id() % group;
                return .{
                    .first = @intFromPtr(first),
                    .row_bytes = stride * @sizeOf(Chunk),
                    .rows_left = @as(i32, @intCast(limit)) - @as(i32, @intCast(thread_row)),
                };
            }

            fn copy(
                rows: RowCopy,
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                first_row: u32,
            ) void {
                rows.copyColumns(count, destination, first_row, 0, row_chunks);
            }

            /// Copies the columns from `first_column` up to `end_column` of `count` rows from
            /// `first_row` on and fills the rows at or past the limit with zeros. The copy reads
            /// no bytes of those rows (PTX ISA section 9.7.10.28.3.1, `src-size`).
            fn copyColumns(
                rows: RowCopy,
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                first_row: u32,
                comptime first_column: u32,
                comptime end_column: u32,
            ) void {
                comptime std.debug.assert(fits(count));
                const thread_slot = Rows.slot(thread.id() / group, thread.id() % group);
                const rows_left = rows.rows_left - @as(i32, @intCast(first_row));
                const pass_bytes = pass_rows * rows.row_bytes;
                var address = rows.first + first_row * rows.row_bytes;
                inline for (0..count / pass_rows) |pass| {
                    const chunk: [*]addrspace(.global) const Chunk = @ptrFromInt(address);
                    const pass_row: u32 = pass * pass_rows;
                    const valid = pass_row < rows_left;
                    inline for (first_column / group..end_column / group) |column_group| {
                        const column: u32 = column_group * group;
                        const slot = thread_slot + comptime Rows.slot(pass_row, column);
                        cp_async.copyZeroFill(
                            .cache_global_level,
                            &destination[slot],
                            &chunk[column],
                            if (valid) @sizeOf(Chunk) else 0,
                        );
                    }
                    address += pass_bytes;
                }
            }
        };
        const copies_rows = RowCopy.fits(block_n);
        const TileRows = if (copies_rows) RowCopy else void;

        pub fn queryTile() *addrspace(.shared) QueryRows.Tile(block_m) {
            return if (query_in_stage) @ptrCast(&tiles.stages[stages - 1]) else &tiles.q;
        }

        /// Returns the tile that stages the output after the last tile of keys and values. A query
        /// tile that holds only some columns leaves the output to the shared memory of the stages.
        pub fn outputStaging() *addrspace(.shared) Rows.Tile(block_m) {
            if (query_chunks == row_chunks) return @ptrCast(queryTile());
            comptime std.debug.assert(@sizeOf(Rows.Tile(block_m)) <= @sizeOf([stages]Stage));
            return @ptrCast(&tiles.stages);
        }

        /// Stages the output rows of each warp in the query tile, which no warp reads after the
        /// last key multiplication, and stores one whole row per warp instruction.
        pub fn storeRows(
            params: ForwardParams,
            tile: anytype,
            finalized: anytype,
            output: anytype,
        ) void {
            comptime std.debug.assert(!specialization.pack_gqa and warps * 16 == block_m);
            const warp_row = thread.warp() * 16;
            const lane = thread.lane();
            const bytes: [*]addrspace(.shared) u8 = @ptrCast(queryTile());
            const own = Rows.slot(warp_row + lane / 4, 0) * @sizeOf(Chunk) + lane % 4 * 4;
            inline for (output, 0..) |fragment, n_tile| {
                inline for (.{ 0, 1 }) |half| {
                    const key = n_tile % Rows.swizzle_columns;
                    const offset = comptime Rows.slot(half * 8, n_tile - key) * @sizeOf(Chunk);
                    const word: *addrspace(.shared) u32 =
                        @ptrCast(@alignCast(bytes + ((own ^ (key * @sizeOf(Chunk))) + offset)));
                    word.* = @backingInt(nvptx.Bf16x2.fromF32(
                        fragment[half * 2] * finalized.row_scale[half],
                        fragment[half * 2 + 1] * finalized.row_scale[half],
                    ));
                }
            }
            nvptx.barWarpSync(.all);
            const sequence = tile.sequence;
            const element_bytes = Sm80FwdTiling.element_bytes;
            const origin = tile.batch * params.o_batch_stride +
                @as(u64, sequence.offset_q) * params.o_row_stride +
                tile.head * params.o_head_stride;
            const row_bytes: u32 = @truncate(params.o_row_stride * element_bytes);
            const first_row = tile.first_row + warp_row;
            const rows_left =
                @as(i32, @bitCast(sequence.seqlen_q)) - @as(i32, @bitCast(first_row));
            const first = params.o + (origin + first_row * params.o_row_stride) * element_bytes +
                lane * @sizeOf(Chunk);
            const staged = Rows.slot(warp_row, lane) * @sizeOf(Chunk);
            inline for (0..16) |row| {
                if (rows_left > row) {
                    const key = row % Rows.swizzle_columns;
                    const offset = @sizeOf(Rows.Tile(row));
                    const source: *addrspace(.shared) const Chunk =
                        @ptrCast(@alignCast(bytes + ((staged ^ (key * @sizeOf(Chunk))) + offset)));
                    const destination: *addrspace(.global) Chunk =
                        @ptrFromInt(first + @as(u64, row_bytes) * row);
                    destination.* = source.*;
                }
            }
            if (elect.groupLeader(4)) {
                fragments.storeLse(
                    params,
                    tile,
                    first_row + lane / 4,
                    sequence.seqlen_q,
                    finalized.lse,
                );
            }
        }

        pub fn kTile(stage: u32) *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.stages[stage].k;
        }

        pub fn vTile(stage: u32) *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.stages[stage].v;
        }

        pub const Load = struct {
            q: [*]addrspace(.global) const Chunk,
            k: [*]addrspace(.global) const Chunk,
            v: [*]addrspace(.global) const Chunk,
            q_stride: u64,
            k_stride: u64,
            v_stride: u64,
            first_tile: u32,
            tile_count: u32,
            first_row: u32,
            seqlen_q: u32,
            seqlen_k: u32,
            q_head_stride: if (specialization.pack_gqa) u64 else void,
            pack_gqa: PackGqaRows,
            key_rows: TileRows,
            value_rows: TileRows,

            pub fn init(params: ForwardParams, _: Args, work: anytype) Load {
                const tile = work.tile;
                const sequence = tile.sequence;
                const q: [*]addrspace(.global) const Chunk = @ptrFromInt(params.q);
                const k: [*]addrspace(.global) const Chunk = @ptrFromInt(params.k);
                const v: [*]addrspace(.global) const Chunk = @ptrFromInt(params.v);
                const q_origin = tile.batch * params.q_batch_stride +
                    @as(u64, sequence.offset_q) * params.q_row_stride +
                    tile.head * params.q_head_stride;
                const k_origin = tile.batch * params.k_batch_stride +
                    @as(u64, sequence.offset_k) * params.k_row_stride +
                    tile.kv_head * params.k_head_stride;
                const v_origin = tile.batch * params.v_batch_stride +
                    @as(u64, sequence.offset_k) * params.v_row_stride +
                    tile.kv_head * params.v_head_stride;
                return .{
                    .q = q + q_origin / chunk_elements,
                    .k = k + k_origin / chunk_elements,
                    .v = v + v_origin / chunk_elements,
                    .q_stride = params.q_row_stride / chunk_elements,
                    .k_stride = params.k_row_stride / chunk_elements,
                    .v_stride = params.v_row_stride / chunk_elements,
                    .first_tile = tile.n_block_min,
                    .tile_count = tile.n_block_max - tile.n_block_min,
                    .first_row = tile.first_row,
                    .seqlen_q = sequence.seqlen_q,
                    .seqlen_k = sequence.seqlen_k,
                    .q_head_stride = if (specialization.pack_gqa) params.q_head_stride /
                        chunk_elements else {},
                    .pack_gqa = tile.pack_gqa,
                    .key_rows = if (TileRows == void) {} else .init(
                        k + k_origin / chunk_elements,
                        params.k_row_stride / chunk_elements,
                        sequence.seqlen_k,
                    ),
                    .value_rows = if (TileRows == void) {} else .init(
                        v + v_origin / chunk_elements,
                        params.v_row_stride / chunk_elements,
                        sequence.seqlen_k,
                    ),
                };
            }

            pub fn reset() void {
                nvptx.barrierSyncAligned(.work_group, null);
                if (single_warp or stages == 1) return;
                if (thread.warp() == 0 and elect.oneSync()) {
                    inline for (&k_full, &v_full, &empty) |*k_stage, *v_stage, *empty_stage| {
                        inline for (.{ k_stage, v_stage, empty_stage }) |barrier| barrier.inval();
                    }
                }
                elect.reconverge();
            }

            pub fn restart(load: *Load, params: ForwardParams, work: anytype) void {
                load.* = .init(params, .{}, work);
                synchronize();
                if (single_warp) {
                    load.key_rows.copy(block_n, kTile(0), load.firstKey(0));
                    load.value_rows.copy(block_n, vTile(0), load.firstKey(0));
                } else if (stages == 1) {
                    load.copyKeys(load.first_tile, 0);
                } else {
                    if (thread.warp() == 0 and elect.oneSync()) {
                        inline for (&k_full, &v_full, &empty) |*k_stage, *v_stage, *empty_stage| {
                            k_stage.inval();
                            v_stage.inval();
                            empty_stage.inval();
                            k_stage.init(threads);
                            v_stage.init(threads);
                            empty_stage.init(warps);
                        }
                        mbarrier.pipelineInitFence();
                    }
                    elect.reconverge();
                    mbarrier.pipelineInitWait(.cta, threads, .other);
                    inline for (0..tiling.prefetch_stages) |prefetched| {
                        const index: u32 = prefetched;
                        if (index < load.tile_count) load.copyTile(load.first_tile + index, index);
                    }
                    return;
                }
                cp_async.commitGroup();
                synchronize();
            }

            fn copyRows(
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                source: [*]addrspace(.global) const Chunk,
                stride: u64,
                first_row: u32,
                limit: u32,
                packing: anytype,
            ) void {
                const rows = .{ count, destination, source, stride, first_row, limit, packing };
                @call(.auto, copyColumns, .{ Rows, row_chunks } ++ rows);
            }

            /// Copies `columns` chunks of each row, starting at the column of `source`.
            fn copyColumns(
                comptime TileLayout: type,
                comptime columns: u32,
                comptime count: u32,
                destination: *addrspace(.shared) TileLayout.Tile(count),
                source: [*]addrspace(.global) const Chunk,
                stride: u64,
                first_row: u32,
                limit: u32,
                packing: anytype,
            ) void {
                const chunks = count * columns;
                inline for (0..(chunks + threads - 1) / threads) |pass| {
                    const linear: u32 = @as(u32, pass) * threads + thread.id();
                    if ((pass + 1) * threads <= chunks or linear < chunks) {
                        const row = linear / columns;
                        const column = linear % columns;
                        const source_row = @min(first_row + row, limit - 1);
                        const offset = if (@TypeOf(packing) == void)
                            source_row * stride
                        else
                            packing.pack_gqa.rowOffset(source_row, stride, packing.head_stride);
                        cp_async.copy(
                            .cache_global_level,
                            &destination[TileLayout.slot(row, column)],
                            &source[offset + column],
                        );
                    }
                }
            }

            fn copyKeys(load: *const Load, n_block: u32, stage: u32) void {
                if (copies_rows) {
                    return load.key_rows.copy(block_n, kTile(stage), n_block * block_n);
                }
                copyRows(
                    block_n,
                    kTile(stage),
                    load.k,
                    load.k_stride,
                    n_block * block_n,
                    load.seqlen_k,
                    {},
                );
            }

            fn copyValues(load: *const Load, n_block: u32, stage: u32) void {
                if (copies_rows) {
                    return load.value_rows.copy(block_n, vTile(stage), n_block * block_n);
                }
                copyRows(
                    block_n,
                    vTile(stage),
                    load.v,
                    load.v_stride,
                    n_block * block_n,
                    load.seqlen_k,
                    {},
                );
            }

            fn copyTile(load: *const Load, n_block: u32, stage: u32) void {
                load.copyKeys(n_block, stage);
                mbarrier.arrive(.cp_async, &k_full[stage]);
                load.copyValues(n_block, stage);
                mbarrier.arrive(.cp_async, &v_full[stage]);
            }

            /// Returns the first key of tile `index`. Past the last tile, it returns the end of
            /// the sequence, from which a copy fills every row with zeros.
            fn firstKey(load: *const Load, index: u32) u32 {
                return if (index < load.tile_count)
                    (load.first_tile + index) * block_n
                else
                    load.seqlen_k;
            }

            /// Waits until no copy group of this thread is in flight, then waits at a barrier. The
            /// barrier makes the copies of all threads visible and shows that every warp has
            /// finished reading the tile before.
            fn synchronize() void {
                cp_async.waitGroup(0);
                if (single_warp) {
                    convergedBarrier();
                } else nvptx.barrierSyncAligned(.work_group, null);
            }

            /// Omits the `bar.warp.sync` of `nvptx.barrierSyncAligned`, after which ptxas emits a
            /// convergence check that ends the basic block. The warps arrive converged.
            fn convergedBarrier() void {
                asm volatile ("barrier.sync.aligned 0;" ::: .{ .memory = true });
            }

            pub fn start(load: *const Load) void {
                if (single_warp) return load.startInParts();
                if (stages > 1 and thread.warp() == 0 and elect.oneSync()) {
                    inline for (&k_full, &v_full, &empty) |*k_stage, *v_stage, *empty_stage| {
                        k_stage.init(threads);
                        v_stage.init(threads);
                        empty_stage.init(warps);
                    }
                    mbarrier.pipelineInitFence();
                }
                if (stages > 1) elect.reconverge();
                if (specialization.pack_gqa) {
                    const packing = .{
                        .pack_gqa = load.pack_gqa,
                        .head_stride = load.q_head_stride,
                    };
                    copyColumns(
                        QueryRows,
                        query_chunks,
                        block_m,
                        queryTile(),
                        load.q + (row_chunks - query_chunks),
                        load.q_stride,
                        load.first_row,
                        load.pack_gqa.end_row,
                        packing,
                    );
                } else {
                    copyColumns(
                        QueryRows,
                        query_chunks,
                        block_m,
                        queryTile(),
                        load.q + (row_chunks - query_chunks),
                        load.q_stride,
                        load.first_row,
                        load.seqlen_q,
                        {},
                    );
                }
                cp_async.commitGroup();
                if (stages == 1) {
                    if (0 < load.tile_count) load.copyKeys(load.first_tile, 0);
                    cp_async.commitGroup();
                    synchronize();
                    return;
                }
                if (0 < load.tile_count) load.copyKeys(load.first_tile, 0);
                mbarrier.pipelineInitWait(.cta, threads, .other);
                if (0 < load.tile_count) {
                    mbarrier.arrive(.cp_async, &k_full[0]);
                    load.copyValues(load.first_tile, 0);
                    mbarrier.arrive(.cp_async, &v_full[0]);
                }
                inline for (1..tiling.prefetch_stages) |prefetched| {
                    const first: u32 = prefetched;
                    if (first < load.tile_count) load.copyTile(load.first_tile + first, first);
                }
                synchronize();
            }

            /// Copies the query and the first keys in column parts, one copy group per part.
            fn startInParts(load: *const Load) void {
                const query_rows: RowCopy = .init(load.q, load.q_stride, load.seqlen_q);
                const first_key = load.firstKey(0);
                inline for (0..prologue_parts) |part| {
                    const first = part * row_chunks / prologue_parts;
                    const end = (part + 1) * row_chunks / prologue_parts;
                    query_rows.copyColumns(block_m, queryTile(), load.first_row, first, end);
                    load.key_rows.copyColumns(block_n, kTile(0), first_key, first, end);
                    cp_async.commitGroup();
                }
            }

            /// Waits for column part `part`, and after the last part copies the first values.
            pub fn awaitPart(load: *const Load, comptime part: u32) void {
                cp_async.waitGroup(prologue_parts - 1 - part);
                convergedBarrier();
                if (part + 1 < prologue_parts) return;
                load.value_rows.copy(block_n, vTile(0), load.firstKey(0));
                cp_async.commitGroup();
            }

            pub fn drain(_: *const Load) void {
                synchronize();
            }

            pub fn queryConsumed(_: *const Load) void {
                if (!single_warp) nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn acquireK(load: *const Load, index: u32) u32 {
                if (single_warp) return 0;
                if (stages == 1) {
                    load.copyValues(load.first_tile + index, 0);
                    cp_async.commitGroup();
                    return 0;
                }
                const refill = index + tiling.prefetch_stages;
                if (refill < load.tile_count) {
                    const refill_stage = refill % stages;
                    if (refill >= stages) mbarrier.wait(
                        .cta,
                        &empty[refill_stage],
                        refill / stages - 1,
                    );
                    load.copyTile(load.first_tile + refill, refill_stage);
                }
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&k_full);
                return state.index();
            }

            pub fn acquireV(_: *const Load, index: u32) void {
                if (stages == 1) return;
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&v_full);
            }

            pub fn release(_: *const Load, index: u32) void {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerRelease(.warp, &empty);
            }

            /// With one stage, the values of a tile load while the warps multiply its keys, and
            /// the keys of the next tile load while the warps multiply the values.
            pub fn releaseKeys(load: *const Load, index: u32) void {
                synchronize();
                if (single_warp) {
                    load.key_rows.copy(block_n, kTile(0), load.firstKey(index + 1));
                } else if (index + 1 < load.tile_count) {
                    load.copyKeys(load.first_tile + index + 1, 0);
                }
                cp_async.commitGroup();
            }

            /// With one warp per scheduler, the values of the next tile load here instead of in
            /// `acquireK`, and past the last tile the copy writes zeros, so the loop holds no
            /// branch for the copy.
            pub fn releaseValues(load: *const Load, index: u32) void {
                synchronize();
                if (single_warp) {
                    load.value_rows.copy(block_n, vTile(0), load.firstKey(index + 1));
                    cp_async.commitGroup();
                }
            }
        };
    };
}

pub fn FlashAttnFwdSm80(
    comptime tiling: Sm80FwdTiling,
    comptime specialization: configurations.Specialization,
    comptime Storage: type,
) type {
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const threads = tiling.threads();
    const k_steps = tiling.head_dim / 16;
    const narrow_step = tiling.head_dim % 16 == 8;
    const mma_m_per_warp = tiling.mma_m_per_warp;
    const warp_rows = 16 * mma_m_per_warp;
    const Layout = Storage.Layout;
    const PairAddresses =
        if (@hasDecl(Layout, "PairAddresses")) Layout.PairAddresses else Layout.Addresses;
    const QueryLayout = if (@hasDecl(Storage, "QueryLayout")) Storage.QueryLayout else Layout;
    const QueryPairs = if (@hasDecl(QueryLayout, "PairAddresses"))
        QueryLayout.PairAddresses
    else
        QueryLayout.Addresses;
    const register_steps = if (!tiling.q_in_regs) 0 else tiling.q_register_steps orelse k_steps;
    const Tile = m_block_work.MBlockWork(block_m, block_n, .{
        .pack_gqa = specialization.pack_gqa,
        .inter_sequence = specialization.inter_sequence,
    });
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_m,
        .row_bytes = 4 * tiling.head_dim,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 1 });
    const Online = fragments.Softmax(block_n);
    const single_warp =
        @hasDecl(Storage, "one_warp_per_scheduler") and Storage.one_warp_per_scheduler;
    const Trace = profile.Recorder(Zone, .{});

    return struct {
        pub const thread_count = threads;
        pub const max_registers = tiling.max_registers;
        pub const separate_v_empty = tiling.separate_v_empty;
        pub const shared_bytes = Storage.shared_bytes;
        pub const Args = Storage.Args;
        pub const Load = Storage.Load;
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(
            scheduler_options.block_m,
            scheduler_options.block_order,
            specialization.sort_batches,
        );

        pub const Work = struct {
            tile: Tile,

            pub const all_groups: Span = .{ .first = 0, .end = block_n / 16 };

            pub fn init(params: ForwardParams) ?Work {
                const scheduler = TileScheduler.create(tile_scheduler.forwardArguments(params));
                const persistent_scheduler: PersistentScheduler = .create();
                const tile = scheduler.getGridWork(
                    tile_scheduler.gridTile(.x, 1, persistent_scheduler.initialWorkTileInfo()),
                ) orelse return null;
                var work: Work = .{ .tile = Tile.init(params, tile) };
                if (specialization.inter_sequence and !work.tile.takeOnlyRun(params)) return null;
                return work;
            }

            pub fn initListed(params: ForwardParams, item: u32, listed_blocks: u32) Work {
                const list: [*]addrspace(.global) const inter_sequence.ListedBlock =
                    @ptrFromInt(params.run_list);
                const listed = list[item % listed_blocks];
                return .{ .tile = Tile.init(params, .{
                    .block = listed.block,
                    .head = item / listed_blocks,
                    .batch = listed.sequence,
                }) };
            }

            pub fn visibleGroups(work: *const Work, n_block: u32) Span {
                const first_key: i32 = @intCast(n_block * block_n);
                const info = work.tile.block_info;
                const row = work.tile.first_row + thread.warp() * warp_rows;
                const first_query: i32 =
                    @intCast(if (specialization.pack_gqa) work.tile.pack_gqa.mIdx(row) else row);
                const last_query: i32 = if (specialization.pack_gqa) @intCast(
                    work.tile.pack_gqa.mIdx(row + warp_rows - 1),
                ) else first_query + warp_rows - 1;
                const col_limit_left = info.colLimitLeft(first_query);
                const col_limit_right = info.colLimitRight(last_query);
                const first = @max(0, @divFloor(col_limit_left - first_key, 16));
                const end = @min(block_n / 16, @divFloor(col_limit_right - first_key + 15, 16));
                return .{ .first = @intCast(first), .end = @intCast(@max(first, end)) };
            }
        };

        pub const Query = if (tiling.q_in_regs) [mma_m_per_warp]struct {
            wide: [register_steps]@Vector(4, u32),
            narrow: if (narrow_step) @Vector(2, u32) else void,
        } else void;
        pub const Scores = [mma_m_per_warp]Online.Scores;

        fn queryAddresses() QueryPairs(block_m) {
            const lane = thread.lane();
            return .init(thread.warp() * warp_rows + lane % 16, lane / 16);
        }

        pub fn loadQuery(load: *const Storage.Load) Query {
            if (!tiling.q_in_regs) return;
            var query: Query = undefined;
            if (tiling.q_register_steps != null) {
                comptime std.debug.assert(!narrow_step);
                const lane = thread.lane();
                const words: [*]addrspace(.global) const u32 = @ptrCast(load.q);
                const row_words = load.q_stride * (@sizeOf(Chunk) / @sizeOf(u32));
                inline for (&query, 0..) |*rows, row_tile| {
                    const row = load.first_row + thread.warp() * warp_rows +
                        @as(u32, row_tile) * 16 + lane / 4;
                    inline for (&rows.wide, 0..) |*fragment, k_step| {
                        var registers: [4]u32 = undefined;
                        inline for (&registers, 0..) |*register, index| {
                            const query_row = row + index % 2 * 8;
                            const column = k_step * 8 + lane % 4 + index / 2 * 4;
                            register.* = if (query_row < load.seqlen_q)
                                words[query_row * row_words + column]
                            else
                                0;
                        }
                        fragment.* = registers;
                    }
                }
                return query;
            }
            const query_addresses = queryAddresses();
            inline for (&query, 0..) |*rows, row_tile| {
                inline for (&rows.wide, 0..) |*fragment, k_step| {
                    fragment.* = nvptx.ldmatrix(.x4, false, query_addresses.chunk(
                        Storage.queryTile(),
                        row_tile * 16,
                        k_step * 2,
                    ));
                }
                if (narrow_step) rows.narrow = nvptx.ldmatrix(
                    .x2,
                    false,
                    query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_steps * 2),
                );
            }
            return query;
        }

        /// Calls `prologue.awaitPart` before the first step of each column part.
        pub fn qkScores(
            query: Query,
            stage: u32,
            visible: Span,
            lanes: anytype,
            prologue: anytype,
        ) Scores {
            @setEvalBranchQuota(100_000);
            const lane = thread.lane();
            const query_addresses = if (@TypeOf(lanes) == void) queryAddresses() else lanes.query;
            const keys = if (@TypeOf(lanes) == void)
                PairAddresses(block_n).init(lane % 8 + lane / 16 * 8, lane / 8 % 2)
            else
                lanes.keys;
            const tile = Storage.kTile(stage);
            var result: Scores = @splat(@splat(@splat(0)));
            inline for (0..k_steps) |k_step_index| {
                const k_step: u32 = k_step_index;
                if (@TypeOf(prologue) != @TypeOf(null)) {
                    const part_steps = k_steps / Storage.prologue_parts;
                    const part = k_step_index / part_steps;
                    if (k_step_index % part_steps == 0) prologue.awaitPart(part);
                }
                var a: [mma_m_per_warp]@Vector(4, u32) = undefined;
                inline for (&a, 0..) |*fragment, row_tile| {
                    fragment.* = if (k_step_index < register_steps)
                        query[row_tile].wide[k_step_index]
                    else
                        nvptx.ldmatrix(.x4, false, query_addresses.chunk(
                            Storage.queryTile(),
                            row_tile * 16,
                            (k_step_index - register_steps) * 2,
                        ));
                }
                inline for (0..block_n / 16) |j| {
                    if (j >= visible.first and j < visible.end) {
                        const b = nvptx.ldmatrix(.x4, false, keys.chunk(tile, j * 16, k_step * 2));
                        const low: @Vector(2, u32) = .{ b[0], b[1] };
                        const high: @Vector(2, u32) = .{ b[2], b[3] };
                        inline for (&result, a) |*rows, fragment| {
                            rows[2 * j] = nvptx.mmaSync(
                                fragments.mma_m16n8k16,
                                fragment,
                                low,
                                rows[2 * j],
                            );
                            rows[2 * j + 1] = nvptx.mmaSync(
                                fragments.mma_m16n8k16,
                                fragment,
                                high,
                                rows[2 * j + 1],
                            );
                        }
                    }
                }
            }
            if (narrow_step) {
                var a: [mma_m_per_warp]@Vector(2, u32) = undefined;
                inline for (&a, 0..) |*fragment, row_tile| {
                    fragment.* = if (tiling.q_in_regs) query[row_tile]
                        .narrow else nvptx.ldmatrix(
                        .x2,
                        false,
                        query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_steps * 2),
                    );
                }
                const narrow_keys = Layout.Addresses(block_n).init(lane % 16, 0);
                inline for (0..block_n / 16) |j| {
                    if (j >= visible.first and j < visible.end) {
                        const b = nvptx.ldmatrix(
                            .x2,
                            false,
                            narrow_keys.chunk(tile, j * 16, k_steps * 2),
                        );
                        inline for (&result, a) |*rows, fragment| {
                            rows[2 * j] = nvptx.mmaSync(
                                fragments.mma_m16n8k8,
                                fragment,
                                @Vector(1, u32){b[0]},
                                rows[2 * j],
                            );
                            rows[2 * j + 1] = nvptx.mmaSync(
                                fragments.mma_m16n8k8,
                                fragment,
                                @Vector(1, u32){b[1]},
                                rows[2 * j + 1],
                            );
                        }
                    }
                }
            }
            return result;
        }

        pub const Running = [mma_m_per_warp]Online;
        pub const Step = [mma_m_per_warp]Online.Step;

        pub fn softmaxStep(running: *Running, scores: Scores, softmax_scale_log2: f32) Step {
            var result: Step = undefined;
            inline for (&result, running, scores) |*rows, *rows_running, rows_scores| {
                rows.* = Online.step(rows_running, rows_scores, softmax_scale_log2);
            }
            return result;
        }

        pub fn maskScores(scores: *Scores, n_block: u32, work: *const Work) void {
            inline for (scores, 0..) |*rows, row_tile| {
                const row = work.tile.first_row +
                    thread.warp() * warp_rows + @as(u32, row_tile) * 16 + thread.lane() / 4;
                const limits = if (specialization.pack_gqa) work.tile.colLimits(
                    row,
                ) else ColLimits.init(work.tile.block_info, row);
                applyMask(rows, n_block * block_n, limits, -std.math.inf(f32));
            }
        }

        pub const Output = [mma_m_per_warp][row_chunks]Accumulator;

        pub fn accumulateOutput(
            output: *Output,
            step: Step,
            stage: u32,
            visible: Span,
            lanes: anytype,
        ) void {
            @setEvalBranchQuota(100_000);
            const lane = thread.lane();
            const pairs = if (@TypeOf(lanes) == void)
                PairAddresses(block_n).init(lane % 8 + lane / 8 % 2 * 8, lane / 16)
            else
                lanes.values;
            inline for (output, step) |*rows, rows_step| {
                if (!single_warp and rows_step.should_rescale) {
                    inline for (rows) |*fragment| fragment.* *= rows_step.row_scale;
                }
            }
            const tile = Storage.vTile(stage);
            inline for (0..block_n / 16) |k_step| {
                if (k_step >= visible.first and k_step < visible.end) {
                    inline for (0..row_chunks / 2) |pair| {
                        const j: u32 = pair;
                        const b = nvptx.ldmatrix(.x4, true, pairs.chunk(tile, k_step * 16, j * 2));
                        const low: @Vector(2, u32) = .{ b[0], b[1] };
                        const high: @Vector(2, u32) = .{ b[2], b[3] };
                        inline for (output, step) |*rows, rows_step| {
                            if (single_warp and k_step == 0) {
                                rows[2 * j] *= rows_step.row_scale;
                                rows[2 * j + 1] *= rows_step.row_scale;
                            }
                            const a = rows_step.probabilities[k_step];
                            rows[2 * j] =
                                nvptx.mmaSync(fragments.mma_m16n8k16, a, low, rows[2 * j]);
                            rows[2 * j + 1] =
                                nvptx.mmaSync(fragments.mma_m16n8k16, a, high, rows[2 * j + 1]);
                        }
                    }
                    if (row_chunks % 2 == 1) {
                        const tail = Layout.Addresses(block_n).init(lane % 16, 0);
                        const b = nvptx.ldmatrix(
                            .x2,
                            true,
                            tail.chunk(tile, k_step * 16, row_chunks - 1),
                        );
                        inline for (output, step) |*rows, rows_step| {
                            rows[row_chunks - 1] = nvptx.mmaSync(
                                fragments.mma_m16n8k16,
                                rows_step.probabilities[k_step],
                                b,
                                rows[row_chunks - 1],
                            );
                        }
                    }
                }
            }
        }

        fn storeOutput(
            params: ForwardParams,
            work: *const Work,
            softmax: Running,
            output: Output,
        ) void {
            var finalized: [mma_m_per_warp]Online.Finalized = undefined;
            inline for (&finalized, softmax) |*rows, rows_softmax| rows.* =
                Online.finalize(rows_softmax, params.softmax_scale_log2);
            const tile = work.tile;
            const sequence = tile.sequence;
            const warp = thread.warp();
            const lane = thread.lane();
            const lane_word = lane % 4;
            const staging = if (@hasDecl(Storage, "outputStaging"))
                Storage.outputStaging()
            else
                Storage.queryTile();
            if (single_warp) return Storage.storeRows(params, tile, finalized[0], output[0]);
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (output, finalized, 0..) |rows, rows_finalized, row_tile| {
                inline for (rows, 0..) |fragment, n_tile| {
                    inline for (.{ 0, 1 }) |half| {
                        const words: [*]addrspace(.shared) u32 = @ptrCast(Layout.chunk(
                            staging,
                            warp * warp_rows + @as(u32, row_tile) * 16 + lane / 4 + half * 8,
                            n_tile,
                            lane / 4,
                        ));
                        words[lane_word] = @backingInt(nvptx.Bf16x2.fromF32(
                            fragment[half * 2] * rows_finalized.row_scale[half],
                            fragment[half * 2 + 1] * rows_finalized.row_scale[half],
                        ));
                    }
                }
            }
            nvptx.barWarpSync(.all);
            const o: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
            const origin = tile.batch * params.o_batch_stride +
                @as(u64, sequence.offset_q) * params.o_row_stride +
                tile.head * params.o_head_stride;
            const row_stride = params.o_row_stride / chunk_elements;
            const row_limit =
                if (specialization.pack_gqa) tile.pack_gqa.end_row else sequence.seqlen_q;
            const OutputCopier = struct {
                o: [*]addrspace(.global) Chunk,
                origin: u64,
                row_stride: u64,
                head_stride: u64,
                first_row: u32,
                warp_row: u32,
                row_limit: u32,
                pack_gqa: @TypeOf(tile.pack_gqa),
                staging: @TypeOf(staging),

                pub fn copy(copier: @This(), tile_row: u32, column: u32) void {
                    const query_row = copier.first_row + copier.warp_row + tile_row;
                    if (query_row < copier.row_limit) {
                        const row_offset = if (specialization.pack_gqa)
                            copier.pack_gqa.rowOffset(
                                query_row,
                                copier.row_stride,
                                copier.head_stride / chunk_elements,
                            )
                        else
                            @as(u64, query_row) * copier.row_stride;
                        copier.o[copier.origin / chunk_elements + row_offset + column] =
                            Layout.chunk(
                                copier.staging,
                                copier.warp_row + tile_row,
                                column,
                                tile_row,
                            ).*;
                    }
                }
            };
            const copier: OutputCopier = .{
                .o = o,
                .origin = origin,
                .row_stride = row_stride,
                .head_stride = params.o_head_stride,
                .first_row = tile.first_row,
                .warp_row = warp * warp_rows,
                .row_limit = row_limit,
                .pack_gqa = tile.pack_gqa,
                .staging = staging,
            };
            smem.copyStagedRows(warp_rows, row_chunks, 32, lane, copier);
            if (elect.groupLeader(4)) {
                inline for (finalized, 0..) |rows_finalized, row_tile| {
                    fragments.storeLse(params, tile, tile.first_row + warp * warp_rows +
                        @as(u32, row_tile) * 16 + lane / 4, row_limit, rows_finalized.lse);
                }
            }
        }

        pub fn run(params: ForwardParams, args: Args) void {
            var work = Work.init(params) orelse return;
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            process(params, args, &work, .first_run, &trace);
        }

        var claimed: [2]u32 addrspace(.shared) = undefined;

        pub fn runListed(params: ForwardParams, args: Args) void {
            const queue: *addrspace(.global) inter_sequence.RunQueue =
                @ptrFromInt(params.run_queue);
            const listed_blocks = queue.listed_blocks;
            const items = listed_blocks * params.num_head;
            var item = @workGroupId(0);
            var slot: u1 = 0;
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            while (item < items) : (slot +%= 1) {
                var work: Work = .initListed(params, item, listed_blocks);
                work.tile.firstRun(params);
                process(params, args, &work, .all_runs, &trace);
                if (thread.id() == 0) {
                    const claims = nvptx.atom(.gpu, .monotonic, .add, &queue.claims, 1);
                    claimed[slot] = nvptx.nctaid(0) + claims;
                }
                Load.reset();
                item = claimed[slot];
            }
        }

        pub fn runPlanned(params: ForwardParams, args: Args) void {
            const queue: *addrspace(.global) inter_sequence.RunQueue =
                @ptrFromInt(params.run_queue);
            const listed_blocks = queue.listed_blocks;
            const items = listed_blocks * params.num_head;
            var item = @workGroupId(0);
            var slot: u1 = 0;
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            while (item < items) : (slot +%= 1) {
                var work: Work = .initListed(params, item, listed_blocks);
                if (work.tile.takeOnlyRun(params)) {
                    process(params, args, &work, .first_run, &trace);
                } else {
                    work.tile.firstRun(params);
                    process(params, args, &work, .all_runs, &trace);
                }
                if (thread.id() == 0) {
                    const claims = nvptx.atom(.gpu, .monotonic, .add, &queue.claims, 1);
                    claimed[slot] = nvptx.nctaid(0) + claims;
                }
                Load.reset();
                item = claimed[slot];
            }
        }

        inline fn process(
            params: ForwardParams,
            args: Args,
            work: *Work,
            comptime runs: enum { first_run, all_runs },
            trace: *Trace,
        ) void {
            trace.begin(.tile);
            var load = Load.init(params, args, work);
            load.start();
            trace.begin(.load_q);
            const query = loadQuery(&load);
            trace.end(.load_q);
            load.queryConsumed();
            var softmax: Running = undefined;
            inline for (&softmax) |*rows| rows.* = Online.init();
            var output: Output = @splat(@splat(@splat(0)));
            // The kernels with swizzled rows keep the shared memory offsets of the lane in
            // registers. ptxas otherwise reads the lane index again inside the loop.
            const lanes = if (Layout.swizzle_columns == 1) {} else lanes: {
                const lane = thread.lane();
                break :lanes .{
                    .query = queryAddresses(),
                    .keys = PairAddresses(block_n).init(lane % 8 + lane / 16 * 8, lane / 8 % 2),
                    .values = PairAddresses(block_n).init(lane % 8 + lane / 8 % 2 * 8, lane / 16),
                };
            };
            const tile_count = work.tile.n_block_max - work.tile.n_block_min;
            var index: u32 = 0;
            if (single_warp and runs == .all_runs and tile_count == 0) {
                load.drain();
            } else if (single_warp and runs == .all_runs) {
                {
                    const n_block = work.tile.n_block_min;
                    trace.begin(.scores);
                    trace.begin(.qk_mma);
                    var scores = qkScores(query, 0, Work.all_groups, lanes, &load);
                    trace.end(.qk_mma);
                    load.releaseKeys(0);
                    if (work.tile.needsMask(n_block)) maskScores(&scores, n_block, work);
                    trace.end(.scores);
                    trace.begin(.softmax);
                    const step = softmaxStep(&softmax, scores, params.softmax_scale_log2);
                    trace.end(.softmax);
                    trace.begin(.output);
                    trace.begin(.pv_mma);
                    accumulateOutput(&output, step, 0, Work.all_groups, lanes);
                    trace.end(.pv_mma);
                    load.releaseValues(0);
                    trace.end(.output);
                }
                index = 1;
                while (true) : (index = 0) {
                    while (index < work.tile.n_block_max - work.tile.n_block_min) : (index += 1) {
                        const n_block = work.tile.n_block_min + index;
                        trace.begin(.scores);
                        trace.begin(.qk_mma);
                        var scores = qkScores(query, 0, Work.all_groups, lanes, null);
                        trace.end(.qk_mma);
                        load.releaseKeys(index);
                        if (work.tile.needsMask(n_block)) maskScores(&scores, n_block, work);
                        trace.end(.scores);
                        trace.begin(.softmax);
                        const step = softmaxStep(&softmax, scores, params.softmax_scale_log2);
                        trace.end(.softmax);
                        trace.begin(.output);
                        trace.begin(.pv_mma);
                        accumulateOutput(&output, step, 0, Work.all_groups, lanes);
                        trace.end(.pv_mma);
                        load.releaseValues(index);
                        trace.end(.output);
                    }
                    if (!work.tile.nextRun(params)) break;
                    load.restart(params, work);
                }
            } else if (single_warp) {
                // The first tile waits for the prologue parts. Then masked tiles before the
                // unmasked ones, the unmasked tiles, and masked tiles after them run in loops.
                const first_block = work.tile.n_block_min;
                const last_block = work.tile.n_block_max;
                const unmasked = work.tile.unmaskedBlocks();
                const middle = std.math.clamp(unmasked[0], first_block, last_block);
                const middle_end = std.math.clamp(unmasked[1], middle, last_block);
                const ends = .{
                    @min(1, tile_count),
                    middle - first_block,
                    middle_end - first_block,
                    tile_count,
                };
                const phases: [4]?bool = .{ null, true, false, true };
                inline for (phases, ends) |phase_masked, end| {
                    while (index < end) : (index += 1) {
                        const n_block = first_block + index;
                        const masked = phase_masked orelse work.tile.needsMask(n_block);
                        const prologue = if (phase_masked == null) &load else null;
                        trace.begin(.scores);
                        trace.begin(.qk_mma);
                        var scores = qkScores(query, 0, Work.all_groups, lanes, prologue);
                        trace.end(.qk_mma);
                        load.releaseKeys(index);
                        if (masked) maskScores(&scores, n_block, work);
                        trace.end(.scores);
                        trace.begin(.softmax);
                        const step = softmaxStep(&softmax, scores, params.softmax_scale_log2);
                        trace.end(.softmax);
                        trace.begin(.output);
                        trace.begin(.pv_mma);
                        accumulateOutput(&output, step, 0, Work.all_groups, lanes);
                        trace.end(.pv_mma);
                        load.releaseValues(index);
                        trace.end(.output);
                    }
                }
                if (tile_count == 0) load.drain();
            } else if (runs == .all_runs) {
                while (true) {
                    runTiles(params, work, &load, query, &softmax, &output, lanes, trace);
                    if (!work.tile.nextRun(params)) break;
                    load.restart(params, work);
                }
            } else runTiles(params, work, &load, query, &softmax, &output, lanes, trace);
            trace.begin(.store);
            storeOutput(params, work, softmax, output);
            trace.end(.store);
            trace.end(.tile);
        }

        inline fn runTiles(
            params: ForwardParams,
            work: *const Work,
            load: *const Load,
            query: Query,
            softmax: *Running,
            output: *Output,
            lanes: anytype,
            trace: *Trace,
        ) void {
            const tile_count = work.tile.n_block_max - work.tile.n_block_min;
            var index: u32 = 0;
            while (index < tile_count) : (index += 1) {
                const n_block = work.tile.n_block_min + index;
                trace.begin(.scores);
                const slot = load.acquireK(index);
                if (work.tile.needsMask(n_block)) {
                    // With square tiles of `cp.async` copies, the warps multiply every key group
                    // of a masked tile, and the mask clears the groups that a warp cannot see.
                    const visible = if (tiling.tma or block_m != block_n)
                        work.visibleGroups(n_block)
                    else
                        Work.all_groups;
                    trace.begin(.qk_mma);
                    var scores = qkScores(query, slot, visible, lanes, null);
                    trace.end(.qk_mma);
                    if (separate_v_empty) load.releaseKeys(index);
                    maskScores(&scores, n_block, work);
                    trace.end(.scores);
                    trace.begin(.softmax);
                    const step = softmaxStep(softmax, scores, params.softmax_scale_log2);
                    trace.end(.softmax);
                    trace.begin(.output);
                    load.acquireV(index);
                    trace.begin(.pv_mma);
                    accumulateOutput(output, step, slot, visible, lanes);
                    trace.end(.pv_mma);
                } else {
                    const visible = Work.all_groups;
                    trace.begin(.qk_mma);
                    const scores = qkScores(query, slot, visible, lanes, null);
                    trace.end(.qk_mma);
                    if (separate_v_empty) load.releaseKeys(index);
                    trace.end(.scores);
                    trace.begin(.softmax);
                    const step = softmaxStep(softmax, scores, params.softmax_scale_log2);
                    trace.end(.softmax);
                    trace.begin(.output);
                    load.acquireV(index);
                    trace.begin(.pv_mma);
                    accumulateOutput(output, step, slot, visible, lanes);
                    trace.end(.pv_mma);
                }
                if (separate_v_empty) load.releaseValues(index) else load.release(index);
                trace.end(.output);
            }
        }
    };
}

const variant = configurations.variants(.flash_fwd_sm80, config.head_dim)[config.variant];
const forward_tiling = variant.tiling;
const Forward = FlashAttnFwdSm80(
    forward_tiling,
    variant.specialization,
    SharedStorage(forward_tiling, variant.specialization),
);

fn launch(params: ForwardParams, args: Forward.Args) void {
    var grid_params = params;
    grid_params.num_m_blocks = nvptx.nctaid(0);
    grid_params.num_head = nvptx.nctaid(1);
    Forward.run(grid_params, args);
}

const names = .{
    "q",                 "k",                "v",                  "o",
    "lse",               "cu_seqlens_q",     "cu_seqlens_k",       "seqlen_q",
    "seqlen_k",          "qhead_per_kvhead", "softmax_scale_log2", "window_size_left",
    "window_size_right",
} ++ entry.strides("q") ++ entry.strides("k") ++ entry.strides("v") ++ entry.strides("o") ++
    .{ "lse_batch_stride", "lse_head_stride", "num_batch" } ++
    (if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}) ++
    if (variant.specialization.inter_sequence) .{
        "visibility_bits", "window_sizes", "run_table", "run_list", "run_queue", "num_head",
    } else .{};
const options: std.lang.CallingConvention.NvptxKernelOptions = .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
};
const Entry = entry.Entry(ForwardParams, Forward.Args, names, options, launch);
const ListedEntry = entry.Entry(ForwardParams, Forward.Args, names, options, Forward.runListed);
const PlannedEntry = entry.Entry(ForwardParams, Forward.Args, names, options, Forward.runPlanned);

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_fwd" });
        profile.exportZones(Zone);
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_fwd_prepare_varlen_num_blocks",
            });
        }
        if (variant.specialization.inter_sequence) {
            @export(&ListedEntry.entry, .{ .name = "flash_attn_fwd_runs" });
            @export(&PlannedEntry.entry, .{ .name = "flash_attn_fwd_planned" });
            @export(&inter_sequence.RunTable(forward_tiling.block_m).build, .{
                .name = "inter_sequence_forward_runs",
            });
            @export(&inter_sequence.transposeVisibility, .{ .name = "inter_sequence_transpose" });
            @export(&inter_sequence.planBlocks, .{ .name = "inter_sequence_plan_blocks" });
            if (variant.specialization.limits.max_seqlen_q == 0 and
                configurations.inter_sequence_megakernel.supports(config.head_dim))
            {
                const Megakernel = inter_sequence_forward.Megakernel(config.head_dim);
                std.debug.assert(Megakernel.shared_bytes == Forward.shared_bytes);
                std.debug.assert(Megakernel.threads == Forward.thread_count);
                const MegakernelEntry = entry.Entry(
                    inter_sequence_forward.Params,
                    inter_sequence_forward.Args,
                    inter_sequence_forward.names,
                    .{
                        .required_work_group_size = .{ .x = Megakernel.threads },
                        .max_registers_per_work_item = Forward.max_registers,
                    },
                    Megakernel.run,
                );
                @export(&MegakernelEntry.entry, .{ .name = "inter_sequence_forward_megakernel" });
            }
            @export(&inter_sequence.countVisibleTokens, .{
                .name = "inter_sequence_count_visible_tokens",
            });
        }
    }
}
