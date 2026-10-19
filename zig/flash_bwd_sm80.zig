const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const config = @import("config");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const FlashAttnBwdSm80Kernel = @import("flash_bwd_kernel_sm80.zig").FlashAttnBwdSm80Kernel;
const NBlockWork = @import("n_block_work.zig").NBlockWork;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const applyMask = @import("mask.zig").applyMask;
const ColLimits = @import("BlockInfo.zig").ColLimits;
const smem = @import("smem.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const RowPair = fragments.RowPair;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;

const vector_reductions = nvptx.gpu.sm >= 90;
/// Without vector reductions, the backward adds each query gradient value with its own reduction,
/// and the fragment order of `dq_accum` lets each of those reductions cover 32 consecutive floats.
pub const fragment_ordered_dq_accum = !vector_reductions;

pub fn SharedStorage(
    comptime tiling: configurations.Sm80BwdTiling,
    comptime loop_schedule: @import("Sm80BwdSchedule.zig"),
) type {
    const padded_chunks = tiling.rowChunks();
    const real_chunks = tiling.head_dim / chunk_elements;
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const query_stages = loop_schedule.queryStages(tiling);
    const threads = tiling.threads();
    const Rows = smem.Layout(padded_chunks);
    const PdS = smem.Layout(block_n / chunk_elements);

    comptime {
        if (loop_schedule.dq_column_splits > 1 and tiling.bulk_reduce) @compileError(
            "only the scalar query gradient reductions split the columns",
        );
        if (loop_schedule.double_buffered_query and (stages != 1 or tiling.alternate_gradient_order))
            @compileError("the second query buffer needs one buffer for the output gradient");
        if (loop_schedule.interleaved_dq_reduction and (!fragment_ordered_dq_accum or tiling.tma or
            !tiling.p_first or !tiling.alternate_gradient_order or
            tiling.head_dim != tiling.headDimPadded())) @compileError(
            "the interleaved dQ reduction needs fragment ordered query gradients, the split " ++
                "gradient order, the probabilities first and an unpadded head dimension",
        );
    }

    return struct {
        const Shared = @This();
        pub const Layout = Rows;
        pub const PdSLayout = PdS;
        pub const shared_bytes = loop_schedule.sharedBytes(tiling);
        pub const Args = struct {};

        const value_tile = tiling.valueTile();
        const PdSTiles = [2]PdS.Tile(block_m);
        const shared_region_bytes = @max(@sizeOf(Rows.Tile(block_n)), @sizeOf(PdSTiles));
        const SharedRegion = [shared_region_bytes / @sizeOf(Chunk)]Chunk;
        const Tiles = struct {
            k: Rows.Tile(block_n),
            v: if (value_tile == .own) Rows.Tile(block_n) else void,
            q: [query_stages]Rows.Tile(block_m),
            do: [stages]Rows.Tile(block_m),
            pds: if (value_tile == .shared_with_pds) SharedRegion else PdSTiles,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);

        pub fn kTile() *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.k;
        }

        pub fn vTile() *addrspace(.shared) Rows.Tile(block_n) {
            return switch (value_tile) {
                .own => &tiles.v,
                .last_query_stage => @ptrCast(&tiles.q[stages - 1]),
                .shared_with_pds => @ptrCast(&tiles.pds),
            };
        }

        pub fn qTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.q[buffer];
        }

        pub fn doTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.do[buffer];
        }

        fn pdsTiles() *addrspace(.shared) PdSTiles {
            return @ptrCast(&tiles.pds);
        }

        pub fn pTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &pdsTiles()[0];
        }

        pub fn dsTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &pdsTiles()[1];
        }

        const TileCopy = smem.RowCopy(Rows, threads, padded_chunks, real_chunks);

        /// Locates the rows of one tensor that a CTA copies into shared memory.
        const RowCopy = struct {
            rows: [*]addrspace(.global) const Chunk,
            row_stride: u64,
            head_stride: u64,
            limit: u32,

            fn init(
                comptime name: []const u8,
                params: BackwardParams,
                batch: u32,
                first_row: u32,
                limit: u32,
            ) RowCopy {
                const chunks: [*]addrspace(.global) const Chunk = @ptrFromInt(@field(params, name));
                const stride = @field(params, name ++ "_row_stride");
                const origin = batch * @field(params, name ++ "_batch_stride") + first_row * stride;
                return .{
                    .rows = chunks + origin / chunk_elements,
                    .row_stride = stride / chunk_elements,
                    .head_stride = @field(params, name ++ "_head_stride") / chunk_elements,
                    .limit = limit,
                };
            }

            fn copy(
                source: RowCopy,
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                head: u32,
                first_row: u32,
            ) void {
                const origin = source.rows + head * source.head_stride;
                const stride = source.row_stride;
                TileCopy.copy(count, destination, origin, stride, first_row, source.limit);
            }
        };

        const copy_rows_per_pass = threads / padded_chunks;
        const thread_rows_copy = stages == 2 and padded_chunks == real_chunks and
            threads % padded_chunks == 0 and block_m % copy_rows_per_pass == 0 and
            copy_rows_per_pass % 8 == 0;

        /// Locates the chunks that a thread copies for the first query block of a work tile, so
        /// that a later block of the same head only adds a multiple of the row stride.
        const ThreadRows = struct {
            q: [*]addrspace(.global) const Chunk,
            do: [*]addrspace(.global) const Chunk,
            rows_left: i32,
            slot: u32,

            fn init(q: RowCopy, do: RowCopy, work: anytype) ThreadRows {
                const tid = @workItemId(0);
                const column = tid % padded_chunks;
                const thread_row = tid / padded_chunks;
                const row = work.m_block_min * block_m + thread_row;
                return .{
                    .q = q.rows + work.first_qhead * q.head_stride + row * q.row_stride + column,
                    .do = do.rows + work.first_qhead * do.head_stride + row * do.row_stride +
                        column,
                    .rows_left = @as(i32, @intCast(q.limit)) - @as(i32, @intCast(row)),
                    .slot = Rows.slot(thread_row, column),
                };
            }
        };

        pub const Load = struct {
            pub const schedule = loop_schedule;

            /// Returns the steps of the main loop that the schedule adds to `Kernel`, which keeps
            /// its tiles in this storage.
            pub fn Steps(comptime Kernel: type) type {
                return @import("Sm80BwdSchedule.zig").Steps(Kernel, Shared, tiling, schedule);
            }

            params: BackwardParams,
            q: RowCopy,
            do: RowCopy,
            thread_rows: if (thread_rows_copy) ThreadRows else void,

            pub fn init(params: BackwardParams, _: Args, work: anytype) Load {
                const sequence = work.sequence;
                const offset = sequence.offset_q;
                const q: RowCopy = .init("q", params, work.batch, offset, sequence.seqlen_q);
                const do: RowCopy = .init("do", params, work.batch, offset, sequence.seqlen_q);
                return .{
                    .params = params,
                    .q = q,
                    .do = do,
                    .thread_rows = if (thread_rows_copy) .init(q, do, work) else {},
                };
            }

            inline fn copyBlock(load: *const Load, buffer: u32, block: u32) void {
                comptime std.debug.assert(thread_rows_copy);
                const rows = load.thread_rows;
                const first_row = block * block_m;
                const rows_left = rows.rows_left - @as(i32, @intCast(first_row));
                inline for (.{ "q", "do" }) |name| {
                    const source = @field(load, name);
                    const tile = &@field(tiles, name)[buffer];
                    var chunk = @field(rows, name) + first_row * source.row_stride;
                    inline for (0..block_m / copy_rows_per_pass) |pass| {
                        const valid = rows_left > pass * copy_rows_per_pass;
                        cp_async.copyZeroFill(
                            .cache_global_level,
                            &tile[rows.slot + comptime Rows.slot(pass * copy_rows_per_pass, 0)],
                            if (valid) &chunk[0] else &source.rows[0],
                            if (valid) @sizeOf(Chunk) else 0,
                        );
                        chunk += copy_rows_per_pass * source.row_stride;
                    }
                }
            }

            fn fetch(load: *const Load, comptime name: []const u8, work: anytype, index: u32) void {
                const step = work.branchedStep(index);
                const buffers = comptime if (std.mem.eql(u8, name, "q")) query_stages else stages;
                const tile = &@field(tiles, name)[index % buffers];
                @field(load, name).copy(block_m, tile, step.head, step.block_info.first_row);
            }

            fn copyQuery(load: *const Load, work: anytype, iteration: u32) void {
                load.fetch("q", work, iteration);
                load.fetch("do", work, iteration);
            }

            pub fn start(load: *const Load, work: anytype) void {
                if (work.iterations() == 0) return;
                const params = load.params;
                const sequence = work.sequence;
                const key_tiles = .{ kTile(), vTile() };
                inline for (.{ "k", "v" }, key_tiles, .{ "q", "do" }) |key_name, key_tile, name| {
                    const keys: RowCopy =
                        .init(key_name, params, work.batch, sequence.offset_k, sequence.seqlen_k);
                    keys.copy(block_n, key_tile, work.kv_head, work.first_key);
                    load.fetch(name, work, 0);
                    cp_async.commitGroup();
                }
                cp_async.waitGroup(1);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn acquireDo(_: *const Load, iteration: u32, work: anytype) void {
                // With the values in registers, `loadValues` has waited for every copy.
                if (schedule.double_buffered_query and tiling.v_in_regs) return;
                if (iteration > 0) return;
                // The copy of the second query block starts later with the interleaved reduction.
                const prefetched = !schedule.interleaved_dq_reduction and work.iterations() > 1;
                if (query_stages == 2 and prefetched) cp_async.waitGroup(
                    1,
                ) else cp_async.waitGroup(0);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            /// Starts the copy of the query block of iteration `next` at the top of the loop, or
            /// between the S and dP GEMMs with the interleaved dQ reduction.
            pub fn prefetch(load: *const Load, next: u32, count: u32, work: anytype) void {
                if (schedule.interleaved_dq_reduction) return;
                load.prefetchQuery(next, count, work);
            }

            pub fn prefetchQuery(load: *const Load, next: u32, count: u32, work: anytype) void {
                if (stages == 2 and next < count) {
                    if (thread_rows_copy and work.qhead_per_kvhead == 1)
                        load.copyBlock(next % stages, next)
                    else
                        load.copyQuery(work, next);
                    cp_async.commitGroup();
                } else if (stages == 1 and query_stages == 2 and next < count) {
                    load.fetch("q", work, next);
                    cp_async.commitGroup();
                }
            }

            pub fn release(load: *const Load, iteration: u32, count: u32, work: anytype) void {
                if (stages == 1 and query_stages == 2) {
                    schedule.loopBarrier();
                    if (iteration + 1 < count) {
                        load.fetch("do", work, iteration + 1);
                        cp_async.commitGroup();
                    }
                } else if (stages == 1 and iteration + 1 < count) {
                    schedule.loopBarrier();
                    load.copyQuery(work, iteration + 1);
                    cp_async.commitGroup();
                }
            }

            pub fn restart(load: *Load, work: anytype) void {
                load.* = .init(load.params, .{}, work);
                load.copyQuery(work, 0);
                cp_async.commitGroup();
                cp_async.waitGroup(0);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn finish(_: *const Load, _: u32, _: u32, _: anytype) void {
                cp_async.waitGroup(0);
                schedule.loopBarrier();
            }
        };
    };
}

pub fn FlashAttnBwdSm80(
    comptime tiling: configurations.Sm80BwdTiling,
    comptime specialization: configurations.Specialization,
    comptime Storage: type,
) type {
    const head_dim = tiling.head_dim;
    const padded_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const sm80_schedule = @hasDecl(Storage.Load, "schedule");
    const query_stages = if (sm80_schedule) Storage.Load.schedule.queryStages(tiling) else stages;
    const threads = tiling.threads();
    const warps = tiling.warps;
    const score_rows = tiling.score_row_tiles;
    const warps_m = block_m / (16 * score_rows);
    const warps_n = warps / warps_m;
    const columns_per_warp = block_n / warps_n;
    const row_tiles = tiling.gradient_row_tiles;
    const warps_kv_n = block_n / (16 * row_tiles);
    const warps_kv_d = warps / warps_kv_n;
    const tiles_kv = padded_chunks / warps_kv_d;
    const warps_dq_m = block_m / (16 * row_tiles);
    const warps_dq_d = warps / warps_dq_m;
    const tiles_dq = padded_chunks / warps_dq_d;
    const k_steps = padded_chunks / 2;
    const Rows = Storage.Layout;
    const PdSLayout = Storage.PdSLayout;
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_n,
        .row_bytes = 8 * head_dim,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 1 });

    comptime {
        if (warps % warps_m != 0 or block_n % warps_n != 0 or columns_per_warp % 16 != 0)
            @compileError("the score tile does not divide among the warps");
        if (warps % warps_kv_n != 0 or padded_chunks % warps_kv_d != 0)
            @compileError("the gradient tile does not divide among the warps");
        if (warps % warps_dq_m != 0 or padded_chunks % warps_dq_d != 0) @compileError(
            "the query gradient does not divide among the warps",
        );
        if (row_tiles > 1 and (vector_reductions or tiling.bulk_reduce)) @compileError(
            "only the scalar query gradient reductions handle several row tiles",
        );
        if (tiling.alternate_gradient_order and (stages != 2 or warps % 2 != 0)) @compileError(
            "the split gradient order needs two query stages and an even warp count",
        );
    }

    return struct {
        pub const thread_count = threads;
        pub const shared_bytes = Storage.shared_bytes;
        pub const pdl = tiling.pdl;
        pub const alternate_gradient_order = tiling.alternate_gradient_order;
        pub const waits_for_scheduler_metadata = specialization.scheduler_metadata;
        pub const Args = Storage.Args;
        pub const Load = Storage.Load;
        pub const Work = NBlockWork(block_m, block_n, .{
            .inter_sequence = specialization.inter_sequence,
        });
        pub const inter_sequence = specialization.inter_sequence;
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(
            scheduler_options.block_m,
            scheduler_options.block_order,
            specialization.sort_batches,
        );

        pub const Step = Work.Step;

        const Terms = struct {
            lse_negated: RowPair,
            dpsum_negated: RowPair,
        };

        const Scores = [score_rows][columns_per_warp / 8]Accumulator;

        const Probabilities = struct {
            p: [score_rows][columns_per_warp / 8]@Vector(2, u32),
            ds: [score_rows][columns_per_warp / 8]@Vector(2, u32),
        };

        fn TransposedOperand(comptime rows: u32, comptime tile_count: u32) type {
            return struct {
                pairs: Rows.Addresses(rows),
                tail: if (tile_count % 2 == 1) Rows.Addresses(rows) else void,

                fn init(first_tile: u32) @This() {
                    const lane = thread.lane();
                    return .{
                        .pairs = .init(lane % 8 + lane / 8 % 2 * 8, first_tile + lane / 16),
                        .tail = if (tile_count % 2 == 1) .init(lane % 16, first_tile) else {},
                    };
                }
            };
        }

        const dq_gemm_tiles = if (tiling.bulk_reduce) Storage.DqAccum.box_columns / 8 else tiles_dq;

        /// Holds the shared memory offsets of the matrix operands that each thread loads. The
        /// kernel computes them once per CTA, so that each `ldmatrix` in the loop adds only a
        /// constant to a register.
        pub const Operands = struct {
            query: Rows.Addresses(block_m),
            key: Rows.Addresses(block_n),
            gradient: TransposedOperand(block_m, tiles_kv),
            query_gradient: TransposedOperand(block_n, dq_gemm_tiles),
            pds_columns: PdSLayout.Addresses(block_m),
            ds_rows: PdSLayout.Addresses(block_m),
            pds_store: PdSLayout.Addresses(block_m),

            pub fn init() Operands {
                const lane = thread.lane();
                const warp_m = thread.warp() % warps_m;
                const warp_n = thread.warp() / warps_m;
                const warp_kv_n = thread.warp() % warps_kv_n;
                const warp_kv_d = thread.warp() / warps_kv_n;
                const warp_dq_m = thread.warp() % warps_dq_m;
                const warp_dq_d = thread.warp() / warps_dq_m;
                const key_row = warp_n * columns_per_warp + lane % 8 + lane / 16 * 8;
                const ds_column = warp_kv_n * 2 * row_tiles + lane / 8 % 2;
                return .{
                    .query = .init(warp_m * 16 * score_rows + lane % 16, lane / 16),
                    .key = .init(key_row, lane / 8 % 2),
                    .gradient = .init(warp_kv_d * tiles_kv),
                    .query_gradient = .init(warp_dq_d * tiles_dq),
                    .pds_columns = .init(lane % 8 + lane / 16 * 8, ds_column),
                    .ds_rows = .init(warp_dq_m * 16 * row_tiles + lane % 16, lane / 16),
                    .pds_store = .init(
                        warp_m * 16 * score_rows + lane / 4,
                        warp_n * (columns_per_warp / 8),
                    ),
                };
            }
        };

        /// Holds the B operand fragments of the dP GEMM, which a warp keeps in registers for the
        /// whole CTA when `tiling.v_in_regs` is set.
        pub const Values = if (tiling.v_in_regs)
            [k_steps][columns_per_warp / 16]@Vector(4, u32)
        else
            void;

        pub fn loadValues(operands: *const Operands) Values {
            if (!tiling.v_in_regs) return {};
            cp_async.waitGroup(0);
            nvptx.barrierSyncAligned(.work_group, null);
            var values: Values = undefined;
            inline for (&values, 0..) |*step_values, k_step| {
                inline for (step_values, 0..) |*b, j| b.* = nvptx.ldmatrix(
                    .x4,
                    false,
                    operands.key.chunk(Storage.vTile(), j * 16, k_step * 2),
                );
            }
            if (tiling.valueTile() != .own) nvptx.barrierSyncAligned(.work_group, null);
            return values;
        }

        /// Multiplies rows of `a_tile` by the key rows in `b_source`, which is either a shared
        /// memory tile or `Values`.
        fn scoresGemm(operands: *const Operands, a_tile: anytype, b_source: anytype) Scores {
            @setEvalBranchQuota(100_000);
            const keys = operands.key;
            var scores: Scores = @splat(@splat(@splat(0)));
            inline for (0..k_steps) |k_step| {
                var a: [score_rows]@Vector(4, u32) = undefined;
                inline for (&a, 0..) |*row_a, tile| row_a.* = nvptx.ldmatrix(
                    .x4,
                    false,
                    operands.query.chunk(a_tile, tile * 16, k_step * 2),
                );
                inline for (0..columns_per_warp / 16) |j| {
                    const b = if (@TypeOf(b_source) == Values)
                        b_source[k_step][j]
                    else
                        nvptx.ldmatrix(.x4, false, keys.chunk(b_source, j * 16, k_step * 2));
                    inline for (&scores, a) |*row_scores, row_a| {
                        row_scores[2 * j] = nvptx.mmaSync(
                            fragments.mma_m16n8k16,
                            row_a,
                            @Vector(2, u32){ b[0], b[1] },
                            row_scores[2 * j],
                        );
                        row_scores[2 * j + 1] = nvptx.mmaSync(
                            fragments.mma_m16n8k16,
                            row_a,
                            @Vector(2, u32){ b[2], b[3] },
                            row_scores[2 * j + 1],
                        );
                    }
                }
            }
            return scores;
        }

        const ValueSource = if (tiling.v_in_regs) Values else @TypeOf(Storage.vTile());

        fn valueSource(values: *const Values) ValueSource {
            return if (tiling.v_in_regs) values.* else Storage.vTile();
        }

        /// Multiplies each A fragment of `a`, one per row tile, by the columns of `b_tile` that
        /// `operand` locates, so that each B fragment serves every row tile.
        pub fn transposedGemm(
            accumulator: anytype,
            comptime tile_count: u32,
            a: anytype,
            operand: anytype,
            b_tile: anytype,
            comptime k_step: u32,
            comptime first_column: u32,
        ) void {
            @setEvalBranchQuota(100_000);
            inline for (0..tile_count / 2) |j| {
                const b = nvptx.ldmatrix(
                    .x4,
                    true,
                    operand.pairs.chunk(b_tile, k_step * 16, first_column + 2 * j),
                );
                inline for (accumulator, a) |*row_accumulator, row_a| {
                    row_accumulator[2 * j] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        row_a,
                        @Vector(2, u32){ b[0], b[1] },
                        row_accumulator[2 * j],
                    );
                    row_accumulator[2 * j + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        row_a,
                        @Vector(2, u32){ b[2], b[3] },
                        row_accumulator[2 * j + 1],
                    );
                }
            }
            if (tile_count % 2 == 1) {
                const b = nvptx.ldmatrix(
                    .x2,
                    true,
                    operand.tail.chunk(b_tile, k_step * 16, first_column + tile_count - 1),
                );
                inline for (accumulator, a) |*row_accumulator, row_a| {
                    row_accumulator[tile_count - 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        row_a,
                        b,
                        row_accumulator[tile_count - 1],
                    );
                }
            }
        }

        pub fn decodeWork(params: BackwardParams) ?Work {
            const splits =
                if (specialization.head_split and params.cu_seqlens_k == 0) nvptx.nctaid(0) /
                ((params.seqlen_k + block_n - 1) / block_n) else 1;
            const persistent_scheduler: PersistentScheduler = .create();
            const ctaid = persistent_scheduler.initialWorkTileInfo();
            const grid = tile_scheduler.gridTile(
                .x,
                splits,
                if (sm80_schedule and !specialization.head_split)
                    Storage.Load.schedule.orderedCtaid(params, ctaid)
                else
                    ctaid,
            );
            const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
            return Work.init(
                params,
                scheduler.getGridWork(grid) orelse return null,
                grid.split_idx,
                grid.num_qhead_splits,
            );
        }

        fn rowQuad(pair: RowPair) Accumulator {
            return .{ pair[0], pair[0], pair[1], pair[1] };
        }

        pub fn computeProbabilities(
            params: BackwardParams,
            iteration: u32,
            work: *const Work,
            step: Step,
            load: *const Storage.Load,
            operands: *const Operands,
            values: *const Values,
        ) Probabilities {
            const warp_m = thread.warp() % warps_m;
            const warp_n = thread.warp() / warps_m;
            const first_row =
                step.block_info.first_row + warp_m * 16 * score_rows + thread.lane() / 4;
            const lse: [*]addrspace(.global) const f32 = @ptrFromInt(params.lse_log2);
            const dpsum: [*]addrspace(.global) const f32 = @ptrFromInt(params.dpsum);
            const base = work.batch * params.lse_batch_stride + step.head * params.lse_head_stride +
                work.sequence.offset_q;
            const seqlen_q = work.sequence.seqlen_q;
            var terms: [score_rows]Terms = @splat(.{
                .lse_negated = @splat(-std.math.inf(f32)),
                .dpsum_negated = @splat(0),
            });
            inline for (&terms, 0..) |*row_terms, tile| {
                inline for (0..2) |half| {
                    const row = first_row + @as(u32, tile) * 16 + @as(u32, half) * 8;
                    if (row < seqlen_q) {
                        row_terms.lse_negated[half] = lse[base + row];
                        row_terms.dpsum_negated[half] = dpsum[base + row];
                    }
                }
            }
            const buffer = iteration % stages;
            const query = Storage.qTile(if (query_stages == stages) buffer else iteration % 2);
            var scores = scoresGemm(operands, query, Storage.kTile());
            load.acquireDo(iteration, work);
            if (sm80_schedule and Storage.Load.schedule.interleaved_dq_reduction) {
                load.prefetchQuery(iteration + 1, work.iterations(), work);
            }
            var d_probabilities = if (tiling.p_first)
                @as(Scores, undefined)
            else
                scoresGemm(operands, Storage.doTile(buffer), valueSource(values));
            if (step.block_info.needsMask(
                step.block_info.colLimitBounds(),
                work.first_key / block_n,
                block_n,
            )) {
                inline for (&scores, 0..) |*row_scores, tile| {
                    const limits = ColLimits.init(step.block_info, first_row + @as(u32, tile) * 16);
                    applyMask(row_scores, work.first_key + warp_n *
                        columns_per_warp, limits, -std.math.inf(f32));
                }
            }
            const scale: Accumulator = @splat(params.softmax_scale_log2);
            var probabilities: Scores = undefined;
            if (tiling.p_first) {
                inline for (&probabilities, scores, terms) |*row_p, row_scores, row_terms| {
                    const lse4 = rowQuad(row_terms.lse_negated);
                    inline for (row_p, row_scores) |*p, score| p.* =
                        fragments.exp2(@mulAdd(Accumulator, score, scale, lse4));
                }
                d_probabilities = scoresGemm(operands, Storage.doTile(buffer), valueSource(values));
            }
            var result: Probabilities = undefined;
            inline for (0..score_rows) |row_tile| {
                const lse4 = rowQuad(terms[row_tile].lse_negated);
                const sum4 = rowQuad(terms[row_tile].dpsum_negated);
                inline for (scores[row_tile], d_probabilities[row_tile], 0..) |score, d_p, tile| {
                    const p = if (tiling.p_first) probabilities[row_tile][tile] else fragments.exp2(
                        @mulAdd(Accumulator, score, scale, lse4),
                    );
                    const ds = p * (d_p + sum4);
                    result.p[row_tile][tile] = .{
                        @backingInt(nvptx.Bf16x2.fromF32(p[0], p[1])),
                        @backingInt(nvptx.Bf16x2.fromF32(p[2], p[3])),
                    };
                    result.ds[row_tile][tile] = .{
                        @backingInt(nvptx.Bf16x2.fromF32(ds[0], ds[1])),
                        @backingInt(nvptx.Bf16x2.fromF32(ds[2], ds[3])),
                    };
                }
            }
            return result;
        }

        pub fn storePdS(probabilities: Probabilities, operands: *const Operands) void {
            const word = thread.lane() % 4;
            inline for (0..score_rows * columns_per_warp / 8) |index| {
                const row_tile = index / (columns_per_warp / 8);
                const tile = index % (columns_per_warp / 8);
                inline for (.{ Storage.pTile(), Storage.dsTile() }, .{
                    probabilities.p[row_tile][tile], probabilities.ds[row_tile][tile],
                }) |destination, pair| {
                    inline for (0..2) |half| {
                        const row = row_tile * 16 + half * 8;
                        const words: [*]addrspace(.shared) u32 =
                            @ptrCast(operands.pds_store.chunk(destination, row, tile));
                        words[word] = pair[half];
                    }
                }
            }
        }

        pub const DkvAccumulators = struct {
            dk: [row_tiles][tiles_kv]Accumulator,
            dv: [row_tiles][tiles_kv]Accumulator,
        };

        pub fn zeroDkv() DkvAccumulators {
            return .{ .dk = @splat(@splat(@splat(0))), .dv = @splat(@splat(@splat(0))) };
        }

        /// Accumulates dK and dV, and adds `query_gradient`, if it is not void, to `dq_accum` in
        /// parts after the products of each step.
        pub fn accumulateDkv(
            accumulators: *DkvAccumulators,
            _: *const Storage.Load,
            iteration: u32,
            operands: *const Operands,
            query_gradient: anytype,
            rows: u32,
        ) void {
            @setEvalBranchQuota(100_000);
            const buffer = iteration % stages;
            const k_steps_kv = block_m / 16;
            const interleaved = @TypeOf(query_gradient) != void;
            inline for (0..k_steps_kv) |k_step| {
                const skips = tiling.skips_padding_rows and !interleaved and k_step > 0;
                if (!skips or k_step * 16 < rows) {
                    var dv_a: [row_tiles]@Vector(4, u32) = undefined;
                    var dk_a: [row_tiles]@Vector(4, u32) = undefined;
                    inline for (&dv_a, &dk_a, 0..) |*p, *ds, tile| {
                        const columns = operands.pds_columns;
                        const row = k_step * 16;
                        const column = tile * 2;
                        const p_chunk = columns.chunk(Storage.pTile(), row, column);
                        const ds_chunk = columns.chunk(Storage.dsTile(), row, column);
                        p.* = nvptx.ldmatrix(.x4, true, p_chunk);
                        ds.* = nvptx.ldmatrix(.x4, true, ds_chunk);
                    }
                    const gradient = operands.gradient;
                    const do_tile = Storage.doTile(buffer);
                    const q_tile = Storage.qTile(buffer);
                    transposedGemm(&accumulators.dv, tiles_kv, dv_a, gradient, do_tile, k_step, 0);
                    transposedGemm(&accumulators.dk, tiles_kv, dk_a, gradient, q_tile, k_step, 0);
                }
                if (@TypeOf(query_gradient) != void) {
                    const reductions = @TypeOf(query_gradient.*).reductions;
                    const part = (reductions + k_steps_kv - 1) / k_steps_kv;
                    query_gradient.add(k_step * part, part);
                }
            }
        }

        pub fn drainDq() void {
            if (tiling.bulk_reduce) Storage.DqAccum.waitFree();
        }

        pub fn accumulateDq(
            params: BackwardParams,
            work: *const Work,
            step: Step,
            load: *const Storage.Load,
            _: u32,
            operands: *const Operands,
        ) void {
            @setEvalBranchQuota(100_000);
            const warp_m = thread.warp() % warps_dq_m;
            const warp_d = thread.warp() / warps_dq_m;
            const lane = thread.lane();
            var a: [block_n / 16][row_tiles]@Vector(4, u32) = undefined;
            inline for (&a, 0..) |*step_a, k_step| {
                inline for (step_a, 0..) |*fragment, tile| fragment.* = nvptx.ldmatrix(
                    .x4,
                    false,
                    operands.ds_rows.chunk(Storage.dsTile(), tile * 16, k_step * 2),
                );
            }
            if (tiling.bulk_reduce) {
                const gradient = Storage.DqAccum;
                const tiles_per_box = gradient.box_columns / 8;
                const row_base = work.sequence.offset_q +
                    if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
                const row = row_base + step.block_info.first_row + warp_m * 16;
                inline for (0..tiles_dq / tiles_per_box) |box| {
                    const first_tile = warp_d * tiles_dq + @as(u32, box) * tiles_per_box;
                    var dq: [1][tiles_per_box]Accumulator = @splat(@splat(@splat(0)));
                    inline for (a, 0..) |fragment, k_step| {
                        transposedGemm(
                            &dq,
                            tiles_per_box,
                            fragment,
                            operands.query_gradient,
                            Storage.kTile(),
                            k_step,
                            box * tiles_per_box,
                        );
                    }
                    gradient.waitFree();
                    inline for (dq[0], 0..) |fragment, tile| {
                        const column = @as(u32, tile) * 8 + lane % 4 * 2;
                        gradient.store(lane / 4, column, .{ fragment[0], fragment[1] });
                        gradient.store(lane / 4 + 8, column, .{ fragment[2], fragment[3] });
                    }
                    if (first_tile * 8 < head_dim) gradient.publish(
                        load,
                        first_tile * 8,
                        step.head,
                        row,
                    );
                }
                return;
            }
            if (fragment_ordered_dq_accum) {
                return Storage.Load.Steps(@This()).reduceDq(params, work, step, operands, a);
            }
            var dq: [row_tiles][tiles_dq]Accumulator = @splat(@splat(@splat(0)));
            inline for (a, 0..) |fragment, k_step| {
                transposedGemm(
                    &dq,
                    tiles_dq,
                    fragment,
                    operands.query_gradient,
                    Storage.kTile(),
                    k_step,
                    0,
                );
            }
            const row = step.block_info.first_row + warp_m * 16 * row_tiles + lane / 4;
            const seqlen_q = work.sequence.seqlen_q;
            const first_column = warp_d * tiles_dq * 8;
            const padded = head_dim != tiling.headDimPadded();
            const base: [*]addrspace(.global) f32 = @ptrFromInt(params.dq_accum);
            const dq_head = base + step.head * params.dq_accum_head_stride;
            const stride = params.dq_accum_row_stride;
            const dq_rows = dq_head + work.batch * params.dq_accum_batch_stride +
                @as(u64, work.sequence.offset_q) * stride;
            const odd = lane % 2 == 1;
            const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
            var values: [tiles_dq]Accumulator = undefined;
            inline for (&values, dq[0]) |*value, fragment| {
                const sent_low = if (odd) fragment[0] else fragment[2];
                const sent_high = if (odd) fragment[1] else fragment[3];
                const received_low = nvptx.shflSync(.bfly, f32, sent_low, 1, clamp, .all);
                const received_high = nvptx.shflSync(.bfly, f32, sent_high, 1, clamp, .all);
                value.* = if (odd)
                    .{ received_low, received_high, fragment[2], fragment[3] }
                else
                    .{ fragment[0], fragment[1], received_low, received_high };
            }
            const r = row + if (odd) @as(u32, 8) else 0;
            if (r < seqlen_q) {
                const dq_row = dq_rows + @as(u64, r) * stride + first_column + (lane % 4 & 2) * 2;
                inline for (values, 0..) |value, tile| {
                    if (!padded or first_column + tile * 8 < head_dim) {
                        fragments.reduceAdd(&dq_row[tile * 8], value);
                    }
                }
            }
        }

        fn reducePair(address: *addrspace(.global) f32, pair: @Vector(2, f32)) void {
            if (vector_reductions) {
                fragments.reduceAdd(address, pair);
            } else {
                const words: [*]addrspace(.global) f32 = @ptrCast(address);
                nvptx.red(.gpu, .monotonic, .add, &words[0], pair[0]);
                nvptx.red(.gpu, .monotonic, .add, &words[1], pair[1]);
            }
        }

        fn accumulateGradients(
            params: BackwardParams,
            work: *const Work,
            accumulators: DkvAccumulators,
        ) void {
            const warp_n = thread.warp() % warps_kv_n;
            const warp_d = thread.warp() / warps_kv_n;
            const lane = thread.lane();
            const scale: Accumulator = @splat(params.softmax_scale);
            const dk: [*]addrspace(.global) f32 = @ptrFromInt(params.dk);
            const dv: [*]addrspace(.global) f32 = @ptrFromInt(params.dv);
            if (fragment_ordered_dq_accum) {
                const decode_again = Storage.Load.schedule.decode_work_twice;
                const decoded = if (decode_again) decodeWork(params).? else work.*;
                const final_work = &decoded;
                const group_row = final_work.sequence.offset_k + final_work.first_key +
                    warp_n * row_tiles * 16 + lane / 4 +
                    if (params.cu_seqlens_k == 0) final_work.batch * params.seqlen_k else 0;
                const lane_offset = @as(u64, group_row - group_row % 8) * head_dim +
                    (group_row % 8 * 4 + lane % 4 + warp_d * tiles_kv * 64);
                const dk_row = dk + final_work.kv_head * params.dk_head_stride + lane_offset;
                const dv_row = dv + final_work.kv_head * params.dv_head_stride + lane_offset;
                inline for (0..row_tiles * tiles_kv) |index| {
                    const row_tile: u32 = index / tiles_kv;
                    const tile_index = index % tiles_kv;
                    const column = (warp_d * tiles_kv + @as(u32, tile_index)) * 8;
                    const dk_scaled = accumulators.dk[row_tile][tile_index] * scale;
                    const dv_values = accumulators.dv[row_tile][tile_index];
                    inline for (.{ 0, 1 }) |half| {
                        const offset = row_tile * 16 + half * 8;
                        const row = warp_n * row_tiles * 16 + lane / 4 + offset;
                        if (row < final_work.key_count and column < head_dim) {
                            inline for (0..2) |parity| {
                                const element = offset * head_dim + tile_index * 64 + parity * 32;
                                const value = half * 2 + parity;
                                const dk_value = dk_scaled[value];
                                const dv_value = dv_values[value];
                                nvptx.red(.gpu, .monotonic, .add, &dk_row[element], dk_value);
                                nvptx.red(.gpu, .monotonic, .add, &dv_row[element], dv_value);
                            }
                        }
                    }
                }
                return;
            }
            const dk_origin =
                work.batch * params.dk_batch_stride + work.kv_head * params.dk_head_stride;
            const dv_origin =
                work.batch * params.dv_batch_stride + work.kv_head * params.dv_head_stride;
            inline for (0..row_tiles * tiles_kv) |index| {
                const row_tile: u32 = index / tiles_kv;
                const tile_index = index % tiles_kv;
                const column = (warp_d * tiles_kv + @as(u32, tile_index)) * 8 + lane % 4 * 2;
                const dk_scaled = accumulators.dk[row_tile][tile_index] * scale;
                const dv_values = accumulators.dv[row_tile][tile_index];
                inline for (.{ 0, 1 }) |half| {
                    const row = (warp_n * row_tiles + row_tile) * 16 + lane / 4 + half * 8;
                    if (row < work.key_count and column < head_dim) {
                        const key_row = work.sequence.offset_k + work.first_key + row;
                        reducePair(&dk[dk_origin + key_row * params.dk_row_stride + column], .{
                            dk_scaled[half * 2], dk_scaled[half * 2 + 1],
                        });
                        reducePair(&dv[dv_origin + key_row * params.dv_row_stride + column], .{
                            dv_values[half * 2], dv_values[half * 2 + 1],
                        });
                    }
                }
            }
        }

        const GradientCopier = struct {
            dk: [*]addrspace(.global) Chunk,
            dv: [*]addrspace(.global) Chunk,
            dk_origin: u64,
            dv_origin: u64,
            params: BackwardParams,
            offset_k: u32,
            work: *const Work,
            warp_row: u32,
            warp_column: u32,

            pub fn copy(copier: GradientCopier, tile_row: u32, tile_column: u32) void {
                const row = copier.warp_row + tile_row;
                const chunk_column = copier.warp_column + tile_column;
                if (row < copier.work.key_count and chunk_column * chunk_elements < head_dim) {
                    const key_row = copier.offset_k + copier.work.first_key + row;
                    copier.dk[
                        (copier.dk_origin +
                            key_row * copier.params.dk_row_stride) / chunk_elements + chunk_column
                    ] = Rows.chunk(Storage.kTile(), row, chunk_column, row).*;
                    copier.dv[
                        (copier.dv_origin +
                            key_row * copier.params.dv_row_stride) / chunk_elements + chunk_column
                    ] = Rows.chunk(Storage.vTile(), row, chunk_column, row).*;
                }
            }
        };

        pub fn storeGradients(
            params: BackwardParams,
            work: *const Work,
            accumulators: DkvAccumulators,
        ) void {
            if (specialization.head_split) return accumulateGradients(params, work, accumulators);
            const warp_n = thread.warp() % warps_kv_n;
            const warp_d = thread.warp() / warps_kv_n;
            const lane = thread.lane();
            const lane_row = lane / 4;
            const sequence = work.sequence;
            const scale: Accumulator = @splat(params.softmax_scale);
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..row_tiles * tiles_kv) |index| {
                const row_tile: u32 = index / tiles_kv;
                const tile_index = index % tiles_kv;
                const chunk_column = warp_d * tiles_kv + @as(u32, tile_index);
                const dk_scaled = accumulators.dk[row_tile][tile_index] * scale;
                const dv_values = accumulators.dv[row_tile][tile_index];
                inline for (.{ 0, 1 }) |half| {
                    const staged_row = (warp_n * row_tiles + row_tile) * 16 + lane_row + half * 8;
                    const dk_words: [*]addrspace(.shared) u32 =
                        @ptrCast(Rows.chunk(Storage.kTile(), staged_row, chunk_column, lane_row));
                    const dv_words: [*]addrspace(.shared) u32 =
                        @ptrCast(Rows.chunk(Storage.vTile(), staged_row, chunk_column, lane_row));
                    dk_words[lane % 4] = @backingInt(
                        nvptx.Bf16x2.fromF32(dk_scaled[half * 2], dk_scaled[half * 2 + 1]),
                    );
                    dv_words[lane % 4] = @backingInt(
                        nvptx.Bf16x2.fromF32(dv_values[half * 2], dv_values[half * 2 + 1]),
                    );
                }
            }
            nvptx.barWarpSync(.all);
            const dk: [*]addrspace(.global) Chunk = @ptrFromInt(params.dk);
            const dv: [*]addrspace(.global) Chunk = @ptrFromInt(params.dv);
            const dk_origin =
                work.batch * params.dk_batch_stride + work.kv_head * params.dk_head_stride;
            const dv_origin =
                work.batch * params.dv_batch_stride + work.kv_head * params.dv_head_stride;
            const copier: GradientCopier = .{
                .dk = dk,
                .dv = dv,
                .dk_origin = dk_origin,
                .dv_origin = dv_origin,
                .params = params,
                .offset_k = sequence.offset_k,
                .work = work,
                .warp_row = warp_n * 16 * row_tiles,
                .warp_column = warp_d * tiles_kv,
            };
            smem.copyStagedRows(16 * row_tiles, tiles_kv, 32, lane, copier);
        }

        pub const Kernel = FlashAttnBwdSm80Kernel(@This());
    };
}

const variant = configurations.variants(.flash_bwd_sm80, config.head_dim)[config.variant];
const backward_tiling = variant.tiling;
const Backward = FlashAttnBwdSm80(
    backward_tiling,
    variant.specialization,
    SharedStorage(backward_tiling, backward_tiling.schedule),
);
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");
const auxiliary: flash_bwd_preprocess.Options = .{
    .varlen_as_one_sequence = variant.specialization.scheduler_metadata,
    .fragment_ordered_dq_accum = fragment_ordered_dq_accum,
};

const Entry = entry.Entry(BackwardParams, Backward.Args, .{
    "q", "k", "v", "do", "lse_log2", "dpsum", "dq_accum", "dk", "dv",
} ++ entry.backward_sequence ++ entry.strides("q") ++ entry.strides("k") ++ entry.strides("v") ++
    entry.strides("do") ++ entry.strides("dk") ++ entry.strides("dv") ++
    entry.strides("dq_accum") ++ .{ "lse_batch_stride", "lse_head_stride", "num_batch" } ++
    (if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}) ++
    if (variant.specialization.inter_sequence) .{ "visibility_bits", "window_sizes" } else .{}, .{
    .required_work_group_size = .{ .x = Backward.thread_count },
}, Backward.Kernel.run);

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_bwd" });
        @export(
            &flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess,
            .{ .name = "flash_attn_bwd_preprocess" },
        );
        @export(
            &flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary)
                .flashAttnBwdPostprocess,
            .{ .name = "flash_attn_bwd_postprocess" },
        );
        if (variant.specialization.scheduler_metadata) {
            @export(&Backward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_bwd_prepare_varlen_num_blocks",
            });
        }
        if (variant.specialization.head_split) @export(
            &flash_bwd_preprocess.flashAttnBwdClear,
            .{ .name = "flash_attn_bwd_clear" },
        );
    }
}
