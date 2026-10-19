const std = @import("std");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const applyMask = @import("mask.zig").applyMask;
const BlockInfo = @import("BlockInfo.zig");
const inter_sequence = @import("inter_sequence.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const megakernel = configurations.inter_sequence_megakernel;
const block_n = megakernel.block_n;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const cta_threads = megakernel.wide_warps * nvptx.warp_size;

pub const Params = struct {
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    o: u64 = 0,
    lse: u64 = 0,
    q_row_stride: u64 = 0,
    q_head_stride: u64 = 0,
    k_row_stride: u64 = 0,
    k_head_stride: u64 = 0,
    v_row_stride: u64 = 0,
    v_head_stride: u64 = 0,
    o_row_stride: u64 = 0,
    o_head_stride: u64 = 0,
    lse_head_stride: u64 = 0,
    cu_seqlens_q: u64 = 0,
    cu_seqlens_k: u64 = 0,
    cu_visible_k: u64 = 0,
    visibility_bits: u64 = 0,
    window_sizes: u64 = 0,
    window_size_left: i32 = 0,
    window_size_right: i32 = 0,
    softmax_scale_log2: f32 = 0,
    num_sequences: u32 = 0,
    num_head: u32 = 0,
    qhead_per_kvhead: u32 = 0,
    sequence_order: u64 = 0,
    wide_min_rows: u32 = 0,
    wide_min_blocks: u32 = 0,
};

pub const Args = struct {};
pub const names = std.meta.fieldNames(Params);

extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
var warp_sums: [cta_threads / nvptx.warp_size]u32 addrspace(.shared) = undefined;
const warp_group_barriers = [2]nvptx.NamedBarrier{ @fromBackingInt(1), @fromBackingInt(2) };

pub fn Megakernel(comptime head_dim: u32) type {
    return struct {
        const Wide = Program(head_dim, megakernel.wide_warps);
        const Narrow = Program(head_dim, megakernel.narrow_warps);

        pub const threads = cta_threads;
        pub const shared_bytes = megakernel.sharedBytes(head_dim);

        comptime {
            const narrow_bytes = 2 * Narrow.group_shared_bytes;
            std.debug.assert(shared_bytes == @max(Wide.group_shared_bytes, narrow_bytes));
            std.debug.assert(Wide.group_threads == threads and 2 * Narrow.group_threads == threads);
        }

        pub fn run(params: Params, _: Args) void {
            const shapes = BlockShapes.init(params, Wide.block_m);
            const ctas = nvptx.nctaid(0);
            const wide: Wide = .{ .index = 0 };
            wide.drain(params, shapes, @workGroupId(0), ctas);
            const narrow: Narrow = .{ .index = thread.warp() / Narrow.warps };
            const wide_last_round = shapes.wide_items % ctas;
            const cta = (@workGroupId(0) + ctas - wide_last_round) % ctas;
            narrow.drain(params, shapes, narrow.index * ctas + cta, 2 * ctas);
        }
    };
}

fn ceilDiv(value: u32, divisor: u32) u32 {
    return (value + divisor - 1) / divisor;
}

fn ctaSum(value: u32) u32 {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    var sum = value;
    inline for (.{ 16, 8, 4, 2, 1 }) |distance| {
        sum += nvptx.shflSync(.bfly, u32, sum, distance, clamp, .all);
    }
    if (thread.lane() == 0) warp_sums[thread.warp()] = sum;
    nvptx.barrierSyncAligned(.work_group, null);
    var total: u32 = 0;
    for (warp_sums) |warp_sum| total += warp_sum;
    nvptx.barrierSyncAligned(.work_group, null);
    return total;
}

const BlockShapes = struct {
    wide_min_rows: u32,
    wide_items: u32,

    fn init(params: Params, wide_rows: u32) BlockShapes {
        const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_q);
        var candidates: u32 = 0;
        var sequence = thread.id();
        while (sequence < params.num_sequences) : (sequence += cta_threads) {
            const rows: u32 = @intCast(cu_q[sequence + 1] - cu_q[sequence]);
            if (rows >= params.wide_min_rows) candidates += ceilDiv(rows, wide_rows);
        }
        const blocks = ctaSum(candidates);
        if (blocks < params.wide_min_blocks) {
            return .{ .wide_min_rows = std.math.maxInt(u32), .wide_items = 0 };
        }
        return .{ .wide_min_rows = params.wide_min_rows, .wide_items = blocks * params.num_head };
    }

    fn isWide(shapes: BlockShapes, rows: u32) bool {
        return rows >= shapes.wide_min_rows;
    }
};

const KeyCursor = struct {
    sequence: u32,
    search_from: u32,
    first_key: u32,
    info: BlockInfo,
    bounds: BlockInfo.ColLimitBounds,
    n_block: u32,
    n_block_end: u32,

    fn init(params: Params, sequence: u32, first_row: u32, row_count: u32) KeyCursor {
        const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_q);
        var cursor: KeyCursor = .{
            .sequence = sequence,
            .search_from = 0,
            .first_key = 0,
            .info = .{
                .first_row = first_row,
                .row_count = row_count,
                .seqlen_q = @intCast(cu_q[sequence + 1] - cu_q[sequence]),
                .seqlen_k = 0,
                .window_size_left = BlockInfo.window_size_unbounded,
                .window_size_right = BlockInfo.window_size_unbounded,
            },
            .bounds = undefined,
            .n_block = 0,
            .n_block_end = 0,
        };
        cursor.findRun(params);
        return cursor;
    }

    fn valid(cursor: *const KeyCursor) bool {
        return cursor.n_block < cursor.n_block_end;
    }

    fn advance(cursor: *KeyCursor, params: Params) void {
        cursor.n_block += 1;
        if (cursor.n_block == cursor.n_block_end) cursor.findRun(params);
    }

    fn findRun(cursor: *KeyCursor, params: Params) void {
        const cu_k: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_k);
        const scalar_window: inter_sequence.Window = .{
            .left = params.window_size_left,
            .right = params.window_size_right,
        };
        const runs: inter_sequence.Runs = .init(
            params.visibility_bits,
            params.window_sizes,
            scalar_window,
            @ptrFromInt(params.cu_seqlens_q),
            cu_k,
            params.num_sequences,
            cursor.sequence,
        );
        while (runs.next(cursor.search_from)) |found| {
            cursor.search_from = found.end;
            const first_key: u32 = @intCast(cu_k[found.first]);
            const window = runs.windowOf(found);
            cursor.info.seqlen_k = @as(u32, @intCast(cu_k[found.end])) - first_key;
            cursor.info.window_size_left = window.left;
            cursor.info.window_size_right = window.right;
            const first_block, const end_block = cursor.info.nBlockMinMax(block_n);
            if (end_block > first_block) {
                cursor.first_key = first_key;
                cursor.n_block = first_block;
                cursor.n_block_end = end_block;
                cursor.bounds = cursor.info.colLimitBounds();
                return;
            }
        }
        cursor.n_block = 0;
        cursor.n_block_end = 0;
    }

    fn tileRow(cursor: *const KeyCursor) u32 {
        return cursor.first_key + cursor.n_block * block_n;
    }

    fn tileRows(cursor: *const KeyCursor) u32 {
        return cursor.info.seqlen_k - cursor.n_block * block_n;
    }

    fn needsMask(cursor: *const KeyCursor) bool {
        return cursor.info.needsMask(cursor.bounds, cursor.n_block, block_n);
    }
};

const OutputRows = struct {
    first_query: u32,
    first_row: u32,
    rows: u32,
    head: u32,
};

fn Program(comptime head_dim: u32, comptime group_warps: u32) type {
    const row_chunks = head_dim / chunk_elements;
    const Rows = smem.Layout(row_chunks);
    const KeyTile = Rows.Tile(block_n);
    const KeyAddresses = Rows.PairAddresses(block_n);
    const Online = fragments.Softmax(block_n);
    const Query = [head_dim / 16]@Vector(4, u32);
    const Output = [row_chunks]Accumulator;
    return struct {
        index: u32,

        const Group = @This();
        pub const warps = group_warps;
        pub const group_threads = warps * nvptx.warp_size;
        pub const block_m = warps * 16;
        const QueryTile = Rows.Tile(block_m);
        const Tiles = struct { q: QueryTile, k: KeyTile, v: KeyTile };
        pub const group_shared_bytes = @sizeOf(Tiles);

        fn id(group: Group) u32 {
            return thread.id() - group.index * group_threads;
        }

        fn warp(group: Group) u32 {
            return thread.warp() - group.index * warps;
        }

        fn sync(group: Group) void {
            if (group_threads == cta_threads) return nvptx.barrierSyncAligned(.work_group, null);
            switch (group.index) {
                0 => nvptx.barrierSyncAligned(warp_group_barriers[0], group_threads),
                else => nvptx.barrierSyncAligned(warp_group_barriers[1], group_threads),
            }
        }

        fn tiles(group: Group) *addrspace(.shared) Tiles {
            const bytes: [*]addrspace(.shared) u8 = @ptrCast(&shared_storage);
            return @ptrCast(@alignCast(bytes + group.index * group_shared_bytes));
        }

        fn drain(
            group: Group,
            params: Params,
            shapes: BlockShapes,
            first_item: u32,
            groups: u32,
        ) void {
            const order: [*]addrspace(.global) const u32 = @ptrFromInt(params.sequence_order);
            const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_q);
            const wide_phase = group_threads == cta_threads;
            var position: u32 = 0;
            var first_of_sequence: u32 = 0;
            var item = first_item;
            while (position < params.num_sequences) {
                const sequence = order[position];
                const rows: u32 = @intCast(cu_q[sequence + 1] - cu_q[sequence]);
                const blocks = if (shapes.isWide(rows) == wide_phase) ceilDiv(rows, block_m) else 0;
                const items = blocks * params.num_head;
                if (item - first_of_sequence >= items) {
                    first_of_sequence += items;
                    position += 1;
                    continue;
                }
                const local = item - first_of_sequence;
                const listed: inter_sequence.ListedBlock = .{
                    .sequence = sequence,
                    .block = local % blocks,
                };
                group.process(params, listed, local / blocks);
                group.sync();
                item += groups;
            }
        }

        const RowCopy = struct {
            slot: u32,
            row: u32,
            column: u32,

            const row_threads = Rows.swizzle_columns;
            const rows_per_pass = group_threads / row_threads;
            const thread_columns = row_chunks / row_threads;

            comptime {
                std.debug.assert(row_chunks % row_threads == 0 and rows_per_pass % 8 == 0);
            }

            fn init(group: Group) RowCopy {
                const row = group.id() / row_threads;
                const column = group.id() % row_threads;
                return .{ .slot = Rows.slot(row, column), .row = row, .column = column };
            }

            fn copy(
                copier: RowCopy,
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                source: RowSource,
                first_row: u32,
                valid_rows: u32,
            ) void {
                comptime std.debug.assert(count % rows_per_pass == 0);
                const rows_left = @as(i32, @bitCast(valid_rows)) - @as(i32, @bitCast(copier.row));
                var address = source.first + first_row * source.row_bytes;
                inline for (0..count / rows_per_pass) |pass| {
                    const pass_row: u32 = pass * rows_per_pass;
                    const size: u32 = if (pass_row < rows_left) @sizeOf(Chunk) else 0;
                    inline for (0..thread_columns) |part| {
                        const column: u32 = part * row_threads;
                        const chunk: *addrspace(.global) const Chunk =
                            @ptrFromInt(address + column * @sizeOf(Chunk));
                        cp_async.copyZeroFill(
                            .cache_global_level,
                            &destination[copier.slot + comptime Rows.slot(pass_row, column)],
                            chunk,
                            size,
                        );
                    }
                    address += source.pass_bytes;
                }
            }
        };

        const RowSource = struct {
            first: u64,
            row_bytes: u64,
            pass_bytes: u64,

            fn init(copier: RowCopy, address: u64, offset: u64, stride: u64) RowSource {
                const element_bytes = @sizeOf(Chunk) / chunk_elements;
                const row_bytes = stride * element_bytes;
                return .{
                    .first = address + offset * element_bytes + copier.row * row_bytes +
                        copier.column * @sizeOf(Chunk),
                    .row_bytes = row_bytes,
                    .pass_bytes = RowCopy.rows_per_pass * row_bytes,
                };
            }
        };

        const Lanes = struct {
            keys: KeyAddresses,
            values: KeyAddresses,

            fn init() Lanes {
                const lane = thread.lane();
                return .{
                    .keys = .init(lane % 8 + lane / 16 * 8, lane / 8 % 2),
                    .values = .init(lane % 8 + lane / 8 % 2 * 8, lane / 16),
                };
            }
        };

        fn loadQuery(group: Group, tile: *addrspace(.shared) QueryTile) Query {
            const lane = thread.lane();
            const addresses =
                Rows.PairAddresses(block_m).init(group.warp() * 16 + lane % 16, lane / 16);
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                fragment.* = nvptx.ldmatrix(.x4, false, addresses.chunk(tile, 0, k_step * 2));
            }
            return query;
        }

        fn qkScores(
            query: Query,
            tile: *addrspace(.shared) KeyTile,
            keys: KeyAddresses,
        ) Online.Scores {
            @setEvalBranchQuota(100_000);
            var scores: Online.Scores = @splat(@splat(0));
            inline for (query, 0..) |fragment, k_step| {
                inline for (0..block_n / 16) |j| {
                    const b = nvptx.ldmatrix(.x4, false, keys.chunk(tile, j * 16, k_step * 2));
                    scores[2 * j] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[0], b[1] },
                        scores[2 * j],
                    );
                    scores[2 * j + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[2], b[3] },
                        scores[2 * j + 1],
                    );
                }
            }
            return scores;
        }

        fn maskScores(
            group: Group,
            scores: *Online.Scores,
            info: BlockInfo,
            first_column: u32,
        ) void {
            const row = info.first_row + group.warp() * 16 + thread.lane() / 4;
            const limits: BlockInfo.ColLimits = .init(info, row);
            applyMask(scores, first_column, limits, -std.math.inf(f32));
        }

        fn accumulate(
            output: *Output,
            step: Online.Step,
            tile: *addrspace(.shared) KeyTile,
            values: KeyAddresses,
        ) void {
            @setEvalBranchQuota(100_000);
            if (step.should_rescale) {
                inline for (output) |*fragment| fragment.* *= step.row_scale;
            }
            inline for (0..block_n / 16) |k_step| {
                inline for (0..row_chunks / 2) |pair| {
                    const b = nvptx.ldmatrix(.x4, true, values.chunk(tile, k_step * 16, pair * 2));
                    const probabilities = step.probabilities[k_step];
                    output[2 * pair] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[0], b[1] },
                        output[2 * pair],
                    );
                    output[2 * pair + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[2], b[3] },
                        output[2 * pair + 1],
                    );
                }
            }
        }

        fn storeOutput(
            group: Group,
            params: Params,
            softmax: Online,
            output: Output,
            tile: *addrspace(.shared) QueryTile,
            block: OutputRows,
        ) void {
            const finalized = Online.finalize(softmax, params.softmax_scale_log2);
            const lane = thread.lane();
            const warp_row = group.warp() * 16;
            inline for (output, 0..) |fragment, n_tile| {
                inline for (.{ 0, 1 }) |half| {
                    const words: [*]addrspace(.shared) u32 = @ptrCast(Rows.chunk(
                        tile,
                        warp_row + lane / 4 + half * 8,
                        n_tile,
                        lane / 4,
                    ));
                    words[lane % 4] = @backingInt(nvptx.Bf16x2.fromF32(
                        fragment[half * 2] * finalized.row_scale[half],
                        fragment[half * 2 + 1] * finalized.row_scale[half],
                    ));
                }
            }
            nvptx.barWarpSync(.all);
            const o: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
            const first = o + (block.first_query * params.o_row_stride +
                block.head * params.o_head_stride) / chunk_elements;
            const row_stride = params.o_row_stride / chunk_elements;
            inline for (0..16 * row_chunks / nvptx.warp_size) |pass| {
                const pass_first: u32 = pass * nvptx.warp_size;
                const index = pass_first + lane;
                const tile_row = index / row_chunks;
                const column = index % row_chunks;
                const row = block.first_row + warp_row + tile_row;
                if (row < block.rows) {
                    first[row * row_stride + column] =
                        Rows.chunk(tile, warp_row + tile_row, column, tile_row).*;
                }
            }
            if (lane % 4 == 0) {
                const lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                const base = block.head * params.lse_head_stride + block.first_query;
                const row = block.first_row + warp_row + lane / 4;
                if (row < block.rows) lse[base + row] = finalized.lse[0];
                if (row + 8 < block.rows) lse[base + row + 8] = finalized.lse[1];
            }
        }

        fn process(
            group: Group,
            params: Params,
            listed: inter_sequence.ListedBlock,
            head: u32,
        ) void {
            const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_q);
            const first_query: u32 = @intCast(cu_q[listed.sequence]);
            const rows = @as(u32, @intCast(cu_q[listed.sequence + 1])) - first_query;
            const first_row = listed.block * block_m;
            const row_count = @min(block_m, rows - first_row);
            const kv_head = head / params.qhead_per_kvhead;
            const storage = group.tiles();
            const copier: RowCopy = .init(group);
            const queries: RowSource = .init(
                copier,
                params.q,
                first_query * params.q_row_stride + head * params.q_head_stride,
                params.q_row_stride,
            );
            const keys: RowSource =
                .init(copier, params.k, kv_head * params.k_head_stride, params.k_row_stride);
            const values: RowSource =
                .init(copier, params.v, kv_head * params.v_head_stride, params.v_row_stride);
            copier.copy(block_m, &storage.q, queries, first_row, row_count);
            cp_async.commitGroup();
            var cursor: KeyCursor = .init(params, listed.sequence, first_row, row_count);
            if (cursor.valid()) {
                copier.copy(block_n, &storage.k, keys, cursor.tileRow(), cursor.tileRows());
            }
            cp_async.commitGroup();
            cp_async.waitGroup(0);
            group.sync();
            const query = group.loadQuery(&storage.q);
            const lanes: Lanes = .init();
            var softmax = Online.init();
            var output: Output = @splat(@splat(0));
            while (cursor.valid()) {
                copier.copy(block_n, &storage.v, values, cursor.tileRow(), cursor.tileRows());
                cp_async.commitGroup();
                var scores = qkScores(query, &storage.k, lanes.keys);
                if (cursor.needsMask()) {
                    group.maskScores(&scores, cursor.info, cursor.n_block * block_n);
                }
                cp_async.waitGroup(0);
                group.sync();
                cursor.advance(params);
                if (cursor.valid()) {
                    copier.copy(block_n, &storage.k, keys, cursor.tileRow(), cursor.tileRows());
                }
                cp_async.commitGroup();
                const step = Online.step(&softmax, scores, params.softmax_scale_log2);
                accumulate(&output, step, &storage.v, lanes.values);
                cp_async.waitGroup(0);
                group.sync();
            }
            group.storeOutput(params, softmax, output, &storage.q, .{
                .first_query = first_query,
                .first_row = first_row,
                .rows = rows,
                .head = head,
            });
        }
    };
}
