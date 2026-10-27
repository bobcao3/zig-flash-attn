const std = @import("std");
const BackwardParams = @import("BackwardParams.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const tile_scheduler = @import("tile_scheduler.zig");
const BlockInfo = @import("BlockInfo.zig");
const inter_sequence = @import("inter_sequence.zig");

pub const Options = struct {
    inter_sequence: bool = false,
};

pub fn NBlockWork(comptime block_m: u32, comptime block_n: u32, comptime options: Options) type {
    return struct {
        batch: u32,
        kv_head: u32,
        qhead_per_kvhead: u32,
        first_qhead: u32,
        sequence: SeqlenInfoQK,
        first_key: u32,
        key_count: u32,
        m_block_min: u32,
        m_block_count: u32,
        block_info: BlockInfo,
        query_batch: if (options.inter_sequence) u32 else void,

        pub const Step = struct {
            head: u32,
            block_info: BlockInfo,
        };

        /// Walks the steps of the current query segment and then of each later one.
        pub const Walk = struct {
            segment: Work,
            position: u32 = 0,

            /// Returns the next step; `segment` then describes the query segment of that step.
            pub fn next(walk: *Walk, params: BackwardParams) Step {
                if (options.inter_sequence and walk.position == walk.segment.iterations()) {
                    _ = walk.segment.nextSegment(params);
                    walk.position = 0;
                }
                defer walk.position += 1;
                return walk.segment.branchedStep(walk.position);
            }
        };

        pub const RowLimits = struct {
            top: i32,
            bot: i32,

            pub fn contains(limits: RowLimits, row: u32) bool {
                const value: i32 = @intCast(row);
                return value >= limits.top and value < limits.bot;
            }
        };

        const Work = @This();

        pub fn init(
            params: BackwardParams,
            tile: tile_scheduler.WorkTileInfo,
            split_idx: u32,
            num_qhead_splits: u32,
        ) Work {
            const sequence = params.sequence(tile.batch);
            const first_key = tile.block * block_n;
            const key_count = @min(block_n, sequence.seqlen_k - first_key);
            const info: BlockInfo = .{
                .first_row = 0,
                .row_count = sequence.seqlen_q,
                .seqlen_q = sequence.seqlen_q,
                .seqlen_k = sequence.seqlen_k,
                .window_size_left = params.window_size_left,
                .window_size_right = params.window_size_right,
            };
            const m_block_min, const m_block_max = info.mBlockMinMax(first_key, key_count, block_m);
            const qhead_per_kvhead = params.qhead_per_kvhead / num_qhead_splits;
            var work: Work = .{
                .batch = tile.batch,
                .kv_head = tile.head,
                .qhead_per_kvhead = qhead_per_kvhead,
                .first_qhead = tile.head * params.qhead_per_kvhead + split_idx * qhead_per_kvhead,
                .sequence = sequence,
                .first_key = first_key,
                .key_count = key_count,
                .m_block_min = m_block_min,
                .m_block_count = m_block_max - m_block_min,
                .block_info = info,
                .query_batch = if (options.inter_sequence) tile.batch else {},
            };
            if (options.inter_sequence) _ = work.findSegment(params, 0);
            return work;
        }

        pub fn queryBatch(work: *const Work) u32 {
            return if (options.inter_sequence) work.query_batch else work.batch;
        }

        pub fn nextSegment(work: *Work, params: BackwardParams) bool {
            return work.findSegment(params, work.query_batch + 1);
        }

        fn findSegment(work: *Work, params: BackwardParams, from: u32) bool {
            comptime std.debug.assert(options.inter_sequence);
            const count = params.num_batch;
            const row: inter_sequence.BitRow = .init(params.visibility_bits, count, work.batch);
            const cu_seqlens_q: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_q);
            var next_sequence = from;
            while (true) {
                const sequence = row.find(next_sequence, true);
                if (sequence >= count) {
                    work.m_block_count = 0;
                    return false;
                }
                next_sequence = sequence + 1;
                const window: inter_sequence.Window = if (sequence == work.batch)
                    inter_sequence.sequenceWindow(params.window_sizes, count, sequence, .{
                        .left = params.window_size_left,
                        .right = params.window_size_right,
                    })
                else
                    .unbounded;
                const offset_q: u32 = @intCast(cu_seqlens_q[sequence]);
                work.sequence.offset_q = offset_q;
                work.sequence.seqlen_q = @as(u32, @intCast(cu_seqlens_q[sequence + 1])) - offset_q;
                work.query_batch = sequence;
                work.block_info = .{
                    .first_row = 0,
                    .row_count = work.sequence.seqlen_q,
                    .seqlen_q = work.sequence.seqlen_q,
                    .seqlen_k = work.sequence.seqlen_k,
                    .window_size_left = window.left,
                    .window_size_right = window.right,
                };
                const m_block_min, const m_block_max =
                    work.block_info.mBlockMinMax(work.first_key, work.key_count, block_m);
                work.m_block_min = m_block_min;
                work.m_block_count = m_block_max - m_block_min;
                if (work.m_block_count > 0) return true;
            }
        }

        pub fn rowLimits(work: *const Work, key: u32) RowLimits {
            if (key >= work.sequence.seqlen_k) return .{ .top = 0, .bot = 0 };
            return .{
                .top = work.block_info.rowLimitTop(@intCast(key)),
                .bot = work.block_info.rowLimitBot(@intCast(key)),
            };
        }

        pub fn unmaskedRowLimits(work: *const Work) RowLimits {
            if (work.key_count < block_n) return .{ .top = 0, .bot = 0 };
            return .{
                .top = work.rowLimits(work.first_key + block_n - 1).top,
                .bot = work.rowLimits(work.first_key).bot - @as(i32, block_m - 1),
            };
        }

        pub fn unmaskedBlocks(work: *const Work) struct { u32, u32 } {
            const limits = work.unmaskedRowLimits();
            const first: i32 = @divFloor(limits.top + (block_m - 1), block_m);
            const end: i32 =
                if (limits.bot > 0) @divFloor(limits.bot + (block_m - 1), block_m) else 0;
            const m_block_min: i32 = @intCast(work.m_block_min);
            const m_block_count: i32 = @intCast(work.m_block_count);
            const begin = std.math.clamp(first - m_block_min, 0, m_block_count);
            return .{ @intCast(begin), @intCast(
                std.math.clamp(end - m_block_min, begin, m_block_count),
            ) };
        }
        pub fn iterations(work: *const Work) u32 {
            return work.m_block_count * work.qhead_per_kvhead;
        }

        /// The iterations of the current query segment and of every later one.
        pub fn remainingIterations(work: Work, params: BackwardParams) u32 {
            var segment = work;
            var total = segment.iterations();
            if (options.inter_sequence) {
                while (segment.nextSegment(params)) total += segment.iterations();
            }
            return total;
        }

        pub fn step(work: *const Work, iteration: u32) Step {
            const first_row = (work.m_block_min + iteration % work.m_block_count) * block_m;
            return .{
                .head = work.first_qhead + iteration / work.m_block_count,
                .block_info = .{
                    .first_row = first_row,
                    .row_count = @min(block_m, work.sequence.seqlen_q - first_row),
                    .seqlen_q = work.sequence.seqlen_q,
                    .seqlen_k = work.sequence.seqlen_k,
                    .window_size_left = work.block_info.window_size_left,
                    .window_size_right = work.block_info.window_size_right,
                },
            };
        }

        pub fn branchedStep(work: *const Work, iteration: u32) Step {
            if (work.qhead_per_kvhead > 1) return work.step(iteration);
            const first_row = (work.m_block_min + iteration) * block_m;
            return .{
                .head = work.first_qhead,
                .block_info = .{
                    .first_row = first_row,
                    .row_count = @min(block_m, work.sequence.seqlen_q - first_row),
                    .seqlen_q = work.sequence.seqlen_q,
                    .seqlen_k = work.sequence.seqlen_k,
                    .window_size_left = work.block_info.window_size_left,
                    .window_size_right = work.block_info.window_size_right,
                },
            };
        }
    };
}
