const std = @import("std");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const applyMask = @import("mask.zig").applyMask;
const BlockInfo = @import("BlockInfo.zig");
const inter_sequence = @import("inter_sequence.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");
const profile = @import("profile.zig");
const Zone = @import("flash_fwd_sm80.zig").Zone;
const FastDivmodDivisor = @import("tile_scheduler.zig").FastDivmodDivisor;
const WorkTileInfo = @import("tile_scheduler.zig").WorkTileInfo;
const large_head_dim = @import("inter_sequence_large_head_dim.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const BlockTable = inter_sequence.BlockTable;
const persistent = configurations.inter_sequence_persistent;
const split_limits = configurations.inter_sequence_splits;
const block_n = persistent.block_n;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const cta_threads = persistent.cta_warps * nvptx.warp_size;

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
    visibility_bits: u64 = 0,
    window_sizes: u64 = 0,
    window_size_left: i32 = 0,
    window_size_right: i32 = 0,
    softmax_scale_log2: f32 = 0,
    num_sequences: u32 = 0,
    num_head: u32 = 0,
    qhead_per_kvhead: u32 = 0,
    block_table_m64: u64 = 0,
    block_table_m128: u64 = 0,
    sequence_records: u64 = 0,
    section_heads: u32 = 0,
    table_capacity: u32 = 0,
    o_accum: u64 = 0,
    o_accum_splits: u32 = 0,
    num_sm: u32 = 0,
    profile: u64 = 0,
};

pub const Args = struct {};
pub const names = names: {
    const fields = std.meta.fieldNames(Params);
    std.debug.assert(std.mem.eql(u8, fields[fields.len - 1], "profile"));
    break :names fields[0 .. fields.len - 1];
};
pub const Trace = profile.Recorder(Zone, .{});

extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
pub var next_tile_idx: [2][2]u32 addrspace(.shared) = undefined;
var arrived_splits: u32 addrspace(.shared) = undefined;
pub var tile_scheduler: TileScheduler addrspace(.shared) = undefined;
const warp_group_barriers = [2]nvptx.NamedBarrier{ @fromBackingInt(1), @fromBackingInt(2) };

pub const TileCountSemaphore = packed struct(u64) {
    fetched_tiles: u32,
    finished_ctas: u32,

    fn add(semaphore: *addrspace(.global) u64, increment: TileCountSemaphore) TileCountSemaphore {
        const previous = nvptx.atom(.gpu, .monotonic, .add, semaphore, @backingInt(increment));
        return @fromBackingInt(previous);
    }

    pub fn finishCta(params: Params) void {
        const m128 = &BlockTable.at(params.block_table_m128).tile_count_semaphore;
        const m64 = &BlockTable.at(params.block_table_m64).tile_count_semaphore;
        nvptx.fence(.seq_cst, .gpu);
        const finished = add(m128, .{ .fetched_tiles = 0, .finished_ctas = 1 }).finished_ctas;
        if (finished + 1 < nvptx.nctaid(0)) return;
        nvptx.fence(.seq_cst, .gpu);
        m128.* = 0;
        m64.* = 0;
    }
};

pub const NextWork = struct {
    leader: bool,
    tile_idx: *addrspace(.shared) u32,

    pub fn prefetch(next: NextWork, params: Params, comptime queue: TileScheduler.Queue) void {
        if (!next.leader) return;
        const table = if (queue == .cta) tile_scheduler.cta_table else params.block_table_m64;
        const semaphore = &BlockTable.at(table).tile_count_semaphore;
        const increment: TileCountSemaphore = .{ .fetched_tiles = 1, .finished_ctas = 0 };
        const fetched = TileCountSemaphore.add(semaphore, increment).fetched_tiles;
        next.tile_idx.* = @min(fetched, tile_scheduler.tileCount(queue));
    }
};

pub fn Pipeline(comptime stages: u32) type {
    return struct {
        write: State = .{},
        read: State = .{},

        pub var k_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        pub var v_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        pub var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub const State = struct {
            index: u32 = 0,
            phase: u32 = 0,

            pub fn advance(state: *State) void {
                const wraps = state.index + 1 == stages;
                state.index = if (wraps) 0 else state.index + 1;
                state.phase ^= @intFromBool(wraps);
            }
        };

        pub fn initBarriers(threads: u32, warps: u32) void {
            if (thread.id() == 0) {
                for (&k_full, &v_full, &empty) |*k, *v, *stage_empty| {
                    k.init(threads);
                    v.init(threads);
                    stage_empty.init(warps);
                }
                mbarrier.pipelineInitFence();
            }
            nvptx.barrierSyncAligned(.work_group, null);
        }
    };
}

const KvPipeline = Pipeline(persistent.stages);

pub fn Kernel(comptime head_dim: u32, comptime packs: bool) type {
    if (persistent.largeHeadDim(head_dim)) return large_head_dim.Kernel(head_dim, packs);
    return struct {
        const CtaWorker = TileWorker(head_dim, persistent.cta_warps, packs);
        const WarpGroupWorker = TileWorker(head_dim, configurations.warps_per_warp_group, packs);

        pub const threads = cta_threads;
        pub const shared_bytes = persistent.sharedBytes(head_dim);

        comptime {
            std.debug.assert(shared_bytes == CtaWorker.shared_bytes);
            std.debug.assert(shared_bytes == 2 * WarpGroupWorker.shared_bytes);
            std.debug.assert(CtaWorker.threads == threads);
            std.debug.assert(2 * WarpGroupWorker.threads == threads);
        }

        pub fn run(params: Params, _: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            trace.begin(.setup);
            if (thread.warp() == 0) {
                const scheduler: TileScheduler =
                    .init(params, packs, TileScheduler.cta_block_m, true);
                if (thread.lane() == 0) tile_scheduler = scheduler;
            }
            KvPipeline.initBarriers(threads, persistent.cta_warps);
            trace.end(.setup);
            const cta: CtaWorker = .{ .index = 0 };
            cta.processTiles(params, &trace);
            nvptx.barrierSyncAligned(.work_group, null);
            const warp_group: WarpGroupWorker = .{ .index = thread.warp() / WarpGroupWorker.warps };
            warp_group.processTiles(params, &trace);
            nvptx.barrierSyncAligned(.work_group, null);
            if (thread.id() == 0) TileCountSemaphore.finishCta(params);
        }
    };
}

pub const Work = struct {
    tile: WorkTileInfo,
    summary: BlockTable.Summary,
    split_idx: u32 = 0,
    num_splits: u32 = 1,
    split_block: u32 = 0,

    pub fn splitNBlocks(work: Work, block_m: u32, n_block_size: u32) [2]u32 {
        const work_unit = BlockTable.work_unit;
        const n_blocks =
            work.summary.block_work / (block_m / work_unit) * (work_unit / n_block_size);
        const first = n_blocks * work.split_idx / work.num_splits;
        if (work.split_idx + 1 == work.num_splits) return .{ first, std.math.maxInt(u32) };
        return .{ first, n_blocks * (work.split_idx + 1) / work.num_splits - first };
    }
};

const BlockRange = struct {
    first: u32,
    end: u32,
    heads: u32,
    section_tiles: FastDivmodDivisor,
    section_heads: FastDivmodDivisor,

    fn init(first: u32, end: u32, heads: u32, section_heads: u32) BlockRange {
        return .{
            .first = first,
            .end = end,
            .heads = heads,
            .section_tiles = .init((end - first) * section_heads),
            .section_heads = .init(section_heads),
        };
    }

    fn tileCount(range: BlockRange) u32 {
        return (range.end - range.first) * range.heads;
    }

    fn locate(
        range: BlockRange,
        params: Params,
        table: u64,
        block_m: u32,
        packing: u32,
        tile_idx: u32,
    ) Work {
        const heads = range.section_heads.divisor;
        const section_tiles = (range.end - range.first) * heads;
        const section = range.section_tiles.divide(tile_idx);
        const local = range.first * heads + tile_idx - section * section_tiles;
        const block = range.section_heads.divide(local);
        const entry = BlockTable.entries(table)[block];
        const summary = BlockTable.summaries(table, params.table_capacity)[block];
        const blocks = (summary.rows * packing + block_m - 1) / block_m;
        const within = local - entry.first_block * heads;
        return .{
            .tile = .{
                .block = blocks - 1 - within / heads,
                .head = within % heads + section * heads,
                .batch = entry.sequence,
            },
            .summary = summary,
        };
    }
};

const Span = struct {
    first: u32,
    end: u32,

    const all: Span = .{ .first = 0, .end = block_n / 16 };

    inline fn holds(span: Span, index: u32) bool {
        return index >= span.first and index < span.end;
    }
};

pub const TileScheduler = struct {
    cta_blocks: BlockRange,
    cta_table: u64,
    block_m: u32,
    first_split_tile: u32,
    num_split_blocks: u32,
    num_splits: u32,
    halved_tiles: u32,
    qhead_per_kvhead: FastDivmodDivisor,
    packing: FastDivmodDivisor,

    pub const Queue = enum { cta, warp_group };
    pub const cta_block_m = persistent.cta_warps * 16;
    pub const warp_group_block_m = configurations.warps_per_warp_group * 16;

    pub fn init(
        params: Params,
        comptime packs: bool,
        comptime block_m: u32,
        comptime has_warp_group_queue: bool,
    ) TileScheduler {
        const cta_table =
            if (block_m == cta_block_m) params.block_table_m128 else params.block_table_m64;
        const cta_table_blocks = BlockTable.at(cta_table).blocks;
        const ctas = nvptx.nctaid(0);
        const packing = if (packs) params.qhead_per_kvhead else 1;
        const heads = params.num_head / packing;
        const section_heads = @max(1, params.section_heads / packing);
        var scheduler: TileScheduler = .{
            .cta_blocks = .init(0, cta_table_blocks, heads, section_heads),
            .cta_table = cta_table,
            .block_m = block_m,
            .qhead_per_kvhead = .init(params.qhead_per_kvhead),
            .packing = .init(packing),
            .first_split_tile = 0,
            .num_split_blocks = 0,
            .num_splits = 1,
            .halved_tiles = 0,
        };
        scheduler.planSplits(params, ctas);
        if (has_warp_group_queue and scheduler.num_split_blocks == 0) {
            const range = scheduler.cta_blocks;
            const cta_tiles = range.tileCount();
            const last_wave = cta_tiles % ctas;
            if (last_wave > 0 and 2 * last_wave <= ctas) {
                const section_tiles = range.end * section_heads;
                const block = (cta_tiles - last_wave) % section_tiles / section_heads;
                const rows = BlockTable.summaries(cta_table, params.table_capacity)[block].rows;
                if (2 * rows * packing > block_m) scheduler.halved_tiles = last_wave;
            }
        }
        return scheduler;
    }

    fn planSplits(scheduler: *TileScheduler, params: Params, ctas: u32) void {
        const range = scheduler.cta_blocks;
        if (range.end == 0 or params.o_accum_splits == 0) return;
        const section_heads = range.section_heads.divisor;
        const section_tiles = range.end * section_heads;
        const last_section = range.tileCount() - section_tiles;
        const summaries = BlockTable.summaries(scheduler.cta_table, params.table_capacity);
        var low: u32 = 0;
        var high = range.end;
        while (low < high) {
            const stride = (high - low + nvptx.warp_size - 1) / nvptx.warp_size;
            const probe = low + thread.lane() * stride;
            const tested = probe < high;
            const short = tested and summaries[probe].block_work < split_limits.min_block_work;
            const found = nvptx.voteBallotSync(short, .all);
            if (found != 0) {
                const lane: u32 = @ctz(found);
                high = low + lane * stride;
                if (lane > 0) low += (lane - 1) * stride + 1;
            } else {
                const last: u32 = 31 - @clz(nvptx.voteBallotSync(tested, .all));
                low += last * stride + 1;
            }
        }
        const long_end = last_section + low * section_heads;
        const first = long_end / ctas * ctas;
        const num_split_blocks = long_end - first;
        if (num_split_blocks == 0) return;
        const remaining = scheduler.ctaWorkBefore(params, range.tileCount()) -
            scheduler.ctaWorkBefore(params, first);
        const summary = summaries[first % section_tiles / section_heads];
        if (2 * summary.rows * scheduler.packing.divisor < scheduler.block_m) return;
        if (remaining >= @as(u64, ctas) * summary.block_work) return;
        var num_splits: u32 = 1;
        var rounds: u32 = 1;
        var count: u32 = 2;
        const max_splits = @min(split_limits.max_splits, params.o_accum_splits / num_split_blocks);
        while (count <= max_splits) {
            const count_rounds = (num_split_blocks * count + ctas - 1) / ctas;
            if (count_rounds <= 2 and count_rounds * num_splits < rounds * count) {
                num_splits = count;
                rounds = count_rounds;
            }
            count += 1;
        }
        if (num_splits < 2) return;
        scheduler.first_split_tile = first;
        scheduler.num_split_blocks = num_split_blocks;
        scheduler.num_splits = num_splits;
    }

    fn ctaWorkBefore(scheduler: TileScheduler, params: Params, tile_idx: u32) u64 {
        const range = scheduler.cta_blocks;
        const table = scheduler.cta_table;
        const capacity = params.table_capacity;
        const heads = range.section_heads.divisor;
        const section_tiles = range.end * heads;
        const section = tile_idx / section_tiles;
        const local = tile_idx - section * section_tiles;
        const block = local / heads;
        const within = local - block * heads;
        const section_work = BlockTable.workBefore(table, capacity, range.end);
        const blocks_work = section * section_work + BlockTable.workBefore(table, capacity, block);
        var work = @as(u64, heads) * blocks_work;
        if (block < range.end) {
            work += within * BlockTable.summaries(table, capacity)[block].block_work;
        }
        return work;
    }

    pub fn tileCount(scheduler: TileScheduler, comptime queue: Queue) u32 {
        return switch (queue) {
            .cta => scheduler.cta_blocks.tileCount() - scheduler.halved_tiles +
                (scheduler.num_splits - 1) * scheduler.num_split_blocks,
            .warp_group => 2 * scheduler.halved_tiles,
        };
    }

    pub fn locate(
        scheduler: TileScheduler,
        params: Params,
        comptime queue: Queue,
        tile_idx: u32,
    ) ?Work {
        const packing = scheduler.packing.divisor;
        if (queue == .cta) {
            const first_split_tile = scheduler.first_split_tile;
            if (tile_idx < first_split_tile) return scheduler.ctaWork(params, tile_idx);
            const split_tile = tile_idx - first_split_tile;
            if (split_tile < scheduler.num_splits * scheduler.num_split_blocks) {
                const split_block = split_tile / scheduler.num_splits;
                var work = scheduler.ctaWork(params, first_split_tile + split_block);
                work.split_idx = split_tile - split_block * scheduler.num_splits;
                work.num_splits = scheduler.num_splits;
                work.split_block = split_block;
                return work;
            }
            const whole_tile = tile_idx - (scheduler.num_splits - 1) * scheduler.num_split_blocks;
            return scheduler.ctaWork(params, whole_tile);
        }
        const cta_tile = scheduler.cta_blocks.tileCount() - scheduler.halved_tiles + tile_idx / 2;
        var work = scheduler.ctaWork(params, cta_tile);
        work.tile.block = 2 * work.tile.block + tile_idx % 2;
        if (work.tile.block * warp_group_block_m >= work.summary.rows * packing) return null;
        return work;
    }

    fn ctaWork(scheduler: TileScheduler, params: Params, tile_idx: u32) Work {
        const packing = scheduler.packing.divisor;
        const table = scheduler.cta_table;
        return scheduler.cta_blocks.locate(params, table, scheduler.block_m, packing, tile_idx);
    }
};

pub fn NBlockCursor(comptime n_block_size: u32) type {
    return struct {
        sequence: u32,
        next_run: u32,
        search_from: u32,
        run_first_key: u32,
        block_info: BlockInfo,
        col_limit_bounds: BlockInfo.ColLimitBounds,
        n_block: u32,
        n_block_max: u32,
        n_block_budget: u32,

        const Cursor = @This();
        const SequenceRecord = inter_sequence.SequenceRecord;

        pub fn init(params: Params, work: Work, first_row: u32, row_count: u32) Cursor {
            var cursor: Cursor = .{
                .sequence = work.tile.batch,
                .next_run = 0,
                .search_from = 0,
                .run_first_key = 0,
                .block_info = .{
                    .first_row = first_row,
                    .row_count = row_count,
                    .seqlen_q = work.summary.rows,
                    .seqlen_k = 0,
                    .window_size_left = BlockInfo.window_size_unbounded,
                    .window_size_right = BlockInfo.window_size_unbounded,
                },
                .col_limit_bounds = undefined,
                .n_block = 0,
                .n_block_max = 0,
                .n_block_budget = std.math.maxInt(u32),
            };
            cursor.findRun(params);
            return cursor;
        }

        pub fn valid(cursor: *const Cursor) bool {
            return cursor.n_block < cursor.n_block_max;
        }

        pub fn advance(cursor: *Cursor, params: Params) void {
            cursor.n_block += 1;
            if (cursor.n_block == cursor.n_block_max) cursor.findRun(params);
        }

        pub fn skip(cursor: *Cursor, params: Params, count: u32) void {
            var left = count;
            while (left > 0 and cursor.valid()) {
                const step = @min(left, cursor.n_block_max - cursor.n_block);
                cursor.n_block += step - 1;
                cursor.advance(params);
                left -= step;
            }
        }

        pub fn limit(cursor: *Cursor, n_blocks: u32) void {
            const taken = @min(cursor.n_block_max - cursor.n_block, n_blocks);
            cursor.n_block_max = cursor.n_block + taken;
            cursor.n_block_budget = n_blocks - taken;
        }

        fn findRun(cursor: *Cursor, params: Params) void {
            if (cursor.n_block_budget == 0) {
                cursor.n_block_max = cursor.n_block;
                return;
            }
            const record = SequenceRecord.at(params.sequence_records, cursor.sequence);
            const last_run = SequenceRecord.max_key_runs - 1;
            const key_run_count = record.key_run_count;
            const record_window = record.window;
            var run = record.key_runs[@min(cursor.next_run, last_run)];
            while (cursor.next_run < key_run_count) {
                cursor.next_run += 1;
                const window: inter_sequence.Window = if (run.banded) record_window else .unbounded;
                if (cursor.setRun(run.first_key, run.first_key + run.keys, window)) return;
                run = record.key_runs[@min(cursor.next_run, last_run)];
            }
            if (cursor.next_run == key_run_count) {
                cursor.next_run += 1;
                cursor.search_from = record.resume_from;
            }
            if (cursor.search_from < params.num_sequences) {
                const cu_k: [*]addrspace(.global) const i32 = @ptrFromInt(params.cu_seqlens_k);
                const runs: inter_sequence.SequenceRuns = .init(
                    params.visibility_bits,
                    params.window_sizes,
                    .{ .left = params.window_size_left, .right = params.window_size_right },
                    @ptrFromInt(params.cu_seqlens_q),
                    cu_k,
                    params.num_sequences,
                    cursor.sequence,
                );
                while (runs.next(cursor.search_from)) |found| {
                    cursor.search_from = found.end;
                    const first_key: u32 = @intCast(cu_k[found.first]);
                    const end_key: u32 = @intCast(cu_k[found.end]);
                    if (cursor.setRun(first_key, end_key, runs.windowOf(found))) return;
                }
                cursor.search_from = params.num_sequences;
            }
            cursor.n_block = 0;
            cursor.n_block_max = 0;
        }

        fn setRun(
            cursor: *Cursor,
            first_key: u32,
            end_key: u32,
            window: inter_sequence.Window,
        ) bool {
            cursor.block_info.seqlen_k = end_key - first_key;
            cursor.block_info.window_size_left = window.left;
            cursor.block_info.window_size_right = window.right;
            const n_block_min, const n_block_max = cursor.block_info.nBlockMinMax(n_block_size);
            if (n_block_max <= n_block_min) return false;
            const taken = @min(n_block_max - n_block_min, cursor.n_block_budget);
            cursor.n_block_budget -= taken;
            cursor.run_first_key = first_key;
            cursor.n_block = n_block_min;
            cursor.n_block_max = n_block_min + taken;
            cursor.col_limit_bounds = cursor.block_info.colLimitBounds();
            return true;
        }

        pub fn firstKey(cursor: *const Cursor) u32 {
            return cursor.run_first_key + cursor.n_block * n_block_size;
        }

        pub fn keysLeft(cursor: *const Cursor) u32 {
            return cursor.block_info.seqlen_k - cursor.n_block * n_block_size;
        }

        pub fn needsMask(cursor: *const Cursor) bool {
            const bounds = cursor.col_limit_bounds;
            return cursor.block_info.needsMask(bounds, cursor.n_block, n_block_size);
        }
    };
}

pub fn QueryRows(comptime packs: bool) type {
    return struct {
        first_query: u32,
        first_row: u32,
        head: u32,
        count: u32,

        const Rows = @This();

        pub fn init(work: Work, block_m: u32) Rows {
            const packing = if (packs) tile_scheduler.packing.divisor else 1;
            const first_row = work.tile.block * block_m;
            return .{
                .first_query = work.summary.first_query,
                .first_row = first_row,
                .head = work.tile.head * packing,
                .count = @min(block_m, work.summary.rows * packing - first_row),
            };
        }

        pub fn tokenCount(rows: Rows) u32 {
            return rows.token(rows.count - 1) + 1 - rows.token(0);
        }

        pub fn token(rows: Rows, row: u32) u32 {
            if (!packs) return rows.first_row + row;
            return tile_scheduler.packing.divide(rows.first_row + row);
        }

        pub fn headOf(rows: Rows, row: u32) u32 {
            if (!packs) return rows.head;
            const packing = tile_scheduler.packing;
            const packed_row = rows.first_row + row;
            return rows.head + packed_row - packing.divide(packed_row) * packing.divisor;
        }

        pub fn offset(rows: Rows, row: u32, row_stride: u64, head_stride: u64) u64 {
            const query: u64 = rows.first_query + rows.token(row);
            return query * row_stride + rows.headOf(row) * head_stride;
        }
    };
}

pub fn maskScores(
    scores: anytype,
    block_info: BlockInfo,
    first_column: u32,
    rows: anytype,
    row: u32,
) void {
    const tokens: @Vector(2, i32) = .{
        @intCast(rows.token(row)),
        @intCast(rows.token(row + 8)),
    };
    const limits: BlockInfo.ColLimits = .{
        .col_limit_left = .{
            block_info.colLimitLeft(tokens[0]),
            block_info.colLimitLeft(tokens[1]),
        },
        .col_limit_right = .{
            block_info.colLimitRight(tokens[0]),
            block_info.colLimitRight(tokens[1]),
        },
    };
    applyMask(scores, first_column, limits, -std.math.inf(f32));
}

pub fn storeHalfRows(
    comptime head_dim: u32,
    params: Params,
    softmax: anytype,
    output: anytype,
    rows: anytype,
    first_row: u32,
    hdim_half: u32,
) void {
    const finalized = softmax.finalize(params.softmax_scale_log2);
    const lane = thread.lane();
    const o: [*]addrspace(.global) u32 = @ptrFromInt(params.o);
    inline for (.{ 0, 1 }) |row_half| {
        const row = first_row + lane / 4 + row_half * 8;
        if (row < rows.count) {
            const offset = rows.offset(row, params.o_row_stride, params.o_head_stride);
            inline for (output, 0..) |fragment, n_tile| {
                const column = hdim_half * (head_dim / 2) + n_tile * 8 + lane % 4 * 2;
                o[(offset + column) / 2] = @backingInt(nvptx.Bf16x2.fromF32(
                    fragment[row_half * 2] * finalized.row_scale[row_half],
                    fragment[row_half * 2 + 1] * finalized.row_scale[row_half],
                ));
            }
            if (hdim_half == 0 and lane % 4 == 0) {
                const lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                const index = rows.headOf(row) * params.lse_head_stride +
                    rows.first_query + rows.token(row);
                lse[index] = finalized.lse[row_half];
            }
        }
    }
}

pub fn combineSplits(
    params: Params,
    softmax: anytype,
    output: anytype,
    work: Work,
    trace: *Trace,
) bool {
    trace.begin(.combine);
    defer trace.end(.combine);
    const OAccum = [output.len + 1][cta_threads]Accumulator;
    const o_accum: [*]addrspace(.global) OAccum = @ptrFromInt(params.o_accum);
    const first_split = work.split_block * work.num_splits;
    const own = &o_accum[first_split + work.split_idx];
    const id = thread.id();
    const caching: nvptx.cache.StoreCaching = .{ .operator = .cache_global_level };
    inline for (output.*, 0..) |fragment, chunk| {
        nvptx.cache.store(caching, &own[chunk][id], fragment);
    }
    const state: Accumulator = .{
        softmax.row_max[0], softmax.row_max[1], softmax.row_sum[0], softmax.row_sum[1],
    };
    nvptx.cache.store(caching, &own[output.len][id], state);
    nvptx.fence(.seq_cst, .gpu);
    nvptx.barrierSyncAligned(.work_group, null);
    if (id == 0) {
        const split_counters: [*]addrspace(.global) u32 = @ptrFromInt(params.o_accum +
            @as(u64, params.o_accum_splits) * @sizeOf(OAccum));
        arrived_splits = nvptx.atom(.gpu, .monotonic, .add, &split_counters[work.split_block], 1);
    }
    nvptx.barrierSyncAligned(.work_group, null);
    if (arrived_splits + 1 < work.num_splits) return false;
    nvptx.fence(.seq_cst, .gpu);
    const loading: nvptx.cache.LoadCaching = .{ .operator = .cache_global_level };
    softmax.* = .init();
    output.* = @splat(@splat(0));
    for (0..work.num_splits) |split_idx| {
        const other = &o_accum[first_split + split_idx];
        const other_state = nvptx.cache.load(loading, &other[output.len][id]);
        const own_scale, const other_scale = softmax.combine(.{
            .row_max = .{ other_state[0], other_state[1] },
            .row_sum = .{ other_state[2], other_state[3] },
        }, params.softmax_scale_log2);
        inline for (output, 0..) |*fragment, chunk| {
            const vector = nvptx.cache.load(loading, &other[chunk][id]);
            fragment.* = fragment.* * own_scale + vector * other_scale;
        }
    }
    return true;
}

fn TileWorker(comptime head_dim: u32, comptime worker_warps: u32, comptime packs: bool) type {
    const row_chunks = head_dim / chunk_elements;
    const Rows = smem.Layout(row_chunks);
    const KvTile = Rows.Tile(block_n);
    const KvAddresses = Rows.PairAddresses(block_n);
    const Online = fragments.Softmax(block_n);
    const Query = [head_dim / 16]@Vector(4, u32);
    const Output = [row_chunks]Accumulator;
    const Cursor = NBlockCursor(block_n);
    return struct {
        index: u32,

        const Worker = @This();
        pub const warps = worker_warps;
        pub const threads = warps * nvptx.warp_size;
        pub const block_m = warps * 16;
        const tile_buffers = persistent.tile_buffers * warps / persistent.cta_warps;
        const whole_cta = warps == persistent.cta_warps;
        const stages = persistent.stages;
        const QueryTile = Rows.Tile(block_m);
        pub const shared_bytes = tile_buffers * @sizeOf(KvTile);
        const slices_small_blocks = whole_cta and !persistent.capsRegisters(head_dim);
        const skips_key_groups = !persistent.capsRegisters(head_dim);
        const queue: TileScheduler.Queue = if (whole_cta) .cta else .warp_group;

        comptime {
            std.debug.assert(tile_buffers == stages * @as(u32, if (whole_cta) 2 else 1));
        }

        fn threadId(worker: Worker) u32 {
            return thread.id() - worker.index * threads;
        }

        fn warp(worker: Worker) u32 {
            return thread.warp() - worker.index * warps;
        }

        fn tileBuffer(worker: Worker, buffer: u32) *addrspace(.shared) KvTile {
            const tiles: [*]addrspace(.shared) KvTile = @ptrCast(@alignCast(&shared_storage));
            return &tiles[worker.index * tile_buffers + buffer];
        }

        fn kTile(worker: Worker, stage: u32) *addrspace(.shared) KvTile {
            return worker.tileBuffer(2 * stage);
        }

        fn vTile(worker: Worker, stage: u32) *addrspace(.shared) KvTile {
            return worker.tileBuffer(2 * stage + 1);
        }

        fn queryTile(worker: Worker, buffer: u32) *addrspace(.shared) QueryTile {
            return @ptrCast(worker.tileBuffer(buffer));
        }

        const Load = struct {
            cursor: Cursor,
            copier: RowCopy,
            keys: RowSource,
            values: RowSource,
            next: NextWork,

            fn advance(load: *Load, params: Params) void {
                load.cursor.advance(params);
                if (!load.cursor.valid()) load.next.prefetch(params, queue);
            }

            fn copyRows(
                load: *Load,
                comptime kind: enum { keys, values },
                destination: *addrspace(.shared) KvTile,
            ) void {
                if (!load.cursor.valid()) return;
                const source = if (kind == .keys) load.keys else load.values;
                const first_key = load.cursor.firstKey();
                load.copier.copy(destination, source, first_key, load.cursor.keysLeft());
            }

            fn copyTile(
                load: *Load,
                worker: Worker,
                params: Params,
                write: *KvPipeline.State,
            ) void {
                if (!load.cursor.valid()) return;
                const first_key = load.cursor.firstKey();
                const keys_left = load.cursor.keysLeft();
                const stage = write.index;
                mbarrier.wait(.cta, &KvPipeline.empty[stage], write.phase ^ 1);
                load.copier.copy(worker.kTile(stage), load.keys, first_key, keys_left);
                mbarrier.arrive(.cp_async, &KvPipeline.k_full[stage]);
                load.copier.copy(worker.vTile(stage), load.values, first_key, keys_left);
                mbarrier.arrive(.cp_async, &KvPipeline.v_full[stage]);
                load.advance(params);
                write.advance();
            }
        };

        fn processTiles(worker: Worker, params: Params, trace: *Trace) void {
            const tile_count = tile_scheduler.tileCount(queue);
            const leader = worker.threadId() == 0;
            var parity: u1 = 0;
            var pipeline: KvPipeline = .{};
            trace.begin(.get_next_work);
            const first: NextWork = .{
                .leader = leader,
                .tile_idx = &next_tile_idx[worker.index][parity],
            };
            first.prefetch(params, queue);
            worker.sync();
            var tile_idx = first.tile_idx.*;
            trace.end(.get_next_work);
            while (tile_idx < tile_count) {
                parity +%= 1;
                const next: NextWork = .{
                    .leader = leader,
                    .tile_idx = &next_tile_idx[worker.index][parity],
                };
                var published = false;
                trace.begin(.tile);
                trace.begin(.locate);
                const located = tile_scheduler.locate(params, queue, tile_idx);
                trace.end(.locate);
                if (located) |work| {
                    published = worker.process(params, work, &pipeline, next, trace);
                } else next.prefetch(params, queue);
                trace.end(.tile);
                trace.begin(.get_next_work);
                if (!published) worker.sync();
                trace.end(.get_next_work);
                tile_idx = next.tile_idx.*;
            }
        }

        fn sync(worker: Worker) void {
            if (whole_cta) return nvptx.barrierSyncAligned(.work_group, null);
            switch (worker.index) {
                0 => nvptx.barrierSyncAligned(warp_group_barriers[0], threads),
                else => nvptx.barrierSyncAligned(warp_group_barriers[1], threads),
            }
        }

        const RowCopy = struct {
            slot: u32,
            row: u32,
            column: u32,

            const row_threads = Rows.swizzle_columns;
            const rows_per_pass = threads / row_threads;
            const thread_columns = row_chunks / row_threads;

            comptime {
                std.debug.assert(row_chunks % row_threads == 0 and rows_per_pass % 8 == 0);
                std.debug.assert(block_n % rows_per_pass == 0);
            }

            fn init(worker: Worker) RowCopy {
                const row = worker.threadId() / row_threads;
                const column = worker.threadId() % row_threads;
                return .{ .slot = Rows.slot(row, column), .row = row, .column = column };
            }

            fn copy(
                copier: RowCopy,
                destination: *addrspace(.shared) KvTile,
                source: RowSource,
                first_row: u32,
                valid_rows: u32,
            ) void {
                const rows_left = @as(i32, @bitCast(valid_rows)) - @as(i32, @bitCast(copier.row));
                var address = source.first + first_row * source.row_bytes;
                inline for (0..block_n / rows_per_pass) |pass| {
                    const pass_row: u32 = pass * rows_per_pass;
                    const size: u32 = if (pass_row < rows_left) @sizeOf(Chunk) else 0;
                    inline for (0..thread_columns) |thread_column| {
                        const column: u32 = thread_column * row_threads;
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

        fn copyQueries(
            copier: RowCopy,
            params: Params,
            destination: *addrspace(.shared) QueryTile,
            rows: QueryRows(packs),
        ) void {
            const element_bytes = @sizeOf(Chunk) / chunk_elements;
            inline for (0..block_m / RowCopy.rows_per_pass) |pass| {
                const pass_row: u32 = pass * RowCopy.rows_per_pass;
                const row = copier.row + pass_row;
                const valid = row < rows.count;
                const source_row = if (valid) row else 0;
                const offset = rows.offset(source_row, params.q_row_stride, params.q_head_stride);
                const address = params.q + element_bytes * offset + copier.column * @sizeOf(Chunk);
                inline for (0..RowCopy.thread_columns) |thread_column| {
                    const column: u32 = thread_column * RowCopy.row_threads;
                    const slot = copier.slot + comptime Rows.slot(pass_row, column);
                    const chunk: *addrspace(.global) const Chunk =
                        @ptrFromInt(address + column * @sizeOf(Chunk));
                    cp_async.copyZeroFill(
                        .cache_global_level,
                        &destination[slot],
                        chunk,
                        if (valid) @sizeOf(Chunk) else 0,
                    );
                }
            }
        }

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
            k: KvAddresses,
            v: KvAddresses,

            fn init() Lanes {
                const lane = thread.lane();
                return .{
                    .k = .init(lane % 8 + lane / 16 * 8, lane / 8 % 2),
                    .v = .init(lane % 8 + lane / 8 % 2 * 8, lane / 16),
                };
            }
        };

        fn loadQuery(tile: *addrspace(.shared) QueryTile, first_row: u32) Query {
            const lane = thread.lane();
            const addresses = Rows.PairAddresses(block_m).init(first_row + lane % 16, lane / 16);
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                fragment.* = nvptx.ldmatrix(.x4, false, addresses.chunk(tile, 0, k_step * 2));
            }
            return query;
        }

        inline fn qkScores(
            query: Query,
            k_tile: *addrspace(.shared) KvTile,
            k_addresses: KvAddresses,
            visible: Span,
        ) Online.Scores {
            @setEvalBranchQuota(100_000);
            var scores: Online.Scores = @splat(@splat(0));
            inline for (query, 0..) |fragment, k_step| {
                inline for (0..block_n / 16) |j| {
                    if (visible.holds(j)) {
                        const b = nvptx.ldmatrix(
                            .x4,
                            false,
                            k_addresses.chunk(k_tile, j * 16, k_step * 2),
                        );
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
            }
            return scores;
        }

        fn visibleGroups(worker: Worker, cursor: *const Cursor, rows: QueryRows(packs)) Span {
            const row = worker.warp() * 16;
            if (row >= rows.count) return .{ .first = 0, .end = 0 };
            const first_key: i32 = @intCast(cursor.n_block * block_n);
            const last_row = @min(row + 15, rows.count - 1);
            const col_limit_left = cursor.block_info.colLimitLeft(@intCast(rows.token(row)));
            const col_limit_right =
                cursor.block_info.colLimitRight(@intCast(rows.token(last_row)));
            const first = @max(0, @divFloor(col_limit_left - first_key, 16));
            const end = @min(block_n / 16, @divFloor(col_limit_right - first_key + 15, 16));
            return .{ .first = @intCast(first), .end = @intCast(@max(first, end)) };
        }

        inline fn accumulateOutput(
            output: *Output,
            step: Online.Step,
            v_tile: *addrspace(.shared) KvTile,
            v_addresses: KvAddresses,
            visible: Span,
        ) void {
            @setEvalBranchQuota(100_000);
            if (step.should_rescale) {
                inline for (output) |*fragment| fragment.* *= step.row_scale;
            }
            inline for (0..block_n / 16) |k_step| {
                if (visible.holds(k_step)) inline for (0..row_chunks / 2) |column_pair| {
                    const b = nvptx.ldmatrix(
                        .x4,
                        true,
                        v_addresses.chunk(v_tile, k_step * 16, column_pair * 2),
                    );
                    const probabilities = step.probabilities[k_step];
                    output[2 * column_pair] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[0], b[1] },
                        output[2 * column_pair],
                    );
                    output[2 * column_pair + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[2], b[3] },
                        output[2 * column_pair + 1],
                    );
                };
            }
        }

        const SliceOnline = fragments.Softmax(16);
        const SliceOutput = [row_chunks / 2]Accumulator;
        const SliceStates = [warps][row_chunks / 2 + 1][nvptx.warp_size]Accumulator;

        fn processSlicedK(
            worker: Worker,
            params: Params,
            rows: QueryRows(packs),
            pipeline: *KvPipeline,
            load: *Load,
            cursor: *Cursor,
            query_tile: *addrspace(.shared) QueryTile,
            trace: *Trace,
        ) void {
            load.copyTile(worker, params, &pipeline.write);
            load.copyTile(worker, params, &pipeline.write);
            cp_async.commitGroup();
            cp_async.waitGroup(1);
            worker.sync();
            const query = loadQuery(query_tile, 0);
            worker.sync();
            trace.end(.load_q);
            const lane = thread.lane();
            const key_slice = worker.warp() % 4;
            const hdim_half = worker.warp() / 4;
            const key_row = key_slice * 16 + lane % 8;
            const k_addresses: KvAddresses = .init(key_row + lane / 16 * 8, lane / 8 % 2);
            const v_addresses: KvAddresses =
                .init(key_row + lane / 8 % 2 * 8, hdim_half * (row_chunks / 2) + lane / 16);
            var softmax = SliceOnline.init();
            var output: SliceOutput = @splat(@splat(0));
            while (cursor.valid()) {
                load.copyTile(worker, params, &pipeline.write);
                const stage = pipeline.read.index;
                const phase = pipeline.read.phase;
                trace.begin(.scores);
                trace.begin(.wait_k);
                mbarrier.wait(.cta, &KvPipeline.k_full[stage], phase);
                trace.end(.wait_k);
                trace.begin(.qk_mma);
                var scores: SliceOnline.Scores = @splat(@splat(0));
                const k_tile = worker.kTile(stage);
                inline for (query, 0..) |fragment, k_step| {
                    const b = nvptx.ldmatrix(
                        .x4,
                        false,
                        k_addresses.chunk(k_tile, 0, k_step * 2),
                    );
                    scores[0] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[0], b[1] },
                        scores[0],
                    );
                    scores[1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[2], b[3] },
                        scores[1],
                    );
                }
                trace.end(.qk_mma);
                if (cursor.needsMask()) {
                    const first_column = cursor.n_block * block_n + key_slice * 16;
                    maskScores(&scores, cursor.block_info, first_column, rows, lane / 4);
                }
                trace.end(.scores);
                trace.begin(.softmax);
                const step = SliceOnline.step(&softmax, scores, params.softmax_scale_log2);
                trace.end(.softmax);
                trace.begin(.output);
                trace.begin(.wait_v);
                mbarrier.wait(.cta, &KvPipeline.v_full[stage], phase);
                trace.end(.wait_v);
                trace.begin(.pv_mma);
                if (step.should_rescale) {
                    inline for (&output) |*fragment| fragment.* *= step.row_scale;
                }
                const v_tile = worker.vTile(stage);
                inline for (0..row_chunks / 4) |column_pair| {
                    const b = nvptx.ldmatrix(
                        .x4,
                        true,
                        v_addresses.chunk(v_tile, 0, column_pair * 2),
                    );
                    output[2 * column_pair] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        step.probabilities[0],
                        @Vector(2, u32){ b[0], b[1] },
                        output[2 * column_pair],
                    );
                    output[2 * column_pair + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        step.probabilities[0],
                        @Vector(2, u32){ b[2], b[3] },
                        output[2 * column_pair + 1],
                    );
                }
                trace.end(.pv_mma);
                trace.end(.output);
                if (lane == 0) _ = KvPipeline.empty[stage].arrive();
                pipeline.read.advance();
                cursor.advance(params);
            }
            trace.begin(.get_next_work);
            worker.sync();
            trace.end(.get_next_work);
            trace.begin(.store);
            const states: *addrspace(.shared) SliceStates =
                @ptrCast(@alignCast(worker.tileBuffer(0)));
            const own = &states[worker.warp()];
            inline for (output, 0..) |fragment, chunk| own[chunk][lane] = fragment;
            own[row_chunks / 2][lane] = .{
                softmax.row_max[0], softmax.row_max[1], softmax.row_sum[0], softmax.row_sum[1],
            };
            worker.sync();
            if (key_slice == 0) {
                for (1..4) |other_slice| {
                    const other = &states[worker.warp() + other_slice];
                    const state = other[row_chunks / 2][lane];
                    const own_scale, const other_scale = softmax.combine(.{
                        .row_max = .{ state[0], state[1] },
                        .row_sum = .{ state[2], state[3] },
                    }, params.softmax_scale_log2);
                    inline for (&output, 0..) |*fragment, chunk| {
                        fragment.* = fragment.* * own_scale + other[chunk][lane] * other_scale;
                    }
                }
                storeHalfRows(head_dim, params, softmax, output, rows, 0, hdim_half);
            }
            worker.sync();
            trace.end(.store);
        }

        fn storeOutput(
            worker: Worker,
            params: Params,
            softmax: Online,
            output: Output,
            rows: QueryRows(packs),
            tile: *addrspace(.shared) QueryTile,
        ) void {
            const finalized = Online.finalize(softmax, params.softmax_scale_log2);
            const lane = thread.lane();
            const warp_row = worker.warp() * 16;
            inline for (output, 0..) |fragment, n_tile| {
                inline for (.{ 0, 1 }) |row_half| {
                    const words: [*]addrspace(.shared) u32 = @ptrCast(Rows.chunk(
                        tile,
                        warp_row + lane / 4 + row_half * 8,
                        n_tile,
                        lane / 4,
                    ));
                    words[lane % 4] = @backingInt(nvptx.Bf16x2.fromF32(
                        fragment[row_half * 2] * finalized.row_scale[row_half],
                        fragment[row_half * 2 + 1] * finalized.row_scale[row_half],
                    ));
                }
            }
            nvptx.barWarpSync(.all);
            const o: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
            inline for (0..16 * row_chunks / nvptx.warp_size) |pass| {
                const pass_first: u32 = pass * nvptx.warp_size;
                const index = pass_first + lane;
                const tile_row = index / row_chunks;
                const column = index % row_chunks;
                const row = warp_row + tile_row;
                if (row < rows.count) {
                    const offset = rows.offset(row, params.o_row_stride, params.o_head_stride);
                    o[offset / chunk_elements + column] =
                        Rows.chunk(tile, row, column, tile_row).*;
                }
            }
            if (lane % 4 == 0) {
                const lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                inline for (.{ 0, 1 }) |row_half| {
                    const row = warp_row + lane / 4 + row_half * 8;
                    if (row < rows.count) {
                        const index = rows.headOf(row) * params.lse_head_stride +
                            rows.first_query + rows.token(row);
                        lse[index] = finalized.lse[row_half];
                    }
                }
            }
        }

        fn process(
            worker: Worker,
            params: Params,
            work: Work,
            pipeline: *KvPipeline,
            next: NextWork,
            trace: *Trace,
        ) bool {
            const rows: QueryRows(packs) = .init(work, block_m);
            const kv_head = tile_scheduler.qhead_per_kvhead.divide(rows.head);
            const copier: RowCopy = .init(worker);
            const free_stage = (pipeline.read.index + stages - 1) % stages;
            const query_buffer = if (whole_cta) 2 * free_stage else tile_buffers - 1;
            const query_tile = worker.queryTile(query_buffer);
            trace.begin(.load_q);
            copyQueries(copier, params, query_tile, rows);
            cp_async.commitGroup();
            trace.begin(.locate);
            var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
            trace.end(.locate);
            if (whole_cta and work.num_splits > 1) {
                const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                cursor.skip(params, first_n_block);
                cursor.limit(n_blocks);
            }
            var load: Load = .{
                .cursor = cursor,
                .copier = copier,
                .keys = .init(
                    copier,
                    params.k,
                    kv_head * params.k_head_stride,
                    params.k_row_stride,
                ),
                .values = .init(
                    copier,
                    params.v,
                    kv_head * params.v_head_stride,
                    params.v_row_stride,
                ),
                .next = next,
            };
            if (!load.cursor.valid()) next.prefetch(params, queue);
            const lanes: Lanes = .init();
            const has_rows = worker.warp() * 16 < rows.count;
            var softmax = Online.init();
            var output: Output = @splat(@splat(0));
            if (whole_cta) {
                if (slices_small_blocks and rows.count <= 16 and work.num_splits == 1) {
                    worker.processSlicedK(
                        params,
                        rows,
                        pipeline,
                        &load,
                        &cursor,
                        query_tile,
                        trace,
                    );
                    return true;
                }
                load.copyTile(worker, params, &pipeline.write);
                cp_async.commitGroup();
                cp_async.waitGroup(1);
                worker.sync();
                const query = loadQuery(query_tile, worker.warp() * 16);
                worker.sync();
                trace.end(.load_q);
                const leader = thread.lane() == 0;
                while (cursor.valid()) {
                    load.copyTile(worker, params, &pipeline.write);
                    const stage = pipeline.read.index;
                    const phase = pipeline.read.phase;
                    trace.begin(.scores);
                    trace.begin(.wait_k);
                    mbarrier.wait(.cta, &KvPipeline.k_full[stage], phase);
                    trace.end(.wait_k);
                    const masked = cursor.needsMask();
                    if (has_rows and (!masked or sees: {
                        const visible = worker.visibleGroups(&cursor, rows);
                        break :sees visible.first < visible.end;
                    })) {
                        trace.begin(.qk_mma);
                        var scores = qkScores(query, worker.kTile(stage), lanes.k, .all);
                        trace.end(.qk_mma);
                        if (masked) {
                            const first_column = cursor.n_block * block_n;
                            const row = worker.warp() * 16 + thread.lane() / 4;
                            maskScores(&scores, cursor.block_info, first_column, rows, row);
                        }
                        trace.end(.scores);
                        trace.begin(.softmax);
                        const step = Online.step(&softmax, scores, params.softmax_scale_log2);
                        trace.end(.softmax);
                        trace.begin(.output);
                        trace.begin(.wait_v);
                        mbarrier.wait(.cta, &KvPipeline.v_full[stage], phase);
                        trace.end(.wait_v);
                        trace.begin(.pv_mma);
                        accumulateOutput(&output, step, worker.vTile(stage), lanes.v, .all);
                        trace.end(.pv_mma);
                        trace.end(.output);
                    } else {
                        trace.end(.scores);
                        trace.begin(.wait_v);
                        mbarrier.wait(.cta, &KvPipeline.v_full[stage], phase);
                        trace.end(.wait_v);
                    }
                    if (leader) _ = KvPipeline.empty[stage].arrive();
                    pipeline.read.advance();
                    cursor.advance(params);
                }
                trace.begin(.get_next_work);
                worker.sync();
                trace.end(.get_next_work);
                if (work.num_splits > 1) {
                    if (!combineSplits(params, &softmax, &output, work, trace)) return true;
                }
                trace.begin(.store);
                const staging = worker.queryTile(2 * ((pipeline.write.index + 1) % stages));
                worker.storeOutput(params, softmax, output, rows, staging);
                trace.end(.store);
                return true;
            }
            load.copyRows(.keys, worker.tileBuffer(0));
            cp_async.commitGroup();
            load.copyRows(.values, worker.tileBuffer(1));
            cp_async.commitGroup();
            if (load.cursor.valid()) load.advance(params);
            cp_async.waitGroup(2);
            worker.sync();
            const query = loadQuery(query_tile, worker.warp() * 16);
            trace.end(.load_q);
            var k_buffer: u32 = 0;
            while (cursor.valid()) {
                const v_buffer = if (k_buffer == tile_buffers - 1) 0 else k_buffer + 1;
                const next_k_buffer = if (k_buffer == 0) tile_buffers - 1 else k_buffer - 1;
                trace.begin(.scores);
                trace.begin(.wait_k);
                cp_async.waitGroup(1);
                worker.sync();
                trace.end(.wait_k);
                load.copyRows(.keys, worker.tileBuffer(next_k_buffer));
                cp_async.commitGroup();
                const masked = cursor.needsMask();
                const partial = skips_key_groups and (masked or !has_rows);
                const visible = if (partial) worker.visibleGroups(&cursor, rows) else Span.all;
                trace.begin(.qk_mma);
                const k_tile = worker.tileBuffer(k_buffer);
                var scores = if (partial)
                    qkScores(query, k_tile, lanes.k, visible)
                else
                    qkScores(query, k_tile, lanes.k, .all);
                trace.end(.qk_mma);
                if (masked) {
                    const first_column = cursor.n_block * block_n;
                    const row = worker.warp() * 16 + thread.lane() / 4;
                    maskScores(&scores, cursor.block_info, first_column, rows, row);
                }
                trace.end(.scores);
                trace.begin(.wait_v);
                cp_async.waitGroup(1);
                worker.sync();
                trace.end(.wait_v);
                load.copyRows(.values, worker.tileBuffer(k_buffer));
                cp_async.commitGroup();
                const next_cursor = load.cursor;
                if (load.cursor.valid()) load.advance(params);
                trace.begin(.softmax);
                const step = Online.step(&softmax, scores, params.softmax_scale_log2);
                trace.end(.softmax);
                trace.begin(.output);
                trace.begin(.pv_mma);
                const v_tile = worker.tileBuffer(v_buffer);
                if (partial) {
                    accumulateOutput(&output, step, v_tile, lanes.v, visible);
                } else accumulateOutput(&output, step, v_tile, lanes.v, .all);
                trace.end(.pv_mma);
                trace.end(.output);
                cursor = next_cursor;
                k_buffer = next_k_buffer;
            }
            cp_async.waitGroup(0);
            worker.sync();
            trace.begin(.store);
            worker.storeOutput(params, softmax, output, rows, query_tile);
            trace.end(.store);
            return false;
        }
    };
}
