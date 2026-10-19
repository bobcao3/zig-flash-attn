const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");

pub const Window = extern struct {
    left: i32,
    right: i32,

    pub const unbounded: Window = .{ .left = -1, .right = -1 };

    pub fn bounded(window: Window) bool {
        return window.left >= 0 or window.right >= 0;
    }
};

pub fn sequenceWindow(window_sizes: u64, count: u32, sequence: u32, scalar: Window) Window {
    if (window_sizes == 0) return scalar;
    const sizes: [*]addrspace(.global) const i32 = @ptrFromInt(window_sizes);
    return .{ .left = sizes[sequence], .right = sizes[count + sequence] };
}

pub fn lastBoundaryAtOrBefore(
    cu_seqlens: [*]addrspace(.global) const i32,
    count: u32,
    token: u32,
) u32 {
    var low: u32 = 0;
    var high: u32 = count + 1;
    while (low < high) {
        const middle = (low + high) / 2;
        if (@as(u32, @intCast(cu_seqlens[middle])) <= token) low = middle + 1 else high = middle;
    }
    return low - 1;
}

pub fn wordsPerRow(count: u32) u32 {
    return (count + 31) / 32;
}

pub const BitRow = struct {
    words: [*]addrspace(.global) const u32,
    count: u32,

    pub fn init(bits: u64, count: u32, row: u32) BitRow {
        const matrix: [*]addrspace(.global) const u32 = @ptrFromInt(bits);
        return .{ .words = matrix + row * wordsPerRow(count), .count = count };
    }

    pub fn find(row: BitRow, from: u32, comptime value: bool) u32 {
        var column = from;
        while (column < row.count) {
            const word = row.words[column / 32];
            const bits = (if (value) word else ~word) >> @intCast(column % 32);
            if (bits != 0) return @min(row.count, column + @ctz(bits));
            column = (column / 32 + 1) * 32;
        }
        return row.count;
    }
};

pub const Run = struct {
    first: u32,
    end: u32,
    banded: bool,
};

pub fn nextRun(row: BitRow, from: u32, self: u32, banded_self: bool, joins_previous: bool) ?Run {
    const first = row.find(from, true);
    if (first >= row.count) return null;
    const end = row.find(first, false);
    if (!banded_self or self < first or self >= end) {
        return .{ .first = first, .end = end, .banded = false };
    }
    if (self == first or joins_previous) {
        return .{ .first = first, .end = self + 1, .banded = true };
    }
    return .{ .first = first, .end = self, .banded = false };
}

pub const Runs = struct {
    row: BitRow,
    self: u32,
    window: Window,
    joins_previous: bool,

    pub fn init(
        visibility_bits: u64,
        window_sizes: u64,
        scalar_window: Window,
        cu_seqlens_q: [*]addrspace(.global) const i32,
        cu_seqlens_k: [*]addrspace(.global) const i32,
        count: u32,
        self: u32,
    ) Runs {
        const window = sequenceWindow(window_sizes, count, self, scalar_window);
        const queries = cu_seqlens_q[self + 1] - cu_seqlens_q[self];
        const keys = cu_seqlens_k[self + 1] - cu_seqlens_k[self];
        return .{
            .row = .init(visibility_bits, count, self),
            .self = self,
            .window = window,
            .joins_previous = window.left < 0 and queries <= keys,
        };
    }

    pub fn next(runs: Runs, from: u32) ?Run {
        return nextRun(runs.row, from, runs.self, runs.window.bounded(), runs.joins_previous);
    }

    pub fn windowOf(runs: Runs, run: Run) Window {
        return if (run.banded) runs.window else .unbounded;
    }
};

pub const RunEntry = extern struct { first_key: u32, key_count: u32, window: Window };
pub const several_runs = std.math.maxInt(u32);
pub const ListedBlock = extern struct { sequence: u32, block: u32 };
pub const RunQueue = extern struct { listed_blocks: u32, claims: u32 };

fn runEntry(runs: Runs, cu_seqlens_k: [*]addrspace(.global) const i32, count: u32) RunEntry {
    const nothing: RunEntry = .{ .first_key = 0, .key_count = 0, .window = .unbounded };
    const run = runs.next(0) orelse return nothing;
    if (runs.row.find(run.end, true) < count) {
        return .{ .first_key = 0, .key_count = several_runs, .window = .unbounded };
    }
    const first_key: u32 = @intCast(cu_seqlens_k[run.first]);
    return .{
        .first_key = first_key,
        .key_count = @as(u32, @intCast(cu_seqlens_k[run.end])) - first_key,
        .window = runs.windowOf(run),
    };
}

pub fn RunTable(comptime block_m: u32) type {
    return struct {
        var listed: u32 addrspace(.shared) = undefined;

        pub fn build(
            cu_seqlens_q: u64,
            cu_seqlens_k: u64,
            visibility_bits: u64,
            window_sizes: u64,
            window_size_left: i32,
            window_size_right: i32,
            run_table: u64,
            run_list: u64,
            run_queue: u64,
            count: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = metadata_threads },
        } }) void {
            const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens_q);
            const cu_k: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens_k);
            const entries: [*]addrspace(.global) RunEntry = @ptrFromInt(run_table);
            const list: [*]addrspace(.global) ListedBlock = @ptrFromInt(run_list);
            const scalar_window: Window = .{ .left = window_size_left, .right = window_size_right };
            if (thread.id() == 0) listed = 0;
            nvptx.barrierSyncAligned(.work_group, null);
            var sequence = thread.id();
            while (sequence < count) : (sequence += metadata_threads) {
                const runs: Runs = .init(
                    visibility_bits,
                    window_sizes,
                    scalar_window,
                    cu_q,
                    cu_k,
                    count,
                    sequence,
                );
                const entry = runEntry(runs, cu_k, count);
                entries[sequence] = entry;
                if (entry.key_count == several_runs) {
                    const rows: u32 = @intCast(cu_q[sequence + 1] - cu_q[sequence]);
                    const blocks = (rows + block_m - 1) / block_m;
                    const first = nvptx.atom(.cta, .monotonic, .add, &listed, blocks);
                    for (0..blocks) |block| {
                        list[first + block] = .{ .sequence = sequence, .block = @intCast(block) };
                    }
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (thread.id() == 0) {
                const queue: *addrspace(.global) RunQueue = @ptrFromInt(run_queue);
                queue.* = .{ .listed_blocks = listed, .claims = 0 };
            }
        }
    };
}

const metadata_threads = 1024;
const transpose_warps = 8;

pub fn transposeVisibility(
    visibility_bits: u64,
    visibility_bits_t: u64,
    count: u32,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = transpose_warps * 32 },
} }) void {
    const words = wordsPerRow(count);
    const block = @workGroupId(0) * transpose_warps + thread.warp();
    if (block >= words * words) return;
    const block_row = block / words;
    const block_column = block % words;
    const lane = thread.lane();
    const source: [*]addrspace(.global) const u32 = @ptrFromInt(visibility_bits);
    const target: [*]addrspace(.global) u32 = @ptrFromInt(visibility_bits_t);
    const row = block_row * 32 + lane;
    const word = if (row < count) source[row * words + block_column] else 0;
    var transposed: u32 = 0;
    inline for (0..32) |bit| {
        const column_bits = nvptx.voteBallotSync(word >> bit & 1 != 0, .all);
        if (lane == bit) transposed = column_bits;
    }
    const column = block_column * 32 + lane;
    if (column < count) target[column * words + block_row] = transposed;
}

fn visibleTokens(row: BitRow, cu_seqlens: [*]addrspace(.global) const i32) u32 {
    var total: u32 = 0;
    var from: u32 = 0;
    while (true) {
        const first = row.find(from, true);
        if (first >= row.count) return total;
        const end = row.find(first, false);
        total +|= @intCast(cu_seqlens[end] - cu_seqlens[first]);
        from = end;
    }
}

fn warpInclusiveSum(value: u32) u32 {
    const up = nvptx.ShuffleClamp.fromWidth(.up, 32);
    var sum = value;
    inline for (.{ 1, 2, 4, 8, 16 }) |distance| {
        const previous = nvptx.shflSync(.up, u32, sum, distance, up, .all);
        if (thread.lane() >= distance) sum +|= previous;
    }
    return sum;
}

var warp_totals: [2][metadata_threads / 32]u32 addrspace(.shared) = undefined;

const BlockSums = struct {
    inclusive: [2]u32,
    total: [2]u32,
};

fn blockInclusiveSums(values: [2]u32) BlockSums {
    const idx = nvptx.ShuffleClamp.fromWidth(.idx, 32);
    var result: BlockSums = undefined;
    inline for (&result.inclusive, values, &warp_totals) |*sum, value, *totals| {
        sum.* = warpInclusiveSum(value);
        if (thread.lane() == 31) totals[thread.warp()] = sum.*;
    }
    nvptx.barrierSyncAligned(.work_group, null);
    inline for (&result.inclusive, &result.total, &warp_totals) |*sum, *total, *totals| {
        const warp_total = totals[thread.lane()];
        const warp_inclusive = warpInclusiveSum(warp_total);
        sum.* +|= nvptx.shflSync(.idx, u32, warp_inclusive - warp_total, thread.warp(), idx, .all);
        total.* = nvptx.shflSync(.idx, u32, warp_inclusive, 31, idx, .all);
    }
    nvptx.barrierSyncAligned(.work_group, null);
    return result;
}

const classes = prepare_scheduler.length_classes;
const BlockShape = enum(u1) { narrow, wide };
pub const PlanScratch = struct {
    class_blocks: [2][classes]u32,
    list_blocks: [2]u32,
    wide_candidates: u32,
};
var plan_scratch: PlanScratch addrspace(.shared) = undefined;

fn ceilDiv(value: u32, divisor: u32) u32 {
    return (value + divisor - 1) / divisor;
}

const PlannedSequence = struct {
    shape: BlockShape,
    blocks: u32,
    class: u32,
};

const BlockRows = struct {
    narrow: u32,
    wide: u32,
    wide_min_sequence: u32,

    fn plan(
        rows_per_block: BlockRows,
        cu_seqlens_q: [*]addrspace(.global) const i32,
        cu_visible_k: [*]addrspace(.global) const i32,
        sequence: u32,
    ) PlannedSequence {
        const rows: u32 = @intCast(cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence]);
        const keys: u32 = @intCast(cu_visible_k[sequence + 1] - cu_visible_k[sequence]);
        const wide = rows >= rows_per_block.wide_min_sequence;
        return .{
            .shape = if (wide) .wide else .narrow,
            .blocks = ceilDiv(rows, if (wide) rows_per_block.wide else rows_per_block.narrow),
            .class = prepare_scheduler.lengthClass(keys),
        };
    }
};

pub const PlanInput = struct {
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    cu_visible_k: u64,
    visibility_bits: u64,
    window_sizes: u64,
    scalar_window: Window,
    run_table: u64,
    narrow_list: u64,
    wide_list: u64,
    count: u32,
    narrow_rows: u32,
    wide_rows: u32,
    wide_min_rows: u32,
    wide_min_blocks: u32,
};

pub const ListBlocks = struct { narrow: u32, wide: u32 };

pub fn planSequences(
    comptime threads: u32,
    input: PlanInput,
    scratch: *addrspace(.shared) PlanScratch,
) ListBlocks {
    const cu_q: [*]addrspace(.global) const i32 = @ptrFromInt(input.cu_seqlens_q);
    const cu_k: [*]addrspace(.global) const i32 = @ptrFromInt(input.cu_seqlens_k);
    const visible_k: [*]addrspace(.global) const i32 = @ptrFromInt(input.cu_visible_k);
    const entries: [*]addrspace(.global) RunEntry = @ptrFromInt(input.run_table);
    const lists: [2][*]addrspace(.global) ListedBlock = .{
        @ptrFromInt(input.narrow_list), @ptrFromInt(input.wide_list),
    };
    const count = input.count;
    var cleared = thread.id();
    while (cleared < 2 * classes) : (cleared += threads) {
        scratch.class_blocks[cleared / classes][cleared % classes] = 0;
    }
    if (thread.id() == 0) scratch.wide_candidates = 0;
    nvptx.barrierSyncAligned(.work_group, null);
    var sequence = thread.id();
    while (sequence < count) : (sequence += threads) {
        if (input.run_table != 0) {
            const runs: Runs = .init(
                input.visibility_bits,
                input.window_sizes,
                input.scalar_window,
                cu_q,
                cu_k,
                count,
                sequence,
            );
            entries[sequence] = runEntry(runs, cu_k, count);
        }
        const rows: u32 = @intCast(cu_q[sequence + 1] - cu_q[sequence]);
        if (rows >= input.wide_min_rows) {
            const blocks = ceilDiv(rows, input.wide_rows);
            _ = nvptx.atom(.cta, .monotonic, .add, &scratch.wide_candidates, blocks);
        }
    }
    nvptx.barrierSyncAligned(.work_group, null);
    const block_rows: BlockRows = .{
        .narrow = input.narrow_rows,
        .wide = input.wide_rows,
        .wide_min_sequence = if (scratch.wide_candidates >= input.wide_min_blocks)
            input.wide_min_rows
        else
            std.math.maxInt(u32),
    };
    sequence = thread.id();
    while (sequence < count) : (sequence += threads) {
        const planned = block_rows.plan(cu_q, visible_k, sequence);
        if (planned.blocks == 0) continue;
        const counter = &scratch.class_blocks[@backingInt(planned.shape)][planned.class];
        _ = nvptx.atom(.cta, .monotonic, .add, counter, planned.blocks);
    }
    nvptx.barrierSyncAligned(.work_group, null);
    if (thread.warp() < 2) {
        const counters = &scratch.class_blocks[thread.warp()];
        const per_lane = classes / 32;
        var blocks: [per_lane]u32 = undefined;
        var lane_blocks: u32 = 0;
        inline for (&blocks, 0..) |*class_count, offset| {
            class_count.* = counters[classes - 1 - (thread.lane() * per_lane + offset)];
            lane_blocks += class_count.*;
        }
        const inclusive = warpInclusiveSum(lane_blocks);
        var start = inclusive - lane_blocks;
        inline for (blocks, 0..) |class_count, offset| {
            counters[classes - 1 - (thread.lane() * per_lane + offset)] = start;
            start += class_count;
        }
        if (thread.lane() == 31) scratch.list_blocks[thread.warp()] = inclusive;
    }
    nvptx.barrierSyncAligned(.work_group, null);
    sequence = thread.id();
    while (sequence < count) : (sequence += threads) {
        const planned = block_rows.plan(cu_q, visible_k, sequence);
        if (planned.blocks == 0) continue;
        const list = @backingInt(planned.shape);
        const counter = &scratch.class_blocks[list][planned.class];
        const first = nvptx.atom(.cta, .monotonic, .add, counter, planned.blocks);
        for (0..planned.blocks) |block| {
            lists[list][first + block] = .{ .sequence = sequence, .block = @intCast(block) };
        }
    }
    return .{ .narrow = scratch.list_blocks[0], .wide = scratch.list_blocks[1] };
}

pub fn planBlocks(
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    cu_visible_k: u64,
    visibility_bits: u64,
    window_sizes: u64,
    window_size_left: i32,
    window_size_right: i32,
    run_table: u64,
    narrow_list: u64,
    wide_list: u64,
    queues: u64,
    count: u32,
    narrow_rows: u32,
    wide_rows: u32,
    wide_min_rows: u32,
    wide_min_blocks: u32,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = metadata_threads },
} }) void {
    const blocks = planSequences(metadata_threads, .{
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_k = cu_seqlens_k,
        .cu_visible_k = cu_visible_k,
        .visibility_bits = visibility_bits,
        .window_sizes = window_sizes,
        .scalar_window = .{ .left = window_size_left, .right = window_size_right },
        .run_table = run_table,
        .narrow_list = narrow_list,
        .wide_list = wide_list,
        .count = count,
        .narrow_rows = narrow_rows,
        .wide_rows = wide_rows,
        .wide_min_rows = wide_min_rows,
        .wide_min_blocks = wide_min_blocks,
    }, &plan_scratch);
    if (thread.id() == 0) {
        const queue: [*]addrspace(.global) RunQueue = @ptrFromInt(queues);
        queue[0] = .{ .listed_blocks = blocks.narrow, .claims = 0 };
        queue[1] = .{ .listed_blocks = blocks.wide, .claims = 0 };
    }
}

pub const MegakernelCounters = extern struct {
    barrier: u32,
    wide_claims: u32,
    narrow_claims: u32,
    wide_blocks: u32,
    narrow_blocks: u32,
};

var order_classes: [classes]u32 addrspace(.shared) = undefined;

fn costClass(
    visibility_bits: u64,
    cu_seqlens_k: [*]addrspace(.global) const i32,
    count: u32,
    sequence: u32,
) u32 {
    const keys = visibleTokens(.init(visibility_bits, count, sequence), cu_seqlens_k);
    return classes - 1 - prepare_scheduler.lengthClass(keys);
}

fn orderSequences(
    visibility_bits: u64,
    cu_seqlens_k: [*]addrspace(.global) const i32,
    sequence_order: u64,
    count: u32,
) void {
    const order: [*]addrspace(.global) u32 = @ptrFromInt(sequence_order);
    var cleared = thread.id();
    while (cleared < classes) : (cleared += metadata_threads) order_classes[cleared] = 0;
    nvptx.barrierSyncAligned(.work_group, null);
    var sequence = thread.id();
    while (sequence < count) : (sequence += metadata_threads) {
        const class = costClass(visibility_bits, cu_seqlens_k, count, sequence);
        _ = nvptx.atom(.cta, .monotonic, .add, &order_classes[class], 1);
    }
    nvptx.barrierSyncAligned(.work_group, null);
    if (thread.warp() == 0) {
        const per_lane = classes / 32;
        var sequences: [per_lane]u32 = undefined;
        var lane_sequences: u32 = 0;
        inline for (&sequences, 0..) |*class_count, offset| {
            class_count.* = order_classes[thread.lane() * per_lane + offset];
            lane_sequences += class_count.*;
        }
        var start = warpInclusiveSum(lane_sequences) - lane_sequences;
        inline for (sequences, 0..) |class_count, offset| {
            order_classes[thread.lane() * per_lane + offset] = start;
            start += class_count;
        }
    }
    nvptx.barrierSyncAligned(.work_group, null);
    sequence = thread.id();
    while (sequence < count) : (sequence += metadata_threads) {
        const class = costClass(visibility_bits, cu_seqlens_k, count, sequence);
        order[nvptx.atom(.cta, .monotonic, .add, &order_classes[class], 1)] = sequence;
    }
}

pub fn countVisibleTokens(
    visibility_bits: u64,
    visibility_bits_t: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    cu_visible_k: u64,
    cu_visible_q: u64,
    sequence_order: u64,
    count: u32,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = metadata_threads },
} }) void {
    const lengths: [2][*]addrspace(.global) const i32 = .{
        @ptrFromInt(cu_seqlens_k), @ptrFromInt(cu_seqlens_q),
    };
    const bits: [2]u64 = .{ visibility_bits, visibility_bits_t };
    const outputs: [2][*]addrspace(.global) i32 = .{
        @ptrFromInt(cu_visible_k), @ptrFromInt(cu_visible_q),
    };
    if (thread.id() == 0) inline for (outputs) |output| {
        output[0] = 0;
    };
    var carry: [2]u32 = .{ 0, 0 };
    var first: u32 = 0;
    while (first < count) : (first += metadata_threads) {
        const sequence = first + thread.id();
        var tokens: [2]u32 = .{ 0, 0 };
        if (sequence < count) inline for (&tokens, bits, lengths) |*total, matrix, cu_seqlens| {
            total.* = visibleTokens(.init(matrix, count, sequence), cu_seqlens);
        };
        const sums = blockInclusiveSums(tokens);
        inline for (&carry, sums.inclusive, sums.total, outputs) |*carried, sum, total, output| {
            const cumulative = @min(carried.* +| sum, std.math.maxInt(i32));
            if (sequence < count) output[sequence + 1] = @intCast(cumulative);
            carried.* +|= total;
        }
    }
    nvptx.barrierSyncAligned(.work_group, null);
    orderSequences(visibility_bits, @ptrFromInt(cu_seqlens_k), sequence_order, count);
}
