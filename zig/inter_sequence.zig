const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");

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
                const run = runs.next(0) orelse {
                    entries[sequence] = .{ .first_key = 0, .key_count = 0, .window = .unbounded };
                    continue;
                };
                if (runs.row.find(run.end, true) < count) {
                    entries[sequence] = .{
                        .first_key = 0,
                        .key_count = several_runs,
                        .window = .unbounded,
                    };
                    const rows: u32 = @intCast(cu_q[sequence + 1] - cu_q[sequence]);
                    const blocks = (rows + block_m - 1) / block_m;
                    const first = nvptx.atom(.cta, .monotonic, .add, &listed, blocks);
                    for (0..blocks) |block| {
                        list[first + block] = .{ .sequence = sequence, .block = @intCast(block) };
                    }
                    continue;
                }
                const first_key: u32 = @intCast(cu_k[run.first]);
                entries[sequence] = .{
                    .first_key = first_key,
                    .key_count = @as(u32, @intCast(cu_k[run.end])) - first_key,
                    .window = runs.windowOf(run),
                };
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

pub fn countVisibleTokens(
    visibility_bits: u64,
    visibility_bits_t: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    cu_visible_k: u64,
    cu_visible_q: u64,
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
}
