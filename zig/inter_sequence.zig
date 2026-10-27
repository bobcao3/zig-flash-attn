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

pub const SequenceRun = struct {
    first: u32,
    end: u32,
    banded: bool,
};

pub const SequenceRuns = struct {
    row: BitRow,
    sequence: u32,
    window: Window,
    joins_previous: bool,

    pub fn init(
        visibility_bits: u64,
        window_sizes: u64,
        scalar_window: Window,
        cu_seqlens_q: [*]addrspace(.global) const i32,
        cu_seqlens_k: [*]addrspace(.global) const i32,
        count: u32,
        sequence: u32,
    ) SequenceRuns {
        const window = sequenceWindow(window_sizes, count, sequence, scalar_window);
        const queries = cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence];
        const keys = cu_seqlens_k[sequence + 1] - cu_seqlens_k[sequence];
        return .{
            .row = .init(visibility_bits, count, sequence),
            .sequence = sequence,
            .window = window,
            .joins_previous = window.left < 0 and queries <= keys,
        };
    }

    pub fn next(runs: SequenceRuns, from: u32) ?SequenceRun {
        const first = runs.row.find(from, true);
        if (first >= runs.row.count) return null;
        const end = runs.row.find(first, false);
        const sequence = runs.sequence;
        if (!runs.window.bounded() or sequence < first or sequence >= end) {
            return .{ .first = first, .end = end, .banded = false };
        }
        if (sequence == first or runs.joins_previous) {
            return .{ .first = first, .end = sequence + 1, .banded = true };
        }
        return .{ .first = first, .end = sequence, .banded = false };
    }

    pub fn windowOf(runs: SequenceRuns, run: SequenceRun) Window {
        return if (run.banded) runs.window else .unbounded;
    }
};

pub const KeyRun = packed struct(u64) {
    first_key: u32,
    keys: u31,
    banded: bool,
};

pub const SequenceRecord = extern struct {
    first_query: u32,
    rows: u32,
    window: Window,
    key_run_count: u32,
    resume_from: u32,
    key_runs: [max_key_runs]KeyRun,

    pub const max_key_runs = 5;

    pub fn at(records: u64, sequence: u32) *addrspace(.global) const SequenceRecord {
        const all: [*]addrspace(.global) const SequenceRecord = @ptrFromInt(records);
        return &all[sequence];
    }

    fn init(sequences: Sequences, windows: Windows, sequence: u32) SequenceRecord {
        const cu_seqlens_q = sequences.cu_seqlens_q;
        const cu_seqlens_k = sequences.cu_seqlens_k;
        const first_query: u32 = @intCast(cu_seqlens_q[sequence]);
        var record: SequenceRecord = .{
            .first_query = first_query,
            .rows = @as(u32, @intCast(cu_seqlens_q[sequence + 1])) - first_query,
            .window = sequenceWindow(windows.sizes, sequences.count, sequence, windows.scalar),
            .key_run_count = 0,
            .resume_from = sequences.count,
            .key_runs = @splat(.{ .first_key = 0, .keys = 0, .banded = false }),
        };
        const runs: SequenceRuns = .init(
            sequences.visibility_bits,
            windows.sizes,
            windows.scalar,
            cu_seqlens_q,
            cu_seqlens_k,
            sequences.count,
            sequence,
        );
        var from: u32 = 0;
        while (runs.next(from)) |run| : (from = run.end) {
            if (record.key_run_count == max_key_runs) {
                record.resume_from = run.first;
                break;
            }
            const first_key: u32 = @intCast(cu_seqlens_k[run.first]);
            const end_key: u32 = @intCast(cu_seqlens_k[run.end]);
            record.key_runs[record.key_run_count] = .{
                .first_key = first_key,
                .keys = @intCast(end_key - first_key),
                .banded = run.banded,
            };
            record.key_run_count += 1;
        }
        return record;
    }
};

const Windows = struct { sizes: u64, scalar: Window };

pub const BlockTable = extern struct {
    blocks: u32,
    tile_count_semaphore: u64,

    pub const Entry = extern struct { sequence: u32, first_block: u32 };

    pub const Summary = extern struct {
        work_before: u64,
        first_query: u32,
        rows: u32,
        block_work: u32,
    };

    pub const block_rows = [_]u32{ 64, 128, 192, 256 };
    pub const work_unit = block_rows[0];
    const m64_max_rows = 384;
    const short_max_rows = 8;

    const SortGroup = enum(u32) { m128, m64, short };

    pub fn at(table: u64) *addrspace(.global) BlockTable {
        return @ptrFromInt(table);
    }

    pub fn entries(table: u64) [*]addrspace(.global) Entry {
        return @ptrFromInt(table + @sizeOf(BlockTable));
    }

    pub fn summaries(table: u64, capacity: u32) [*]addrspace(.global) Summary {
        return @ptrFromInt(table + @sizeOf(BlockTable) + @as(u64, capacity) * @sizeOf(Entry));
    }

    fn bytes(capacity: u32) u64 {
        return @sizeOf(BlockTable) + @as(u64, capacity) * (@sizeOf(Entry) + @sizeOf(Summary));
    }

    pub fn workBefore(table: u64, capacity: u32, block: u32) u64 {
        const blocks = at(table).blocks;
        if (block < blocks) return summaries(table, capacity)[block].work_before;
        if (blocks == 0) return 0;
        const last = summaries(table, capacity)[blocks - 1];
        return last.work_before + last.block_work;
    }

    fn blockWork(block_m: u32, keys: u32) u32 {
        return block_m / work_unit * ((keys + work_unit - 1) / work_unit);
    }
};

const Workspace = struct {
    fn table(workspace: u64, capacity: u32, comptime block_m: u32) u64 {
        const index = comptime std.mem.indexOfScalar(u32, &BlockTable.block_rows, block_m).?;
        return workspace + index * BlockTable.bytes(capacity);
    }

    fn order(workspace: u64, capacity: u32) [*]addrspace(.global) u32 {
        return @ptrFromInt(workspace + BlockTable.block_rows.len * BlockTable.bytes(capacity));
    }
};

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

fn saturatingAdd(left: u32, right: u32) u32 {
    const sum = left +% right;
    return if (sum < left) std.math.maxInt(u32) else sum;
}

fn warpInclusiveSums(comptime count: usize, values: [count]u32) [count]u32 {
    const up = nvptx.ShuffleClamp.fromWidth(.up, 32);
    var sums = values;
    inline for (.{ 1, 2, 4, 8, 16 }) |distance| {
        var previous: [count]u32 = undefined;
        inline for (&previous, sums) |*value, sum| {
            value.* = nvptx.shflSync(.up, u32, sum, distance, up, .all);
        }
        if (thread.lane() >= distance) {
            inline for (&sums, previous) |*sum, value| sum.* = saturatingAdd(sum.*, value);
        }
    }
    return sums;
}

fn Sums(comptime count: usize) type {
    return struct { inclusive: [count]u32, total: [count]u32 };
}

fn ctaSums(
    comptime threads: u32,
    comptime count: usize,
    values: [count]u32,
    totals: *addrspace(.shared) [count][threads / nvptx.warp_size]u32,
) Sums(count) {
    const warps = threads / nvptx.warp_size;
    const idx = nvptx.ShuffleClamp.fromWidth(.idx, 32);
    var result: Sums(count) = undefined;
    result.inclusive = warpInclusiveSums(count, values);
    if (thread.lane() == nvptx.warp_size - 1) {
        inline for (totals, result.inclusive) |*warp_totals, sum| warp_totals[thread.warp()] = sum;
    }
    nvptx.barrierSyncAligned(.work_group, null);
    var warp_totals: [count]u32 = @splat(0);
    if (thread.lane() < warps) {
        inline for (&warp_totals, totals) |*warp_total, *all| warp_total.* = all[thread.lane()];
    }
    const warp_sums = warpInclusiveSums(count, warp_totals);
    var earlier: [count]u32 = undefined;
    inline for (&earlier, warp_sums, warp_totals) |*value, sum, warp_total| {
        value.* = nvptx.shflSync(.idx, u32, sum - warp_total, thread.warp(), idx, .all);
    }
    inline for (&result.total, warp_sums) |*total, sum| {
        total.* = nvptx.shflSync(.idx, u32, sum, 31, idx, .all);
    }
    inline for (&result.inclusive, earlier) |*sum, value| sum.* = saturatingAdd(sum.*, value);
    nvptx.barrierSyncAligned(.work_group, null);
    return result;
}

var token_totals: [2][metadata_threads / 32]u32 addrspace(.shared) = undefined;

const length_classes = prepare_scheduler.length_classes;
const groups = std.enums.values(BlockTable.SortGroup).len;
const order_classes: u32 = groups * length_classes;

const Sequences = struct {
    cu_seqlens_q: [*]addrspace(.global) const i32,
    cu_seqlens_k: [*]addrspace(.global) const i32,
    cu_visible_k: [*]addrspace(.global) const i32,
    visibility_bits: u64,
    count: u32,
    packing: u32,
    sorts_by_group: bool,
};

const table_threads = 256;

const TableBuilder = struct {
    const threads = table_threads;
    const warps = threads / nvptx.warp_size;
    const classes_per_thread = (order_classes + threads - 1) / threads;
    const tables = BlockTable.block_rows.len;

    const SortedSequence = struct {
        sequence: u32,
        rows: u32,
        keys: u32,
        class: u32,
        first_query: u32 = 0,

        const absent: SortedSequence = .{
            .sequence = 0,
            .rows = 0,
            .keys = 0,
            .class = order_classes,
        };

        fn load(sequences: Sequences, sequence: u32, comptime summarize: bool) SortedSequence {
            if (sequence >= sequences.count) return .absent;
            const cu_seqlens_q = sequences.cu_seqlens_q;
            const cu_visible_k = sequences.cu_visible_k;
            const rows: u32 = @intCast(cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence]);
            const keys: u32 = @intCast(cu_visible_k[sequence + 1] - cu_visible_k[sequence]);
            const group: BlockTable.SortGroup = group: {
                const block_rows = BlockTable.block_rows;
                if (!sequences.sorts_by_group) break :group .m128;
                if (rows <= BlockTable.short_max_rows) break :group .short;
                if (rows <= block_rows[0]) break :group .m64;
                const last_block_rows = rows % block_rows[1];
                const fills = last_block_rows == 0 or last_block_rows > block_rows[0];
                break :group if (fills or rows >= BlockTable.m64_max_rows) .m128 else .m64;
            };
            const last_group_class = (@backingInt(group) + 1) * length_classes - 1;
            var sorted: SortedSequence = .{
                .sequence = sequence,
                .rows = rows,
                .keys = keys,
                .class = last_group_class - prepare_scheduler.lengthClass(keys),
            };
            if (summarize) sorted.first_query = @intCast(cu_seqlens_q[sequence]);
            return sorted;
        }
    };

    const SortedSequences = struct {
        sequence: [threads]u32,
        rows: [threads]u32,
        keys: [threads]u32,
        first_query: [threads]u32,
        first_blocks: [tables][threads]u32,
        work_before: [tables][threads]u64,

        fn put(
            sorted: *addrspace(.shared) SortedSequences,
            index: u32,
            sequence: SortedSequence,
        ) void {
            sorted.sequence[index] = sequence.sequence;
            sorted.rows[index] = sequence.rows;
            sorted.keys[index] = sequence.keys;
            sorted.first_query[index] = sequence.first_query;
        }
    };

    const Shared = struct {
        class_ends: [order_classes]u32,
        warp_classes: [warps][order_classes]u16 align(4),
        warp_totals: [2 * tables][warps]u32,
        sorted: SortedSequences,
    };

    fn sequenceAt(sequences: Sequences, own: SortedSequence, first: u32) SortedSequence {
        if (first == 0) return own;
        return .load(sequences, first + thread.id(), false);
    }

    fn countClasses(
        shared: *addrspace(.shared) Shared,
        sequences: Sequences,
        own: SortedSequence,
    ) void {
        var class = thread.id();
        while (class < order_classes) : (class += threads) shared.class_ends[class] = 0;
        const words: *addrspace(.shared) [warps * order_classes / 2]u32 =
            @ptrCast(&shared.warp_classes);
        var word = thread.id();
        while (word < words.len) : (word += threads) words[word] = 0;
        nvptx.barrierSyncAligned(.work_group, null);
        var first: u32 = 0;
        while (first < sequences.count) : (first += threads) {
            const sequence = sequenceAt(sequences, own, first);
            if (sequence.class < order_classes) {
                _ = nvptx.atom(.cta, .monotonic, .add, &shared.class_ends[sequence.class], 1);
            }
        }
        nvptx.barrierSyncAligned(.work_group, null);
        const first_class = thread.id() * classes_per_thread;
        var counts: [classes_per_thread]u32 = @splat(0);
        var thread_count: u32 = 0;
        inline for (&counts, 0..) |*count, offset| {
            if (first_class + offset < order_classes) {
                count.* = shared.class_ends[first_class + offset];
            }
            thread_count += count.*;
        }
        const sums = ctaSums(threads, 1, .{thread_count}, shared.warp_totals[0..1]);
        var start = sums.inclusive[0] - thread_count;
        inline for (counts, 0..) |count, offset| {
            if (first_class + offset < order_classes) {
                shared.class_ends[first_class + offset] = start;
            }
            start += count;
        }
        nvptx.barrierSyncAligned(.work_group, null);
    }

    fn orderSequences(
        shared: *addrspace(.shared) Shared,
        sequences: Sequences,
        own: SortedSequence,
        order: [*]addrspace(.global) u32,
    ) void {
        const warp = thread.warp();
        var first: u32 = 0;
        while (first < sequences.count) : (first += threads) {
            const sequence = first + thread.id();
            const present = sequence < sequences.count;
            const class = sequenceAt(sequences, own, first).class;
            const peers = nvptx.matchAnySync(u32, class, .all);
            const earlier_peers: u32 = @popCount(peers & nvptx.lanemaskLt());
            const leads = present and earlier_peers == 0;
            if (leads) shared.warp_classes[warp][class] = @popCount(peers);
            nvptx.barrierSyncAligned(.work_group, null);
            var position: u32 = 0;
            var closes = false;
            if (present) {
                var before: u32 = 0;
                var after: u32 = 0;
                for (&shared.warp_classes, 0..) |*counts, other| {
                    if (other < warp) before += counts[class];
                    if (other > warp) after += counts[class];
                }
                position = shared.class_ends[class] + before + earlier_peers;
                if (sequences.count <= threads) {
                    shared.sorted.put(position, own);
                } else order[position] = sequence;
                closes = after == 0 and peers & nvptx.lanemaskGt() == 0;
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (closes) shared.class_ends[class] = position + 1;
            if (leads) shared.warp_classes[warp][class] = 0;
            nvptx.barrierSyncAligned(.work_group, null);
        }
    }

    fn placeBlocks(
        shared: *addrspace(.shared) Shared,
        sequences: Sequences,
        order: [*]addrspace(.global) const u32,
        workspace: u64,
        capacity: u32,
        comptime block_rows: []const u32,
        comptime summarize: bool,
    ) void {
        const sorted = &shared.sorted;
        const single = sequences.count <= threads;
        var carried_blocks: [block_rows.len]u32 = @splat(0);
        var carried_work: [block_rows.len]u64 = @splat(0);
        var first: u32 = 0;
        while (first < sequences.count) : (first += threads) {
            const position = first + thread.id();
            var placed: SortedSequence = .absent;
            if (position < sequences.count) {
                placed = if (single) .{
                    .sequence = sorted.sequence[position],
                    .rows = sorted.rows[position],
                    .keys = sorted.keys[position],
                    .class = 0,
                    .first_query = sorted.first_query[position],
                } else .load(sequences, order[position], summarize);
            }
            const tables_used = block_rows.len;
            var values: [tables_used * @as(u32, if (summarize) 2 else 1)]u32 = undefined;
            inline for (block_rows, 0..) |block_m, table| {
                const blocks = (placed.rows * sequences.packing + block_m - 1) / block_m;
                values[table] = blocks;
                if (summarize) {
                    const work = blocks *| BlockTable.blockWork(block_m, placed.keys);
                    values[tables_used + table] = work;
                }
            }
            const totals = shared.warp_totals[0..values.len];
            const sums = ctaSums(threads, values.len, values, totals);
            if (!single) sorted.put(thread.id(), placed);
            inline for (0..tables_used) |table| {
                const block_sum = sums.inclusive[table] - values[table];
                sorted.first_blocks[table][thread.id()] = carried_blocks[table] + block_sum;
                if (summarize) {
                    const work = tables_used + table;
                    const work_sum = sums.inclusive[work] - values[work];
                    sorted.work_before[table][thread.id()] = carried_work[table] + work_sum;
                    carried_work[table] += sums.total[work];
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            var spans: [tables_used]BlockSpan = undefined;
            inline for (&spans, &carried_blocks, 0..) |*span, *carried, table| {
                const end = carried.* + sums.total[table];
                span.* = .{ .first = carried.*, .end = @min(end, capacity) };
                carried.* = end;
            }
            const count = @min(threads, sequences.count - first);
            fill(sorted, count, workspace, capacity, block_rows, summarize, spans);
            nvptx.barrierSyncAligned(.work_group, null);
        }
        inline for (block_rows, 0..) |block_m, table| {
            if (thread.id() == table) {
                BlockTable.at(Workspace.table(workspace, capacity, block_m)).* = .{
                    .blocks = @min(carried_blocks[table], capacity),
                    .tile_count_semaphore = 0,
                };
            }
        }
    }

    const BlockSpan = struct { first: u32, end: u32 };

    fn fill(
        sorted: *addrspace(.shared) const SortedSequences,
        count: u32,
        workspace: u64,
        capacity: u32,
        comptime block_rows: []const u32,
        comptime summarize: bool,
        spans: [block_rows.len]BlockSpan,
    ) void {
        var total: u32 = 0;
        for (spans) |span| total += span.end -| span.first;
        var flat_block = thread.id();
        while (flat_block < total) : (flat_block += threads) {
            var rest = flat_block;
            inline for (block_rows, spans, 0..) |block_m, span, table| {
                const blocks = span.end -| span.first;
                if (rest < blocks) {
                    const block = span.first + rest;
                    const address = Workspace.table(workspace, capacity, block_m);
                    const first_blocks = &sorted.first_blocks[table];
                    var low: u32 = 0;
                    var high = count;
                    while (low < high) {
                        const middle = (low + high) / 2;
                        if (first_blocks[middle] <= block) low = middle + 1 else high = middle;
                    }
                    const index = low - 1;
                    const first_block = first_blocks[index];
                    storeWords(BlockTable.Entry, &BlockTable.entries(address)[block], .{
                        .sequence = sorted.sequence[index],
                        .first_block = first_block,
                    });
                    if (summarize) {
                        const block_work = BlockTable.blockWork(block_m, sorted.keys[index]);
                        const summaries = BlockTable.summaries(address, capacity);
                        storeWords(BlockTable.Summary, &summaries[block], .{
                            .work_before = sorted.work_before[table][index] +
                                @as(u64, block - first_block) * block_work,
                            .first_query = sorted.first_query[index],
                            .rows = sorted.rows[index],
                            .block_work = block_work,
                        });
                    }
                    break;
                }
                rest -= blocks;
            }
        }
    }

    fn storeWords(comptime T: type, target: *addrspace(.global) T, value: T) void {
        const Words = @Vector(if (@sizeOf(T) % 16 == 0) 4 else 2, u32);
        comptime std.debug.assert(@sizeOf(T) % @sizeOf(Words) == 0);
        const count = @sizeOf(T) / @sizeOf(Words);
        const words: *const [count]Words = @ptrCast(@alignCast(&value));
        const targets: *addrspace(.global) [count]Words = @ptrCast(@alignCast(target));
        inline for (targets, words) |*slot, word| {
            nvptx.cache.store(.{ .operator = .write_back }, slot, word);
        }
    }
};

pub fn TableKernel(comptime block_rows: []const u32, comptime summarize: bool) type {
    return struct {
        var shared: TableBuilder.Shared addrspace(.shared) = undefined;

        pub fn build(
            cu_seqlens_q: u64,
            cu_seqlens_k: u64,
            cu_visible_k: u64,
            visibility_bits: u64,
            workspace: u64,
            capacity: u32,
            count: u32,
            split_counters: u64,
            split_counter_count: u32,
            sequence_records: u64,
            window_sizes: u64,
            window_size_left: i32,
            window_size_right: i32,
            packing: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = table_threads },
        } }) void {
            if (comptime nvptx.gpu.sm >= 90) nvptx.griddepcontrol(.launch_dependents);
            const sequences: Sequences = .{
                .cu_seqlens_q = @ptrFromInt(cu_seqlens_q),
                .cu_seqlens_k = @ptrFromInt(cu_seqlens_k),
                .cu_visible_k = @ptrFromInt(cu_visible_k),
                .visibility_bits = visibility_bits,
                .count = count,
                .packing = packing,
                .sorts_by_group = block_rows.len == 1,
            };
            if (@workGroupId(0) > 0) {
                const sequence = (@workGroupId(0) - 1) * table_threads + thread.id();
                if (sequence >= count) return;
                const windows: Windows = .{
                    .sizes = window_sizes,
                    .scalar = .{ .left = window_size_left, .right = window_size_right },
                };
                const targets: [*]addrspace(.global) SequenceRecord = @ptrFromInt(sequence_records);
                const record: SequenceRecord = .init(sequences, windows, sequence);
                TableBuilder.storeWords(SequenceRecord, &targets[sequence], record);
                return;
            }
            const counters: [*]addrspace(.global) u32 = @ptrFromInt(split_counters);
            var counter = thread.id();
            while (counter < split_counter_count) : (counter += table_threads) {
                counters[counter] = 0;
            }
            const own: TableBuilder.SortedSequence = .load(sequences, thread.id(), summarize);
            const order = Workspace.order(workspace, capacity);
            TableBuilder.countClasses(&shared, sequences, own);
            TableBuilder.orderSequences(&shared, sequences, own, order);
            TableBuilder.placeBlocks(
                &shared,
                sequences,
                order,
                workspace,
                capacity,
                block_rows,
                summarize,
            );
        }
    };
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
            const row: BitRow = .init(matrix, count, sequence);
            var from: u32 = 0;
            while (true) {
                const run_first = row.find(from, true);
                if (run_first >= row.count) break;
                const run_end = row.find(run_first, false);
                total.* +|= @intCast(cu_seqlens[run_end] - cu_seqlens[run_first]);
                from = run_end;
            }
        };
        const sums = ctaSums(metadata_threads, 2, tokens, &token_totals);
        inline for (&carry, sums.inclusive, sums.total, outputs) |*carried, sum, total, output| {
            const cumulative = @min(carried.* +| sum, std.math.maxInt(i32));
            if (sequence < count) output[sequence + 1] = @intCast(cumulative);
            carried.* +|= total;
        }
    }
}
