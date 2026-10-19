const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");
const elect = @import("elect.zig");
const configurations = @import("configurations.zig");

pub const Entry = @Vector(2, u32);

const mantissa_bits = 2;
pub const length_classes = @bitSizeOf(u32) << mantissa_bits;

pub fn lookup(scheduler_metadata: u64, num_batch: u32, block: u32) ?Entry {
    const table: [*]addrspace(.global) const Entry = @ptrFromInt(scheduler_metadata);
    const entry = table[block];
    return if (entry[0] < num_batch) entry else null;
}

pub fn lengthClass(length: u32) u32 {
    if (length == 0) return 0;
    const exponent: u5 = @intCast(31 - @clz(length));
    const mantissa: u32 =
        @intCast(@as(u64, length) << mantissa_bits >> exponent & ((1 << mantissa_bits) - 1));
    return @as(u32, exponent) << mantissa_bits | mantissa;
}

fn warpInclusiveSum(value: u32) u32 {
    const up = nvptx.ShuffleClamp.fromWidth(.up, 32);
    var sum = value;
    inline for (.{ 1, 2, 4, 8, 16 }) |distance| {
        const previous_sum = nvptx.shflSync(.up, u32, sum, distance, up, .all);
        if (thread.lane() >= distance) sum += previous_sum;
    }
    return sum;
}

pub fn PrepareScheduler(
    comptime block_m: u32,
    comptime block_order: configurations.BlockOrder,
    comptime sort: bool,
) type {
    return struct {
        pub const threads = 1024;
        const warps = threads / nvptx.warp_size;
        const classes_per_lane = length_classes / nvptx.warp_size;
        const class_stride = warps + 1;
        const idx = nvptx.ShuffleClamp.fromWidth(.idx, 32);

        var inclusive: [threads]u32 addrspace(.shared) = undefined;
        var batches: [threads]u32 addrspace(.shared) = undefined;
        var warp_blocks: [warps]u32 addrspace(.shared) = undefined;
        var class_counts: [length_classes]u32 addrspace(.shared) = undefined;
        var class_warps: [length_classes * class_stride]u32 addrspace(.shared) = undefined;

        comptime {
            std.debug.assert(warps == nvptx.warp_size);
        }

        fn sortedPosition(cu_seqlens_other: u64, first_batch: u32, num_batch: u32) u32 {
            const valid = first_batch + thread.id() < num_batch;
            const m_cu_seqlens_other: [*]addrspace(.global) const i32 =
                @ptrFromInt(cu_seqlens_other);
            const length: u32 =
                if (valid) @intCast(m_cu_seqlens_other[first_batch + thread.id() + 1] -
                    m_cu_seqlens_other[first_batch + thread.id()]) else 0;
            const class = length_classes - 1 - lengthClass(length);
            var cleared = thread.id();
            while (cleared < class_warps.len) : (cleared += threads) class_warps[cleared] = 0;
            nvptx.barrierSyncAligned(.work_group, null);
            const peers = nvptx.matchAnySync(u32, if (valid) class else length_classes, .all);
            if (valid) class_warps[class * class_stride + thread.warp()] = @popCount(peers);
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..classes_per_lane) |offset| {
                const column = (thread.warp() * classes_per_lane + offset) * class_stride;
                const count = class_warps[column + thread.lane()];
                const warp_inclusive = warpInclusiveSum(count);
                class_warps[column + thread.lane()] = warp_inclusive - count;
                const total =
                    nvptx.shflSync(.idx, u32, warp_inclusive, nvptx.warp_size - 1, idx, .all);
                if (elect.oneSync()) class_counts[thread.warp() * classes_per_lane + offset] =
                    total;
                elect.reconverge();
            }
            nvptx.barrierSyncAligned(.work_group, null);
            var lane_count: u32 = 0;
            inline for (
                0..classes_per_lane,
            ) |offset| lane_count += class_counts[thread.lane() * classes_per_lane + offset];
            var start = warpInclusiveSum(lane_count) - lane_count;
            var class_start: u32 = 0;
            inline for (0..classes_per_lane) |offset| {
                const offset_start =
                    nvptx.shflSync(.idx, u32, start, class / classes_per_lane, idx, .all);
                if (class % classes_per_lane == offset) class_start = offset_start;
                start += class_counts[thread.lane() * classes_per_lane + offset];
            }
            const below: u32 = @popCount(peers & nvptx.lanemaskLt());
            if (!valid) return thread.id();
            return class_start + class_warps[class * class_stride + thread.warp()] + below;
        }

        pub fn prepareVarlenNumBlocks(
            scheduler_metadata: u64,
            cu_seqlens: u64,
            cu_seqlens_other: u64,
            num_batch: u32,
            num_block: u32,
        ) callconv(.{ .nvptx_kernel = .{ .required_work_group_size = .{ .x = threads } } }) void {
            if (nvptx.gpu.sm >= 90) nvptx.griddepcontrol(.launch_dependents);
            const m_cu_seqlens: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens);
            const table: [*]addrspace(.global) Entry = @ptrFromInt(scheduler_metadata);
            const end_block = (@workGroupId(0) + 1) * threads;
            var entry: Entry = .{ num_batch, 0 };
            var first_block: u32 = 0;
            var first_batch: u32 = 0;
            while (first_batch < num_batch and first_block < end_block) : (first_batch += threads) {
                const blocks: u32 = if (first_batch + thread.id() < num_batch)
                    (@as(u32, @intCast(m_cu_seqlens[first_batch + thread.id() + 1] -
                        m_cu_seqlens[first_batch + thread.id()])) + block_m - 1) / block_m
                else
                    0;
                const ordered_blocks = if (sort) ordered: {
                    const position = sortedPosition(cu_seqlens_other, first_batch, num_batch);
                    batches[position] = first_batch + thread.id();
                    inclusive[position] = blocks;
                    nvptx.barrierSyncAligned(.work_group, null);
                    break :ordered inclusive[thread.id()];
                } else blocks;
                const sum = warpInclusiveSum(ordered_blocks);
                const warp_sum = nvptx.shflSync(.idx, u32, sum, nvptx.warp_size - 1, idx, .all);
                if (elect.oneSync()) warp_blocks[thread.warp()] = warp_sum;
                elect.reconverge();
                nvptx.barrierSyncAligned(.work_group, null);
                const warp_count = warp_blocks[thread.lane()];
                const warp_inclusive = warpInclusiveSum(warp_count);
                const preceding = nvptx.shflSync(
                    .idx,
                    u32,
                    warp_inclusive - warp_count,
                    thread.warp(),
                    idx,
                    .all,
                );
                inclusive[thread.id()] = preceding + sum;
                nvptx.barrierSyncAligned(.work_group, null);
                const window_blocks =
                    nvptx.shflSync(.idx, u32, warp_inclusive, nvptx.warp_size - 1, idx, .all);
                if (end_block - threads + thread.id() >= first_block and
                    end_block - threads + thread.id() - first_block < window_blocks)
                {
                    const local = end_block - threads + thread.id() - first_block;
                    var low: u32 = 0;
                    var high: u32 = threads - 1;
                    while (low < high) {
                        const middle = (low + high) / 2;
                        if (inclusive[middle] > local) high = middle else low = middle + 1;
                    }
                    const owner_inclusive = inclusive[low];
                    const owner_blocks = owner_inclusive - if (low > 0) inclusive[low - 1] else 0;
                    entry = .{
                        if (sort) batches[low] else first_batch + low,
                        if (block_order == .descending) owner_inclusive -
                            1 - local else local - (owner_inclusive - owner_blocks),
                    };
                }
                first_block += window_blocks;
                nvptx.barrierSyncAligned(.work_group, null);
            }
            if (end_block - threads +
                thread.id() < num_block) table[end_block - threads + thread.id()] = entry;
        }
    };
}

test "length classes rise with the length and stay below the class count" {
    var previous: u32 = 0;
    var length: u32 = 0;
    while (length < 1 << 16) : (length += 1) {
        const class = lengthClass(length);
        try std.testing.expect(class >= previous and class < length_classes);
        previous = class;
    }
    for ([_]u32{
        1 << 20, (1 << 20) + (1 << 18), std.math.maxInt(i32), std.math.maxInt(u32),
    }) |large| {
        try std.testing.expect(lengthClass(large) < length_classes);
    }
    try std.testing.expect(lengthClass(1 << 20) < lengthClass((1 << 20) + (1 << 18)));
}
