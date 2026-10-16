const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");

pub const capacity = 512;

pub const Swizzle = enum(u3) {
    none = 0,
    bytes_128 = 2,
    bytes_64 = 4,
    bytes_32 = 6,

    pub fn fromBytes(bytes: u32) Swizzle {
        return switch (bytes) {
            128 => .bytes_128,
            64 => .bytes_64,
            32 => .bytes_32,
            else => .none,
        };
    }
};

pub const Major = enum(u1) { k = 0, mn = 1 };

pub fn sharedAddress(pointer: anytype) u32 {
    return @intCast(@intFromPtr(pointer));
}

pub fn sharedDescriptor(start: u32, leading_byte_offset: u32, stride_byte_offset: u32, swizzle: Swizzle) u64 {
    return encode(start) |
        encode(leading_byte_offset) << 16 |
        encode(stride_byte_offset) << 32 |
        1 << 46 |
        @as(u64, @backingInt(swizzle)) << 61;
}

fn encode(bytes: u32) u64 {
    return (bytes & 0x3ffff) >> 4;
}

pub fn kMajorDescriptor(start: u32, comptime swizzle_bytes: u32) u64 {
    return sharedDescriptor(start, 16, 8 * swizzle_bytes, comptime Swizzle.fromBytes(swizzle_bytes));
}

pub fn kMajorOffsets(comptime steps: u32, comptime rows: u32, comptime swizzle_bytes: u32) [steps]u32 {
    const steps_per_box = swizzle_bytes / 32;
    var offsets: [steps]u32 = undefined;
    for (&offsets, 0..) |*offset, step| {
        offset.* = (step / steps_per_box * rows * swizzle_bytes + step % steps_per_box * 32) >> 4;
    }
    return offsets;
}

pub fn rowOffsets(comptime steps: u32, comptime row_bytes: u32) [steps]u32 {
    var offsets: [steps]u32 = undefined;
    for (&offsets, 0..) |*offset, step| offset.* = step * 16 * row_bytes >> 4;
    return offsets;
}

pub fn instructionDescriptor(
    comptime m: u32,
    comptime n: u32,
    comptime a_major: Major,
    comptime b_major: Major,
) u32 {
    const f32_accumulator = 1 << 4;
    const bf16_a = 1 << 7;
    const bf16_b = 1 << 10;
    return f32_accumulator | bf16_a | bf16_b |
        @as(u32, @backingInt(a_major)) << 15 |
        @as(u32, @backingInt(b_major)) << 16 |
        (n >> 3) << 17 |
        (m >> 4) << 24;
}

pub fn address(lane: u32, column: u32) u32 {
    return lane << 16 | column;
}

pub fn warpAddress(base: u32, column: u32) u32 {
    return base + address(thread.warp() % 4 * 32, column);
}

pub fn allocate(slot: *addrspace(.shared) u32, comptime columns: u32) void {
    asm volatile (std.fmt.comptimePrint(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%[slot]], {d};\n" ++
                "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
            .{columns},
        )
        :
        : [slot] "r" (slot),
        : .{ .memory = true });
}

pub fn deallocate(base: u32, comptime columns: u32) void {
    asm volatile (std.fmt.comptimePrint("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %[base], {d};", .{columns})
        :
        : [base] "r" (base),
        : .{ .memory = true });
}

pub fn load(comptime count: u32, tensor_address: u32) @Vector(count, f32) {
    return asm volatile (std.fmt.comptimePrint("tcgen05.ld.sync.aligned.32x32b.x{d}.b32 %[d], [%[a]];", .{count})
        : [d] "=f" (-> @Vector(count, f32)),
        : [a] "r" (tensor_address),
        : .{ .memory = true });
}

pub fn store(tensor_address: u32, values: anytype) void {
    const count = @typeInfo(@TypeOf(values)).vector.len;
    const text = std.fmt.comptimePrint("tcgen05.st.sync.aligned.32x32b.x{d}.b32 [%[a]], %[v];", .{count});
    switch (@typeInfo(@TypeOf(values)).vector.child) {
        f32 => asm volatile (text
            :
            : [a] "r" (tensor_address),
              [v] "f" (values),
            : .{ .memory = true }),
        u32 => asm volatile (text
            :
            : [a] "r" (tensor_address),
              [v] "r" (values),
            : .{ .memory = true }),
        else => @compileError("tcgen05.st takes f32 or u32 vectors"),
    }
}

pub const piece_columns = 32;

pub fn loadColumns(comptime count: u32, tensor_address: u32) @Vector(count, f32) {
    var values: @Vector(count, f32) = undefined;
    inline for (0..count / piece_columns) |piece_index| {
        const piece: u32 = piece_index;
        const part = load(piece_columns, tensor_address + piece * piece_columns);
        inline for (0..piece_columns) |column| values[piece * piece_columns + column] = part[column];
    }
    return values;
}

pub fn storeColumns(tensor_address: u32, values: anytype) void {
    const Values = @TypeOf(values);
    const Element = @typeInfo(Values).vector.child;
    const count = @typeInfo(Values).vector.len;
    comptime var start: u32 = 0;
    inline while (start < count) {
        const width = comptime @min(piece_columns, std.math.floorPowerOfTwo(u32, count - start));
        var part: @Vector(width, Element) = undefined;
        inline for (0..width) |column| part[column] = values[start + column];
        store(tensor_address + start, part);
        start += width;
    }
}

pub fn waitLoad() void {
    asm volatile ("tcgen05.wait::ld.sync.aligned;" ::: .{ .memory = true });
}

pub fn waitStore() void {
    asm volatile ("tcgen05.wait::st.sync.aligned;" ::: .{ .memory = true });
}

pub fn fenceBeforeSync() void {
    asm volatile ("tcgen05.fence::before_thread_sync;" ::: .{ .memory = true });
}

pub fn fenceAfterSync() void {
    asm volatile ("tcgen05.fence::after_thread_sync;" ::: .{ .memory = true });
}

fn mmaStepsText(
    comptime cta_group: u32,
    comptime tensor_a: bool,
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
) []const u8 {
    comptime var text: []const u8 = "{\n.reg .pred leader, accumulate;\n.reg .b32 a_low, a_high, b_low, b_high, a_step, b_step;\n" ++
        ".reg .b64 a, b;\nelect.sync _|leader, -1;\nsetp.ne.b32 accumulate, %[accumulate], 0;\n" ++
        "mov.b64 {b_low, b_high}, %[b];\n";
    if (!tensor_a) text = text ++ "mov.b64 {a_low, a_high}, %[a];\n";
    inline for (a_offsets, b_offsets, 0..) |a_offset, b_offset, step| {
        const enable = if (step == 0) "accumulate" else "1";
        text = text ++ std.fmt.comptimePrint("add.u32 b_step, b_low, {d};\nmov.b64 b, {{b_step, b_high}};\n", .{b_offset});
        if (tensor_a) {
            text = text ++ std.fmt.comptimePrint(
                "@leader tcgen05.mma.cta_group::{d}.kind::f16 [%[d]], [%[a] + {d}], b, %[instruction], {s};\n",
                .{ cta_group, a_offset, enable },
            );
        } else {
            text = text ++ std.fmt.comptimePrint(
                "add.u32 a_step, a_low, {d};\nmov.b64 a, {{a_step, a_high}};\n" ++
                    "@leader tcgen05.mma.cta_group::{d}.kind::f16 [%[d]], a, b, %[instruction], {s};\n",
                .{ a_offset, cta_group, enable },
            );
        }
    }
    return text ++ "}";
}

pub fn mmaSsSteps(
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    d: u32,
    a: u64,
    b: u64,
    instruction: u32,
    accumulate: bool,
) void {
    asm volatile (mmaStepsText(1, false, a_offsets, b_offsets)
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn mmaTsSteps(
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    d: u32,
    a: u32,
    b: u64,
    instruction: u32,
    accumulate: bool,
) void {
    asm volatile (mmaStepsText(1, true, a_offsets, b_offsets)
        :
        : [d] "r" (d),
          [a] "r" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn mmaSs(d: u32, a: u64, b: u64, instruction: u32, accumulate: bool) void {
    asm volatile ("{\n.reg .pred accumulate;\nsetp.ne.b32 accumulate, %[accumulate], 0;\n" ++
            "tcgen05.mma.cta_group::1.kind::f16 [%[d]], %[a], %[b], %[instruction], accumulate;\n}"
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn mmaTs(d: u32, a: u32, b: u64, instruction: u32, accumulate: bool) void {
    asm volatile ("{\n.reg .pred accumulate;\nsetp.ne.b32 accumulate, %[accumulate], 0;\n" ++
            "tcgen05.mma.cta_group::1.kind::f16 [%[d]], [%[a]], %[b], %[instruction], accumulate;\n}"
        :
        : [d] "r" (d),
          [a] "r" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn allocate2Sm(slot: *addrspace(.shared) u32, comptime columns: u32) void {
    asm volatile (std.fmt.comptimePrint(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%[slot]], {d};\n" ++
                "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;",
            .{columns},
        )
        :
        : [slot] "r" (slot),
        : .{ .memory = true });
}

pub fn deallocate2Sm(base: u32, comptime columns: u32) void {
    asm volatile (std.fmt.comptimePrint("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %[base], {d};", .{columns})
        :
        : [base] "r" (base),
        : .{ .memory = true });
}

pub fn mmaSsSteps2Sm(
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    d: u32,
    a: u64,
    b: u64,
    instruction: u32,
    accumulate: bool,
) void {
    asm volatile (mmaStepsText(2, false, a_offsets, b_offsets)
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn mmaTsSteps2Sm(
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    d: u32,
    a: u32,
    b: u64,
    instruction: u32,
    accumulate: bool,
) void {
    asm volatile (mmaStepsText(2, true, a_offsets, b_offsets)
        :
        : [d] "r" (d),
          [a] "r" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

const both_ctas = 0b11;

pub const Commit = enum { cta_group_1, cta_group_2 };

pub fn commit(comptime group: Commit, barrier: *addrspace(.shared) nvptx.Mbarrier) void {
    switch (group) {
        .cta_group_1 => asm volatile ("{\n.reg .pred leader;\nelect.sync _|leader, -1;\n" ++
                "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%[barrier]];\n}"
            :
            : [barrier] "r" (barrier),
            : .{ .memory = true }),
        .cta_group_2 => asm volatile (std.fmt.comptimePrint(
                "{{\n.reg .pred leader;\n.reg .b16 mask;\nmov.b16 mask, {d};\nelect.sync _|leader, -1;\n" ++
                    "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%[barrier]], mask;\n}}",
                .{both_ctas},
            )
            :
            : [barrier] "r" (barrier),
            : .{ .memory = true }),
    }
}
