const std = @import("std");
const nvptx = std.nvptx;

pub const Swizzle = enum(u2) {
    none = 0,
    bytes_128 = 1,
    bytes_64 = 2,
    bytes_32 = 3,
};

pub fn descriptor(
    start: u32,
    leading_byte_offset: u32,
    stride_byte_offset: u32,
    swizzle: Swizzle,
) u64 {
    return encode(start) |
        encode(leading_byte_offset) << 16 |
        encode(stride_byte_offset) << 32 |
        @as(u64, @backingInt(swizzle)) << 62;
}

pub fn address(pointer: anytype) u32 {
    return @intCast(@intFromPtr(pointer));
}

pub fn Operands(comptime swizzle_bytes: u32) type {
    const swizzle: Swizzle = switch (swizzle_bytes) {
        128 => .bytes_128,
        64 => .bytes_64,
        32 => .bytes_32,
        else => unreachable,
    };
    const steps_per_box = swizzle_bytes / 32;
    return struct {
        pub fn kStepOffset(comptime rows: u32, comptime k_step: u32) u32 {
            return k_step / steps_per_box * rows * swizzle_bytes + k_step % steps_per_box * 32;
        }

        pub fn kMajor(base: u32) u64 {
            return descriptor(base, 16, 8 * swizzle_bytes, swizzle);
        }

        pub fn mnMajor(base: u32, comptime rows: u32) u64 {
            return descriptor(base, rows * swizzle_bytes, 8 * swizzle_bytes, swizzle);
        }

        pub fn mmaKSteps(
            comptime n: comptime_int,
            accumulator: *Accumulator(n),
            comptime k_steps: u32,
            a: u64,
            comptime a_rows: u32,
            b: u64,
            comptime b_rows: u32,
        ) void {
            inline for (0..k_steps) |k_step| {
                const a_step = advance(a, kStepOffset(a_rows, k_step));
                mmaShared(
                    n,
                    accumulator,
                    a_step,
                    advance(b, kStepOffset(b_rows, k_step)),
                    k_step != 0,
                    false,
                    false,
                );
            }
        }
    };
}

pub fn advance(base: u64, comptime bytes: u32) u64 {
    const words: [2]u32 = @bitCast(base);
    return @bitCast([2]u32{ words[0] + bytes / 16, words[1] });
}

fn encode(bytes: u32) u64 {
    return (bytes & 0x3ffff) >> 4;
}

pub const fence = nvptx.wgmma.fenceSyncAligned;
pub const commit = nvptx.wgmma.commitGroupSyncAligned;
pub const wait = nvptx.wgmma.waitGroupSyncAligned;

pub fn fenceOperand(accumulator: anytype) void {
    var value = accumulator.*;
    asm volatile (""
        : [value] "+f" (value),
    );
    accumulator.* = value;
}

pub fn Accumulator(comptime n: comptime_int) type {
    return @Vector(n / 2, f32);
}

pub fn packPairs(comptime n: u32, accumulator: Accumulator(n)) [n / 16]@Vector(4, u32) {
    var result: [n / 16]@Vector(4, u32) = undefined;
    inline for (&result, 0..) |*fragment, step| {
        const left = step * 8;
        const right = step * 8 + 4;
        fragment.* = .{
            @backingInt(nvptx.Bf16x2.fromF32(accumulator[left], accumulator[left + 1])),
            @backingInt(nvptx.Bf16x2.fromF32(accumulator[left + 2], accumulator[left + 3])),
            @backingInt(nvptx.Bf16x2.fromF32(accumulator[right], accumulator[right + 1])),
            @backingInt(nvptx.Bf16x2.fromF32(accumulator[right + 2], accumulator[right + 3])),
        };
    }
    return result;
}

pub fn storeMatrices(comptime transposed: bool, destination: u32, fragment: @Vector(4, u32)) void {
    asm volatile ("stmatrix.sync.aligned.m8n8.x4" ++
            (if (transposed) ".trans" else "") ++ ".shared.b16 [%[destination]], %[fragment];"
        :
        : [destination] "r" (destination),
          [fragment] "r" (fragment),
        : .{ .memory = true });
}

pub fn scaledBy(comptime n: u32, accumulator: Accumulator(n), scale: f32) Accumulator(n) {
    var result: Accumulator(n) = undefined;
    inline for (0..n / 2) |index| result[index] = accumulator[index] * scale;
    return result;
}

fn instruction(
    comptime n: comptime_int,
    comptime a_in_registers: bool,
    comptime accumulate: bool,
    comptime transpose_a: bool,
    comptime transpose_b: bool,
) []const u8 {
    return std.fmt.comptimePrint(
        "{{\n.reg .pred p;\nsetp.ne.b32 p, {d}, 0;\n" ++
            "wgmma.mma_async.sync.aligned.m64n{d}k16.f32.bf16.bf16 " ++
            "%[d], %[a], %[b], p, 1, 1, {s}{d};\n}}",
        .{
            @intFromBool(accumulate),
            n,
            if (a_in_registers) "" else if (transpose_a) "1, " else "0, ",
            @intFromBool(transpose_b),
        },
    );
}

pub fn mmaShared(
    comptime n: comptime_int,
    accumulator: *Accumulator(n),
    a_descriptor: u64,
    b_descriptor: u64,
    comptime accumulate: bool,
    comptime transpose_a: bool,
    comptime transpose_b: bool,
) void {
    var value = accumulator.*;
    asm volatile (instruction(n, false, accumulate, transpose_a, transpose_b)
        : [d] "+f" (value),
        : [a] "l" (a_descriptor),
          [b] "l" (b_descriptor),
    );
    accumulator.* = value;
}

pub fn mmaRegisters(
    comptime n: comptime_int,
    accumulator: *Accumulator(n),
    a_registers: @Vector(4, u32),
    b_descriptor: u64,
    comptime accumulate: bool,
    comptime transpose_b: bool,
) void {
    var value = accumulator.*;
    asm volatile (instruction(n, true, accumulate, false, transpose_b)
        : [d] "+f" (value),
        : [a] "r" (a_registers),
          [b] "l" (b_descriptor),
    );
    accumulator.* = value;
}
