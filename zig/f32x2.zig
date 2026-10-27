const std = @import("std");
const nvptx = std.nvptx;
const maskR2p = @import("mask.zig").maskR2p;
const thread = @import("thread.zig");
const Chunk = @import("fragments.zig").Chunk;
const pivot_probability = @import("fragments.zig").pivot_probability;

pub const Pair = @Vector(2, f32);

pub fn splat(value: f32) Pair {
    return .{ value, value };
}

pub fn fma(a: Pair, b: Pair, c: Pair) Pair {
    return nvptx.math.fmaF32x2(.nearest_even, true, a, b, c);
}

pub fn fmaKeepingSubnormals(a: Pair, b: Pair, c: Pair) Pair {
    return nvptx.math.fmaF32x2(.nearest_even, false, a, b, c);
}

pub fn add(a: Pair, b: Pair) Pair {
    return nvptx.math.addF32x2(.nearest_even, true, a, b);
}

pub fn addKeepingSubnormals(a: Pair, b: Pair) Pair {
    return nvptx.math.addF32x2(.nearest_even, false, a, b);
}

pub fn mul(a: Pair, b: Pair) Pair {
    return nvptx.math.mulF32x2(.nearest_even, true, a, b);
}

pub fn max3(a: f32, b: f32, c: f32) f32 {
    return nvptx.math.max3F32(.{}, a, b, c);
}

pub fn exp2f(x: Pair) Pair {
    return .{ nvptx.math.ex2ApproxF32(true, x[0]), nvptx.math.ex2ApproxF32(true, x[1]) };
}

pub fn ex2Emulation(x: Pair) Pair {
    return asm ("{\n" ++
            ".reg .f32 low, high;\n" ++
            ".reg .b64 v, rounded, whole, fraction, magic, c1, c2, c3, one, p;\n" ++
            ".reg .b32 r0, r1, p0, p1;\n" ++
            "mov.b64 v, %[x];\n" ++
            "mov.b64 {low, high}, v;\n" ++
            "max.ftz.f32 low, low, 0fC2FE0000;\n" ++
            "max.ftz.f32 high, high, 0fC2FE0000;\n" ++
            "mov.b64 v, {low, high};\n" ++
            "mov.b64 magic, 0x4B4000004B400000;\n" ++
            "add.rm.ftz.f32x2 rounded, v, magic;\n" ++
            "sub.rn.ftz.f32x2 whole, rounded, magic;\n" ++
            "sub.rn.ftz.f32x2 fraction, v, whole;\n" ++
            "mov.b64 c3, 0x3D9DF09D3D9DF09D;\n" ++
            "mov.b64 c2, 0x3E6906A43E6906A4;\n" ++
            "mov.b64 c1, 0x3F31F5193F31F519;\n" ++
            "mov.b64 one, 0x3F8000003F800000;\n" ++
            "fma.rn.ftz.f32x2 p, fraction, c3, c2;\n" ++
            "fma.rn.ftz.f32x2 p, p, fraction, c1;\n" ++
            "fma.rn.ftz.f32x2 p, p, fraction, one;\n" ++
            "mov.b64 {r0, r1}, rounded;\n" ++
            "mov.b64 {p0, p1}, p;\n" ++
            "shl.b32 r0, r0, 23;\n" ++
            "shl.b32 r1, r1, 23;\n" ++
            "add.s32 p0, p0, r0;\n" ++
            "add.s32 p1, p1, r1;\n" ++
            "mov.b64 p, {p0, p1};\n" ++
            "mov.b64 %[d], p;\n" ++
            "}"
        : [d] "=f" (-> Pair),
        : [x] "f" (x),
    );
}

pub fn exponentiate(scores: anytype, offsets: anytype, scale_log2: f32) @TypeOf(scores) {
    const count = @typeInfo(@TypeOf(scores)).vector.len;
    const scale = splat(scale_log2);
    var probabilities: @TypeOf(scores) = undefined;
    inline for (0..count / 4) |quad| {
        const scalar = @TypeOf(offsets) == f32;
        const offset: @Vector(4, f32) = if (scalar) @splat(offsets) else @bitCast(offsets[quad]);
        inline for (0..2) |pair| {
            const column = quad * 4 + pair * 2;
            const exponent = fma(
                .{ scores[column], scores[column + 1] },
                scale,
                .{ offset[pair * 2], offset[pair * 2 + 1] },
            );
            const value = exp2f(exponent);
            inline for (0..2) |element| probabilities[column + element] = value[element];
        }
    }
    return probabilities;
}

pub fn dsPiece(
    comptime piece: u32,
    probabilities: anytype,
    dp: anytype,
    dpsum_negated: anytype,
    interior: bool,
    lower: i32,
    upper: i32,
) @Vector(16, u32) {
    var ds: @Vector(32, f32) = undefined;
    inline for (0..8) |quad| {
        const scalar = @TypeOf(dpsum_negated) == f32;
        const sums: [2]Pair = if (scalar) @splat(splat(dpsum_negated)) else sums: {
            const sum: @Vector(4, f32) = @bitCast(dpsum_negated[piece * 8 + quad]);
            break :sums .{ .{ sum[0], sum[1] }, .{ sum[2], sum[3] } };
        };
        inline for (0..2) |pair| {
            const column = quad * 4 + pair * 2;
            const value = mul(
                .{ probabilities[piece * 32 + column], probabilities[piece * 32 + column + 1] },
                add(.{ dp[piece][column], dp[piece][column + 1] }, sums[pair]),
            );
            inline for (0..2) |element| ds[column + element] = value[element];
        }
    }
    if (!interior) maskR2p(&ds, lower - @as(i32, piece * 32), upper - @as(i32, piece * 32), 0);
    return @import("fragments.zig").packBf16(32, ds);
}

pub fn quadAt(
    values: anytype,
    comptime index: u32,
) @Vector(4, @typeInfo(@TypeOf(values)).vector.child) {
    const first = index * 4;
    return .{ values[first], values[first + 1], values[first + 2], values[first + 3] };
}

pub fn largest(values: anytype) f32 {
    const count = @typeInfo(@TypeOf(values)).vector.len;
    var level: [count]f32 = values;
    comptime var remaining = count;
    inline while (remaining > 1) {
        const next = (remaining + 2) / 3;
        inline for (0..next) |index| {
            const first = 3 * index;
            level[index] = if (first + 2 < remaining)
                max3(level[first], level[first + 1], level[first + 2])
            else if (first + 1 < remaining)
                @max(level[first], level[first + 1])
            else
                level[first];
        }
        remaining = next;
    }
    return level[0];
}

pub fn warpHasPivot(probabilities: anytype) bool {
    return nvptx.voteSync(.any, largest(probabilities) > pivot_probability, .all);
}

pub fn excludePivots(probabilities: anytype, keys: [*]addrspace(.global) i32, key: i32) void {
    inline for (0..@typeInfo(@TypeOf(probabilities.*)).vector.len) |column| {
        if (probabilities[column] > pivot_probability) {
            keys[column] = key;
            probabilities[column] = 0;
        }
    }
}

pub fn widenBf16(word: u32) Pair {
    return nvptx.Bf16x2.toF32(@fromBackingInt(word));
}

pub fn sumBf16(words: anytype) Pair {
    const count = @typeInfo(@TypeOf(words)).vector.len;
    var level: [count]Pair = undefined;
    inline for (0..count) |word| level[word] = widenBf16(words[word]);
    comptime var remaining = count;
    inline while (remaining > 1) {
        const next = (remaining + 1) / 2;
        inline for (0..remaining / 2) |index| {
            level[index] = addKeepingSubnormals(level[2 * index], level[2 * index + 1]);
        }
        if (remaining % 2 == 1) level[next - 1] = level[remaining - 1];
        remaining = next;
    }
    return level[0];
}

/// Returns, in each lane, the sum over the rows of the BF16 column `chunk * 8 + thread.lane() / 4`
/// of a tile with 128-byte rows that stores logical chunk `c` of row `r` at chunk `c ^ r % 8`.
/// The caller passes the same `chunk` in lanes that differ only in bits 2 to 4 of `thread.lane()`.
pub fn swizzledColumnSum(tile: anytype, chunk: u32) f32 {
    const rows = @typeInfo(@TypeOf(tile.*)).array.len / 8;
    const slice = thread.lane() / 4;
    // Quarter warp q reads rows q and q + 4 of each group of 8 rows, so that the 16-byte loads of
    // its 8 lanes touch 8 different groups of 4 banks.
    const row = slice % 2 * 4 + slice / 2;
    var sums: [4]Pair = @splat(splat(0));
    inline for (0..rows / 8) |pass| {
        const words: Chunk = tile[(pass * 8 + row) * 8 + (chunk ^ row)];
        inline for (0..4) |word| {
            sums[word] = addKeepingSubnormals(sums[word], widenBf16(words[word]));
        }
    }
    var level: [8]f32 = undefined;
    inline for (0..4) |word| {
        level[2 * word] = sums[word][0];
        level[2 * word + 1] = sums[word][1];
    }
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    inline for (.{ 16, 8, 4 }, .{ 4, 2, 1 }) |offset, half| {
        const upper = thread.lane() & offset != 0;
        inline for (0..half) |index| {
            const kept = if (upper) level[index + half] else level[index];
            const sent = if (upper) level[index] else level[index + half];
            level[index] = kept + nvptx.shflSync(.bfly, f32, sent, offset, clamp, .all);
        }
    }
    return level[0];
}
