const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");

pub const Chunk = @Vector(4, u32);
pub const Accumulator = @Vector(4, f32);
pub const RowPair = @Vector(2, f32);

pub const mma_m16n8k16: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .accumulator = .f32,
    .element = .bf16,
};

pub const mma_m16n8k8: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k8,
    .accumulator = .f32,
    .element = .bf16,
};

pub const rescale_threshold = 8.0;

pub fn exp2(x: anytype) @TypeOf(x) {
    var result: @TypeOf(x) = undefined;
    inline for (0..@typeInfo(@TypeOf(x)).vector.len) |i| {
        result[i] = nvptx.math.ex2ApproxF32(true, x[i]);
    }
    return result;
}

pub fn packBf16(comptime count: u32, values: anytype) @Vector(count / 2, u32) {
    var packed_values: @Vector(count / 2, u32) = undefined;
    inline for (0..count / 2) |pair| {
        packed_values[pair] =
            @backingInt(nvptx.Bf16x2.fromF32(values[2 * pair], values[2 * pair + 1]));
    }
    return packed_values;
}

pub fn scaledChunk(values: anytype, comptime chunk: u32, scale: f32) Chunk {
    var words: Chunk = undefined;
    inline for (0..4) |word| {
        words[word] = @backingInt(nvptx.Bf16x2.fromF32(
            values[chunk * 8 + word * 2] * scale,
            values[chunk * 8 + word * 2 + 1] * scale,
        ));
    }
    return words;
}

pub fn storeScaledChunks(
    comptime head_dim: u32,
    pointer: u64,
    element: u64,
    column: u32,
    values: @Vector(32, f32),
    scale: f32,
) void {
    const out: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
    inline for (0..4) |part| {
        const first = column + part * 8;
        if (first < head_dim) out[(element + first) / 8] = scaledChunk(values, part, scale);
    }
}

pub fn reduceAdd(address: *addrspace(.global) f32, values: anytype) void {
    switch (@typeInfo(@TypeOf(values)).vector.len) {
        2 => asm volatile ("red.global.add.v2.f32 [%[address]], {%[a], %[b]};"
            :
            : [address] "l" (address),
              [a] "f" (values[0]),
              [b] "f" (values[1]),
            : .{ .memory = true }),
        4 => asm volatile ("red.global.add.v4.f32 [%[address]], {%[a], %[b], %[c], %[d]};"
            :
            : [address] "l" (address),
              [a] "f" (values[0]),
              [b] "f" (values[1]),
              [c] "f" (values[2]),
              [d] "f" (values[3]),
            : .{ .memory = true }),
        else => comptime unreachable,
    }
}

pub fn shuffleXor(x: RowPair, comptime mask: u32) RowPair {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return .{
        nvptx.shflSync(.bfly, f32, x[0], mask, clamp, .all),
        nvptx.shflSync(.bfly, f32, x[1], mask, clamp, .all),
    };
}

pub fn quadAllreduce(comptime op: enum { max, sum }, x: RowPair) RowPair {
    const first = switch (op) {
        .max => @max(x, shuffleXor(x, 1)),
        .sum => x + shuffleXor(x, 1),
    };
    return switch (op) {
        .max => @max(first, shuffleXor(first, 2)),
        .sum => first + shuffleXor(first, 2),
    };
}

pub fn Softmax(comptime block_n: comptime_int) type {
    return struct {
        row_max: RowPair,
        row_sum: RowPair,

        pub const Scores = [block_n / 8]Accumulator;
        pub const Probabilities = [block_n / 16]@Vector(4, u32);

        pub fn init() @This() {
            return .{ .row_max = @splat(-std.math.inf(f32)), .row_sum = @splat(0) };
        }

        pub const Step = struct {
            row_scale: Accumulator,
            should_rescale: bool,
            probabilities: Probabilities,
        };

        pub const Exponentials = struct {
            row_scale: Accumulator,
            should_rescale: bool,
            values: Scores,
        };

        pub const Finalized = struct {
            row_scale: RowPair,
            lse: RowPair,
        };

        pub fn exponentials(softmax: *@This(), scores: Scores, scale_log2: f32) Exponentials {
            var max = softmax.row_max;
            inline for (scores) |fragment| {
                const even: RowPair = .{ fragment[0], fragment[2] };
                const odd: RowPair = .{ fragment[1], fragment[3] };
                max = @max(max, @max(even, odd));
            }
            max = quadAllreduce(.max, max);
            const scale: RowPair = @splat(scale_log2);
            const minus_infinity: RowPair = @splat(-std.math.inf(f32));
            const scaled_max =
                @select(f32, max != minus_infinity, max * scale, @as(RowPair, @splat(0)));
            const previous_scaled_max = softmax.row_max * scale;
            const exponent = previous_scaled_max - scaled_max;
            const raise_max = exponent < @as(RowPair, @splat(-rescale_threshold));
            const row_scale = @select(f32, raise_max, exp2(exponent), @as(RowPair, @splat(1)));
            const offset = @select(f32, raise_max, scaled_max, previous_scaled_max);
            const should_rescale = nvptx.voteSync(.any, @reduce(
                .Or,
                raise_max & (softmax.row_max != minus_infinity),
            ), .all);
            var sum = softmax.row_sum * row_scale;
            const offsets: Accumulator = .{ offset[0], offset[0], offset[1], offset[1] };
            const scale4: Accumulator = @splat(scale_log2);
            var values: Scores = undefined;
            inline for (&values, scores) |*value, fragment| {
                value.* = exp2(@mulAdd(Accumulator, fragment, scale4, -offsets));
                sum += RowPair{ value[0] + value[1], value[2] + value[3] };
            }
            softmax.row_max = @select(f32, raise_max, max, softmax.row_max);
            softmax.row_sum = sum;
            return .{
                .row_scale = .{ row_scale[0], row_scale[0], row_scale[1], row_scale[1] },
                .should_rescale = should_rescale,
                .values = values,
            };
        }

        pub fn pack(values: Scores) Probabilities {
            var probabilities: Probabilities = undefined;
            inline for (&probabilities, 0..) |*fragment, k_step| {
                const left = values[2 * k_step];
                const right = values[2 * k_step + 1];
                fragment.* = .{
                    @backingInt(nvptx.Bf16x2.fromF32(left[0], left[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(left[2], left[3])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[0], right[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[2], right[3])),
                };
            }
            return probabilities;
        }

        pub fn step(softmax: *@This(), scores: Scores, scale_log2: f32) Step {
            const exponentiated = exponentials(softmax, scores, scale_log2);
            return .{
                .row_scale = exponentiated.row_scale,
                .should_rescale = exponentiated.should_rescale,
                .probabilities = pack(exponentiated.values),
            };
        }

        pub fn finalize(softmax: @This(), scale_log2: f32) Finalized {
            const sum = quadAllreduce(.sum, softmax.row_sum);
            const non_empty_row = sum > @as(RowPair, @splat(0));
            const scale: RowPair = @splat(scale_log2);
            const ln2: RowPair = @splat(std.math.ln2);
            const log2_sum: RowPair = .{
                nvptx.math.lg2ApproxF32(true, sum[0]), nvptx.math.lg2ApproxF32(true, sum[1]),
            };
            const log_sum_exp = (softmax.row_max * scale + log2_sum) * ln2;
            return .{
                .row_scale = @select(
                    f32,
                    non_empty_row,
                    @as(RowPair, @splat(1)) / sum,
                    @as(RowPair, @splat(0)),
                ),
                .lse = @select(
                    f32,
                    non_empty_row,
                    log_sum_exp,
                    @as(RowPair, @splat(-std.math.inf(f32))),
                ),
            };
        }
    };
}

pub fn storeLse(params: anytype, work: anytype, row: u32, row_limit: u32, lse: RowPair) void {
    const m_lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
    const base = work.batch * params.lse_batch_stride +
        work.head * params.lse_head_stride + work.sequence.offset_q;
    if (@TypeOf(work.pack_gqa) != void) {
        inline for (0..2) |half| {
            const packed_row = row + @as(u32, half) * 8;
            if (packed_row < row_limit)
                m_lse[base + work.pack_gqa.rowOffset(packed_row, 1, params.lse_head_stride)] =
                    lse[half];
        }
        return;
    }
    if (row < row_limit) m_lse[base + row] = lse[0];
    if (row + 8 < row_limit) m_lse[base + row + 8] = lse[1];
}

pub fn lseIndex(params: anytype, work: anytype, row: u32) u64 {
    const row_offset = if (@TypeOf(work.pack_gqa) != void) work.pack_gqa.rowOffset(
        row,
        1,
        params.lse_head_stride,
    ) else row;
    return work.batch * params.lse_batch_stride + work.head * params.lse_head_stride +
        work.sequence.offset_q + row_offset;
}

pub fn storeLseRow(
    params: anytype,
    work: anytype,
    first: u32,
    row: u32,
    valid_rows: u32,
    lse: f32,
) void {
    if (@TypeOf(work.pack_gqa) != void) {
        const out: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
        if (row < valid_rows) out[lseIndex(params, work, first + row)] = lse;
        return;
    }
    const lane = thread.lane();
    const warp_first = row - lane;
    const clamp = nvptx.ShuffleClamp.fromWidth(.idx, 32);
    var quad: @Vector(4, f32) = undefined;
    inline for (0..4) |element_index| {
        const element: u32 = element_index;
        quad[element] = nvptx.shflSync(.idx, f32, lse, (lane * 4 + element) % 32, clamp, .all);
    }
    const address = params.lse + 4 * lseIndex(params, work, first + warp_first);
    if (address % 16 == 0 and warp_first + 32 <= valid_rows) {
        const out: [*]addrspace(.global) @Vector(4, f32) = @ptrFromInt(address);
        if (lane < 8) out[lane] = quad;
    } else if (row < valid_rows) {
        const out: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
        out[lseIndex(params, work, first + row)] = lse;
    }
}

pub fn gather(address: u64, index: u64, rows: u32) Chunk {
    const source: [*]addrspace(.global) const Chunk = @ptrFromInt(address);
    const lane = thread.lane();
    const first = index / 4;
    const end = index + rows;
    const shift: u32 = @intCast(index % 4);
    const current: Chunk = if ((first + lane) * 4 < end) source[first + lane] else @splat(0);
    const next: Chunk = if ((first + lane + 1) * 4 < end) source[first + lane + 1] else @splat(0);
    const values: [8]u32 = .{
        current[0], current[1], current[2], current[3],
        next[0],    next[1],    next[2],    next[3],
    };
    var result: Chunk = undefined;
    inline for (0..4) |element| {
        var value = values[element];
        inline for (1..4) |offset| {
            if (shift == offset) value = values[element + offset];
        }
        result[element] = value;
    }
    return result;
}
