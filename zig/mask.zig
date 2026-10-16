const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");
const ColLimits = @import("BlockInfo.zig").ColLimits;

const mask_r2p_chunk_size = 32;

fn sm90ColToR2pIdx(col_limit: i32) i32 {
    return @divFloor(col_limit, 8) * 2 + @min(@mod(col_limit, 8), 2);
}

pub fn maskR2p(values: anytype, col_limit_left: i32, col_limit_right: i32, comptime fill: f32) void {
    const count = @typeInfo(@TypeOf(values.*)).vector.len;
    inline for (0..(count + mask_r2p_chunk_size - 1) / mask_r2p_chunk_size) |s| {
        const masked_right = @as(i32, (s + 1) * mask_r2p_chunk_size) - col_limit_right;
        const masked_left = col_limit_left - @as(i32, s * mask_r2p_chunk_size);
        const below = nvptx.math.shf(.right, .clamp, 0xffff_ffff, 0, if (masked_right > 0) @bitCast(masked_right) else 0);
        const above = nvptx.math.shf(.left, .clamp, 0, 0xffff_ffff, if (masked_left > 0) @bitCast(masked_left) else 0);
        const keep = below & above;
        inline for (s * mask_r2p_chunk_size..@min(count, (s + 1) * mask_r2p_chunk_size)) |column| {
            values[column] = if (keep & (@as(u32, 1) << column % mask_r2p_chunk_size) != 0) values[column] else fill;
        }
    }
}

pub fn applyMask(acc_s: anytype, first_column: u32, limits: ColLimits, comptime fill: f32) void {
    const flat = @typeInfo(@TypeOf(acc_s.*)) == .vector;
    const tiles = if (flat) @typeInfo(@TypeOf(acc_s.*)).vector.len / 4 else acc_s.len;
    const thr_col_offset: i32 = @intCast(first_column + thread.lane() % 4 * 2);
    inline for (0..2) |half| {
        var row: @Vector(2 * tiles, f32) = undefined;
        inline for (0..tiles) |tile| {
            inline for (0..2) |pair| {
                row[2 * tile + pair] = if (flat) acc_s[tile * 4 + 2 * half + pair] else acc_s[tile][2 * half + pair];
            }
        }
        maskR2p(
            &row,
            sm90ColToR2pIdx(limits.col_limit_left[half] - thr_col_offset),
            sm90ColToR2pIdx(limits.col_limit_right[half] - thr_col_offset),
            fill,
        );
        inline for (0..tiles) |tile| {
            inline for (0..2) |pair| {
                if (flat) {
                    acc_s[tile * 4 + 2 * half + pair] = row[2 * tile + pair];
                } else {
                    acc_s[tile][2 * half + pair] = row[2 * tile + pair];
                }
            }
        }
    }
}
