const BlockInfo = @This();

first_row: u32,
row_count: u32,
seqlen_q: u32,
seqlen_k: u32,
window_size_left: i32,
window_size_right: i32,

pub const window_size_unbounded: i32 = -1;

pub fn causalRowOffset(info: BlockInfo) i32 {
    return @as(i32, @intCast(info.seqlen_k)) - @as(i32, @intCast(info.seqlen_q));
}

pub fn colLimitLeft(info: BlockInfo, row: i32) i32 {
    const causal_row_offset = info.causalRowOffset();
    if (info.window_size_left == window_size_unbounded) return 0;
    return @max(0, row + causal_row_offset - info.window_size_left);
}

pub fn colLimitRight(info: BlockInfo, row: i32) i32 {
    const causal_row_offset = info.causalRowOffset();
    const seqlen_k: i32 = @intCast(info.seqlen_k);
    if (info.window_size_right == window_size_unbounded) return seqlen_k;
    return @min(seqlen_k, row + causal_row_offset + info.window_size_right + 1);
}

pub fn rowLimitTop(info: BlockInfo, col: i32) i32 {
    const causal_row_offset = info.causalRowOffset();
    if (info.window_size_right == window_size_unbounded) return 0;
    return @max(0, col - causal_row_offset - info.window_size_right);
}

pub fn rowLimitBot(info: BlockInfo, col: i32) i32 {
    const causal_row_offset = info.causalRowOffset();
    const seqlen_q: i32 = @intCast(info.seqlen_q);
    if (info.window_size_left == window_size_unbounded) return seqlen_q;
    return @min(seqlen_q, col - causal_row_offset + info.window_size_left + 1);
}

pub fn nBlockMinMax(info: BlockInfo, comptime block_n: u32) struct { u32, u32 } {
    if (info.row_count == 0) return .{ 0, 0 };
    const seqlen_k: i32 = @intCast(info.seqlen_k);
    const last_row: i32 = @intCast(info.first_row + info.row_count - 1);
    const col_limit_left = info.colLimitLeft(@intCast(info.first_row));
    const col_limit_right = info.colLimitRight(last_row);
    if (col_limit_right <= col_limit_left or col_limit_left >= seqlen_k) return .{ 0, 0 };
    return .{
        @as(u32, @intCast(col_limit_left)) / block_n,
        (@as(u32, @intCast(col_limit_right)) + (block_n - 1)) / block_n,
    };
}

pub fn mBlockMinMax(info: BlockInfo, first_col: u32, col_count: u32, comptime block_m: u32) struct {
    u32,
    u32,
} {
    const seqlen_q: i32 = @intCast(info.seqlen_q);
    const row_limit_top = info.rowLimitTop(@intCast(first_col));
    const row_limit_bot = info.rowLimitBot(@intCast(first_col + col_count - 1));
    if (row_limit_bot <= row_limit_top or row_limit_top >= seqlen_q) return .{ 0, 0 };
    return .{
        @as(u32, @intCast(row_limit_top)) / block_m,
        (@as(u32, @intCast(row_limit_bot)) + (block_m - 1)) / block_m,
    };
}

pub fn colLimitBounds(info: BlockInfo) ColLimitBounds {
    return .{
        .col_limit_left_max = info.colLimitLeft(@intCast(info.first_row + info.row_count - 1)),
        .col_limit_right_min = info.colLimitRight(@intCast(info.first_row)),
    };
}

pub fn needsMask(
    info: BlockInfo,
    bounds: ColLimitBounds,
    n_block: u32,
    comptime block_n: u32,
) bool {
    const first_col: i32 = @intCast(n_block * block_n);
    const end_col: i32 = first_col + block_n;
    return first_col < bounds.col_limit_left_max or end_col > bounds.col_limit_right_min or
        end_col > @as(i32, @intCast(info.seqlen_k));
}

pub fn unmaskedBlocks(info: BlockInfo, bounds: ColLimitBounds, comptime block_n: u32) struct {
    u32,
    u32,
} {
    const left: u32 = @intCast(@max(0, bounds.col_limit_left_max));
    const right: u32 =
        @intCast(@max(0, @min(bounds.col_limit_right_min, @as(i32, @intCast(info.seqlen_k)))));
    return .{ (left + block_n - 1) / block_n, right / block_n };
}

pub const ColLimitBounds = struct {
    col_limit_left_max: i32,
    col_limit_right_min: i32,
};

pub const ColLimits = struct {
    col_limit_left: @Vector(2, i32),
    col_limit_right: @Vector(2, i32),

    pub fn init(info: BlockInfo, row: u32) ColLimits {
        const rows: @Vector(2, i32) = .{ @intCast(row), @intCast(row + 8) };
        return .{
            .col_limit_left = .{ info.colLimitLeft(rows[0]), info.colLimitLeft(rows[1]) },
            .col_limit_right = .{ info.colLimitRight(rows[0]), info.colLimitRight(rows[1]) },
        };
    }
};
