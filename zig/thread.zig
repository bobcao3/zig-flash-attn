const std = @import("std");
const nvptx = std.nvptx;

pub fn id() u32 {
    return @workItemId(0);
}

pub fn warp() u32 {
    return id() / 32;
}

pub fn ctaid() [3]u32 {
    return .{ @workGroupId(0), @workGroupId(1), @workGroupId(2) };
}

pub fn lane() u32 {
    return nvptx.laneid();
}
