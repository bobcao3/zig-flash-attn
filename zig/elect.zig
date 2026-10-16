const nvptx = @import("std").nvptx;
const thread = @import("thread.zig");

pub fn oneSync() bool {
    if (nvptx.gpu.sm >= 90) return nvptx.electSync(.all);
    return thread.lane() == @ctz(nvptx.activemask());
}

pub fn groupLeader(comptime group_lanes: u32) bool {
    const first = thread.lane() / group_lanes * group_lanes;
    const mask: u32 = (@as(u32, 0xffff_ffff) >> @intCast(32 - group_lanes)) << @intCast(first);
    return thread.lane() == @ctz(nvptx.activemask() & mask);
}

pub fn reconverge() void {
    nvptx.barWarpSync(.all);
}
