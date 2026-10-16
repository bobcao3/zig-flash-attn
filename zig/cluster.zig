const std = @import("std");
const nvptx = std.nvptx;

pub const rank = nvptx.clusterCtarank;
pub const count = nvptx.clusterNctarank;

pub fn map(pointer: anytype, target: u32) u32 {
    const address: u32 = if (@TypeOf(pointer) == u32) pointer else @intCast(@intFromPtr(pointer));
    return nvptx.mapaSharedCluster(address, target);
}

pub fn rankIn2Cta() u32 {
    return rank() & 1;
}

pub fn leaderAddress(pointer: *addrspace(.shared) nvptx.Mbarrier) u32 {
    return map(pointer, rank() & ~@as(u32, 1));
}

pub fn arriveReleaseCluster(cluster_address: u32) void {
    nvptx.mbarrier.arriveCluster(cluster_address, .{ .semantics = .release, .scope = .cluster });
}

pub fn completeTxReleaseCluster(cluster_address: u32, bytes: u32) void {
    nvptx.fence(.acq_rel, .cluster);
    nvptx.mbarrier.completeTxCluster(cluster_address, bytes, .cluster);
}

pub fn wait() void {
    nvptx.barrierClusterWait(false, true);
    nvptx.fenceProxyAsyncSyncRestrict(.acquire);
}

pub fn sync() void {
    nvptx.barrierClusterArrive(.release, true);
    nvptx.barrierClusterWait(true, true);
}

pub fn peerAddress(pointer: anytype) u32 {
    return map(pointer, rankIn2Cta() ^ 1);
}

pub fn tryCancel(
    response: *addrspace(.shared) @Vector(4, u32),
    barrier: *addrspace(.shared) nvptx.Mbarrier,
    comptime multicast: bool,
) void {
    asm volatile ("clusterlaunchcontrol.try_cancel.async.shared::cta" ++
            ".mbarrier::complete_tx::bytes" ++
            (if (multicast) ".multicast::cluster::all" else "") ++
            ".b128 [%[response]], [%[barrier]];"
        :
        : [response] "r" (response),
          [barrier] "r" (barrier),
        : .{ .memory = true });
}

pub fn firstCtaid(
    response: *addrspace(.shared) const @Vector(4, u32),
    comptime dimension: u32,
) u32 {
    return asm volatile ("{\n.reg .b128 response;\n.reg .pred canceled;\n.reg .b32 ctaid;\n" ++
            "ld.shared.b128 response, [%[address]];\n" ++
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\n" ++
            "mov.b32 ctaid, -1;\n" ++
            "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::" ++
            "xyz"[dimension..][0..1] ++ ".b32.b128 ctaid, response;\n" ++
            "mov.b32 %[ctaid], ctaid;\n}"
        : [ctaid] "=r" (-> u32),
        : [address] "r" (response),
        : .{ .memory = true });
}

pub fn load(cluster_address: u32) @Vector(4, f32) {
    return asm volatile ("ld.shared::cluster.v4.f32 %[values], [%[address]];"
        : [values] "=f" (-> @Vector(4, f32)),
        : [address] "r" (cluster_address),
        : .{ .memory = true });
}

pub fn arriveExpectTx(cluster_address: u32, comptime bytes: u32) void {
    nvptx.mbarrier.arriveExpectTxCluster(cluster_address, bytes, null);
}
