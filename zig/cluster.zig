const std = @import("std");
const nvptx = std.nvptx;

pub fn rankIn2Cta() u32 {
    return asm ("mov.u32 %[rank], %cluster_ctarank;"
        : [rank] "=r" (-> u32),
    ) & 1;
}

pub fn leaderAddress(pointer: *addrspace(.shared) nvptx.Mbarrier) u32 {
    const target = asm ("mov.u32 %[rank], %cluster_ctarank;"
        : [rank] "=r" (-> u32),
    ) & ~@as(u32, 1);
    return asm ("mapa.shared::cluster.u32 %[mapped], %[local], %[target];"
        : [mapped] "=r" (-> u32),
        : [local] "r" (pointer),
          [target] "r" (target),
    );
}

pub fn arrive(cluster_address: u32) void {
    asm volatile ("mbarrier.arrive.shared::cluster.b64 _, [%[address]];"
        :
        : [address] "r" (cluster_address),
        : .{ .memory = true });
}

pub fn arriveReleaseCluster(cluster_address: u32) void {
    asm volatile ("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%[address]];"
        :
        : [address] "r" (cluster_address),
        : .{ .memory = true });
}

pub fn completeTxReleaseCluster(cluster_address: u32, bytes: u32) void {
    asm volatile ("fence.acq_rel.cluster;\nmbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [%[address]], %[bytes];"
        :
        : [address] "r" (cluster_address),
          [bytes] "r" (bytes),
        : .{ .memory = true });
}

pub fn arriveRelaxed() void {
    asm volatile ("barrier.cluster.arrive.relaxed.aligned;" ::: .{ .memory = true });
}

pub fn wait() void {
    asm volatile ("barrier.cluster.wait.aligned;\nfence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" ::: .{ .memory = true });
}

pub fn sync() void {
    asm volatile ("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: .{ .memory = true });
}

pub fn count() u32 {
    return asm ("mov.u32 %[count], %cluster_nctarank;"
        : [count] "=r" (-> u32),
    );
}

pub fn peerAddress(pointer: anytype) u32 {
    return asm ("mapa.shared::cluster.u32 %[mapped], %[local], %[target];"
        : [mapped] "=r" (-> u32),
        : [local] "r" (pointer),
          [target] "r" (rankIn2Cta() ^ 1),
    );
}

pub fn tryCancel(response: *addrspace(.shared) @Vector(4, u32), barrier: *addrspace(.shared) nvptx.Mbarrier, comptime multicast: bool) void {
    asm volatile ("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes" ++
            (if (multicast) ".multicast::cluster::all" else "") ++ ".b128 [%[response]], [%[barrier]];"
        :
        : [response] "r" (response),
          [barrier] "r" (barrier),
        : .{ .memory = true });
}

pub fn firstCtaid(response: *addrspace(.shared) const @Vector(4, u32), comptime dimension: u32) u32 {
    return asm volatile ("{\n.reg .b128 response;\n.reg .pred canceled;\n.reg .b32 ctaid;\n" ++
            "ld.shared.b128 response, [%[address]];\n" ++
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\n" ++
            "mov.b32 ctaid, -1;\n" ++
            "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::" ++ "xyz"[dimension..][0..1] ++ ".b32.b128 ctaid, response;\n" ++
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
    asm volatile (std.fmt.comptimePrint("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%[address]], {d};", .{bytes})
        :
        : [address] "r" (cluster_address),
        : .{ .memory = true });
}
