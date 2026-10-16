const std = @import("std");
const nvptx = std.nvptx;
const Mbarrier = nvptx.Mbarrier;
const cluster = @import("cluster.zig");
const elect = @import("elect.zig");

pub const Order = enum { cta, cluster };

pub const Arrival = enum {
    thread,
    warp,
    warp_synced,
    cp_async,
    leader_cta,
    leader_release_cluster,
    peer_release_cluster,
};

pub const Scope = enum { cta, cluster, launch };

pub const Waiter = enum { initializer, other };

pub fn wait(comptime order: Order, barrier: *addrspace(.shared) Mbarrier, phase: u32) void {
    const parity = phase & 1;
    if (comptime nvptx.gpu.sm < 90) {
        comptime std.debug.assert(order == .cta);
        while (!barrier.testWaitPhase(parity)) {}
        return;
    }
    switch (order) {
        .cta => while (!barrier.tryWaitPhase(parity, null, null)) {},
        .cluster => while (!barrier.tryWaitPhase(phase, .{
            .semantics = .acquire,
            .scope = .cluster,
        }, null)) {},
    }
}

pub fn waitSuspendHint(
    barrier: *addrspace(.shared) Mbarrier,
    phase: u32,
    comptime nanoseconds: u32,
) void {
    if (nanoseconds == 0) return wait(.cta, barrier, phase);
    while (!barrier.tryWaitPhase(phase & 1, null, nanoseconds)) {}
}

pub fn waitBackoff(
    barrier: *addrspace(.shared) Mbarrier,
    phase: u32,
    comptime nanoseconds: u32,
) void {
    if (nanoseconds == 0) return wait(.cta, barrier, phase);
    while (!barrier.tryWaitParity(@truncate(phase))) nvptx.nanosleep(nanoseconds);
}

pub fn arrive(comptime arrival: Arrival, barrier: *addrspace(.shared) Mbarrier) void {
    switch (arrival) {
        .thread => _ = barrier.arrive(),
        .cp_async => nvptx.cp_async.mbarrierArriveNoinc(barrier),
        .warp, .warp_synced, .leader_cta, .leader_release_cluster, .peer_release_cluster => {
            if (arrival != .warp) nvptx.barWarpSync(.all);
            if (elect.oneSync()) switch (arrival) {
                .warp, .warp_synced => _ = barrier.arrive(),
                .leader_cta => nvptx.mbarrier.arriveCluster(cluster.leaderAddress(barrier), null),
                .leader_release_cluster => cluster.arriveReleaseCluster(
                    cluster.leaderAddress(barrier),
                ),
                .peer_release_cluster => cluster.arriveReleaseCluster(cluster.peerAddress(barrier)),
                else => comptime unreachable,
            };
            elect.reconverge();
        },
    }
}

pub fn arrivals(comptime arrival: Arrival, comptime warps: u32, ctas: u32) u32 {
    return switch (arrival) {
        .thread, .cp_async => 32 * warps,
        .warp, .warp_synced, .peer_release_cluster => warps,
        .leader_cta, .leader_release_cluster => ctas * warps,
    };
}

pub fn pipelineInitFence() void {
    if (nvptx.gpu.sm < 90) return;
    nvptx.fenceMbarrierInit();
    nvptx.fenceProxyAsync(.shared);
}

pub fn pipelineInitArrive(comptime scope: Scope) void {
    switch (scope) {
        .cta => {},
        .cluster => nvptx.barrierClusterArrive(.relaxed, true),
        .launch => if (cluster.count() > 1) nvptx.barrierClusterArrive(.relaxed, true),
    }
}

pub fn pipelineInitWait(comptime scope: Scope, comptime thread_count: u32, waiter: Waiter) void {
    switch (scope) {
        .cta => waitCta(thread_count, waiter),
        .cluster => cluster.wait(),
        .launch => if (cluster.count() > 1) cluster.wait() else waitCta(thread_count, waiter),
    }
}

fn waitCta(comptime thread_count: u32, waiter: Waiter) void {
    switch (waiter) {
        .initializer => nvptx.barrierArriveAligned(.work_group, thread_count),
        .other => nvptx.barrierSyncAligned(.work_group, thread_count),
    }
}

pub fn PipelineState(comptime stages: u32) type {
    comptime std.debug.assert(stages >= 1);
    return struct {
        count: u32,

        const Barriers = *addrspace(.shared) [stages]Mbarrier;

        pub fn index(state: @This()) u32 {
            return if (stages == 1) 0 else state.count % stages;
        }

        pub fn phase(state: @This()) u32 {
            return if (stages == 1) state.count else state.count / stages;
        }

        pub fn consumerWait(state: @This(), full: Barriers) void {
            wait(.cta, &full[state.index()], state.phase());
        }

        pub fn consumerRelease(state: @This(), comptime arrival: Arrival, empty: Barriers) void {
            arrive(arrival, &empty[state.index()]);
        }
    };
}

test "the pipeline state gives the slot and the phase parity of a ring" {
    inline for (1..5) |stages| {
        for (0..64) |count| {
            const state: PipelineState(stages) = .{ .count = @intCast(count) };
            try std.testing.expectEqual(count % stages, state.index());
            try std.testing.expectEqual(count / stages & 1, state.phase() & 1);
            if (count >= stages) try std.testing.expectEqual(
                (count / stages - 1) & 1,
                (state.phase() ^ 1) & 1,
            );
        }
    }
}

test "the arrival counts match the arrive calls" {
    try std.testing.expectEqual(@as(u32, 128), arrivals(.cp_async, 4, 1));
    try std.testing.expectEqual(@as(u32, 4), arrivals(.warp, 4, 1));
    try std.testing.expectEqual(@as(u32, 8), arrivals(.leader_cta, 4, 2));
    try std.testing.expectEqual(@as(u32, 4), arrivals(.peer_release_cluster, 4, 2));
}
