const std = @import("std");
const nvptx = std.nvptx;
const thread = @import("thread.zig");
const elect = @import("elect.zig");
const tma = @import("tma.zig");
const mbarrier = @import("mbarrier.zig");
const tcgen05 = @import("tcgen05.zig");
const cluster = @import("cluster.zig");

pub fn WarpSpecializedKernel(comptime Params: type, comptime Roles: type) type {
    return struct {
        comptime {
            var role_registers: u32 = 0;
            var next_warp: u32 = 0;
            for (Roles.roles) |role| {
                _, const warps, const registers = role;
                role_registers += warps * registers * 32;
                next_warp += warps;
            }
            std.debug.assert(role_registers <= Roles.max_registers * Roles.thread_count);
            std.debug.assert(next_warp == Roles.thread_count / 32);
            std.debug.assert(Roles.initializer_warp < next_warp);
            if (@hasDecl(Roles, "tmem_warps")) {
                std.debug.assert(Roles.tmem_warps >> (Roles.thread_count / 32) == 0);
                std.debug.assert(Roles.tmem_warps >> Roles.tmem_allocator_warp & 1 == 1);
                std.debug.assert(Roles.tmem_warps >> Roles.initializer_warp & 1 == 0);
            }
        }

        pub fn run(params: Params, args: Roles.Args) void {
            if (@hasDecl(Roles, "prepare")) Roles.prepare(params, args);
            Roles.initialize();
            if (@hasDecl(Roles, "tmem_warps") and thread.warp() == Roles.tmem_allocator_warp) {
                Roles.allocateTmem();
                tcgen05.fenceBeforeSync();
            }
            mbarrier.pipelineInitArrive(Roles.init_scope);
            const initializes = thread.warp() == Roles.initializer_warp;
            if (initializes) {
                if (elect.oneSync()) tma.prefetchMaps(args);
                elect.reconverge();
            }
            mbarrier.pipelineInitWait(
                Roles.init_scope,
                Roles.thread_count,
                if (initializes) .initializer else .other,
            );
            const tmem = @hasDecl(Roles, "tmem_warps");
            if (tmem and Roles.tmem_warps >> @intCast(thread.warp()) & 1 != 0) {
                const clustered = Roles.init_scope == .launch and cluster.count() > 1;
                if (Roles.init_scope == .cluster or clustered) {
                    const tmem_threads = comptime @as(u32, @popCount(Roles.tmem_warps)) * 32;
                    nvptx.barrierSyncAligned(Roles.tmem_barrier, tmem_threads);
                }
                tcgen05.fenceAfterSync();
            }
            comptime var first_warp: u32 = 0;
            inline for (Roles.roles) |role| {
                const runRole, const warps, const registers = role;
                if (thread.warp() >= first_warp and thread.warp() < first_warp + warps) {
                    if (registers > Roles.max_registers) {
                        asm volatile (std.fmt.comptimePrint(
                                "setmaxnreg.inc.sync.aligned.u32 {d};",
                                .{registers},
                            ));
                    } else if (registers < Roles.max_registers) {
                        asm volatile (std.fmt.comptimePrint(
                                "setmaxnreg.dec.sync.aligned.u32 {d};",
                                .{registers},
                            ));
                    }
                    runRole(params, args);
                }
                first_warp += warps;
            }
        }
    };
}
