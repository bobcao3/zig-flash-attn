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
                role_registers += role.warps * role.registers * 32;
                std.debug.assert(role.first_warp == next_warp);
                next_warp += role.warps;
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
            mbarrier.pipelineInitWait(Roles.init_scope, Roles.thread_count, if (initializes) .initializer else .other);
            if (@hasDecl(Roles, "tmem_warps")) handOffTmem();
            inline for (Roles.roles) |role| {
                if (thread.warp() >= role.first_warp and thread.warp() < role.first_warp + role.warps) {
                    if (role.registers > Roles.max_registers) {
                        asm volatile (std.fmt.comptimePrint("setmaxnreg.inc.sync.aligned.u32 {d};", .{role.registers}));
                    } else if (role.registers < Roles.max_registers) {
                        asm volatile (std.fmt.comptimePrint("setmaxnreg.dec.sync.aligned.u32 {d};", .{role.registers}));
                    }
                    role.Role.run(params, args);
                }
            }
        }

        fn handOffTmem() void {
            if (Roles.tmem_warps >> @intCast(thread.warp()) & 1 == 0) return;
            switch (Roles.init_scope) {
                .cta => {},
                .cluster => syncTmemWarps(),
                .launch => if (cluster.count() > 1) syncTmemWarps(),
            }
            tcgen05.fenceAfterSync();
        }

        fn syncTmemWarps() void {
            nvptx.barrierSyncAligned(Roles.tmem_barrier, comptime @as(u32, @popCount(Roles.tmem_warps)) * 32);
        }
    };
}
