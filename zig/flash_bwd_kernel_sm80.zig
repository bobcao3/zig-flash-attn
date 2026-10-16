const std = @import("std");
const nvptx = std.nvptx;
const BackwardParams = @import("BackwardParams.zig");
const grid_dependency = @import("grid_dependency.zig");
const thread = @import("thread.zig");

pub fn FlashAttnBwdSm80Kernel(comptime Components: type) type {
    return struct {
        pub fn run(params: BackwardParams, args: Components.Args) void {
            if (Components.pdl) {
                grid_dependency.launchDependents();
                if (Components.waits_for_scheduler_metadata) grid_dependency.wait();
            }
            var work = Components.decode(params) orelse {
                if (Components.pdl) grid_dependency.wait();
                return;
            };
            var load = Components.Load.init(params, args, &work);
            load.start(&work);
            if (Components.pdl) grid_dependency.wait();
            var gradients = Components.Dkv.zero();
            const iterations = work.iterations();
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                const step = work.step(iteration);
                load.prefetch(iteration + 1, iterations, &work);
                const terms = Components.Scores.rowTerms(params, &work, step);
                const probabilities = Components.Scores.compute(params, iteration, &work, step, &load, terms);
                Components.PdS.store(probabilities);
                nvptx.barrierSyncAligned(.work_group, null);
                if (Components.alternate_gradient_order and thread.warp() >= Components.thread_count / 64) {
                    Components.Dq.accumulate(params, &work, step, &load, iteration);
                    Components.Dkv.accumulate(&gradients, &load, iteration);
                } else if (Components.alternate_gradient_order) {
                    Components.Dkv.accumulate(&gradients, &load, iteration);
                    Components.Dq.accumulate(params, &work, step, &load, iteration);
                } else {
                    Components.Dkv.accumulate(&gradients, &load, iteration);
                    load.release(iteration, iterations, &work);
                    Components.Dq.accumulate(params, &work, step, &load, iteration);
                }
                load.finish(iteration, iterations, &work);
            }
            Components.Dq.drain();
            Components.Epilogue.store(params, &work, gradients);
        }
    };
}
