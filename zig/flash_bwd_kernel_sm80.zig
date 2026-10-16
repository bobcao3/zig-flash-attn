const nvptx = @import("std").nvptx;
const BackwardParams = @import("BackwardParams.zig");
const thread = @import("thread.zig");

pub fn FlashAttnBwdSm80Kernel(comptime Components: type) type {
    return struct {
        pub fn run(params: BackwardParams, args: Components.Args) void {
            if (Components.pdl) {
                nvptx.griddepcontrol(.launch_dependents);
                if (Components.waits_for_scheduler_metadata) nvptx.griddepcontrol(.wait);
            }
            var work = Components.decodeWork(params) orelse {
                if (Components.pdl) nvptx.griddepcontrol(.wait);
                return;
            };
            var load = Components.Load.init(params, args, &work);
            load.start(&work);
            if (Components.pdl) nvptx.griddepcontrol(.wait);
            var gradients = Components.zeroDkv();
            const iterations = work.iterations();
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                const step = work.step(iteration);
                load.prefetch(iteration + 1, iterations, &work);
                const probabilities =
                    Components.computeProbabilities(params, iteration, &work, step, &load);
                Components.storePdS(probabilities);
                nvptx.barrierSyncAligned(.work_group, null);
                if (Components.alternate_gradient_order and
                    thread.warp() >= Components.thread_count / 64)
                {
                    Components.accumulateDq(params, &work, step, &load, iteration);
                    Components.accumulateDkv(&gradients, &load, iteration);
                } else if (Components.alternate_gradient_order) {
                    Components.accumulateDkv(&gradients, &load, iteration);
                    Components.accumulateDq(params, &work, step, &load, iteration);
                } else {
                    Components.accumulateDkv(&gradients, &load, iteration);
                    load.release(iteration, iterations, &work);
                    Components.accumulateDq(params, &work, step, &load, iteration);
                }
                load.finish(iteration, iterations, &work);
            }
            Components.drainDq();
            Components.storeGradients(params, &work, gradients);
        }
    };
}
