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
            const operands: Components.Operands = .init();
            const values = Components.loadValues(&operands);
            var gradients = Components.zeroDkv();
            const iterations = work.iterations();
            const scheduled = @hasDecl(Components.Load, "Steps");
            const Steps = if (scheduled) Components.Load.Steps(Components) else void;
            const dq_first = scheduled and Steps.schedule.interleaved_dq_reduction and
                thread.warp() >= Components.thread_count / 64;
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                const step = work.branchedStep(iteration);
                load.prefetch(iteration + 1, iterations, &work);
                const probabilities =
                    Components.computeProbabilities(
                        params,
                        iteration,
                        &work,
                        step,
                        &load,
                        &operands,
                        &values,
                    );
                Components.storePdS(probabilities, &operands);
                if (scheduled) {
                    Steps.schedule.loopBarrier();
                } else {
                    nvptx.barrierSyncAligned(.work_group, null);
                }
                if (dq_first) {
                    Steps.accumulateDqAndDkv(
                        params,
                        &work,
                        step,
                        &gradients,
                        &load,
                        iteration,
                        &operands,
                    );
                } else if (Components.alternate_gradient_order and
                    thread.warp() >= Components.thread_count / 64)
                {
                    Components.accumulateDq(params, &work, step, &load, iteration, &operands);
                    Components.accumulateDkv(&gradients, &load, iteration, &operands, {});
                } else if (Components.alternate_gradient_order) {
                    Components.accumulateDkv(&gradients, &load, iteration, &operands, {});
                    Components.accumulateDq(params, &work, step, &load, iteration, &operands);
                } else if (scheduled and Steps.schedule.double_buffered_query) {
                    Steps.accumulateDvAndRelease(
                        &gradients,
                        &load,
                        iteration,
                        iterations,
                        &work,
                        &operands,
                    );
                    Components.accumulateDq(params, &work, step, &load, iteration, &operands);
                    Steps.accumulateDk(&gradients, iteration, &operands);
                } else {
                    Components.accumulateDkv(&gradients, &load, iteration, &operands, {});
                    load.release(iteration, iterations, &work);
                    Components.accumulateDq(params, &work, step, &load, iteration, &operands);
                }
                load.finish(iteration, iterations, &work);
            }
            Components.drainDq();
            Components.storeGradients(params, &work, gradients);
        }
    };
}
