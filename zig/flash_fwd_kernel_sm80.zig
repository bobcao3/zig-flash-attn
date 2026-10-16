const ForwardParams = @import("ForwardParams.zig");

pub fn FlashAttnFwdSm80Kernel(comptime Components: type) type {
    return struct {
        pub fn run(params: ForwardParams, args: Components.Args) void {
            var work = Components.Work.init(params) orelse return;
            var load = Components.Load.init(params, args, &work);
            load.start();
            const query = Components.Qk.loadQuery();
            load.queryConsumed();
            var softmax = Components.Softmax.init();
            var output = Components.Pv.zero();
            const tile_count = work.tile.n_block_max - work.tile.n_block_min;
            var index: u32 = 0;
            while (index < tile_count) : (index += 1) {
                const n_block = work.tile.n_block_min + index;
                const slot = load.acquireK(index);
                if (work.tile.needsMask(n_block)) {
                    const visible = work.visibleGroups(n_block);
                    var scores = Components.Qk.scores(query, slot, visible);
                    if (Components.separate_v_empty) load.releaseKeys(index);
                    Components.Softmax.mask(&scores, n_block, &work);
                    const step = Components.Softmax.step(&softmax, scores, params.softmax_scale_log2);
                    load.acquireV(index);
                    Components.Pv.accumulate(&output, step, slot, visible);
                } else {
                    const visible = Components.Work.all_groups;
                    const scores = Components.Qk.scores(query, slot, visible);
                    if (Components.separate_v_empty) load.releaseKeys(index);
                    const step = Components.Softmax.step(&softmax, scores, params.softmax_scale_log2);
                    load.acquireV(index);
                    Components.Pv.accumulate(&output, step, slot, visible);
                }
                if (Components.separate_v_empty) load.releaseValues(index) else load.release(index);
            }
            Components.Epilogue.store(params, &work, softmax, output);
        }
    };
}
