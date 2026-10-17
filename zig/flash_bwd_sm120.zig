const builtin = @import("builtin");
const config = @import("config");
const BackwardParams = @import("BackwardParams.zig");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const tma = @import("tma.zig");
const sm80 = @import("flash_bwd_sm80.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");

const variant = configurations.backwardSm120Variant(
    config.head_dim,
    @field(configurations.Mode, config.mode),
).?;
const backward_tiling = variant.tiling;
const Storage = tma.BackwardStorage(backward_tiling);
const Backward = sm80.FlashAttnBwdSm80(backward_tiling, variant.specialization, Storage);
const auxiliary: flash_bwd_preprocess.Options = .{
    .pdl = Backward.pdl,
    .varlen_as_one_sequence = variant.specialization.scheduler_metadata,
    .fragment_ordered_dq_accum = sm80.fragment_ordered_dq_accum,
};

const bulk = backward_tiling.bulk_reduce;
const Entry = entry.Entry(BackwardParams, Backward.Args, .{ "q_map", "k_map", "v_map", "do_map" } ++
    (if (bulk) .{"dq_accum_map"} else .{}) ++ .{ "lse_log2", "dpsum" } ++
    (if (bulk) .{} else .{"dq_accum"}) ++ .{ "dk", "dv" } ++ entry.backward_sequence ++
    entry.strides("dk") ++ entry.strides("dv") ++ (if (bulk) .{} else entry.strides("dq_accum")) ++
    .{ "lse_batch_stride", "lse_head_stride", "num_batch" } ++
    if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}, .{
    .required_work_group_size = .{ .x = Backward.thread_count },
}, Backward.Kernel.run);

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_bwd" });
        @export(
            &flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess,
            .{ .name = "flash_attn_bwd_preprocess" },
        );
        @export(
            &flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary)
                .flashAttnBwdPostprocess,
            .{ .name = "flash_attn_bwd_postprocess" },
        );
        if (variant.specialization.scheduler_metadata) {
            @export(&Backward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_bwd_prepare_varlen_num_blocks",
            });
        }
        if (variant.specialization.head_split) @export(
            &flash_bwd_preprocess.flashAttnBwdClear,
            .{ .name = "flash_attn_bwd_clear" },
        );
    }
}
