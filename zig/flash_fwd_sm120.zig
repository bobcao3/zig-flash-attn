const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const configurations = @import("configurations.zig");
const tma = @import("tma.zig");
const sm80 = @import("flash_fwd_sm80.zig");
const entry = @import("entry.zig");
const ForwardParams = @import("ForwardParams.zig");
const profile = @import("profile.zig");

const variant = configurations.variants(.flash_fwd_sm120, config.head_dim)[config.variant];
const forward_tiling = variant.tiling;
const Forward = sm80.FlashAttnFwdSm80(
    forward_tiling,
    variant.specialization,
    tma.SharedStorage(forward_tiling, variant.specialization),
);

fn launch(params: ForwardParams, args: Forward.Args) void {
    var grid_params = params;
    grid_params.num_m_blocks = std.nvptx.nctaid(0);
    grid_params.num_head = std.nvptx.nctaid(1);
    Forward.run(grid_params, args);
}

const Entry = entry.Entry(ForwardParams, Forward.Args, .{
    "q_map",             "k_map",            "v_map",              "o",
    "lse",               "cu_seqlens_q",     "cu_seqlens_k",       "seqlen_q",
    "seqlen_k",          "qhead_per_kvhead", "softmax_scale_log2", "window_size_left",
    "window_size_right",
} ++ entry.strides("o") ++ .{ "lse_batch_stride", "lse_head_stride", "num_batch" } ++
    if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}, .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
}, launch);

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_fwd" });
        profile.exportZones(sm80.Zone);
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_fwd_prepare_varlen_num_blocks",
            });
        }
    }
}
