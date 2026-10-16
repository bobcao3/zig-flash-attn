const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const configurations = @import("configurations.zig");
const tma = @import("tma.zig");
const sm80 = @import("flash_fwd_sm80.zig");

const variant = configurations.forwardSm120Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const forward_tiling = variant.tiling;
const Forward = sm80.FlashAttnFwdSm80(forward_tiling, variant.specialization, tma.SharedStorage(forward_tiling, variant.specialization));

fn flashAttnFwd(
    q_map: tma.TensorMap,
    k_map: tma.TensorMap,
    v_map: tma.TensorMap,
    o: u64,
    lse: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    seqlen_q: u32,
    seqlen_k: u32,
    qhead_per_kvhead: u32,
    softmax_scale_log2: f32,
    window_size_left: i32,
    window_size_right: i32,
    o_batch_stride: u64,
    o_row_stride: u64,
    o_head_stride: u64,
    lse_batch_stride: u64,
    lse_head_stride: u64,
    num_batch: u32,
    scheduler_metadata: if (variant.specialization.scheduler_metadata) u64 else void,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
} }) void {
    @setEvalBranchQuota(100_000);
    Forward.Kernel.run(.{
        .q = 0,
        .k = 0,
        .v = 0,
        .o = o,
        .lse = lse,
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_k = cu_seqlens_k,
        .seqlen_q = seqlen_q,
        .seqlen_k = seqlen_k,
        .qhead_per_kvhead = qhead_per_kvhead,
        .softmax_scale_log2 = softmax_scale_log2,
        .window_size_left = window_size_left,
        .window_size_right = window_size_right,
        .q_batch_stride = 0,
        .q_row_stride = 0,
        .q_head_stride = 0,
        .k_batch_stride = 0,
        .k_row_stride = 0,
        .k_head_stride = 0,
        .v_batch_stride = 0,
        .v_row_stride = 0,
        .v_head_stride = 0,
        .o_batch_stride = o_batch_stride,
        .o_row_stride = o_row_stride,
        .o_head_stride = o_head_stride,
        .lse_batch_stride = lse_batch_stride,
        .lse_head_stride = lse_head_stride,
        .num_m_blocks = std.nvptx.nctaid(0),
        .num_head = std.nvptx.nctaid(1),
        .num_batch = num_batch,
        .scheduler_metadata = if (variant.specialization.scheduler_metadata) scheduler_metadata else 0,
    }, .{ .q = &q_map, .k = &k_map, .v = &v_map });
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&flashAttnFwd, .{ .name = "flash_attn_fwd" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_fwd_prepare_varlen_num_blocks" });
        }
    }
}
