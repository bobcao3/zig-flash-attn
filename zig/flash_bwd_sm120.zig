const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const configurations = @import("configurations.zig");
const tma = @import("tma.zig");
const sm80 = @import("flash_bwd_sm80.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");

const variant = configurations.backwardSm120Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const backward_tiling = variant.tiling;
const Storage = tma.BackwardStorage(backward_tiling);
const Backward = sm80.FlashAttnBwdSm80(backward_tiling, variant.specialization, Storage);
const auxiliary: flash_bwd_preprocess.Options = .{ .pdl = Backward.pdl, .varlen_as_one_sequence = variant.specialization.scheduler_metadata };

fn flashAttnBwdAtomic(
    q_map: tma.TensorMap,
    k_map: tma.TensorMap,
    v_map: tma.TensorMap,
    do_map: tma.TensorMap,
    lse_log2: u64,
    dpsum: u64,
    dq_accum: u64,
    dk: u64,
    dv: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    seqlen_q: u32,
    seqlen_k: u32,
    qhead_per_kvhead: u32,
    softmax_scale_log2: f32,
    softmax_scale: f32,
    window_size_left: i32,
    window_size_right: i32,
    dk_batch_stride: u64,
    dk_row_stride: u64,
    dk_head_stride: u64,
    dv_batch_stride: u64,
    dv_row_stride: u64,
    dv_head_stride: u64,
    dq_accum_batch_stride: u64,
    dq_accum_row_stride: u64,
    dq_accum_head_stride: u64,
    lse_batch_stride: u64,
    lse_head_stride: u64,
    num_batch: u32,
    scheduler_metadata: if (variant.specialization.scheduler_metadata) u64 else void,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = Backward.thread_count },
} }) void {
    @setEvalBranchQuota(100_000);
    Backward.Kernel.run(.{
        .q = 0,
        .k = 0,
        .v = 0,
        .do = 0,
        .lse_log2 = lse_log2,
        .dpsum = dpsum,
        .dq_accum = dq_accum,
        .dk = dk,
        .dv = dv,
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_k = cu_seqlens_k,
        .seqlen_q = seqlen_q,
        .seqlen_k = seqlen_k,
        .qhead_per_kvhead = qhead_per_kvhead,
        .softmax_scale_log2 = softmax_scale_log2,
        .softmax_scale = softmax_scale,
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
        .do_batch_stride = 0,
        .do_row_stride = 0,
        .do_head_stride = 0,
        .dk_batch_stride = dk_batch_stride,
        .dk_row_stride = dk_row_stride,
        .dk_head_stride = dk_head_stride,
        .dv_batch_stride = dv_batch_stride,
        .dv_row_stride = dv_row_stride,
        .dv_head_stride = dv_head_stride,
        .dq_accum_batch_stride = dq_accum_batch_stride,
        .dq_accum_row_stride = dq_accum_row_stride,
        .dq_accum_head_stride = dq_accum_head_stride,
        .lse_batch_stride = lse_batch_stride,
        .lse_head_stride = lse_head_stride,
        .num_batch = num_batch,
        .scheduler_metadata = if (variant.specialization.scheduler_metadata) scheduler_metadata else 0,
    }, .{ .q = &q_map, .k = &k_map, .v = &v_map, .do = &do_map, .dq_accum = {} });
}

fn flashAttnBwdBulk(
    q_map: tma.TensorMap,
    k_map: tma.TensorMap,
    v_map: tma.TensorMap,
    do_map: tma.TensorMap,
    dq_accum_map: tma.TensorMap,
    lse_log2: u64,
    dpsum: u64,
    dk: u64,
    dv: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    seqlen_q: u32,
    seqlen_k: u32,
    qhead_per_kvhead: u32,
    softmax_scale_log2: f32,
    softmax_scale: f32,
    window_size_left: i32,
    window_size_right: i32,
    dk_batch_stride: u64,
    dk_row_stride: u64,
    dk_head_stride: u64,
    dv_batch_stride: u64,
    dv_row_stride: u64,
    dv_head_stride: u64,
    lse_batch_stride: u64,
    lse_head_stride: u64,
    num_batch: u32,
    scheduler_metadata: if (variant.specialization.scheduler_metadata) u64 else void,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = Backward.thread_count },
} }) void {
    @setEvalBranchQuota(100_000);
    Backward.Kernel.run(.{
        .q = 0,
        .k = 0,
        .v = 0,
        .do = 0,
        .lse_log2 = lse_log2,
        .dpsum = dpsum,
        .dq_accum = 0,
        .dk = dk,
        .dv = dv,
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_k = cu_seqlens_k,
        .seqlen_q = seqlen_q,
        .seqlen_k = seqlen_k,
        .qhead_per_kvhead = qhead_per_kvhead,
        .softmax_scale_log2 = softmax_scale_log2,
        .softmax_scale = softmax_scale,
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
        .do_batch_stride = 0,
        .do_row_stride = 0,
        .do_head_stride = 0,
        .dk_batch_stride = dk_batch_stride,
        .dk_row_stride = dk_row_stride,
        .dk_head_stride = dk_head_stride,
        .dv_batch_stride = dv_batch_stride,
        .dv_row_stride = dv_row_stride,
        .dv_head_stride = dv_head_stride,
        .dq_accum_batch_stride = 0,
        .dq_accum_row_stride = 0,
        .dq_accum_head_stride = 0,
        .lse_batch_stride = lse_batch_stride,
        .lse_head_stride = lse_head_stride,
        .num_batch = num_batch,
        .scheduler_metadata = if (variant.specialization.scheduler_metadata) scheduler_metadata else 0,
    }, .{ .q = &q_map, .k = &k_map, .v = &v_map, .do = &do_map, .dq_accum = &dq_accum_map });
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(if (backward_tiling.bulk_reduce) &flashAttnBwdBulk else &flashAttnBwdAtomic, .{ .name = "flash_attn_bwd" });
        @export(&flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess, .{ .name = "flash_attn_bwd_preprocess" });
        @export(&flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary).flashAttnBwdPostprocess, .{ .name = "flash_attn_bwd_postprocess" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Backward.PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_bwd_prepare_varlen_num_blocks" });
        }
        if (variant.specialization.head_split) @export(&flash_bwd_preprocess.flashAttnBwdClear, .{ .name = "flash_attn_bwd_clear" });
    }
}
