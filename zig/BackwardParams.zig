const BackwardParams = @This();
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");

q: u64 = 0,
k: u64 = 0,
v: u64 = 0,
do: u64 = 0,
lse_log2: u64 = 0,
dpsum: u64 = 0,
pivot_sum: u64 = 0,
pivot_key: u64 = 0,
dq_accum: u64 = 0,
dk: u64 = 0,
dv: u64 = 0,
cu_seqlens_q: u64 = 0,
cu_seqlens_k: u64 = 0,
seqlen_q: u32 = 0,
seqlen_k: u32 = 0,
qhead_per_kvhead: u32 = 0,
softmax_scale_log2: f32 = 0,
softmax_scale: f32 = 0,
window_size_left: i32 = 0,
window_size_right: i32 = 0,
q_batch_stride: u64 = 0,
q_row_stride: u64 = 0,
q_head_stride: u64 = 0,
k_batch_stride: u64 = 0,
k_row_stride: u64 = 0,
k_head_stride: u64 = 0,
v_batch_stride: u64 = 0,
v_row_stride: u64 = 0,
v_head_stride: u64 = 0,
do_batch_stride: u64 = 0,
do_row_stride: u64 = 0,
do_head_stride: u64 = 0,
dk_batch_stride: u64 = 0,
dk_row_stride: u64 = 0,
dk_head_stride: u64 = 0,
dv_batch_stride: u64 = 0,
dv_row_stride: u64 = 0,
dv_head_stride: u64 = 0,
dq_accum_batch_stride: u64 = 0,
dq_accum_row_stride: u64 = 0,
dq_accum_head_stride: u64 = 0,
lse_batch_stride: u64 = 0,
lse_head_stride: u64 = 0,
num_batch: u32 = 0,
scheduler_metadata: u64 = 0,
visibility_bits: u64 = 0,
window_sizes: u64 = 0,

/// `pivot_sum` holds `pivot_sum_scale` times the row sums of the BF16 dS. Kernels that sum a row in
/// parts add the parts with `red.add.f32`, which flushes subnormal inputs and results to zero (PTX
/// ISA 9.7.15.6). An FP32 sum of BF16 numbers is a multiple of 2^-133, the smallest BF16 subnormal
/// number, so each scaled part and each total of scaled parts is zero or at least 2^-126, the
/// smallest normal FP32 number, in magnitude. The scaled sums overflow once a row sum or one of its
/// parts reaches 2^121 in magnitude.
pub const pivot_sum_scale: f32 = 0x1p7;

pub const Gradient = enum { dk, dv };

pub fn gradientElement(
    params: BackwardParams,
    comptime gradient: Gradient,
    batch: u32,
    kv_head: u32,
    row: u64,
) u64 {
    return switch (gradient) {
        .dv => batch * params.dv_batch_stride +
            row * params.dv_row_stride + kv_head * params.dv_head_stride,
        .dk => batch * params.dk_batch_stride +
            row * params.dk_row_stride + kv_head * params.dk_head_stride,
    };
}

pub fn sequence(params: BackwardParams, batch: u32) SeqlenInfoQK {
    return .resolve(
        params.cu_seqlens_q,
        params.cu_seqlens_k,
        params.seqlen_q,
        params.seqlen_k,
        batch,
    );
}
