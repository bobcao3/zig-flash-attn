const std = @import("std");
const nvptx = std.nvptx;
const config = @import("config");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const tile_scheduler = @import("tile_scheduler.zig");
const BlockInfo = @import("BlockInfo.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;

const preprocess_rows = 32;
pub const auxiliary_threads = 256;
const preprocess_lanes = auxiliary_threads / preprocess_rows;
pub const row_chunks = config.head_dim / chunk_elements;
const passes = (row_chunks + preprocess_lanes - 1) / preprocess_lanes;
pub const accumulator_elements = configurations.Sm80FwdTiling.chunk_bytes / @sizeOf(f32);

pub const Options = struct {
    clear_dq_accum: bool = true,
    pdl: bool = false,
    varlen_as_one_sequence: bool = false,
};

pub fn rowTile(
    comptime rows: u32,
    comptime varlen_as_one_sequence: bool,
    cu_seqlens_q: u64,
    num_batch: u32,
) ?tile_scheduler.WorkTileInfo {
    const Scheduler = tile_scheduler.SingleTileScheduler(.{
        .block_m = rows,
        .row_bytes = 0,
        .pack_gqa = false,
        .block_order = .ascending,
        .l2_swizzle = &.{},
    });
    const persistent_scheduler: tile_scheduler.PersistentScheduler(
        .single_tile,
        .{ .axis = .x, .ctas = 1 },
    ) = .create();
    const grid = tile_scheduler.gridTile(.x, 1, persistent_scheduler.initialWorkTileInfo());
    if (varlen_as_one_sequence and
        cu_seqlens_q != 0) return .{ .block = grid.block, .head = grid.head, .batch = 0 };
    const scheduler = Scheduler.create(.{
        .num_block = grid.num_block,
        .num_head = grid.num_head,
        .num_batch = num_batch,
        .seqlen_other = 0,
        .qhead_per_kvhead = 1,
        .window_size_left = BlockInfo.window_size_unbounded,
        .window_size_right = BlockInfo.window_size_unbounded,
        .cu_seqlens = cu_seqlens_q,
        .cu_seqlens_other = 0,
    });
    return scheduler.getGridWork(grid);
}

pub fn rowSequence(
    comptime varlen_as_one_sequence: bool,
    cu_seqlens_q: u64,
    num_batch: u32,
    seqlen_q: u32,
    batch: u32,
) SeqlenInfoQK {
    return if (varlen_as_one_sequence)
        SeqlenInfoQK.flatRows(cu_seqlens_q, num_batch, seqlen_q)
    else
        SeqlenInfoQK.resolve(cu_seqlens_q, cu_seqlens_q, seqlen_q, seqlen_q, batch);
}

pub fn FlashAttnBwdPreprocess(comptime options: Options) type {
    return struct {
        pub fn flashAttnBwdPreprocess(
            o: u64,
            do: u64,
            lse: u64,
            lse_log2: u64,
            dpsum: u64,
            dq_accum: u64,
            cu_seqlens_q: u64,
            seqlen_q: u32,
            o_batch_stride: u64,
            o_row_stride: u64,
            o_head_stride: u64,
            do_batch_stride: u64,
            do_row_stride: u64,
            do_head_stride: u64,
            lse_batch_stride: u64,
            lse_head_stride: u64,
            dq_accum_batch_stride: u64,
            dq_accum_row_stride: u64,
            dq_accum_head_stride: u64,
            num_batch: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = auxiliary_threads },
        } }) void {
            if (options.pdl) nvptx.griddepcontrol(.launch_dependents);
            const tile = rowTile(
                preprocess_rows,
                options.varlen_as_one_sequence,
                cu_seqlens_q,
                num_batch,
            ) orelse return;
            const batch = tile.batch;
            const block = tile.block;
            const head = tile.head;
            const sequence = rowSequence(
                options.varlen_as_one_sequence,
                cu_seqlens_q,
                num_batch,
                seqlen_q,
                batch,
            );
            const tid = thread.id();
            const lane = tid % preprocess_lanes;
            const row = block * preprocess_rows + tid / preprocess_lanes;
            const valid = row < sequence.seqlen_q;
            const clamped = if (valid) row else 0;
            const o_words: [*]addrspace(.global) const Chunk = @ptrFromInt(o);
            const do_words: [*]addrspace(.global) const Chunk = @ptrFromInt(do);
            const o_base = (batch * o_batch_stride + (sequence.offset_q + clamped) * o_row_stride +
                head * o_head_stride) / chunk_elements;
            const do_base = (batch * do_batch_stride + (sequence.offset_q + clamped) *
                do_row_stride + head * do_head_stride) / chunk_elements;
            const index =
                batch * lse_batch_stride + head * lse_head_stride + sequence.offset_q + row;
            var o_values: [passes]Chunk = @splat(@splat(0));
            var do_values: [passes]Chunk = @splat(@splat(0));
            inline for (0..passes) |pass| {
                const column = lane + pass * preprocess_lanes;
                if (column < row_chunks) {
                    o_values[pass] = o_words[o_base + column];
                    do_values[pass] = do_words[do_base + column];
                }
            }
            var sum: f32 = 0;
            inline for (o_values, do_values) |o_chunk, do_chunk| {
                inline for (0..4) |word| {
                    const o_pair = nvptx.Bf16x2.toF32(@fromBackingInt(o_chunk[word]));
                    const do_pair = nvptx.Bf16x2.toF32(@fromBackingInt(do_chunk[word]));
                    sum += o_pair[0] * do_pair[0] + o_pair[1] * do_pair[1];
                }
            }
            const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
            inline for (.{ 1, 2, 4 }) |mask| {
                sum += nvptx.shflSync(.bfly, f32, sum, mask, clamp, .all);
            }
            if (valid) {
                const lse_in: [*]addrspace(.global) const f32 = @ptrFromInt(lse);
                const lse_out: [*]addrspace(.global) f32 = @ptrFromInt(lse_log2);
                const sum_out: [*]addrspace(.global) f32 = @ptrFromInt(dpsum);
                if (elect.groupLeader(preprocess_lanes)) {
                    const value = lse_in[index];
                    lse_out[index] =
                        if (value == -std.math.inf(f32)) -std.math.inf(f32) else value *
                        -std.math.log2e;
                    sum_out[index] = -sum;
                }
                if (options.clear_dq_accum) {
                    const zeros: [*]addrspace(.global) Chunk = @ptrFromInt(dq_accum);
                    const accum_base = (batch * dq_accum_batch_stride +
                        (sequence.offset_q + row) * dq_accum_row_stride +
                        head * dq_accum_head_stride) / accumulator_elements;
                    var zero_column: u32 = lane;
                    while (zero_column < config.head_dim /
                        accumulator_elements) : (zero_column += preprocess_lanes)
                    {
                        zeros[accum_base + zero_column] = @splat(0);
                    }
                }
            }
        }
    };
}

pub fn flashAttnBwdClear(accumulator: u64, chunks: u32) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = auxiliary_threads },
} }) void {
    const zeros: [*]addrspace(.global) Chunk = @ptrFromInt(accumulator);
    const stride = nvptx.nctaid(0) * auxiliary_threads;
    var index = @workGroupId(0) * auxiliary_threads + thread.id();
    while (index < chunks) : (index += stride) zeros[index] = @splat(0);
}
