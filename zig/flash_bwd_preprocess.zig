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
    fragment_ordered_dq_accum: bool = false,
};

const dq_accum_row_chunks = config.head_dim / accumulator_elements;

/// Returns the index of a 16-byte chunk of `dq_accum` in the fragment order, given the float
/// offsets of a head and of a row within the head, and the chunk within the row. The fragment order
/// stores each group of 8 rows of a head as blocks of 8 columns. A block holds 64 floats: 4 even
/// columns of each of its 8 rows, then 4 odd columns of each row. A lane of a matrix multiply
/// accumulator holds one value of each column parity in one row, so a warp that adds one
/// accumulator value covers 32 consecutive floats.
pub fn fragmentOrderedChunk(head_offset: u64, row_offset: u64, chunk: u64) u64 {
    const linear = row_offset / accumulator_elements + chunk;
    const row = linear / dq_accum_row_chunks;
    const column = linear % dq_accum_row_chunks;
    return head_offset / accumulator_elements + (row - row % 8) * dq_accum_row_chunks +
        column / 2 * 16 + column % 2 * 8 + row % 8;
}

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

/// Returns the row of a fragment ordered `dq_accum` that holds the row `row` of `sequence`, which is
/// the sequence `batch` of the batch, or all sequences in a row with `varlen_as_one_sequence`. The
/// rows of sequence `b` of a varlen batch start at its offset rounded down to a multiple of 8, plus
/// `8 * b`, so that no group of 8 rows holds two sequences. The accumulator then needs 8 more rows
/// per sequence.
pub fn accumulatorRow(
    comptime options: Options,
    cu_seqlens_q: u64,
    num_batch: u32,
    sequence: SeqlenInfoQK,
    batch: u32,
    row: u32,
) u32 {
    const flat_row = sequence.offset_q + row;
    if (cu_seqlens_q == 0) return flat_row;
    if (!options.varlen_as_one_sequence) return flat_row - sequence.offset_q % 8 + 8 * batch;
    const starts: [*]addrspace(.global) const u32 = @ptrFromInt(cu_seqlens_q);
    var first: u32 = 0;
    var last: u32 = num_batch;
    while (last - first > 1) {
        const middle = (first + last) / 2;
        if (starts[middle] <= flat_row) first = middle else last = middle;
    }
    return flat_row - starts[first] % 8 + 8 * first;
}

pub fn FlashAttnBwdPreprocess(comptime options: Options) type {
    return struct {
        pub fn flashAttnBwdPreprocess(
            o: u64,
            do: u64,
            lse: u64,
            lse_log2: u64,
            dpsum: u64,
            pivot_sum: u64,
            pivot_key: u64,
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
            dk_states: u64,
            dk_state_chunks: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = auxiliary_threads },
        } }) void {
            if (options.pdl) nvptx.griddepcontrol(.launch_dependents);
            clearGridChunks(dk_states, dk_state_chunks);
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
                const pivot_sums: [*]addrspace(.global) f32 = @ptrFromInt(pivot_sum);
                const pivot_keys: [*]addrspace(.global) i32 = @ptrFromInt(pivot_key);
                if (elect.groupLeader(preprocess_lanes)) {
                    const value = lse_in[index];
                    lse_out[index] =
                        if (value == -std.math.inf(f32)) -std.math.inf(f32) else value *
                        -std.math.log2e;
                    sum_out[index] = -sum;
                    pivot_sums[index] = 0;
                    pivot_keys[index] = -1;
                }
                if (options.clear_dq_accum) {
                    const zeros: [*]addrspace(.global) Chunk = @ptrFromInt(dq_accum);
                    const row_offset = batch * dq_accum_batch_stride +
                        (if (options.fragment_ordered_dq_accum)
                            accumulatorRow(options, cu_seqlens_q, num_batch, sequence, batch, row)
                        else
                            sequence.offset_q + row) * dq_accum_row_stride;
                    const head_offset = head * dq_accum_head_stride;
                    const accum_base = (row_offset + head_offset) / accumulator_elements;
                    var zero_column: u32 = lane;
                    while (zero_column < dq_accum_row_chunks) : (zero_column += preprocess_lanes) {
                        zeros[
                            if (options.fragment_ordered_dq_accum)
                                fragmentOrderedChunk(head_offset, row_offset, zero_column)
                            else
                                accum_base + zero_column
                        ] = @splat(0);
                    }
                }
            }
        }
    };
}

fn clearGridChunks(pointer: u64, chunks: u32) void {
    const zeros: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
    const blocks = nvptx.nctaid(0) * nvptx.nctaid(1) * nvptx.nctaid(2);
    const block = @workGroupId(0) +
        nvptx.nctaid(0) * (@workGroupId(1) + nvptx.nctaid(1) * @workGroupId(2));
    var index = block * auxiliary_threads + thread.id();
    while (index < chunks) : (index += blocks * auxiliary_threads) zeros[index] = @splat(0);
}

pub fn flashAttnBwdClear(accumulator: u64, chunks: u32) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = auxiliary_threads },
} }) void {
    const zeros: [*]addrspace(.global) Chunk = @ptrFromInt(accumulator);
    const stride = nvptx.nctaid(0) * auxiliary_threads;
    var index = @workGroupId(0) * auxiliary_threads + thread.id();
    while (index < chunks) : (index += stride) zeros[index] = @splat(0);
}
