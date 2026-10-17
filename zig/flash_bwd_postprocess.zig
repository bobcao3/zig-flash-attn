const nvptx = @import("std").nvptx;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const thread = @import("thread.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");

const Chunk = fragments.Chunk;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const auxiliary_threads = flash_bwd_preprocess.auxiliary_threads;
const row_chunks = flash_bwd_preprocess.row_chunks;
const accumulator_elements = flash_bwd_preprocess.accumulator_elements;
const fragmentOrderedChunk = flash_bwd_preprocess.fragmentOrderedChunk;
const postprocess_rows = 16;

pub fn FlashAttnBwdPostprocessConvertDq(comptime options: flash_bwd_preprocess.Options) type {
    return struct {
        pub fn flashAttnBwdPostprocess(
            dq_accum: u64,
            dq: u64,
            cu_seqlens_q: u64,
            seqlen_q: u32,
            softmax_scale: f32,
            dq_accum_batch_stride: u64,
            dq_accum_row_stride: u64,
            dq_accum_head_stride: u64,
            dq_batch_stride: u64,
            dq_row_stride: u64,
            dq_head_stride: u64,
            num_batch: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = auxiliary_threads },
        } }) void {
            const tile = flash_bwd_preprocess.rowTile(
                postprocess_rows,
                options.varlen_as_one_sequence,
                cu_seqlens_q,
                num_batch,
            ) orelse return;
            const batch = tile.batch;
            const block = tile.block;
            const head = tile.head;
            const sequence = flash_bwd_preprocess.rowSequence(
                options.varlen_as_one_sequence,
                cu_seqlens_q,
                num_batch,
                seqlen_q,
                batch,
            );
            if (options.pdl) nvptx.griddepcontrol(.wait);
            const input: [*]addrspace(.global) const fragments.Accumulator = @ptrFromInt(dq_accum);
            const output: [*]addrspace(.global) Chunk = @ptrFromInt(dq);
            var item: u32 = thread.id();
            while (item < postprocess_rows * row_chunks) : (item += auxiliary_threads) {
                const row = block * postprocess_rows + item / row_chunks;
                const column = item % row_chunks;
                if (row < sequence.seqlen_q) {
                    const fragment_order = options.fragment_ordered_dq_accum;
                    const row_offset = batch * dq_accum_batch_stride +
                        (if (fragment_order) flash_bwd_preprocess.accumulatorRow(
                            options,
                            cu_seqlens_q,
                            num_batch,
                            sequence,
                            batch,
                            row,
                        ) else sequence.offset_q + row) * dq_accum_row_stride;
                    const head_offset = head * dq_accum_head_stride;
                    const chunk = column * (chunk_elements / accumulator_elements);
                    const source = (row_offset + head_offset) / accumulator_elements + chunk;
                    const scale: fragments.Accumulator = @splat(softmax_scale);
                    const low = input[
                        if (fragment_order)
                            fragmentOrderedChunk(head_offset, row_offset, chunk)
                        else
                            source
                    ] * scale;
                    const high = input[
                        if (fragment_order)
                            fragmentOrderedChunk(head_offset, row_offset, chunk + 1)
                        else
                            source + 1
                    ] * scale;
                    const values: [8]f32 = if (fragment_order)
                        .{ low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3] }
                    else
                        .{ low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3] };
                    const destination =
                        (batch * dq_batch_stride + (sequence.offset_q + row) * dq_row_stride +
                            head * dq_head_stride) / chunk_elements + column;
                    var words: [4]u32 = undefined;
                    inline for (&words, 0..) |*word, pair| word.* = @backingInt(
                        nvptx.Bf16x2.fromF32(values[2 * pair], values[2 * pair + 1]),
                    );
                    output[destination] = words;
                }
            }
        }
    };
}
