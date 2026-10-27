const std = @import("std");
const nvptx = std.nvptx;
const config = @import("config");
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const thread = @import("thread.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const BackwardParams = @import("BackwardParams.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const auxiliary_threads = flash_bwd_preprocess.auxiliary_threads;
const row_chunks = flash_bwd_preprocess.row_chunks;
const accumulator_elements = flash_bwd_preprocess.accumulator_elements;
const fragmentOrderedChunk = flash_bwd_preprocess.fragmentOrderedChunk;
const postprocess_rows = 16;
const vector_reductions = nvptx.gpu.sm >= 90;

const Row = [chunk_elements]f32;

/// Returns the floats of the 8-column chunk `column` of an accumulator row in column order. The
/// accumulator has the layout of `dq_accum`, in the fragment order when `fragment_order` is set.
fn loadAccumulatorRow(
    comptime fragment_order: bool,
    input: [*]addrspace(.global) const Accumulator,
    head_offset: u64,
    row_offset: u64,
    column: u32,
) Row {
    const chunk = column * (chunk_elements / accumulator_elements);
    const source = (row_offset + head_offset) / accumulator_elements + chunk;
    const low = input[
        if (fragment_order) fragmentOrderedChunk(head_offset, row_offset, chunk) else source
    ];
    const high = input[
        if (fragment_order) fragmentOrderedChunk(head_offset, row_offset, chunk + 1) else source + 1
    ];
    return if (fragment_order)
        .{ low[0], high[0], low[1], high[1], low[2], high[2], low[3], high[3] }
    else
        .{ low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3] };
}

const Update = enum { store, reduce, compare_and_swap };

const exact_add_limit: f32 = 0x1p-100;

fn addAccumulator(
    comptime update: Update,
    address: *addrspace(.global) f32,
    values: Accumulator,
) void {
    if (update == .compare_and_swap) {
        // red.add.f32 flushes subnormal inputs and results to zero (PTX ISA 9.7.15.6), so the
        // kernel adds a value below `exact_add_limit` in magnitude with compare-and-swap. A flush
        // in the add of a larger value loses less than 2^-126, which is less than 2^-26 times
        // that value.
        if (vector_reductions) {
            var large = true;
            inline for (0..4) |index| large = large and @abs(values[index]) >= exact_add_limit;
            if (large) return fragments.reduceAdd(address, values);
        }
        const floats: [*]addrspace(.global) f32 = @ptrCast(address);
        const words: [*]addrspace(.global) u32 = @ptrCast(address);
        inline for (0..4) |index| {
            const value = values[index];
            if (@abs(value) >= exact_add_limit) {
                nvptx.red(.gpu, .monotonic, .add, &floats[index], value);
            } else if (value != 0) {
                var old = nvptx.atomCas(.gpu, .monotonic, &words[index], 0, 0);
                while (true) {
                    const new: u32 = @bitCast(@as(f32, @bitCast(old)) + value);
                    const seen = nvptx.atomCas(.gpu, .monotonic, &words[index], old, new);
                    if (seen == old) break;
                    old = seen;
                }
            }
        }
    } else if (update == .store) {
        const chunk: *addrspace(.global) Accumulator = @ptrCast(@alignCast(address));
        chunk.* = values;
    } else if (vector_reductions) {
        fragments.reduceAdd(address, values);
    } else {
        const words: [*]addrspace(.global) f32 = @ptrCast(address);
        inline for (0..4) |index| nvptx.red(.gpu, .monotonic, .add, &words[index], values[index]);
    }
}

/// Adds `values`, the chunk `column` of a row in column order, to an accumulator with the layout
/// that `loadAccumulatorRow` reads, or stores them there when `update` is `.store`.
fn addAccumulatorRow(
    comptime fragment_order: bool,
    comptime update: Update,
    output: [*]addrspace(.global) f32,
    head_offset: u64,
    row_offset: u64,
    column: u32,
    values: Row,
) void {
    const chunk = column * (chunk_elements / accumulator_elements);
    if (fragment_order) {
        const even = fragmentOrderedChunk(head_offset, row_offset, chunk) * accumulator_elements;
        const odd = fragmentOrderedChunk(head_offset, row_offset, chunk + 1) * accumulator_elements;
        addAccumulator(update, &output[even], .{ values[0], values[2], values[4], values[6] });
        addAccumulator(update, &output[odd], .{ values[1], values[3], values[5], values[7] });
    } else {
        const first = row_offset + head_offset + chunk * accumulator_elements;
        addAccumulator(update, &output[first], values[0..4].*);
        addAccumulator(update, &output[first + accumulator_elements], values[4..8].*);
    }
}

fn joinRow(halves: [2]Accumulator) Row {
    const low: [4]f32 = halves[0];
    const high: [4]f32 = halves[1];
    return low ++ high;
}

fn unpackBf16Row(chunk: Chunk) Row {
    var values: Row = undefined;
    inline for (0..4) |pair| {
        const unpacked = nvptx.Bf16x2.toF32(@fromBackingInt(chunk[pair]));
        values[2 * pair] = unpacked[0];
        values[2 * pair + 1] = unpacked[1];
    }
    return values;
}

fn packBf16Row(values: Row) Chunk {
    var words: Chunk = undefined;
    inline for (0..4) |pair| words[pair] = @backingInt(
        nvptx.Bf16x2.fromF32(values[2 * pair], values[2 * pair + 1]),
    );
    return words;
}

pub fn FlashAttnBwdPostprocessConvertDq(comptime options: flash_bwd_preprocess.Options) type {
    return struct {
        var tile_keys: [postprocess_rows]i32 addrspace(.shared) = undefined;
        const warp_rows = if (nvptx.warp_size % row_chunks == 0)
            nvptx.warp_size / row_chunks
        else
            0;
        const group_rows = if (warp_rows > 0) warp_rows else postprocess_rows;
        var tile_gradients: [postprocess_rows * row_chunks][2]Accumulator addrspace(.shared) =
            undefined;

        /// Converts the rows of `dq_accum` to `dq`. The main pass leaves out the dS of the pivot
        /// key of a row, which equals the negated `pivot_sum` of the row divided by
        /// `BackwardParams.pivot_sum_scale`. With a nonzero `pivot_sum`, the kernel adds that dS
        /// times the pivot key row to the values that it loads from `dq_accum` before it scales
        /// and converts them, and it adds that dS times the softmax scale times the query row to
        /// the row of `dk_accum` that belongs to the pivot key. The rows of a warp, or of a work
        /// group when rows span warps, that share a pivot key add the sum of their products to
        /// `dk_accum`. When `dk_states` is nonzero, `dk_states` holds one word per 8-column chunk
        /// of `dk_accum`, zero where `dk_accum` holds no sum yet. The first thread that claims a
        /// chunk stores its sum there. The other threads add their sums after that store.
        pub fn flashAttnBwdPostprocess(
            dq_accum: u64,
            dq: u64,
            pivot_sum: u64,
            pivot_key: u64,
            q: u64,
            k: u64,
            dk_accum: u64,
            dk_states: u64,
            cu_seqlens_q: u64,
            seqlen_q: u32,
            qhead_per_kvhead: u32,
            softmax_scale: f32,
            dq_accum_batch_stride: u64,
            dq_accum_row_stride: u64,
            dq_accum_head_stride: u64,
            dq_batch_stride: u64,
            dq_row_stride: u64,
            dq_head_stride: u64,
            q_batch_stride: u64,
            q_row_stride: u64,
            q_head_stride: u64,
            k_batch_stride: u64,
            k_row_stride: u64,
            k_head_stride: u64,
            dk_accum_batch_stride: u64,
            dk_accum_row_stride: u64,
            dk_accum_head_stride: u64,
            lse_batch_stride: u64,
            lse_head_stride: u64,
            num_batch: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = auxiliary_threads },
            .min_work_groups_per_sm = if (nvptx.gpu.sm == 90) 2048 / auxiliary_threads else null,
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
            const fragment_order = options.fragment_ordered_dq_accum;
            const input: [*]addrspace(.global) const Accumulator = @ptrFromInt(dq_accum);
            const output: [*]addrspace(.global) Chunk = @ptrFromInt(dq);
            const keys: [*]addrspace(.global) const i32 = @ptrFromInt(pivot_key);
            const sums: [*]addrspace(.global) const f32 = @ptrFromInt(pivot_sum);
            const key_words: [*]addrspace(.global) const Chunk = @ptrFromInt(k);
            const query_words: [*]addrspace(.global) const Chunk = @ptrFromInt(q);
            const key_gradients: [*]addrspace(.global) f32 = @ptrFromInt(dk_accum);
            const key_states: [*]addrspace(.global) u32 = @ptrFromInt(dk_states);
            const kv_head = head / qhead_per_kvhead;
            const key_head_offset = kv_head * dk_accum_head_stride;
            const scatters = pivot_sum != 0;
            var has_pivots = false;
            var item: u32 = thread.id();
            while (item < postprocess_rows * row_chunks) : (item += auxiliary_threads) {
                const row = block * postprocess_rows + item / row_chunks;
                const column = item % row_chunks;
                if (scatters and column == 0) tile_keys[item / row_chunks] = -1;
                if (row < sequence.seqlen_q) {
                    const index =
                        batch * lse_batch_stride + head * lse_head_stride + sequence.offset_q + row;
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
                    var values =
                        loadAccumulatorRow(fragment_order, input, head_offset, row_offset, column);
                    const key: i32 = if (pivot_sum != 0) keys[index] else -1;
                    if (key >= 0) {
                        const key_row: u64 = @intCast(key);
                        const pivot_ds = sums[index] * (-1.0 / BackwardParams.pivot_sum_scale);
                        const key_values = unpackBf16Row(key_words[
                            (batch * k_batch_stride + key_row * k_row_stride +
                                kv_head * k_head_stride) / chunk_elements + column
                        ]);
                        const query_values = unpackBf16Row(query_words[
                            (batch * q_batch_stride + (sequence.offset_q + row) * q_row_stride +
                                head * q_head_stride) / chunk_elements + column
                        ]);
                        const scaled_ds = pivot_ds * softmax_scale;
                        var key_gradient: Row = undefined;
                        inline for (&values, &key_gradient, key_values, query_values) |
                            *value,
                            *gradient,
                            key_value,
                            query_value,
                        | {
                            value.* = @mulAdd(f32, pivot_ds, key_value, value.*);
                            gradient.* = scaled_ds * query_value;
                        }
                        tile_gradients[item] = .{ key_gradient[0..4].*, key_gradient[4..8].* };
                        if (column == 0) tile_keys[item / row_chunks] = key;
                        has_pivots = true;
                    }
                    inline for (&values) |*value| value.* *= softmax_scale;
                    const destination =
                        (batch * dq_batch_stride + (sequence.offset_q + row) * dq_row_stride +
                            head * dq_head_stride) / chunk_elements + column;
                    output[destination] = packBf16Row(values);
                }
            }
            if (!scatters) return;
            if (warp_rows > 0) {
                if (!nvptx.voteSync(.any, has_pivots, .all)) return;
                nvptx.barWarpSync(.all);
            } else if (!nvptx.barrierRedAligned(.@"or", .work_group, null, has_pivots)) return;
            item = thread.id();
            while (item < postprocess_rows * row_chunks) : (item += auxiliary_threads) {
                const tile_row = item / row_chunks;
                const column = item % row_chunks;
                const key = tile_keys[tile_row];
                if (key < 0) continue;
                const first_row = if (warp_rows > 0) tile_row - tile_row % warp_rows else 0;
                var first = true;
                inline for (0..group_rows) |offset| {
                    const other = first_row + offset;
                    if (other < tile_row and tile_keys[other] == key) first = false;
                }
                if (first) {
                    var sum = joinRow(tile_gradients[item]);
                    inline for (0..group_rows) |offset| {
                        const other = first_row + offset;
                        if (other > tile_row and tile_keys[other] == key) {
                            const values = joinRow(tile_gradients[other * row_chunks + column]);
                            inline for (&sum, values) |*total, value| total.* += value;
                        }
                    }
                    const key_row: u64 = @intCast(key);
                    const key_row_offset =
                        batch * dk_accum_batch_stride + key_row * dk_accum_row_stride;
                    const flat_row = (key_row_offset + key_head_offset) / config.head_dim;
                    const state = &key_states[flat_row * row_chunks + column];
                    if (dk_states != 0 and nvptx.atomCas(.gpu, .monotonic, state, 0, 1) == 0) {
                        addAccumulatorRow(
                            fragment_order,
                            .store,
                            key_gradients,
                            key_head_offset,
                            key_row_offset,
                            column,
                            sum,
                        );
                        _ = nvptx.atom(.gpu, .release, .exch, state, 2);
                        continue;
                    }
                    if (dk_states != 0) {
                        while (nvptx.atom(.gpu, .acquire, .@"or", state, 0) != 2) {}
                    }
                    if (dk_states != 0) addAccumulatorRow(
                        fragment_order,
                        .compare_and_swap,
                        key_gradients,
                        key_head_offset,
                        key_row_offset,
                        column,
                        sum,
                    ) else addAccumulatorRow(
                        fragment_order,
                        .reduce,
                        key_gradients,
                        key_head_offset,
                        key_row_offset,
                        column,
                        sum,
                    );
                }
            }
        }
    };
}

pub fn FlashAttnBwdFixDk(comptime options: flash_bwd_preprocess.Options) type {
    return struct {
        /// Adds the 8-column chunks of the FP32 `dk_accum` whose words in `dk_states` are nonzero
        /// to the BF16 `dk`.
        /// `dk_accum` holds `accumulator_rows` rows per head, and its row r of a head belongs to
        /// the key `r % batch_rows` of the batch `r / batch_rows`.
        pub fn flashAttnBwdFixDk(
            dk_accum: u64,
            dk_states: u64,
            dk: u64,
            rows: u32,
            accumulator_rows: u32,
            batch_rows: u32,
            dk_batch_stride: u64,
            dk_row_stride: u64,
            dk_head_stride: u64,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = auxiliary_threads },
        } }) void {
            if (options.pdl) nvptx.griddepcontrol(.wait);
            const input: [*]addrspace(.global) const Accumulator = @ptrFromInt(dk_accum);
            const states: [*]addrspace(.global) const u32 = @ptrFromInt(dk_states);
            const output: [*]addrspace(.global) Chunk = @ptrFromInt(dk);
            const stride = nvptx.nctaid(0) * auxiliary_threads;
            var item = @workGroupId(0) * auxiliary_threads + thread.id();
            while (item < rows * row_chunks) : (item += stride) {
                if (states[item] != 0) {
                    const flat_row = item / row_chunks;
                    const head = flat_row / accumulator_rows;
                    const row = flat_row % accumulator_rows;
                    const column = item % row_chunks;
                    const corrections = loadAccumulatorRow(
                        options.fragment_ordered_dq_accum,
                        input,
                        @as(u64, head) * accumulator_rows * config.head_dim,
                        @as(u64, row) * config.head_dim,
                        column,
                    );
                    const destination = (row / batch_rows * dk_batch_stride +
                        row % batch_rows * dk_row_stride + head * dk_head_stride) /
                        chunk_elements + column;
                    var values = unpackBf16Row(output[destination]);
                    inline for (&values, corrections) |*value, correction| value.* += correction;
                    output[destination] = packBf16Row(values);
                }
            }
        }
    };
}
