const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const fragments = @import("fragments.zig");
const thread = @import("thread.zig");

const Accumulator = fragments.Accumulator;
const Sm80BwdSchedule = @This();

/// Gives the query tile a second buffer, so that its next copy overlaps the current GEMMs.
double_buffered_query: bool = false,
/// Makes the upper half of the warps add dQ in parts between their dK and dV GEMM steps, and
/// starts the copy of the next query block between the S and dP GEMMs.
interleaved_dq_reduction: bool = false,
/// Splits the dQ GEMM into column parts; a warp adds each part before it computes the next.
dq_column_splits: u32 = 1,
/// Omits the `bar.warp.sync` before the CTA barriers of the main loop.
plain_loop_barriers: bool = false,
/// Starts the CTAs of the first key blocks of a bounded mask first, across several heads.
key_block_major: bool = false,
/// Decodes the work tile again after the main loop, which then holds fewer registers.
decode_work_twice: bool = false,

pub fn init(head_dim: u32) Sm80BwdSchedule {
    return switch (head_dim) {
        64 => .{ .decode_work_twice = true },
        96 => .{
            .double_buffered_query = true,
            .plain_loop_barriers = true,
            .key_block_major = true,
            .decode_work_twice = true,
        },
        128 => .{ .interleaved_dq_reduction = true, .decode_work_twice = true },
        256 => .{ .dq_column_splits = 2 },
        else => .{},
    };
}

pub fn queryStages(schedule: Sm80BwdSchedule, tiling: configurations.Sm80BwdTiling) u32 {
    return if (schedule.double_buffered_query) 2 else tiling.stages;
}

pub fn sharedBytes(schedule: Sm80BwdSchedule, tiling: configurations.Sm80BwdTiling) u32 {
    const query_buffers = schedule.queryStages(tiling) - tiling.stages;
    return tiling.sharedBytes() + query_buffers * tiling.tileBytes(tiling.block_m);
}

pub inline fn loopBarrier(comptime schedule: Sm80BwdSchedule) void {
    if (schedule.plain_loop_barriers) {
        asm volatile ("barrier.sync.aligned 0;" ::: .{ .memory = true });
    } else {
        nvptx.barrierSyncAligned(.work_group, null);
    }
}

/// Under a causal mask the first key blocks have the most query blocks. A group of heads holds
/// four key blocks per SM, and the first group takes the remainder, so that the last groups are
/// full.
pub fn orderedCtaid(
    comptime schedule: Sm80BwdSchedule,
    params: BackwardParams,
    ctaid: [3]u32,
) [3]u32 {
    if (!schedule.key_block_major) return ctaid;
    const bounded = params.window_size_left >= 0 or params.window_size_right >= 0;
    return if (bounded and nvptx.nctaid(0) >= 16) blockMajor(ctaid) else ctaid;
}

fn blockMajor(ctaid: [3]u32) [3]u32 {
    const blocks = nvptx.nctaid(0);
    const heads = nvptx.nctaid(1);
    const head_count = heads * nvptx.nctaid(2);
    const group_size = @min(head_count, (4 * nvptx.nsmid() + blocks - 1) / blocks);
    const first_group = head_count - (head_count - 1) / group_size * group_size;
    const linear = ctaid[0] + blocks * (ctaid[1] + heads * ctaid[2]);
    const in_first = linear < first_group * blocks;
    const rest = linear - first_group * blocks;
    const group = if (in_first) 0 else rest / (group_size * blocks);
    const within = if (in_first) linear else rest - group * group_size * blocks;
    const group_heads = if (in_first) first_group else group_size;
    const first_head = if (in_first) 0 else first_group + group * group_size;
    const head = first_head + within % group_heads;
    return .{ within / group_heads, head % heads, head / heads };
}

pub fn Steps(
    comptime Backward: type,
    comptime Storage: type,
    comptime tiling: configurations.Sm80BwdTiling,
    comptime loop_schedule: Sm80BwdSchedule,
) type {
    const head_dim = tiling.head_dim;
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const row_tiles = tiling.gradient_row_tiles;
    const warps_kv_n = block_n / (16 * row_tiles);
    const tiles_kv = tiling.rowChunks() / (tiling.warps / warps_kv_n);
    const warps_dq_m = block_m / (16 * row_tiles);
    const tiles_dq = tiling.rowChunks() / (tiling.warps / warps_dq_m);
    const query_stages = loop_schedule.queryStages(tiling);
    const Operands = Backward.Operands;
    const Work = Backward.Work;
    const DkvAccumulators = Backward.DkvAccumulators;
    const transposedGemm = Backward.transposedGemm;

    return struct {
        pub const schedule = loop_schedule;

        pub fn accumulateDqAndDkv(
            params: BackwardParams,
            work: *const Work,
            step: Backward.Step,
            accumulators: *DkvAccumulators,
            load: *const Storage.Load,
            iteration: u32,
            operands: *const Operands,
        ) void {
            const dq = queryGradientGemm(operands, {}, tiles_dq, 0);
            const gradient: QueryGradient(tiles_dq) = .init(params, work, step, dq, 0);
            Backward.accumulateDkv(accumulators, load, iteration, operands, &gradient);
        }

        pub fn reduceDq(
            params: BackwardParams,
            work: *const Work,
            step: Backward.Step,
            operands: *const Operands,
            a: DsFragments,
        ) void {
            const split_tiles = tiles_dq / schedule.dq_column_splits;
            inline for (0..schedule.dq_column_splits) |split| {
                const dq = queryGradientGemm(operands, a, split_tiles, split * split_tiles);
                const gradient: QueryGradient(split_tiles) =
                    .init(params, work, step, dq, split * split_tiles);
                gradient.add(0, QueryGradient(split_tiles).reductions);
            }
        }

        pub fn accumulateDk(
            accumulators: *DkvAccumulators,
            iteration: u32,
            operands: *const Operands,
        ) void {
            inline for (0..block_m / 16) |k_step| {
                const q_tile = Storage.qTile(iteration % query_stages);
                keyGradientStep(&accumulators.dk, Storage.dsTile(), q_tile, operands, k_step);
            }
        }

        fn keyGradientStep(
            accumulator: *[row_tiles][tiles_kv]Accumulator,
            a_tile: anytype,
            b_tile: anytype,
            operands: *const Operands,
            comptime k_step: u32,
        ) void {
            @setEvalBranchQuota(100_000);
            var a: [row_tiles]@Vector(4, u32) = undefined;
            inline for (&a, 0..) |*fragment, tile| fragment.* = nvptx.ldmatrix(
                .x4,
                true,
                operands.pds_columns.chunk(a_tile, k_step * 16, tile * 2),
            );
            transposedGemm(accumulator, tiles_kv, a, operands.gradient, b_tile, k_step, 0);
        }

        /// The warps load the operands of the last `release_k_steps` steps before `load.release`
        /// and multiply them after it.
        pub fn accumulateDvAndRelease(
            accumulators: *DkvAccumulators,
            load: *const Storage.Load,
            iteration: u32,
            count: u32,
            work: *const Work,
            operands: *const Operands,
        ) void {
            @setEvalBranchQuota(100_000);
            const k_steps_kv = block_m / 16;
            const release_k_steps = 2;
            comptime std.debug.assert(tiles_kv % 2 == 0);
            const do_tile = Storage.doTile(iteration % stages);
            inline for (0..k_steps_kv - release_k_steps) |k_step| {
                keyGradientStep(&accumulators.dv, Storage.pTile(), do_tile, operands, k_step);
            }
            var a: [release_k_steps][row_tiles]@Vector(4, u32) = undefined;
            var b: [release_k_steps][tiles_kv / 2]@Vector(4, u32) = undefined;
            inline for (&a, &b, k_steps_kv - release_k_steps..) |*step_a, *step_b, k_step| {
                inline for (step_a, 0..) |*p, tile| p.* = nvptx.ldmatrix(
                    .x4,
                    true,
                    operands.pds_columns.chunk(Storage.pTile(), k_step * 16, tile * 2),
                );
                inline for (step_b, 0..) |*fragment, j| fragment.* = nvptx.ldmatrix(
                    .x4,
                    true,
                    operands.gradient.pairs.chunk(do_tile, k_step * 16, 2 * j),
                );
            }
            load.release(iteration, count, work);
            inline for (a, b) |step_a, step_b| {
                inline for (step_b, 0..) |fragment, j| {
                    inline for (&accumulators.dv, step_a) |*row_accumulator, row_a| {
                        row_accumulator[2 * j] = nvptx.mmaSync(
                            fragments.mma_m16n8k16,
                            row_a,
                            @Vector(2, u32){ fragment[0], fragment[1] },
                            row_accumulator[2 * j],
                        );
                        row_accumulator[2 * j + 1] = nvptx.mmaSync(
                            fragments.mma_m16n8k16,
                            row_a,
                            @Vector(2, u32){ fragment[2], fragment[3] },
                            row_accumulator[2 * j + 1],
                        );
                    }
                }
            }
        }

        pub const DsFragments = [block_n / 16][row_tiles]@Vector(4, u32);

        fn queryGradientGemm(
            operands: *const Operands,
            a: anytype,
            comptime tile_count: u32,
            comptime first_tile: u32,
        ) [row_tiles][tile_count]Accumulator {
            @setEvalBranchQuota(100_000);
            var dq: [row_tiles][tile_count]Accumulator = @splat(@splat(@splat(0)));
            inline for (0..block_n / 16) |k_step| {
                var step_a: [row_tiles]@Vector(4, u32) = undefined;
                if (@TypeOf(a) == DsFragments) {
                    step_a = a[k_step];
                } else inline for (&step_a, 0..) |*fragment, tile| {
                    fragment.* = nvptx.ldmatrix(
                        .x4,
                        false,
                        operands.ds_rows.chunk(Storage.dsTile(), tile * 16, k_step * 2),
                    );
                }
                const keys = Storage.kTile();
                const operand = operands.query_gradient;
                transposedGemm(&dq, tile_count, step_a, operand, keys, k_step, first_tile);
            }
            return dq;
        }

        fn QueryGradient(comptime tile_count: u32) type {
            return struct {
                dq: [row_tiles][tile_count]Accumulator,
                address: [*]addrspace(.global) f32,
                first_column: u32,

                const Gradient = @This();
                pub const reductions = row_tiles * 2 * tile_count * 2;
                const padded = head_dim != tiling.headDimPadded();

                fn init(
                    params: BackwardParams,
                    work: *const Work,
                    step: Backward.Step,
                    dq: [row_tiles][tile_count]Accumulator,
                    comptime first_tile: u32,
                ) Gradient {
                    const warp_m = thread.warp() % warps_dq_m;
                    const warp_d = thread.warp() / warps_dq_m;
                    const lane = thread.lane();
                    const row = step.block_info.first_row + warp_m * 16 * row_tiles + lane / 4;
                    const first_column = (warp_d * tiles_dq + first_tile) * 8;
                    const base: [*]addrspace(.global) f32 = @ptrFromInt(params.dq_accum);
                    const sequence_rows = if (params.cu_seqlens_q == 0) params.seqlen_q else 8;
                    const group_row =
                        row + work.sequence.offset_q / 8 * 8 + work.queryBatch() * sequence_rows;
                    return .{
                        .dq = dq,
                        .address = base + step.head * params.dq_accum_head_stride +
                            @as(u64, group_row - group_row % 8) * head_dim +
                            (group_row % 8 * 4 + lane % 4 + first_column * 8),
                        .first_column = first_column,
                    };
                }

                /// Rows past the end of the sequence hold zeros, because their probabilities and
                /// output gradients are zero, so the warps add them to the padding or to the next
                /// sequence without a check.
                pub fn add(
                    gradient: *const Gradient,
                    comptime first: u32,
                    comptime count: u32,
                ) void {
                    inline for (first..@min(reductions, first + count)) |index| {
                        const row_tile = index / (2 * tile_count * 2);
                        const half = index / (tile_count * 2) % 2;
                        const tile = index / 2 % tile_count;
                        const parity = index % 2;
                        const offset = row_tile * 16 + half * 8;
                        if (!padded or gradient.first_column + tile * 8 < head_dim) nvptx.red(
                            .gpu,
                            .monotonic,
                            .add,
                            &gradient.address[offset * head_dim + tile * 64 + parity * 32],
                            gradient.dq[row_tile][tile][half * 2 + parity],
                        );
                    }
                }
            };
        }
    };
}
