const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const config = @import("config");
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const FlashAttnBwdSm80Kernel = @import("flash_bwd_kernel_sm80.zig").FlashAttnBwdSm80Kernel;
const NBlockWork = @import("n_block_work.zig").NBlockWork;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const applyMask = @import("mask.zig").applyMask;
const ColLimits = @import("BlockInfo.zig").ColLimits;
const smem = @import("smem.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const RowPair = fragments.RowPair;
const chunk_bytes = configurations.Sm80FwdTiling.chunk_bytes;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;

const vector_reductions = nvptx.gpu.sm >= 90;

pub fn SharedStorage(comptime tiling: configurations.Sm80BwdTiling) type {
    const padded_chunks = tiling.rowChunks();
    const real_chunks = tiling.head_dim / chunk_elements;
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const threads = tiling.threads();
    const Rows = smem.Layout(padded_chunks);
    const PdS = smem.Layout(block_n / chunk_elements);

    return struct {
        pub const Layout = Rows;
        pub const PdSLayout = PdS;
        pub const shared_bytes = tiling.sharedBytes();
        pub const Args = struct {};

        const Tiles = struct {
            k: Rows.Tile(block_n),
            v: Rows.Tile(block_n),
            q: [stages]Rows.Tile(block_m),
            do: [stages]Rows.Tile(block_m),
            p: PdS.Tile(block_m),
            ds: PdS.Tile(block_m),
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);

        pub fn kTile() *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.k;
        }

        pub fn vTile() *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.v;
        }

        pub fn qTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.q[buffer];
        }

        pub fn doTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.do[buffer];
        }

        pub fn pTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &tiles.p;
        }

        pub fn dsTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &tiles.ds;
        }

        pub const Load = struct {
            params: BackwardParams,

            pub fn init(params: BackwardParams, args: Args, work: anytype) Load {
                _ = args;
                _ = work;
                return .{ .params = params };
            }

            fn copyRows(
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                base: u64,
                origin: u64,
                stride: u64,
                first_row: u32,
                limit: u32,
            ) void {
                const source: [*]addrspace(.global) const Chunk = @ptrFromInt(base);
                const chunks = count * padded_chunks;
                const tid = thread.id();
                inline for (0..(chunks + threads - 1) / threads) |pass| {
                    const linear: u32 = @as(u32, pass) * threads + tid;
                    if ((pass + 1) * threads <= chunks or linear < chunks) {
                        const row = linear / padded_chunks;
                        const column = linear % padded_chunks;
                        const valid = first_row + row < limit and column < real_chunks;
                        const source_row = if (valid) first_row + row else 0;
                        const source_column = if (valid) column else 0;
                        cp_async.copyZeroFill(
                            .cache_global_level,
                            &destination[Rows.slot(row, column)],
                            &source[origin / chunk_elements + source_row * stride / chunk_elements + source_column],
                            if (valid) chunk_bytes else 0,
                        );
                    }
                }
            }

            fn copyQ(load: *const Load, work: anytype, iteration: u32) void {
                const step = work.step(iteration);
                const params = load.params;
                const sequence = work.sequence;
                const q_origin = work.batch * params.q_batch_stride + sequence.offset_q * params.q_row_stride +
                    step.head * params.q_head_stride;
                copyRows(block_m, &tiles.q[iteration % stages], params.q, q_origin, params.q_row_stride, step.block_info.first_row, sequence.seqlen_q);
            }

            fn copyDo(load: *const Load, work: anytype, iteration: u32) void {
                const step = work.step(iteration);
                const params = load.params;
                const sequence = work.sequence;
                const do_origin = work.batch * params.do_batch_stride + sequence.offset_q * params.do_row_stride +
                    step.head * params.do_head_stride;
                copyRows(block_m, &tiles.do[iteration % stages], params.do, do_origin, params.do_row_stride, step.block_info.first_row, sequence.seqlen_q);
            }

            fn copyQuery(load: *const Load, work: anytype, iteration: u32) void {
                load.copyQ(work, iteration);
                load.copyDo(work, iteration);
            }

            pub fn start(load: *const Load, work: anytype) void {
                if (work.iterations() == 0) return;
                const params = load.params;
                const sequence = work.sequence;
                const k_origin = work.batch * params.k_batch_stride + sequence.offset_k * params.k_row_stride +
                    work.kv_head * params.k_head_stride;
                const v_origin = work.batch * params.v_batch_stride + sequence.offset_k * params.v_row_stride +
                    work.kv_head * params.v_head_stride;
                copyRows(block_n, &tiles.k, params.k, k_origin, params.k_row_stride, work.first_key, sequence.seqlen_k);
                load.copyQ(work, 0);
                cp_async.commitGroup();
                copyRows(block_n, &tiles.v, params.v, v_origin, params.v_row_stride, work.first_key, sequence.seqlen_k);
                load.copyDo(work, 0);
                cp_async.commitGroup();
                cp_async.waitGroup(1);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn acquireDo(load: *const Load, iteration: u32, work: anytype) void {
                _ = load;
                if (iteration > 0) return;
                if (stages == 2 and work.iterations() > 1) cp_async.waitGroup(1) else cp_async.waitGroup(0);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn prefetch(load: *const Load, next: u32, count: u32, work: anytype) void {
                if (stages == 2 and next < count) {
                    load.copyQuery(work, next);
                    cp_async.commitGroup();
                }
            }

            pub fn release(load: *const Load, iteration: u32, count: u32, work: anytype) void {
                if (stages == 1 and iteration + 1 < count) {
                    nvptx.barrierSyncAligned(.work_group, null);
                    load.copyQuery(work, iteration + 1);
                    cp_async.commitGroup();
                }
            }

            pub fn finish(load: *const Load, iteration: u32, count: u32, work: anytype) void {
                _ = load;
                _ = iteration;
                _ = count;
                _ = work;
                cp_async.waitGroup(0);
                nvptx.barrierSyncAligned(.work_group, null);
            }
        };
    };
}

pub fn FlashAttnBwdSm80(comptime tiling: configurations.Sm80BwdTiling, comptime specialization: configurations.Specialization, comptime Storage: type) type {
    const head_dim = tiling.head_dim;
    const padded_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const threads = tiling.threads();
    const warps = tiling.warps;
    const warps_m = block_m / 16;
    const warps_n = warps / warps_m;
    const columns_per_warp = block_n / warps_n;
    const warps_kv_n = block_n / 16;
    const warps_kv_d = warps / warps_kv_n;
    const tiles_kv = padded_chunks / warps_kv_d;
    const warps_dq_d = warps / warps_m;
    const tiles_dq = padded_chunks / warps_dq_d;
    const k_steps = padded_chunks / 2;
    const Rows = Storage.Layout;
    const PdSLayout = Storage.PdSLayout;
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_n,
        .row_bytes = 8 * head_dim,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const PersistentScheduler = tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 1 });

    comptime {
        if (warps % warps_m != 0 or block_n % warps_n != 0 or columns_per_warp % 16 != 0)
            @compileError("the score tile does not divide among the warps");
        if (warps % warps_kv_n != 0 or padded_chunks % warps_kv_d != 0)
            @compileError("the gradient tile does not divide among the warps");
        if (padded_chunks % warps_dq_d != 0) @compileError("the query gradient does not divide among the warps");
        if (tiling.alternate_gradient_order and (stages != 2 or warps % 2 != 0)) @compileError("the split gradient order needs two query stages and an even warp count");
    }

    return struct {
        pub const thread_count = threads;
        pub const shared_bytes = Storage.shared_bytes;
        pub const pdl = tiling.pdl;
        pub const alternate_gradient_order = tiling.alternate_gradient_order;
        pub const waits_for_scheduler_metadata = specialization.scheduler_metadata;
        pub const Args = Storage.Args;
        pub const Load = Storage.Load;
        pub const Work = NBlockWork(block_m, block_n);
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(scheduler_options.block_m, scheduler_options.block_order, specialization.sort_batches);

        const Step = Work.Step;

        const Terms = struct {
            lse_negated: RowPair,
            dpsum_negated: RowPair,
        };

        const Probabilities = struct {
            p: [columns_per_warp / 8]@Vector(2, u32),
            ds: [columns_per_warp / 8]@Vector(2, u32),
        };

        fn scoresGemm(a_tile: anytype, b_tile: anytype) [columns_per_warp / 8]Accumulator {
            @setEvalBranchQuota(100_000);
            const warp_m = thread.warp() % warps_m;
            const warp_n = thread.warp() / warps_m;
            const lane = thread.lane();
            const lane_row = lane % 8;
            var scores: [columns_per_warp / 8]Accumulator = @splat(@splat(0));
            inline for (0..k_steps) |step_index| {
                const k_step: u32 = step_index;
                const a = nvptx.ldmatrix(.x4, false, Rows.chunk(a_tile, warp_m * 16 + lane % 16, k_step * 2 + lane / 16, lane_row));
                inline for (0..columns_per_warp / 16) |pair_index| {
                    const j: u32 = pair_index;
                    const row = warp_n * columns_per_warp + j * 16 + lane_row + lane / 16 * 8;
                    const b = nvptx.ldmatrix(.x4, false, Rows.chunk(b_tile, row, k_step * 2 + lane / 8 % 2, lane_row));
                    scores[2 * j] = nvptx.mmaSync(fragments.mma_m16n8k16, a, @Vector(2, u32){ b[0], b[1] }, scores[2 * j]);
                    scores[2 * j + 1] = nvptx.mmaSync(fragments.mma_m16n8k16, a, @Vector(2, u32){ b[2], b[3] }, scores[2 * j + 1]);
                }
            }
            return scores;
        }

        fn transposedGemm(
            accumulator: anytype,
            comptime tile_count: u32,
            a: anytype,
            b_tile: anytype,
            comptime k_step: u32,
            first_tile: u32,
        ) void {
            @setEvalBranchQuota(100_000);
            const lane = thread.lane();
            const lane_row = lane % 8;
            inline for (0..tile_count / 2) |pair_index| {
                const j: u32 = pair_index;
                const b = nvptx.ldmatrix(.x4, true, Rows.chunk(b_tile, k_step * 16 + lane_row + lane / 8 % 2 * 8, first_tile + 2 * j + lane / 16, lane_row));
                accumulator[2 * j] = nvptx.mmaSync(fragments.mma_m16n8k16, a, @Vector(2, u32){ b[0], b[1] }, accumulator[2 * j]);
                accumulator[2 * j + 1] = nvptx.mmaSync(fragments.mma_m16n8k16, a, @Vector(2, u32){ b[2], b[3] }, accumulator[2 * j + 1]);
            }
            if (tile_count % 2 == 1) {
                const b = nvptx.ldmatrix(.x2, true, Rows.chunk(b_tile, k_step * 16 + lane % 16, first_tile + tile_count - 1, lane_row));
                accumulator[tile_count - 1] = nvptx.mmaSync(fragments.mma_m16n8k16, a, b, accumulator[tile_count - 1]);
            }
        }

        pub fn decode(params: BackwardParams) ?Work {
            const splits = if (specialization.head_split and params.cu_seqlens_k == 0) nvptx.nctaid(0) / ((params.seqlen_k + block_n - 1) / block_n) else 1;
            const persistent_scheduler: PersistentScheduler = .create();
            const grid = tile_scheduler.gridTile(.x, splits, persistent_scheduler.initialWorkTileInfo());
            const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
            return Work.init(params, scheduler.getGridWork(grid) orelse return null, grid.split_idx, grid.num_qhead_splits);
        }

        pub const Scores = struct {
            pub fn rowTerms(params: BackwardParams, work: *const Work, step: Step) Terms {
                const row = step.block_info.first_row + thread.warp() % warps_m * 16 + thread.lane() / 4;
                const lse: [*]addrspace(.global) const f32 = @ptrFromInt(params.lse_log2);
                const dpsum: [*]addrspace(.global) const f32 = @ptrFromInt(params.dpsum);
                const base = work.batch * params.lse_batch_stride + step.head * params.lse_head_stride +
                    work.sequence.offset_q;
                const seqlen_q = work.sequence.seqlen_q;
                var terms: Terms = .{ .lse_negated = @splat(-std.math.inf(f32)), .dpsum_negated = @splat(0) };
                inline for (0..2) |half| {
                    const half_row = row + @as(u32, half) * 8;
                    if (half_row < seqlen_q) {
                        terms.lse_negated[half] = lse[base + half_row];
                        terms.dpsum_negated[half] = dpsum[base + half_row];
                    }
                }
                return terms;
            }

            pub fn compute(
                params: BackwardParams,
                iteration: u32,
                work: *const Work,
                step: Step,
                load: *const Storage.Load,
                terms: Terms,
            ) Probabilities {
                const buffer = iteration % stages;
                const warp_m = thread.warp() % warps_m;
                const warp_n = thread.warp() / warps_m;
                var scores = scoresGemm(Storage.qTile(buffer), Storage.kTile());
                load.acquireDo(iteration, work);
                var d_probabilities = if (tiling.p_first)
                    @as([columns_per_warp / 8]Accumulator, undefined)
                else
                    scoresGemm(Storage.doTile(buffer), Storage.vTile());
                if (step.block_info.needsMask(step.block_info.colLimitBounds(), work.first_key / block_n, block_n)) {
                    const limits = ColLimits.init(step.block_info, step.block_info.first_row + warp_m * 16 + thread.lane() / 4);
                    applyMask(&scores, work.first_key + warp_n * columns_per_warp, limits, -std.math.inf(f32));
                }
                const scale: Accumulator = @splat(params.softmax_scale_log2);
                var result: Probabilities = undefined;
                const lse4: Accumulator = .{ terms.lse_negated[0], terms.lse_negated[0], terms.lse_negated[1], terms.lse_negated[1] };
                const sum4: Accumulator = .{ terms.dpsum_negated[0], terms.dpsum_negated[0], terms.dpsum_negated[1], terms.dpsum_negated[1] };
                var probabilities: [columns_per_warp / 8]Accumulator = undefined;
                if (tiling.p_first) {
                    inline for (&probabilities, scores) |*p, score| p.* = fragments.exp2(@mulAdd(Accumulator, score, scale, lse4));
                    d_probabilities = scoresGemm(Storage.doTile(buffer), Storage.vTile());
                }
                inline for (scores, d_probabilities, 0..) |score, d_probability, tile| {
                    const p = if (tiling.p_first) probabilities[tile] else fragments.exp2(@mulAdd(Accumulator, score, scale, lse4));
                    const ds = p * (d_probability + sum4);
                    result.p[tile] = .{
                        @backingInt(nvptx.Bf16x2.fromF32(p[0], p[1])),
                        @backingInt(nvptx.Bf16x2.fromF32(p[2], p[3])),
                    };
                    result.ds[tile] = .{
                        @backingInt(nvptx.Bf16x2.fromF32(ds[0], ds[1])),
                        @backingInt(nvptx.Bf16x2.fromF32(ds[2], ds[3])),
                    };
                }
                return result;
            }
        };

        pub const PdS = struct {
            pub fn store(probabilities: Probabilities) void {
                const warp_m = thread.warp() % warps_m;
                const warp_n = thread.warp() / warps_m;
                const lane = thread.lane();
                const row = warp_m * 16 + lane / 4;
                inline for (0..columns_per_warp / 8) |tile_index| {
                    const tile: u32 = tile_index;
                    const column = warp_n * (columns_per_warp / 8) + tile;
                    PdSLayout.word(Storage.pTile(), row, column, lane % 4).* = probabilities.p[tile][0];
                    PdSLayout.word(Storage.pTile(), row + 8, column, lane % 4).* = probabilities.p[tile][1];
                    PdSLayout.word(Storage.dsTile(), row, column, lane % 4).* = probabilities.ds[tile][0];
                    PdSLayout.word(Storage.dsTile(), row + 8, column, lane % 4).* = probabilities.ds[tile][1];
                }
            }
        };

        const DkvAccumulators = struct {
            dk: [tiles_kv]Accumulator,
            dv: [tiles_kv]Accumulator,
        };

        pub const Dkv = struct {
            pub fn zero() DkvAccumulators {
                return .{ .dk = @splat(@splat(0)), .dv = @splat(@splat(0)) };
            }

            pub fn accumulate(accumulators: *DkvAccumulators, load: *const Storage.Load, iteration: u32) void {
                @setEvalBranchQuota(100_000);
                _ = load;
                const buffer = iteration % stages;
                const warp_n = thread.warp() % warps_kv_n;
                const warp_d = thread.warp() / warps_kv_n;
                const lane = thread.lane();
                const lane_row = lane % 8;
                inline for (0..block_m / 16) |k_step| {
                    const row = @as(u32, k_step) * 16 + lane_row + lane / 16 * 8;
                    const column = warp_n * 2 + lane / 8 % 2;
                    const dv_a = nvptx.ldmatrix(.x4, true, PdSLayout.chunk(Storage.pTile(), row, column, lane_row));
                    const dk_a = nvptx.ldmatrix(.x4, true, PdSLayout.chunk(Storage.dsTile(), row, column, lane_row));
                    transposedGemm(&accumulators.dv, tiles_kv, dv_a, Storage.doTile(buffer), k_step, warp_d * tiles_kv);
                    transposedGemm(&accumulators.dk, tiles_kv, dk_a, Storage.qTile(buffer), k_step, warp_d * tiles_kv);
                }
            }
        };

        pub const Dq = struct {
            pub fn drain() void {
                if (tiling.bulk_reduce) Storage.DqAccum.waitFree();
            }

            pub fn accumulate(
                params: BackwardParams,
                work: *const Work,
                step: Step,
                load: *const Storage.Load,
                iteration: u32,
            ) void {
                @setEvalBranchQuota(100_000);
                _ = iteration;
                const warp_m = thread.warp() % warps_m;
                const warp_d = thread.warp() / warps_m;
                const lane = thread.lane();
                const lane_row = lane % 8;
                if (tiling.bulk_reduce) {
                    const gradient = Storage.DqAccum;
                    const tiles_per_box = gradient.box_columns / 8;
                    const row_base = work.sequence.offset_q + if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
                    const row = row_base + step.block_info.first_row + warp_m * 16;
                    var a: [block_n / 16]@Vector(4, u32) = undefined;
                    inline for (&a, 0..) |*fragment, k_step| {
                        fragment.* = nvptx.ldmatrix(.x4, false, PdSLayout.chunk(Storage.dsTile(), warp_m * 16 + lane % 16, @as(u32, k_step) * 2 + lane / 16, lane_row));
                    }
                    inline for (0..tiles_dq / tiles_per_box) |box| {
                        const first_tile = warp_d * tiles_dq + @as(u32, box) * tiles_per_box;
                        var dq: [tiles_per_box]Accumulator = @splat(@splat(0));
                        inline for (a, 0..) |fragment, k_step| {
                            transposedGemm(&dq, tiles_per_box, fragment, Storage.kTile(), k_step, first_tile);
                        }
                        gradient.waitFree();
                        inline for (dq, 0..) |fragment, tile| {
                            const column = @as(u32, tile) * 8 + lane % 4 * 2;
                            gradient.store(lane / 4, column, .{ fragment[0], fragment[1] });
                            gradient.store(lane / 4 + 8, column, .{ fragment[2], fragment[3] });
                        }
                        if (first_tile * 8 < head_dim) gradient.publish(load, first_tile * 8, step.head, row);
                    }
                    return;
                }
                var dq: [tiles_dq]Accumulator = @splat(@splat(0));
                inline for (0..block_n / 16) |k_step| {
                    const a = nvptx.ldmatrix(.x4, false, PdSLayout.chunk(Storage.dsTile(), warp_m * 16 + lane % 16, @as(u32, k_step) * 2 + lane / 16, lane_row));
                    transposedGemm(&dq, tiles_dq, a, Storage.kTile(), k_step, warp_d * tiles_dq);
                }
                const row = step.block_info.first_row + warp_m * 16 + lane / 4;
                const seqlen_q = work.sequence.seqlen_q;
                const origin = work.batch * params.dq_accum_batch_stride + step.head * params.dq_accum_head_stride;
                const base: [*]addrspace(.global) f32 = @ptrFromInt(params.dq_accum);
                const row_index = origin + @as(u64, work.sequence.offset_q) * params.dq_accum_row_stride;
                inline for (dq, 0..) |fragment, tile_index| {
                    const tile: u32 = tile_index;
                    const first_column = (warp_d * tiles_dq + tile) * 8;
                    if (first_column < head_dim) {
                        if (vector_reductions) {
                            const odd = lane % 2 == 1;
                            const sent_low = if (odd) fragment[0] else fragment[2];
                            const sent_high = if (odd) fragment[1] else fragment[3];
                            const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
                            const received_low = nvptx.shflSync(.bfly, f32, sent_low, 1, clamp, .all);
                            const received_high = nvptx.shflSync(.bfly, f32, sent_high, 1, clamp, .all);
                            const values: Accumulator = if (odd)
                                .{ received_low, received_high, fragment[2], fragment[3] }
                            else
                                .{ fragment[0], fragment[1], received_low, received_high };
                            const r = row + if (odd) @as(u32, 8) else 0;
                            if (r < seqlen_q) {
                                const column = first_column + (lane % 4 & 2) * 2;
                                const address = &base[row_index + @as(u64, r) * params.dq_accum_row_stride + column];
                                asm volatile ("red.global.add.v4.f32 [%[address]], {%[a], %[b], %[c], %[d]};"
                                    :
                                    : [address] "l" (address),
                                      [a] "f" (values[0]),
                                      [b] "f" (values[1]),
                                      [c] "f" (values[2]),
                                      [d] "f" (values[3]),
                                    : .{ .memory = true });
                            }
                        } else {
                            const column = first_column + lane % 4 * 2;
                            inline for (.{ 0, 1 }) |half| {
                                const r = row + half * 8;
                                if (r < seqlen_q) {
                                    const index = row_index + @as(u64, r) * params.dq_accum_row_stride + column;
                                    nvptx.red(.gpu, .monotonic, .add, &base[index], fragment[half * 2]);
                                    nvptx.red(.gpu, .monotonic, .add, &base[index + 1], fragment[half * 2 + 1]);
                                }
                            }
                        }
                    }
                }
            }
        };

        pub const Epilogue = struct {
            fn reducePair(address: *addrspace(.global) f32, pair: @Vector(2, f32)) void {
                if (vector_reductions) {
                    asm volatile ("red.global.add.v2.f32 [%[address]], {%[a], %[b]};"
                        :
                        : [address] "l" (address),
                          [a] "f" (pair[0]),
                          [b] "f" (pair[1]),
                        : .{ .memory = true });
                } else {
                    const words: [*]addrspace(.global) f32 = @ptrCast(address);
                    nvptx.red(.gpu, .monotonic, .add, &words[0], pair[0]);
                    nvptx.red(.gpu, .monotonic, .add, &words[1], pair[1]);
                }
            }

            fn accumulate(params: BackwardParams, work: *const Work, accumulators: DkvAccumulators) void {
                const warp_n = thread.warp() % warps_kv_n;
                const warp_d = thread.warp() / warps_kv_n;
                const lane = thread.lane();
                const scale: Accumulator = @splat(params.softmax_scale);
                const dk: [*]addrspace(.global) f32 = @ptrFromInt(params.dk);
                const dv: [*]addrspace(.global) f32 = @ptrFromInt(params.dv);
                const dk_origin = work.batch * params.dk_batch_stride + work.kv_head * params.dk_head_stride;
                const dv_origin = work.batch * params.dv_batch_stride + work.kv_head * params.dv_head_stride;
                inline for (0..tiles_kv) |tile_index| {
                    const column = (warp_d * tiles_kv + @as(u32, tile_index)) * 8 + lane % 4 * 2;
                    const dk_scaled = accumulators.dk[tile_index] * scale;
                    const dv_values = accumulators.dv[tile_index];
                    inline for (.{ 0, 1 }) |half| {
                        const row = warp_n * 16 + lane / 4 + half * 8;
                        if (row < work.key_count and column < head_dim) {
                            const key_row = work.sequence.offset_k + work.first_key + row;
                            reducePair(&dk[dk_origin + key_row * params.dk_row_stride + column], .{ dk_scaled[half * 2], dk_scaled[half * 2 + 1] });
                            reducePair(&dv[dv_origin + key_row * params.dv_row_stride + column], .{ dv_values[half * 2], dv_values[half * 2 + 1] });
                        }
                    }
                }
            }

            pub fn store(params: BackwardParams, work: *const Work, accumulators: DkvAccumulators) void {
                if (specialization.head_split) return accumulate(params, work, accumulators);
                const warp_n = thread.warp() % warps_kv_n;
                const warp_d = thread.warp() / warps_kv_n;
                const lane = thread.lane();
                const lane_row = lane / 4;
                const sequence = work.sequence;
                const scale: Accumulator = @splat(params.softmax_scale);
                nvptx.barrierSyncAligned(.work_group, null);
                inline for (0..tiles_kv) |tile_index| {
                    const chunk_column = warp_d * tiles_kv + @as(u32, tile_index);
                    const dk_scaled = accumulators.dk[tile_index] * scale;
                    inline for (.{ 0, 1 }) |half| {
                        const staged_row = warp_n * 16 + lane_row + half * 8;
                        const dk_words: [*]addrspace(.shared) u32 = @ptrCast(Rows.chunk(Storage.kTile(), staged_row, chunk_column, lane_row));
                        const dv_words: [*]addrspace(.shared) u32 = @ptrCast(Rows.chunk(Storage.vTile(), staged_row, chunk_column, lane_row));
                        dk_words[lane % 4] = @backingInt(nvptx.Bf16x2.fromF32(dk_scaled[half * 2], dk_scaled[half * 2 + 1]));
                        dv_words[lane % 4] = @backingInt(nvptx.Bf16x2.fromF32(
                            accumulators.dv[tile_index][half * 2],
                            accumulators.dv[tile_index][half * 2 + 1],
                        ));
                    }
                }
                nvptx.barWarpSync(.all);
                const dk: [*]addrspace(.global) Chunk = @ptrFromInt(params.dk);
                const dv: [*]addrspace(.global) Chunk = @ptrFromInt(params.dv);
                const dk_origin = work.batch * params.dk_batch_stride + work.kv_head * params.dk_head_stride;
                const dv_origin = work.batch * params.dv_batch_stride + work.kv_head * params.dv_head_stride;
                inline for (0..(16 * tiles_kv + 31) / 32) |pass| {
                    const linear: u32 = @as(u32, pass) * 32 + lane;
                    if ((pass + 1) * 32 <= 16 * tiles_kv or linear < 16 * tiles_kv) {
                        const row = warp_n * 16 + linear / tiles_kv;
                        const chunk_column = warp_d * tiles_kv + linear % tiles_kv;
                        if (row < work.key_count and chunk_column * chunk_elements < head_dim) {
                            const key_row = sequence.offset_k + work.first_key + row;
                            dk[(dk_origin + key_row * params.dk_row_stride) / chunk_elements + chunk_column] =
                                Rows.chunk(Storage.kTile(), row, chunk_column, row).*;
                            dv[(dv_origin + key_row * params.dv_row_stride) / chunk_elements + chunk_column] =
                                Rows.chunk(Storage.vTile(), row, chunk_column, row).*;
                        }
                    }
                }
            }
        };

        pub const Kernel = FlashAttnBwdSm80Kernel(@This());
    };
}

const variant = configurations.backwardSm80Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const backward_tiling = variant.tiling;
const Backward = FlashAttnBwdSm80(backward_tiling, variant.specialization, SharedStorage(backward_tiling));
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");
const auxiliary: flash_bwd_preprocess.Options = .{ .varlen_as_one_sequence = variant.specialization.scheduler_metadata };

fn flashAttnBwd(
    q: u64,
    k: u64,
    v: u64,
    do: u64,
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
    q_batch_stride: u64,
    q_row_stride: u64,
    q_head_stride: u64,
    k_batch_stride: u64,
    k_row_stride: u64,
    k_head_stride: u64,
    v_batch_stride: u64,
    v_row_stride: u64,
    v_head_stride: u64,
    do_batch_stride: u64,
    do_row_stride: u64,
    do_head_stride: u64,
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
        .q = q,
        .k = k,
        .v = v,
        .do = do,
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
        .q_batch_stride = q_batch_stride,
        .q_row_stride = q_row_stride,
        .q_head_stride = q_head_stride,
        .k_batch_stride = k_batch_stride,
        .k_row_stride = k_row_stride,
        .k_head_stride = k_head_stride,
        .v_batch_stride = v_batch_stride,
        .v_row_stride = v_row_stride,
        .v_head_stride = v_head_stride,
        .do_batch_stride = do_batch_stride,
        .do_row_stride = do_row_stride,
        .do_head_stride = do_head_stride,
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
    }, .{});
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&flashAttnBwd, .{ .name = "flash_attn_bwd" });
        @export(&flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess, .{ .name = "flash_attn_bwd_preprocess" });
        @export(&flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary).flashAttnBwdPostprocess, .{ .name = "flash_attn_bwd_postprocess" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Backward.PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_bwd_prepare_varlen_num_blocks" });
        }
        if (variant.specialization.head_split) @export(&flash_bwd_preprocess.flashAttnBwdClear, .{ .name = "flash_attn_bwd_clear" });
    }
}
