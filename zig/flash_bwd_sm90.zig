const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const NBlockWork = @import("n_block_work.zig").NBlockWork;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const applyMask = @import("mask.zig").applyMask;
const ColLimits = @import("BlockInfo.zig").ColLimits;
const tma = @import("tma.zig");
const wgmma = @import("wgmma.zig");
const elect = @import("elect.zig");
const mbarrier = @import("mbarrier.zig");
const cluster = @import("cluster.zig");
const thread = @import("thread.zig");
const smem = @import("smem.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");
const grid_dependency = @import("grid_dependency.zig");
const cp_async = nvptx.cp_async;

const Accumulator = fragments.Accumulator;
const Chunk = fragments.Chunk;
const pds_swizzle_bytes = 32;
const producer_warps = configurations.Sm90BwdTiling.producer_warps;
const consumer_warps = configurations.Sm90BwdTiling.consumer_warps;
const consumer_threads = consumer_warps * 32;
const split_units = 2 * consumer_warps;
const reduce_warps = 1;
const staging_threads = (configurations.warps_per_warp_group + reduce_warps) * 32;
const load_arrivals = mbarrier.arrivals(.cp_async, 1, 1) + 1;
const role_registers = configurations.registerRequirements(consumer_warps / configurations.warps_per_warp_group);

const NamedBarrier = enum(u4) {
    consumers = 1,
    dq_empty = 2,
    dq_full = 6,
    _,

    fn of(base: NamedBarrier, group: u32) NamedBarrier {
        return @fromBackingInt(@intCast(@backingInt(base) + group));
    }
};

fn stagingRow(row: u32) u32 {
    return row / 8 * 8 + row % 4 * 2 + row % 8 / 4;
}

const RelativeRowLimits = struct {
    top: [2]i32,
    bot: [2]i32,
};

fn clusterRank() u32 {
    return asm ("mov.u32 %[rank], %cluster_ctarank;"
        : [rank] "=r" (-> u32),
    );
}

fn splitOwner(tensor: u32, unit: u32) u32 {
    return (tensor * consumer_warps + unit) >> @intCast(std.math.log2(split_units) - @ctz(cluster.count()));
}

fn consumerWarp() u32 {
    return thread.warp() - producer_warps;
}

fn consumerGroup() u32 {
    return consumerWarp() / 4;
}

fn publishClearedRows() void {
    nvptx.fenceProxyAsync(.shared);
    nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
}

fn schedulerOptions(comptime tiling: configurations.Sm90BwdTiling, comptime specialization: configurations.Specialization) tile_scheduler.Options {
    return .{
        .block_m = tiling.block_n,
        .row_bytes = 8 * tiling.head_dim,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
}

fn splitWork(
    comptime Work: type,
    comptime tiling: configurations.Sm90BwdTiling,
    comptime specialization: configurations.Specialization,
    params: BackwardParams,
) ?Work {
    const TileScheduler = tile_scheduler.TileScheduler(schedulerOptions(tiling, specialization));
    const PersistentScheduler = tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .y, .ctas = null });
    const persistent_scheduler: PersistentScheduler = .create();
    const grid = tile_scheduler.gridTile(.y, cluster.count(), persistent_scheduler.initialWorkTileInfo());
    const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
    return Work.init(params, scheduler.getGridWork(grid) orelse return null, grid.split_idx, grid.num_qhead_splits);
}

fn exchangePartials(comptime quads: u32, base: u32, values: *[2][quads]@Vector(4, f32), owned: [2]bool, splits: u32) void {
    const consumer = thread.id() - producer_warps * 32;
    inline for (0..2) |tensor| {
        inline for (0..quads) |quad| {
            const slot: *addrspace(.shared) @Vector(4, f32) = @ptrFromInt(base + (@as(u32, tensor * quads + quad) * consumer_threads + consumer) * 16);
            slot.* = values[tensor][quad];
        }
    }
    cluster.sync();
    const rank = clusterRank();
    inline for (0..2) |tensor| {
        if (owned[tensor]) {
            var peer: u32 = 0;
            while (peer < splits) : (peer += 1) {
                if (peer != rank) {
                    const remote = asm ("mapa.shared::cluster.u32 %[mapped], %[local], %[peer];"
                        : [mapped] "=r" (-> u32),
                        : [local] "r" (base),
                          [peer] "r" (peer),
                    );
                    inline for (0..quads) |quad| {
                        const partial = cluster.load(remote + (@as(u32, tensor * quads + quad) * consumer_threads + consumer) * 16);
                        inline for (0..4) |element| values[tensor][quad][element] += partial[element];
                    }
                }
            }
        }
    }
    cluster.sync();
}

pub fn FlashAttnBwdSm90(comptime tiling: configurations.Sm90BwdTiling, comptime specialization: configurations.Specialization) type {
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const swizzle_bytes = tiling.swizzleBytes();
    const Operands = wgmma.Operands(swizzle_bytes);
    const k_steps = padded / 16;
    const query_steps = block_m / 16;
    const key_steps = block_n / 16;
    const Layout = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const ds_swizzle_bytes = 128;
    const DsOperands = wgmma.Operands(ds_swizzle_bytes);
    const DsLayout = tma.SwizzledLayout(block_m / 8, ds_swizzle_bytes);
    const ds_box_chunks = DsLayout.swizzle_columns;
    const Work = NBlockWork(block_m, block_n);
    const dq_split = tiling.dq_split;
    const dq_groups: u32 = if (dq_split == .none) 1 else 2;
    const dq_columns = if (dq_split == .by_columns) padded / 2 else padded;
    const dq_rows = if (dq_split == .by_rows) 64 else block_m;
    const staging_boxes = tiling.stagingBoxes();
    const staging_buffers = tiling.staging_buffers;
    const box_columns = configurations.Sm90BwdTiling.dq_accum_box_columns;
    const early_values = block_m == 64;
    const early_exp_tiles = tiling.early_exp_tiles;
    const separate_do_pipeline = tiling.separate_do_pipeline;
    comptime std.debug.assert(block_n == 128 and dq_rows == 64);

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            do: *const tma.TensorMap,
            dq_accum: *const tma.TensorMap,
        };

        const Tiles = struct {
            k: Layout.Tile(block_n),
            v: Layout.Tile(block_n),
            q: [stages]Layout.Tile(block_m),
            do: [stages]Layout.Tile(block_m),
            ds: [stages]DsLayout.Tile(block_n),
            staging: [staging_buffers][staging_boxes][block_m * box_columns / 4]Chunk,
            lse: [stages][block_m]f32,
            dpsum: [stages][block_m]f32,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var kv_full: nvptx.Mbarrier addrspace(.shared) = undefined;
        var query_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var query_empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var do_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var do_empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub const init_scope: mbarrier.Scope = .cta;
        pub const initializer_warp: u32 = 0;
        pub const roles = .{
            .{ .first_warp = 0, .warps = producer_warps, .registers = role_registers[0], .Role = Producer },
            .{ .first_warp = producer_warps, .warps = consumer_warps, .registers = role_registers[1], .Role = Consumer },
        };

        pub fn initialize() void {
            if (thread.warp() == initializer_warp) {
                if (elect.oneSync()) {
                    kv_full.init(1);
                    inline for (0..stages) |stage| {
                        query_full[stage].init(load_arrivals);
                        query_empty[stage].init(consumer_warps);
                        if (separate_do_pipeline) {
                            do_full[stage].init(load_arrivals);
                            do_empty[stage].init(consumer_warps);
                        }
                    }
                    mbarrier.pipelineInitFence();
                }
                elect.reconverge();
            }
        }

        pub const Producer = struct {
            pub fn run(params: BackwardParams, args: Args) void {
                switch (thread.warp()) {
                    0 => load(params, args),
                    1 => reduceDq(params, args),
                    else => {},
                }
            }

            fn load(params: BackwardParams, args: Args) void {
                const lane = thread.lane();
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                const iterations = work.iterations();
                if (iterations == 0) return;
                const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
                if (elect.oneSync()) {
                    _ = kv_full.arriveExpectTx(comptime 2 * tiling.tileBytes(block_n));
                    const row = work.sequence.offset_k + work.first_key;
                    Layout.copyTile(&tiles.k, args.k, &kv_full, work.kv_head, row, batch);
                    Layout.copyTile(&tiles.v, args.v, &kv_full, work.kv_head, row, batch);
                }
                elect.reconverge();
                grid_dependency.wait();
                const lse: [*]addrspace(.global) const f32 = @ptrFromInt(params.lse_log2);
                const dpsum: [*]addrspace(.global) const f32 = @ptrFromInt(params.dpsum);
                var iteration: u32 = 0;
                while (iteration < iterations) : (iteration += 1) {
                    const step = work.step(iteration);
                    const stage = iteration % stages;
                    const reuse = iteration / stages;
                    if (reuse > 0) {
                        while (!query_empty[stage].tryWaitParity(@intCast((reuse - 1) & 1))) {}
                    }
                    const row = work.sequence.offset_q + step.block_info.first_row;
                    if (elect.oneSync()) {
                        _ = query_full[stage].arriveExpectTx(comptime (if (separate_do_pipeline) 1 else 2) * tiling.tileBytes(block_m));
                        Layout.copyTile(&tiles.q[stage], args.q, &query_full[stage], step.head, row, batch);
                        if (!separate_do_pipeline) Layout.copyTile(&tiles.do[stage], args.do, &query_full[stage], step.head, row, batch);
                    }
                    elect.reconverge();
                    const vectors: u64 = work.batch * params.lse_batch_stride + step.head * params.lse_head_stride + row;
                    inline for (0..block_m / 32) |element| {
                        const column = element * 32 + lane;
                        cp_async.copy(.cache_all_levels, &tiles.lse[stage][column], &lse[vectors + column]);
                        if (!separate_do_pipeline) cp_async.copy(.cache_all_levels, &tiles.dpsum[stage][column], &dpsum[vectors + column]);
                    }
                    cp_async.mbarrierArriveNoinc(&query_full[stage]);
                    if (separate_do_pipeline) {
                        if (reuse > 0) {
                            while (!do_empty[stage].tryWaitParity(@intCast((reuse - 1) & 1))) {}
                        }
                        if (elect.oneSync()) {
                            _ = do_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_m));
                            Layout.copyTile(&tiles.do[stage], args.do, &do_full[stage], step.head, row, batch);
                        }
                        elect.reconverge();
                        inline for (0..block_m / 32) |element| {
                            const column = element * 32 + lane;
                            cp_async.copy(.cache_all_levels, &tiles.dpsum[stage][column], &dpsum[vectors + column]);
                        }
                        cp_async.mbarrierArriveNoinc(&do_full[stage]);
                    }
                }
            }

            fn reduceDq(params: BackwardParams, args: Args) void {
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                grid_dependency.wait();
                const iterations = work.iterations();
                const row_base = work.sequence.offset_q +
                    if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
                inline for (0..staging_buffers) |buffer| {
                    if (buffer < iterations) releaseStaging(buffer);
                }
                var iteration: u32 = 0;
                while (iteration < iterations) : (iteration += 1) {
                    const step = work.step(iteration);
                    const row = row_base + step.block_info.first_row;
                    const buffer = iteration % staging_buffers;
                    inline for (0..dq_groups) |group| {
                        nvptx.barrierSyncAligned(NamedBarrier.dq_full.of(@as(u32, group) * staging_buffers + buffer), staging_threads);
                        if (dq_split == .by_columns) {
                            if (elect.oneSync()) reduceBoxes(args, buffer, group * dq_columns / box_columns, dq_columns / box_columns, step.head, row);
                            elect.reconverge();
                        }
                    }
                    if (dq_split != .by_columns) {
                        if (elect.oneSync()) reduceBoxes(args, buffer, 0, staging_boxes, step.head, row);
                        elect.reconverge();
                    }
                    tma.waitBulkReads();
                    if (iteration + staging_buffers < iterations) releaseStaging(buffer);
                }
            }

            fn releaseStaging(buffer: u32) void {
                inline for (0..dq_groups) |group| {
                    nvptx.barrierArriveAligned(NamedBarrier.dq_empty.of(@as(u32, group) * staging_buffers + buffer), staging_threads);
                }
            }

            fn reduceBoxes(args: Args, buffer: u32, comptime first: u32, comptime count: u32, head: u32, row: u32) void {
                inline for (first..first + count) |box| {
                    tma.reduceInterleavedBox(args.dq_accum, &tiles.staging[buffer][box][0], box * box_columns, row, head);
                }
                tma.commitBulkGroup();
            }
        };

        const Gradients = struct {
            dk: wgmma.Accumulator(padded),
            dv: wgmma.Accumulator(padded),
        };

        const RowVectors = if (early_values) [block_m / 8][2]f32 else *addrspace(.shared) const [block_m]f32;

        fn rowPair(vectors: *addrspace(.shared) const [block_m]f32, comptime tile: u32) [2]f32 {
            const pairs: *addrspace(.shared) const [block_m / 8][8]f32 = @ptrFromInt(wgmma.address(vectors) + thread.lane() % 4 * 8);
            return pairs[tile][0..2].*;
        }

        fn rowVectors(vectors: *addrspace(.shared) const [block_m]f32) RowVectors {
            if (!early_values) return vectors;
            var result: RowVectors = undefined;
            inline for (0..block_m / 8) |tile| result[tile] = rowPair(vectors, tile);
            return result;
        }

        fn tilePair(vectors: RowVectors, comptime tile: u32) [2]f32 {
            return if (early_values) vectors[tile] else rowPair(vectors, tile);
        }

        fn exponentiate(
            comptime masked: bool,
            comptime first_tile: u32,
            comptime end_tile: u32,
            result: *wgmma.Accumulator(block_m),
            scores: wgmma.Accumulator(block_m),
            lse_negated: RowVectors,
            limits: RelativeRowLimits,
            softmax_scale_log2: f32,
        ) void {
            inline for (first_tile..end_tile) |tile| {
                const lse_pair = tilePair(lse_negated, tile);
                inline for (0..4) |element| {
                    const index = tile * 4 + element;
                    const p = nvptx.math.ex2ApproxF32(true, @mulAdd(f32, scores[index], softmax_scale_log2, lse_pair[element % 2]));
                    const column: i32 = tile * 8 + element % 2;
                    result[index] = if (masked and (column < limits.top[element / 2] or column >= limits.bot[element / 2])) 0 else p;
                }
            }
        }

        fn computeDs(
            comptime masked: bool,
            probabilities: wgmma.Accumulator(block_m),
            dp: wgmma.Accumulator(block_m),
            dpsum_negated: RowVectors,
        ) wgmma.Accumulator(block_m) {
            var result: wgmma.Accumulator(block_m) = undefined;
            inline for (0..block_m / 8) |tile| {
                const dpsum_pair = tilePair(dpsum_negated, tile);
                inline for (0..4) |element| {
                    const index = tile * 4 + element;
                    const p = probabilities[index];
                    const ds = p * (dp[index] + dpsum_pair[element % 2]);
                    result[index] = if (masked and p == 0) 0 else ds;
                }
            }
            return result;
        }

        const PdS = struct {
            probabilities: [query_steps]@Vector(4, u32),
            ds: [query_steps]@Vector(4, u32),
        };

        fn computePdS(
            comptime masked: bool,
            leading: *const wgmma.Accumulator(block_m),
            scores: wgmma.Accumulator(block_m),
            dp: *wgmma.Accumulator(block_m),
            lse_negated: RowVectors,
            dpsum: *addrspace(.shared) const [block_m]f32,
            limits: RelativeRowLimits,
            softmax_scale_log2: f32,
            ds_base: u32,
            matrix_base: u32,
        ) PdS {
            var probabilities = leading.*;
            wgmma.wait(0);
            wgmma.fenceOperand(dp);
            exponentiate(masked, early_exp_tiles, block_m / 8, &probabilities, scores, lse_negated, limits, softmax_scale_log2);
            const dpsum_negated = rowVectors(dpsum);
            const ds_registers = wgmma.packPairs(block_m, computeDs(masked, probabilities, dp.*, dpsum_negated));
            inline for (0..query_steps) |query_step| {
                const chunk = query_step * 2;
                wgmma.storeMatrices(
                    false,
                    ds_base + @as(u32, chunk / ds_box_chunks * (block_n * ds_swizzle_bytes)) +
                        (matrix_base ^ @as(u32, chunk % ds_box_chunks * 16)),
                    ds_registers[query_step],
                );
            }
            return .{ .probabilities = wgmma.packPairs(block_m, probabilities), .ds = ds_registers };
        }

        fn issueGradient(
            gradient: *wgmma.Accumulator(padded),
            registers: [query_steps]@Vector(4, u32),
            rows: u64,
        ) void {
            wgmma.fence();
            inline for (0..query_steps) |query_step| {
                wgmma.mmaRegisters(
                    padded,
                    gradient,
                    registers[query_step],
                    wgmma.advance(rows, query_step * 16 * swizzle_bytes),
                    true,
                    true,
                );
            }
            wgmma.commit();
        }

        fn dqStagingChunks() [4]u32 {
            const lane = thread.lane();
            const group = consumerGroup();
            const row = stagingRow((if (dq_split == .by_rows) group * 64 else 0) + consumerWarp() % 4 * 16 + lane / 4);
            const first_box = if (dq_split == .by_columns) group * dq_columns / box_columns else 0;
            const row_address = wgmma.address(&tiles.staging[0]) + first_box * (block_m * box_columns * 4) + row * box_columns * 4 + lane % 2 * 8;
            const swizzle_offset = (lane % 4 / 2 ^ row % 8) * 16;
            var chunks: [4]u32 = undefined;
            inline for (0..4) |chunk| chunks[chunk] = row_address + (@as(u32, chunk * 32) ^ swizzle_offset);
            return chunks;
        }

        fn storeDq(chunks: [4]u32, buffer: u32, accumulator: wgmma.Accumulator(dq_columns)) void {
            const buffer_offset = buffer * @sizeOf(@TypeOf(tiles.staging[0]));
            inline for (0..dq_columns / 8) |tile| {
                const box_offset = tile * 8 / box_columns * (block_m * box_columns * 4);
                inline for (0..2) |half| {
                    const destination: *addrspace(.shared) @Vector(2, f32) = @ptrFromInt(
                        chunks[tile % 4] + buffer_offset + box_offset + half * 8 * box_columns * 4,
                    );
                    destination.* = .{ accumulator[tile * 4 + half * 2], accumulator[tile * 4 + half * 2 + 1] };
                }
            }
        }

        pub const Consumer = struct {
            pub fn run(params: BackwardParams, _: Args) void {
                if (dq_groups == 2 or consumerGroup() == 0) process(true, params) else process(false, params);
            }

            fn process(comptime writes_dq: bool, params: BackwardParams) void {
                @setEvalBranchQuota(100_000);
                const lane = thread.lane();
                const group = consumerGroup();
                const warp_in_group = consumerWarp() % 4;
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                const iterations = work.iterations();
                var gradients: Gradients = .{ .dk = @splat(0), .dv = @splat(0) };
                if (iterations > 0) {
                    while (!kv_full.tryWaitParity(0)) {}
                    if (params.cu_seqlens_k != 0 and work.key_count < block_n) {
                        Layout.clearRows(&tiles.k, work.key_count, consumer_threads, thread.id() - producer_warps * 32);
                        publishClearedRows();
                    }
                    const key_row = work.first_key + group * 64 + warp_in_group * 16 + lane / 4;
                    const matrix_row = group * 64 + warp_in_group * 16 + lane % 16;
                    const placement: Placement = .{
                        .unmasked = work.unmaskedRowLimits(),
                        .near = work.rowLimits(key_row),
                        .far = work.rowLimits(key_row + 8),
                        .matrix_base = matrix_row * ds_swizzle_bytes +
                            smem.swizzledColumn(ds_swizzle_bytes, lane / 16, matrix_row) * 16,
                        .dq_chunks = dqStagingChunks(),
                        .query_box = if (dq_split == .by_rows) group else 0,
                        .key_box = if (dq_split == .by_columns) group * dq_columns / (swizzle_bytes / 2) else 0,
                        .key = Operands.kMajor(wgmma.address(&tiles.k) + group * 64 * swizzle_bytes),
                        .value = Operands.kMajor(wgmma.address(&tiles.v) + group * 64 * swizzle_bytes),
                        .column_offset = lane % 4 * 2,
                        .dq_empty = NamedBarrier.dq_empty.of(group * staging_buffers),
                        .dq_full = NamedBarrier.dq_full.of(group * staging_buffers),
                    };
                    var iteration: u32 = 0;
                    if (tiling.separate_masking_iterations) {
                        const unmasked_begin, const unmasked_end = work.unmaskedBlocks();
                        while (iteration < iterations) {
                            const head_end = iteration + work.m_block_count;
                            const middle_end = iteration + unmasked_end;
                            var masked_end = iteration + unmasked_begin;
                            while (true) {
                                while (iteration < masked_end) : (iteration += 1) {
                                    processBlock(writes_dq, true, params, work, iteration, placement, &gradients);
                                }
                                if (masked_end == head_end) break;
                                while (iteration < middle_end) : (iteration += 1) {
                                    processBlock(writes_dq, false, params, work, iteration, placement, &gradients);
                                }
                                masked_end = head_end;
                            }
                        }
                    } else {
                        while (iteration < iterations) : (iteration += 1) {
                            processBlock(writes_dq, null, params, work, iteration, placement, &gradients);
                        }
                    }
                }
                store(params, work, gradients);
            }

            const Placement = struct {
                unmasked: Work.RowLimits,
                near: Work.RowLimits,
                far: Work.RowLimits,
                matrix_base: u32,
                dq_chunks: [4]u32,
                query_box: u32,
                key_box: u32,
                key: u64,
                value: u64,
                column_offset: u32,
                dq_empty: NamedBarrier,
                dq_full: NamedBarrier,
            };

            fn processBlock(
                comptime writes_dq: bool,
                comptime phase_mask: ?bool,
                params: BackwardParams,
                work: Work,
                iteration: u32,
                placement: Placement,
                gradients: *Gradients,
            ) void {
                @setEvalBranchQuota(100_000);
                const unmasked = placement.unmasked;
                const near = placement.near;
                const far = placement.far;
                const matrix_base = placement.matrix_base;
                const query_box = placement.query_box;
                const key_box = placement.key_box;
                const key = placement.key;
                const value = placement.value;
                const step = work.step(iteration);
                const stage = iteration % stages;
                while (!query_full[stage].tryWaitParity(@intCast((iteration / stages) & 1))) {}
                const lse = rowVectors(&tiles.lse[stage]);
                const query_base = wgmma.address(&tiles.q[stage]);
                const do_base = wgmma.address(&tiles.do[stage]);
                const query = Operands.kMajor(query_base);
                const do = Operands.kMajor(do_base);
                var scores: wgmma.Accumulator(block_m) = undefined;
                var dp: wgmma.Accumulator(block_m) = undefined;
                wgmma.fence();
                inline for (0..k_steps) |k_step| {
                    wgmma.mmaShared(
                        block_m,
                        &scores,
                        wgmma.advance(key, Operands.kStepOffset(block_n, k_step)),
                        wgmma.advance(query, Operands.kStepOffset(block_m, k_step)),
                        k_step != 0,
                        false,
                        false,
                    );
                }
                wgmma.commit();
                if (separate_do_pipeline) {
                    while (!do_full[stage].tryWaitParity(@intCast((iteration / stages) & 1))) {}
                    wgmma.fence();
                }
                inline for (0..k_steps) |k_step| {
                    wgmma.mmaShared(
                        block_m,
                        &dp,
                        wgmma.advance(value, Operands.kStepOffset(block_n, k_step)),
                        wgmma.advance(do, Operands.kStepOffset(block_m, k_step)),
                        k_step != 0,
                        false,
                        false,
                    );
                }
                wgmma.commit();
                wgmma.wait(1);
                wgmma.fenceOperand(&scores);
                const masked = phase_mask orelse !unmasked.contains(step.block_info.first_row);
                const column_offset: i32 = @intCast(step.block_info.first_row + placement.column_offset);
                const limits: RelativeRowLimits = .{
                    .top = .{ near.top - column_offset, far.top - column_offset },
                    .bot = .{ near.bot - column_offset, far.bot - column_offset },
                };
                const ds_base = wgmma.address(&tiles.ds[stage]);
                const dpsum = &tiles.dpsum[stage];
                var leading: wgmma.Accumulator(block_m) = undefined;
                if (masked)
                    exponentiate(true, 0, early_exp_tiles, &leading, scores, lse, limits, params.softmax_scale_log2)
                else
                    exponentiate(false, 0, early_exp_tiles, &leading, scores, lse, limits, params.softmax_scale_log2);
                const pds = if (masked)
                    computePdS(true, &leading, scores, &dp, lse, dpsum, limits, params.softmax_scale_log2, ds_base, matrix_base)
                else
                    computePdS(false, &leading, scores, &dp, lse, dpsum, limits, params.softmax_scale_log2, ds_base, matrix_base);
                const ds_registers = pds.ds;
                issueGradient(&gradients.dv, pds.probabilities, Operands.mnMajor(do_base, block_m));
                nvptx.fenceProxyAsync(.shared);
                nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                var dq_accumulator: wgmma.Accumulator(dq_columns) = undefined;
                wgmma.fence();
                if (writes_dq) {
                    const scores_transposed = DsOperands.mnMajor(ds_base + query_box * (block_n * ds_swizzle_bytes), block_n);
                    const key_rows = Operands.mnMajor(wgmma.address(&tiles.k) + key_box * (block_n * swizzle_bytes), block_n);
                    inline for (0..key_steps) |key_step| {
                        wgmma.mmaShared(
                            dq_columns,
                            &dq_accumulator,
                            wgmma.advance(scores_transposed, key_step * 16 * ds_swizzle_bytes),
                            wgmma.advance(key_rows, key_step * 16 * swizzle_bytes),
                            key_step != 0,
                            true,
                            true,
                        );
                    }
                    wgmma.commit();
                }
                const query_rows = Operands.mnMajor(query_base, block_m);
                inline for (0..query_steps) |query_step| {
                    wgmma.mmaRegisters(
                        padded,
                        &gradients.dk,
                        ds_registers[query_step],
                        wgmma.advance(query_rows, query_step * 16 * swizzle_bytes),
                        true,
                        true,
                    );
                }
                wgmma.commit();
                if (writes_dq or separate_do_pipeline) wgmma.wait(1);
                if (separate_do_pipeline) {
                    if (elect.oneSync()) _ = do_empty[stage].arrive();
                    elect.reconverge();
                }
                if (writes_dq) {
                    wgmma.fenceOperand(&dq_accumulator);
                    const buffer = iteration % staging_buffers;
                    nvptx.barrierSyncAligned(placement.dq_empty.of(buffer), staging_threads);
                    storeDq(placement.dq_chunks, buffer, dq_accumulator);
                    nvptx.fenceProxyAsync(.shared);
                    nvptx.barrierArriveAligned(placement.dq_full.of(buffer), staging_threads);
                }
                wgmma.wait(0);
                wgmma.fenceOperand(&gradients.dk);
                wgmma.fenceOperand(&gradients.dv);
                if (elect.oneSync()) _ = query_empty[stage].arrive();
                elect.reconverge();
            }

            fn store(params: BackwardParams, work: Work, gradients: Gradients) void {
                nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                const splits = cluster.count();
                const rank = clusterRank();
                var finished = gradients;
                if (splits > 1) {
                    var values: [2][padded / 8]@Vector(4, f32) = undefined;
                    inline for (0..padded / 8) |quad| {
                        inline for (0..4) |element| {
                            values[0][quad][element] = gradients.dv[quad * 4 + element];
                            values[1][quad][element] = gradients.dk[quad * 4 + element];
                        }
                    }
                    const owned = [2]bool{ splitOwner(0, consumerWarp()) == rank, splitOwner(1, consumerWarp()) == rank };
                    exchangePartials(padded / 8, wgmma.address(&tiles.k), &values, owned, splits);
                    inline for (0..padded / 8) |quad| {
                        inline for (0..4) |element| {
                            finished.dv[quad * 4 + element] = values[0][quad][element];
                            finished.dk[quad * 4 + element] = values[1][quad][element];
                        }
                    }
                }
                if (splitOwner(1, consumerWarp()) == rank) stageRows(&tiles.k, finished.dk, params.softmax_scale);
                if (splitOwner(0, consumerWarp()) == rank) stageRows(&tiles.v, finished.dv, 1);
                nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                copyRows(&tiles.k, 1, work, params.dk, params.dk_batch_stride, params.dk_row_stride, params.dk_head_stride);
                copyRows(&tiles.v, 0, work, params.dv, params.dv_batch_stride, params.dv_row_stride, params.dv_head_stride);
            }

            fn stageRows(tile: *addrspace(.shared) Layout.Tile(block_n), gradient: wgmma.Accumulator(padded), scale: f32) void {
                const lane = thread.lane();
                const row = consumerWarp() * 16 + lane % 16;
                const pairs = wgmma.packPairs(padded, wgmma.scaledBy(padded, gradient, scale));
                inline for (pairs, 0..) |fragment, step| {
                    wgmma.storeMatrices(false, wgmma.address(Layout.chunk(tile, row, @as(u32, step) * 2 + lane / 16, 0)), fragment);
                }
            }

            fn copyRows(
                tile: *addrspace(.shared) Layout.Tile(block_n),
                tensor: u32,
                work: Work,
                address: u64,
                batch_stride: u64,
                row_stride: u64,
                head_stride: u64,
            ) void {
                const chunks_per_row = head_dim / 8;
                const output: [*]addrspace(.global) Chunk = @ptrFromInt(address);
                const origin = work.batch * batch_stride + work.kv_head * head_stride +
                    @as(u64, work.sequence.offset_k + work.first_key) * row_stride;
                const consumer = thread.id() - producer_warps * 32;
                inline for (0..(block_n * chunks_per_row + consumer_threads - 1) / consumer_threads) |pass| {
                    const index = @as(u32, pass) * consumer_threads + consumer;
                    const row = index / chunks_per_row;
                    if (row < work.key_count and splitOwner(tensor, row / 16) == clusterRank() and ((pass + 1) * consumer_threads <= block_n * chunks_per_row or index < block_n * chunks_per_row)) {
                        const column = index % chunks_per_row;
                        output[(origin + row * row_stride) / 8 + column] = Layout.chunk(tile, row, column, 0).*;
                    }
                }
            }
        };

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}

pub fn FlashAttnBwdSm90SinglePdsStage(comptime tiling: configurations.Sm90BwdTiling, comptime specialization: configurations.Specialization) type {
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const q_stage = tiling.stages;
    const swizzle_bytes = tiling.swizzleBytes();
    const Operands = wgmma.Operands(swizzle_bytes);
    const PdSOperands = wgmma.Operands(pds_swizzle_bytes);
    const k_steps = padded / 16;
    const query_steps = block_m / 16;
    const key_steps = block_n / 16;
    const keys_per_compute_group = block_n / 2;
    const group_columns = padded / 2;
    const halves = group_columns / 64;
    const score_rows = block_m * pds_swizzle_bytes;
    const Layout = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const Work = NBlockWork(block_m, block_n);
    const staging_boxes = tiling.residentStagingBoxes();
    const box_columns = configurations.Sm90BwdTiling.dq_accum_box_columns;
    comptime std.debug.assert(tiling.staging_buffers == 1 and tiling.doStage() == 1 and keys_per_compute_group % 8 == 0);

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            do: *const tma.TensorMap,
            dq_accum: *const tma.TensorMap,
        };

        const Tiles = struct {
            k: Layout.Tile(block_n),
            v: Layout.Tile(block_n),
            q: [q_stage]Layout.Tile(block_m),
            do: Layout.Tile(block_m),
            probabilities: [block_m * block_n / 8]Chunk,
            ds: [block_m * block_n / 8]Chunk,
            staging: [staging_boxes][block_m * box_columns / 4]Chunk,
            lse: [q_stage][block_m]f32,
            dpsum: [block_m]f32,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var kv_full: nvptx.Mbarrier addrspace(.shared) = undefined;
        var query_full: [q_stage]nvptx.Mbarrier addrspace(.shared) = undefined;
        var query_empty: [q_stage]nvptx.Mbarrier addrspace(.shared) = undefined;
        var do_full: nvptx.Mbarrier addrspace(.shared) = undefined;
        var do_empty: nvptx.Mbarrier addrspace(.shared) = undefined;

        pub const init_scope: mbarrier.Scope = .cta;
        pub const initializer_warp: u32 = 0;
        pub const roles = .{
            .{ .first_warp = 0, .warps = producer_warps, .registers = role_registers[0], .Role = Producer },
            .{ .first_warp = producer_warps, .warps = consumer_warps, .registers = role_registers[1], .Role = Consumer },
        };

        pub fn initialize() void {
            if (thread.warp() == initializer_warp) {
                if (elect.oneSync()) {
                    kv_full.init(1);
                    inline for (0..q_stage) |stage| {
                        query_full[stage].init(load_arrivals);
                        query_empty[stage].init(consumer_warps);
                    }
                    do_full.init(load_arrivals);
                    do_empty.init(consumer_warps);
                    mbarrier.pipelineInitFence();
                }
                elect.reconverge();
            }
        }

        fn waitParity(barrier: *addrspace(.shared) nvptx.Mbarrier, parity: u32) void {
            while (!barrier.tryWaitParity(@intCast(parity & 1))) {}
        }

        pub const Producer = struct {
            pub fn run(params: BackwardParams, args: Args) void {
                switch (thread.warp()) {
                    0 => load(params, args),
                    1 => reduceDq(params, args),
                    else => {},
                }
            }

            fn load(params: BackwardParams, args: Args) void {
                const lane = thread.lane();
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                const iterations = work.iterations();
                if (iterations == 0) return;
                const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
                if (elect.oneSync()) {
                    _ = kv_full.arriveExpectTx(comptime 2 * tiling.tileBytes(block_n));
                    const row = work.sequence.offset_k + work.first_key;
                    Layout.copyTile(&tiles.k, args.k, &kv_full, work.kv_head, row, batch);
                    Layout.copyTile(&tiles.v, args.v, &kv_full, work.kv_head, row, batch);
                }
                elect.reconverge();
                grid_dependency.wait();
                const lse: [*]addrspace(.global) const f32 = @ptrFromInt(params.lse_log2);
                const dpsum: [*]addrspace(.global) const f32 = @ptrFromInt(params.dpsum);
                var iteration: u32 = 0;
                while (iteration < iterations) : (iteration += 1) {
                    const step = work.step(iteration);
                    const stage = iteration % q_stage;
                    const row = work.sequence.offset_q + step.block_info.first_row;
                    const vectors: u64 = work.batch * params.lse_batch_stride + step.head * params.lse_head_stride + row;
                    if (iteration >= q_stage) waitParity(&query_empty[stage], iteration / q_stage - 1);
                    if (elect.oneSync()) {
                        _ = query_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_m));
                        Layout.copyTile(&tiles.q[stage], args.q, &query_full[stage], step.head, row, batch);
                    }
                    elect.reconverge();
                    inline for (0..block_m / 32) |element| {
                        const column = element * 32 + lane;
                        cp_async.copy(.cache_all_levels, &tiles.lse[stage][column], &lse[vectors + column]);
                    }
                    cp_async.mbarrierArriveNoinc(&query_full[stage]);
                    if (iteration > 0) waitParity(&do_empty, iteration - 1);
                    if (elect.oneSync()) {
                        _ = do_full.arriveExpectTx(comptime tiling.tileBytes(block_m));
                        Layout.copyTile(&tiles.do, args.do, &do_full, step.head, row, batch);
                    }
                    elect.reconverge();
                    inline for (0..block_m / 32) |element| {
                        const column = element * 32 + lane;
                        cp_async.copy(.cache_all_levels, &tiles.dpsum[column], &dpsum[vectors + column]);
                    }
                    cp_async.mbarrierArriveNoinc(&do_full);
                }
            }

            fn reduceDq(params: BackwardParams, args: Args) void {
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                grid_dependency.wait();
                const iterations = work.iterations();
                const row_base = work.sequence.offset_q +
                    if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
                if (iterations > 0) nvptx.barrierArriveAligned(NamedBarrier.dq_empty.of(0), staging_threads);
                var iteration: u32 = 0;
                while (iteration < iterations) : (iteration += 1) {
                    const step = work.step(iteration);
                    const row = row_base + step.block_info.first_row;
                    inline for (0..2) |group| {
                        nvptx.barrierSyncAligned(NamedBarrier.dq_full.of(group), staging_threads);
                        if (elect.oneSync()) {
                            inline for (0..staging_boxes) |box| {
                                tma.reduceInterleavedBox(args.dq_accum, &tiles.staging[box][0], (group * staging_boxes + box) * box_columns, row, step.head);
                            }
                            tma.commitBulkGroup();
                        }
                        elect.reconverge();
                        tma.waitBulkReads();
                        if (group == 0) {
                            nvptx.barrierArriveAligned(NamedBarrier.dq_empty.of(1), staging_threads);
                        } else if (iteration + 1 < iterations) {
                            nvptx.barrierArriveAligned(NamedBarrier.dq_empty.of(0), staging_threads);
                        }
                    }
                }
            }
        };

        const Gradients = struct {
            dk: [halves]wgmma.Accumulator(block_n),
            dv: [halves]wgmma.Accumulator(block_n),
        };

        fn exponentiate(
            scores: wgmma.Accumulator(keys_per_compute_group),
            stage: u32,
            softmax_scale_log2: f32,
        ) wgmma.Accumulator(keys_per_compute_group) {
            const row = consumerWarp() % 4 * 16 + thread.lane() / 4;
            const lse_negated = [2]f32{ tiles.lse[stage][row], tiles.lse[stage][row + 8] };
            var result: wgmma.Accumulator(keys_per_compute_group) = undefined;
            inline for (0..keys_per_compute_group / 8) |tile| {
                inline for (0..4) |element| {
                    const index = tile * 4 + element;
                    result[index] = nvptx.math.ex2ApproxF32(true, @mulAdd(f32, scores[index], softmax_scale_log2, lse_negated[element / 2]));
                }
            }
            return result;
        }

        fn computeDs(
            comptime masked: bool,
            probabilities: wgmma.Accumulator(keys_per_compute_group),
            dp: wgmma.Accumulator(keys_per_compute_group),
        ) wgmma.Accumulator(keys_per_compute_group) {
            const row = consumerWarp() % 4 * 16 + thread.lane() / 4;
            const dpsum_negated = [2]f32{ tiles.dpsum[row], tiles.dpsum[row + 8] };
            var result: wgmma.Accumulator(keys_per_compute_group) = undefined;
            inline for (0..keys_per_compute_group / 8) |tile| {
                inline for (0..4) |element| {
                    const index = tile * 4 + element;
                    const p = probabilities[index];
                    const ds = p * (dp[index] + dpsum_negated[element / 2]);
                    result[index] = if (masked and p == 0) 0 else ds;
                }
            }
            return result;
        }

        fn storePdS(destination: *addrspace(.shared) [block_m * block_n / 8]Chunk, values: wgmma.Accumulator(keys_per_compute_group)) void {
            const lane = thread.lane();
            const row = consumerWarp() % 4 * 16 + lane / 4;
            const base = wgmma.address(destination) + lane % 4 * 4;
            inline for (0..keys_per_compute_group / 8) |tile| {
                const chunk = consumerGroup() * (keys_per_compute_group / 8) + @as(u32, tile);
                inline for (0..2) |half| {
                    const r = row + @as(u32, half) * 8;
                    const word: *addrspace(.shared) u32 = @ptrFromInt(base + chunk / 2 * score_rows + r * pds_swizzle_bytes +
                        smem.swizzledColumn(pds_swizzle_bytes, chunk % 2, r) * 16);
                    word.* = @backingInt(nvptx.Bf16x2.fromF32(values[tile * 4 + half * 2], values[tile * 4 + half * 2 + 1]));
                }
            }
        }

        fn storeDq(accumulator: wgmma.Accumulator(group_columns)) void {
            const lane = thread.lane();
            const row = stagingRow(consumerWarp() % 4 * 16 + lane / 4);
            const row_address = wgmma.address(&tiles.staging) + row * box_columns * 4 + lane % 2 * 8;
            const swizzle_offset = (lane % 4 / 2 ^ row % 8) * 16;
            inline for (0..group_columns / 8) |tile| {
                const column = tile * 8;
                const box_offset = column / box_columns * (block_m * box_columns * 4);
                const chunk: u32 = tile % 4 * 32;
                inline for (0..2) |half| {
                    const destination: *addrspace(.shared) @Vector(2, f32) = @ptrFromInt(
                        row_address + box_offset + half * 8 * box_columns * 4 + (chunk ^ swizzle_offset),
                    );
                    destination.* = .{ accumulator[tile * 4 + half * 2], accumulator[tile * 4 + half * 2 + 1] };
                }
            }
        }

        pub const Consumer = struct {
            pub fn run(params: BackwardParams, _: Args) void {
                @setEvalBranchQuota(100_000);
                const group = consumerGroup();
                const work = splitWork(Work, tiling, specialization, params) orelse return;
                const iterations = work.iterations();
                var gradients: Gradients = .{ .dk = @splat(@splat(0)), .dv = @splat(@splat(0)) };
                if (iterations > 0) {
                    while (!kv_full.tryWaitParity(0)) {}
                    if (params.cu_seqlens_k != 0 and work.key_count < block_n) {
                        Layout.clearRows(&tiles.k, work.key_count, consumer_threads, thread.id() - producer_warps * 32);
                        publishClearedRows();
                    }
                    const unmasked = work.unmaskedRowLimits();
                    const first_column = work.first_key + group * keys_per_compute_group;
                    const key_base = wgmma.address(&tiles.k) + group * keys_per_compute_group * swizzle_bytes;
                    const value_base = wgmma.address(&tiles.v) + group * keys_per_compute_group * swizzle_bytes;
                    const do_base = wgmma.address(&tiles.do);
                    const probabilities_base = wgmma.address(&tiles.probabilities);
                    const ds_base = wgmma.address(&tiles.ds);
                    const key_rows_base = wgmma.address(&tiles.k) + group * halves * (block_n * swizzle_bytes);
                    var iteration: u32 = 0;
                    while (iteration < iterations) : (iteration += 1) {
                        const step = work.step(iteration);
                        const stage = iteration % q_stage;
                        waitParity(&query_full[stage], iteration / q_stage);
                        waitParity(&do_full, iteration);
                        if (params.cu_seqlens_q != 0 and step.block_info.row_count < block_m) {
                            Layout.clearRows(&tiles.q[stage], step.block_info.row_count, consumer_threads, thread.id() - producer_warps * 32);
                            Layout.clearRows(&tiles.do, step.block_info.row_count, consumer_threads, thread.id() - producer_warps * 32);
                            publishClearedRows();
                        }
                        const query_base = wgmma.address(&tiles.q[stage]);
                        var scores: wgmma.Accumulator(keys_per_compute_group) = undefined;
                        var dp: wgmma.Accumulator(keys_per_compute_group) = undefined;
                        wgmma.fence();
                        inline for (0..k_steps) |k_step| {
                            wgmma.mmaShared(
                                keys_per_compute_group,
                                &scores,
                                Operands.kMajor(query_base + Operands.kStepOffset(block_m, k_step)),
                                Operands.kMajor(key_base + Operands.kStepOffset(block_n, k_step)),
                                k_step != 0,
                                false,
                                false,
                            );
                        }
                        wgmma.commit();
                        inline for (0..k_steps) |k_step| {
                            wgmma.mmaShared(
                                keys_per_compute_group,
                                &dp,
                                Operands.kMajor(do_base + Operands.kStepOffset(block_m, k_step)),
                                Operands.kMajor(value_base + Operands.kStepOffset(block_n, k_step)),
                                k_step != 0,
                                false,
                                false,
                            );
                        }
                        wgmma.commit();
                        wgmma.wait(1);
                        wgmma.fenceOperand(&scores);
                        const masked = !unmasked.contains(step.block_info.first_row);
                        var probabilities = exponentiate(scores, stage, params.softmax_scale_log2);
                        if (masked) {
                            const row = step.block_info.first_row + consumerWarp() % 4 * 16 + thread.lane() / 4;
                            var limits = ColLimits.init(step.block_info, row);
                            if (row >= step.block_info.seqlen_q) limits.col_limit_right[0] = 0;
                            if (row + 8 >= step.block_info.seqlen_q) limits.col_limit_right[1] = 0;
                            applyMask(&probabilities, first_column, limits, 0);
                        }
                        nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                        storePdS(&tiles.probabilities, probabilities);
                        wgmma.wait(0);
                        wgmma.fenceOperand(&dp);
                        const ds = if (masked)
                            computeDs(true, probabilities, dp)
                        else
                            computeDs(false, probabilities, dp);
                        storePdS(&tiles.ds, ds);
                        nvptx.fenceProxyAsync(.shared);
                        nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                        var dq: wgmma.Accumulator(group_columns) = undefined;
                        wgmma.fence();
                        inline for (0..halves) |half| {
                            inline for (0..query_steps) |query_step| {
                                wgmma.mmaShared(
                                    block_n,
                                    &gradients.dv[half],
                                    Operands.mnMajor(do_base + (group * halves + @as(u32, half)) * (block_m * swizzle_bytes) + @as(u32, query_step * 16 * swizzle_bytes), block_m),
                                    PdSOperands.mnMajor(probabilities_base + @as(u32, query_step * 16 * pds_swizzle_bytes), block_m),
                                    true,
                                    true,
                                    true,
                                );
                            }
                        }
                        wgmma.commit();
                        inline for (0..key_steps) |key_step| {
                            wgmma.mmaShared(
                                group_columns,
                                &dq,
                                PdSOperands.kMajor(ds_base + PdSOperands.kStepOffset(block_m, key_step)),
                                Operands.mnMajor(key_rows_base + @as(u32, key_step * 16 * swizzle_bytes), block_n),
                                key_step != 0,
                                false,
                                true,
                            );
                        }
                        wgmma.commit();
                        inline for (0..halves) |half| {
                            inline for (0..query_steps) |query_step| {
                                wgmma.mmaShared(
                                    block_n,
                                    &gradients.dk[half],
                                    Operands.mnMajor(query_base + (group * halves + @as(u32, half)) * (block_m * swizzle_bytes) + @as(u32, query_step * 16 * swizzle_bytes), block_m),
                                    PdSOperands.mnMajor(ds_base + @as(u32, query_step * 16 * pds_swizzle_bytes), block_m),
                                    true,
                                    true,
                                    true,
                                );
                            }
                        }
                        wgmma.commit();
                        wgmma.wait(1);
                        wgmma.fenceOperand(&dq);
                        if (elect.oneSync()) _ = do_empty.arrive();
                        elect.reconverge();
                        nvptx.barrierSyncAligned(NamedBarrier.dq_empty.of(group), staging_threads);
                        storeDq(dq);
                        nvptx.fenceProxyAsync(.shared);
                        nvptx.barrierArriveAligned(NamedBarrier.dq_full.of(group), staging_threads);
                        wgmma.wait(0);
                        inline for (0..halves) |half| {
                            wgmma.fenceOperand(&gradients.dk[half]);
                            wgmma.fenceOperand(&gradients.dv[half]);
                        }
                        if (elect.oneSync()) _ = query_empty[stage].arrive();
                        elect.reconverge();
                    }
                }
                store(params, work, gradients);
            }

            fn store(params: BackwardParams, work: Work, gradients: Gradients) void {
                nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                const splits = cluster.count();
                const rank = clusterRank();
                var finished = gradients;
                const quads = block_n / 8;
                if (splits > 1) {
                    var values: [2][halves * quads]@Vector(4, f32) = undefined;
                    inline for (0..halves) |half| {
                        inline for (0..quads) |quad| {
                            inline for (0..4) |element| {
                                values[0][half * quads + quad][element] = gradients.dv[half][quad * 4 + element];
                                values[1][half * quads + quad][element] = gradients.dk[half][quad * 4 + element];
                            }
                        }
                    }
                    const owned = [2]bool{ splitOwner(0, consumerWarp()) == rank, splitOwner(1, consumerWarp()) == rank };
                    exchangePartials(halves * quads, wgmma.address(&tiles.k), &values, owned, splits);
                    inline for (0..halves) |half| {
                        inline for (0..quads) |quad| {
                            inline for (0..4) |element| {
                                finished.dv[half][quad * 4 + element] = values[0][half * quads + quad][element];
                                finished.dk[half][quad * 4 + element] = values[1][half * quads + quad][element];
                            }
                        }
                    }
                }
                inline for (0..halves) |half| {
                    if (splitOwner(1, consumerWarp()) == rank) stageRows(&tiles.k, half, finished.dk[half], params.softmax_scale);
                    if (splitOwner(0, consumerWarp()) == rank) stageRows(&tiles.v, half, finished.dv[half], 1);
                }
                nvptx.barrierSyncAligned(NamedBarrier.consumers, consumer_threads);
                copyRows(&tiles.k, 1, work, params.dk, params.dk_batch_stride, params.dk_row_stride, params.dk_head_stride);
                copyRows(&tiles.v, 0, work, params.dv, params.dv_batch_stride, params.dv_row_stride, params.dv_head_stride);
            }

            fn stageRows(tile: *addrspace(.shared) Layout.Tile(block_n), comptime half: u32, gradient: wgmma.Accumulator(block_n), scale: f32) void {
                const lane = thread.lane();
                const matrix = lane / 8;
                const column = (consumerGroup() * group_columns + half * 64 + consumerWarp() % 4 * 16) / 8 + matrix % 2;
                const pairs = wgmma.packPairs(block_n, wgmma.scaledBy(block_n, gradient, scale));
                inline for (pairs, 0..) |fragment, step| {
                    const key = (@as(u32, step) * 2 + matrix / 2) * 8 + lane % 8;
                    wgmma.storeMatrices(true, wgmma.address(Layout.chunk(tile, key, column, 0)), fragment);
                }
            }

            fn copyRows(
                tile: *addrspace(.shared) Layout.Tile(block_n),
                tensor: u32,
                work: Work,
                address: u64,
                batch_stride: u64,
                row_stride: u64,
                head_stride: u64,
            ) void {
                const chunks_per_row = head_dim / 8;
                const output: [*]addrspace(.global) Chunk = @ptrFromInt(address);
                const origin = work.batch * batch_stride + work.kv_head * head_stride +
                    @as(u64, work.sequence.offset_k + work.first_key) * row_stride;
                const consumer = thread.id() - producer_warps * 32;
                inline for (0..(block_n * chunks_per_row + consumer_threads - 1) / consumer_threads) |pass| {
                    const index = @as(u32, pass) * consumer_threads + consumer;
                    const row = index / chunks_per_row;
                    const column = index % chunks_per_row;
                    const unit = column * 8 / group_columns * 4 + column * 8 % 64 / 16;
                    if (row < work.key_count and splitOwner(tensor, unit) == clusterRank() and ((pass + 1) * consumer_threads <= block_n * chunks_per_row or index < block_n * chunks_per_row)) {
                        output[(origin + row * row_stride) / 8 + column] = Layout.chunk(tile, row, column, 0).*;
                    }
                }
            }
        };

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}

const variant = configurations.backwardSm90Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const backward_tiling = variant.tiling;

comptime {
    std.debug.assert(configurations.launchBackwardSm90(config.head_dim, @field(configurations.Mode, config.mode)).?.max_qhead_splits <= split_units);
}

const auxiliary: flash_bwd_preprocess.Options = .{ .pdl = true, .varlen_as_one_sequence = variant.specialization.scheduler_metadata };
const Backward = if (backward_tiling.pds_stage == 1)
    FlashAttnBwdSm90SinglePdsStage(backward_tiling, variant.specialization)
else
    FlashAttnBwdSm90(backward_tiling, variant.specialization);

fn flashAttnBwd(
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
    .max_registers_per_work_item = Backward.max_registers,
} }) void {
    @setEvalBranchQuota(100_000);
    grid_dependency.launchDependents();
    if (variant.specialization.scheduler_metadata) grid_dependency.wait();
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
        @export(&flashAttnBwd, .{ .name = "flash_attn_bwd" });
        @export(&flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess, .{ .name = "flash_attn_bwd_preprocess" });
        @export(&flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary).flashAttnBwdPostprocess, .{ .name = "flash_attn_bwd_postprocess" });
        if (variant.specialization.scheduler_metadata) {
            const options = schedulerOptions(backward_tiling, variant.specialization);
            const PrepareScheduler = prepare_scheduler.PrepareScheduler(options.block_m, options.block_order, variant.specialization.sort_batches);
            @export(&PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_bwd_prepare_varlen_num_blocks" });
        }
    }
}
