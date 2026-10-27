const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const dsPiece = @import("f32x2.zig").dsPiece;
const tma = @import("tma.zig");
const tcgen05 = @import("tcgen05.zig");
const f32x2 = @import("f32x2.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");
const cluster = @import("cluster.zig");
const mbarrier = @import("mbarrier.zig");
const BlockInfo = @import("BlockInfo.zig");
const tile_scheduler = @import("tile_scheduler.zig");

const Chunk = fragments.Chunk;
const Mbarrier = nvptx.Mbarrier;

const NamedBarrier = enum(u4) { pivot_exchange = 2 };

pub fn FlashAttnBwdSm100Dq(comptime tiling: configurations.Sm100BwdTiling) type {
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const swizzle_bytes = tiling.swizzleBytes();
    const swizzle = tcgen05.Swizzle.fromBytes(swizzle_bytes);
    const box_columns = tiling.boxColumns();
    const boxes = padded / box_columns;
    const query_rows = tiling.dq_block_m / 2;
    const cluster_rows = tiling.dq_block_m;
    const key_rows = 2 * tiling.block_n;
    const half_keys = key_rows / 2;
    const half_columns = padded / 2;
    const half_boxes = half_columns / box_columns;
    const compute_groups = configurations.Sm100BwdTiling.compute_groups;
    const other_warps = configurations.Sm100BwdTiling.other_warps;
    const num_regs = configurations.Sm100BwdTiling.num_regs_separate_dq;
    const keys_per_compute_group = key_rows / compute_groups;
    const group_steps = keys_per_compute_group / 16;
    const pieces = keys_per_compute_group / tcgen05.piece_columns;
    const dq_pieces = half_columns / tcgen05.piece_columns;
    const s_column = 0;
    const dp_column = key_rows;
    const dq_column = 2 * key_rows;
    const compute_warps = 4 * compute_groups;
    const empty_warps = 4;
    const other_warp = compute_warps + empty_warps;
    const Scores = @Vector(keys_per_compute_group, f32);
    const TileScheduler = tile_scheduler.TileScheduler(.{
        .block_m = cluster_rows,
        .row_bytes = 4 * head_dim,
        .pack_gqa = false,
        .block_order = .ascending,
        .l2_swizzle = tiling.l2_swizzle,
    });
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 2 });

    return struct {
        pub const thread_count = tiling.threads();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        pub const init_scope: mbarrier.Scope = .cluster;
        pub const initializer_warp: u32 = other_warp + 1;
        pub const tmem_warps: u32 = (1 << compute_warps) - 1 | 1 << other_warp;
        pub const tmem_allocator_warp: u32 = other_warp;
        pub const tmem_barrier: nvptx.NamedBarrier = @fromBackingInt(1);
        pub const roles = .{
            .{ runCompute, compute_warps, num_regs.compute },
            .{ runEmpty, empty_warps, num_regs.empty },
            .{ runOther, other_warps, num_regs.other },
        };
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            do: *const tma.TensorMap,
            dq: u64,
            dq_batch_stride: u64,
            dq_row_stride: u64,
            dq_head_stride: u64,
        };

        const Tiles = struct {
            q: [query_rows * padded / 8]Chunk,
            do: [query_rows * padded / 8]Chunk,
            k: [half_keys * padded / 8]Chunk,
            v: [half_keys * padded / 8]Chunk,
            kt: [key_rows * half_columns / 8]Chunk,
            pivot_sums: [compute_groups][query_rows]f32,
            pivot_keys: [compute_groups][query_rows]i32,
        };

        const RowPivot = struct {
            sum: f32,
            key: i32,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) <= tiling.sharedBytes());
            std.debug.assert(dq_column + padded <= tcgen05.capacity);
            std.debug.assert(other_warp + other_warps == thread_count / 32);
            for (
                .{ "q", "do", "k", "v", "kt" },
            ) |field| std.debug.assert(@offsetOf(Tiles, field) % smem.alignment == 0);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var q_full: Mbarrier addrspace(.shared) = undefined;
        var do_full: Mbarrier addrspace(.shared) = undefined;
        var k_full: Mbarrier addrspace(.shared) = undefined;
        var v_full: Mbarrier addrspace(.shared) = undefined;
        var kt_full: Mbarrier addrspace(.shared) = undefined;
        var k_empty: Mbarrier addrspace(.shared) = undefined;
        var v_empty: Mbarrier addrspace(.shared) = undefined;
        var kt_empty: Mbarrier addrspace(.shared) = undefined;
        var kt_tma_full: Mbarrier addrspace(.shared) = undefined;
        var s_full: Mbarrier addrspace(.shared) = undefined;
        var dp_full: Mbarrier addrspace(.shared) = undefined;
        var dq_full: Mbarrier addrspace(.shared) = undefined;
        var s_empty: Mbarrier addrspace(.shared) = undefined;
        var ds_ready: Mbarrier addrspace(.shared) = undefined;
        var tmem_base: u32 addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            tcgen05.allocate2Sm(&tmem_base, tcgen05.capacity);
        }

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                inline for (.{
                    &q_full,      &do_full, &k_full,  &v_full,  &k_empty, &v_empty, &kt_empty,
                    &kt_tma_full, &s_full,  &dp_full, &dq_full,
                }) |barrier| {
                    barrier.init(1);
                }
                kt_full.init(2);
                s_empty.init(2 * compute_warps);
                ds_ready.init(2 * compute_warps);
                mbarrier.pipelineInitFence();
            }
            elect.reconverge();
        }

        pub fn finish() void {
            tcgen05.fenceBeforeSync();
            cluster.sync();
            if (thread.warp() == other_warp) {
                tcgen05.fenceAfterSync();
                tcgen05.deallocate2Sm(tmem_base, tcgen05.capacity);
            }
        }

        const Work = struct {
            batch: u32,
            head: u32,
            kv_head: u32,
            sequence: SeqlenInfoQK,
            first_row: u32,
            block_info: BlockInfo,
            n_block_min: u32,
            n_block_max: u32,

            fn decode(params: BackwardParams) ?Work {
                const persistent_scheduler: PersistentScheduler = .create();
                const grid =
                    tile_scheduler.gridTile(.x, 2, persistent_scheduler.initialWorkTileInfo());
                const scheduler = TileScheduler.create(.{
                    .num_block = grid.num_block,
                    .num_head = grid.num_head,
                    .num_batch = params.num_batch,
                    .num_qhead_splits = grid.num_qhead_splits,
                    .seqlen_other = params.seqlen_k,
                    .qhead_per_kvhead = params.qhead_per_kvhead,
                    .window_size_left = params.window_size_left,
                    .window_size_right = params.window_size_right,
                    .cu_seqlens = params.cu_seqlens_q,
                    .cu_seqlens_other = params.cu_seqlens_k,
                });
                const tile = scheduler.getGridWork(grid) orelse return null;
                const sequence = params.sequence(tile.batch);
                const first_row = tile.block * cluster_rows;
                const info: BlockInfo = .{
                    .first_row = first_row,
                    .row_count = @min(cluster_rows, sequence.seqlen_q - first_row),
                    .seqlen_q = sequence.seqlen_q,
                    .seqlen_k = sequence.seqlen_k,
                    .window_size_left = params.window_size_left,
                    .window_size_right = params.window_size_right,
                };
                const kv_head = tile.head / params.qhead_per_kvhead;
                const n_block_min, const n_block_max = info.nBlockMinMax(key_rows);
                return .{
                    .batch = tile.batch,
                    .head = tile.head,
                    .kv_head = kv_head,
                    .sequence = sequence,
                    .first_row = first_row,
                    .block_info = info,
                    .n_block_min = n_block_min,
                    .n_block_max = n_block_max,
                };
            }

            fn tensorBatch(work: Work, params: BackwardParams) u32 {
                return if (params.cu_seqlens_q == 0) work.batch else 0;
            }
        };

        pub fn runOther(params: BackwardParams, args: Args) void {
            switch (thread.warp() - other_warp) {
                0 => runMma(params),
                1 => runLoad(params, args),
                else => {},
            }
        }

        pub fn runEmpty(_: BackwardParams, _: Args) void {}

        fn copyTransposed(
            params: BackwardParams,
            args: Args,
            work: Work,
            first_key: u32,
            index: u32,
        ) void {
            mbarrier.wait(.cta, &kt_empty, index ^ 1);
            if (elect.oneSync()) {
                _ = kt_tma_full.arriveExpectTx(key_rows * half_columns * 2);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = key_rows,
                    .box_rows = tiling.block_n,
                    .count = half_boxes,
                }, &tiles.kt, args.k, &kt_tma_full, .{
                    cluster.rankIn2Cta() * half_columns, work.kv_head, first_key,
                    work.tensorBatch(
                        params,
                    ),
                });
            }
            elect.reconverge();
            mbarrier.wait(.cta, &kt_tma_full, index);
            const valid = SeqlenInfoQK.rowsBeforeEnd(
                params.cu_seqlens_k,
                params.num_batch,
                first_key,
                key_rows,
            );
            if (valid < key_rows) {
                tma.clearBoxRows(box_columns, key_rows, half_boxes, &tiles.kt, valid);
                nvptx.fenceProxyAsync(.shared);
            }
            mbarrier.arrive(.leader_cta, &kt_full);
        }

        fn runLoad(params: BackwardParams, args: Args) void {
            const work = Work.decode(params) orelse return;
            const count = work.n_block_max - work.n_block_min;
            const batch = work.tensorBatch(params);
            const own_row =
                work.sequence.offset_q + work.first_row + cluster.rankIn2Cta() * query_rows;
            if (elect.oneSync()) {
                if (cluster.rankIn2Cta() == 0) _ =
                    q_full.arriveExpectTx(2 * query_rows * padded * 2);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = query_rows,
                    .box_rows = tiling.block_m,
                    .count = boxes,
                }, &tiles.q, args.q, cluster.leaderAddress(
                    &q_full,
                ), .{ 0, work.head, own_row, batch });
            }
            elect.reconverge();
            var index: u32 = 0;
            while (index < count) : (index += 1) {
                const first_key = work.sequence.offset_k + (work.n_block_min + index) * key_rows;
                const own_key = first_key + cluster.rankIn2Cta() * half_keys;
                mbarrier.wait(.cta, &k_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        k_full.arriveExpectTx(2 * half_keys * padded * 2);
                    tma.copyBoxes(.{
                        .columns = box_columns,
                        .rows = half_keys,
                        .box_rows = tiling.block_n,
                        .count = boxes,
                    }, &tiles.k, args.k, cluster.leaderAddress(
                        &k_full,
                    ), .{ 0, work.kv_head, own_key, batch });
                    if (index == 0) {
                        if (cluster.rankIn2Cta() == 0) _ =
                            do_full.arriveExpectTx(2 * query_rows * padded * 2);
                        tma.copyBoxes(.{
                            .columns = box_columns,
                            .rows = query_rows,
                            .box_rows = tiling.block_m,
                            .count = boxes,
                        }, &tiles.do, args.do, cluster.leaderAddress(
                            &do_full,
                        ), .{ 0, work.head, own_row, batch });
                    }
                }
                elect.reconverge();
                mbarrier.wait(.cta, &v_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        v_full.arriveExpectTx(2 * half_keys * padded * 2);
                    tma.copyBoxes(.{
                        .columns = box_columns,
                        .rows = half_keys,
                        .box_rows = tiling.block_n,
                        .count = boxes,
                    }, &tiles.v, args.v, cluster.leaderAddress(
                        &v_full,
                    ), .{ 0, work.kv_head, own_key, batch });
                }
                elect.reconverge();
                copyTransposed(params, args, work, first_key, index);
            }
        }

        const dq_instruction = tcgen05.instructionDescriptor(cluster_rows, padded, .k, .mn);

        fn operandOffsets() [key_rows / 16]u32 {
            var offsets: [key_rows / 16]u32 = undefined;
            for (&offsets, 0..) |*offset, step| offset.* =
                step / group_steps * keys_per_compute_group + step % group_steps * 8;
            return offsets;
        }

        fn mmaScores(tmem: u32, comptime column: u32, a: anytype, b: anytype) void {
            tcgen05.mmaKMajor(2, padded, query_rows, half_keys, tmem + column, a, b);
        }

        fn mmaDq(tmem: u32, index: u32) void {
            mbarrier.wait(.cta, &ds_ready, index);
            mbarrier.wait(.cta, &kt_full, index);
            tcgen05.fenceAfterSync();
            tcgen05.mmaTsSteps2Sm(
                &comptime operandOffsets(),
                &comptime tcgen05.rowOffsets(key_rows / 16, swizzle_bytes),
                tmem + dq_column,
                tmem + dp_column,
                tcgen05.sharedDescriptor(
                    tcgen05.sharedAddress(&tiles.kt),
                    key_rows * swizzle_bytes,
                    8 * swizzle_bytes,
                    swizzle,
                ),
                dq_instruction,
                index > 0,
            );
            tcgen05.commit(.cta_group_2, &kt_empty);
        }

        fn runMma(params: BackwardParams) void {
            const work = Work.decode(params) orelse return;
            const count = work.n_block_max - work.n_block_min;
            if (cluster.rankIn2Cta() != 0 or count == 0) return;
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            mbarrier.wait(.cta, &q_full, 0);
            var index: u32 = 0;
            while (index < count) : (index += 1) {
                mbarrier.wait(.cta, &k_full, index);
                if (index > 0) mbarrier.wait(.cta, &s_empty, index - 1);
                tcgen05.fenceAfterSync();
                mmaScores(tmem, s_column, &tiles.q, &tiles.k);
                tcgen05.commit(.cta_group_2, &s_full);
                tcgen05.commit(.cta_group_2, &k_empty);
                if (index > 0) mmaDq(tmem, index - 1);
                if (index == 0) mbarrier.wait(.cta, &do_full, 0);
                mbarrier.wait(.cta, &v_full, index);
                tcgen05.fenceAfterSync();
                mmaScores(tmem, dp_column, &tiles.do, &tiles.v);
                tcgen05.commit(.cta_group_2, &dp_full);
                tcgen05.commit(.cta_group_2, &v_empty);
            }
            mmaDq(tmem, count - 1);
            tcgen05.commit(.cta_group_2, &dq_full);
        }

        fn columnGroup() u32 {
            return thread.warp() / 4;
        }

        fn computeDs(
            probabilities: Scores,
            dp_address: u32,
            dpsum_negated: f32,
            interior: bool,
            lower: i32,
            upper: i32,
        ) f32x2.Pair {
            const dp = tcgen05.loadPieces(pieces, dp_address);
            tcgen05.waitLoad();
            var sum = f32x2.splat(0);
            inline for (0..pieces) |piece_index| {
                const piece: u32 = piece_index;
                const words =
                    dsPiece(piece, probabilities, dp, dpsum_negated, interior, lower, upper);
                tcgen05.store(dp_address + piece * 16, words);
                sum = f32x2.addKeepingSubnormals(sum, f32x2.sumBf16(words));
            }
            return sum;
        }

        fn takePivot(probabilities: *Scores, first_key: i32, pivot_key: *i32) void {
            inline for (0..keys_per_compute_group) |column| {
                if (probabilities[column] > fragments.pivot_probability) {
                    pivot_key.* = first_key + @as(i32, column);
                    probabilities[column] = 0;
                }
            }
        }

        fn shareRowPivot(partial: RowPivot) RowPivot {
            const row = thread.warp() % 4 * 32 + thread.lane();
            tiles.pivot_sums[columnGroup()][row] = partial.sum;
            tiles.pivot_keys[columnGroup()][row] = partial.key;
            nvptx.barrierSyncAligned(NamedBarrier.pivot_exchange, compute_warps * 32);
            return .{
                .sum = tiles.pivot_sums[0][row] + tiles.pivot_sums[1][row],
                .key = @max(tiles.pivot_keys[0][row], tiles.pivot_keys[1][row]),
            };
        }

        fn addPivotTerm(
            params: BackwardParams,
            work: Work,
            values: @Vector(32, f32),
            comptime piece: u32,
            pivot: RowPivot,
        ) @Vector(32, f32) {
            const keys: [*]addrspace(.global) const Chunk = @ptrFromInt(params.k);
            const column = columnGroup() * half_columns + piece * 32;
            const element = work.batch * params.k_batch_stride +
                @as(u64, @intCast(pivot.key)) * params.k_row_stride +
                work.kv_head * params.k_head_stride + column;
            // `pivot.sum` can be an FP32 subnormal number, and `fma.ftz` flushes subnormal inputs
            // to zero (PTX ISA, fma).
            const weight = f32x2.splat(-pivot.sum);
            var result = values;
            inline for (0..4) |chunk| {
                if (column + chunk * 8 < head_dim) {
                    const words = keys[element / 8 + chunk];
                    inline for (0..4) |word| {
                        const first = chunk * 8 + word * 2;
                        const value = f32x2.fmaKeepingSubnormals(
                            weight,
                            f32x2.widenBf16(words[word]),
                            .{ result[first], result[first + 1] },
                        );
                        result[first] = value[0];
                        result[first + 1] = value[1];
                    }
                }
            }
            return result;
        }

        fn storeDq(
            params: BackwardParams,
            args: Args,
            work: Work,
            query: u32,
            accumulated: @Vector(32, f32),
            comptime piece: u32,
            pivot: RowPivot,
        ) void {
            const values = if (pivot.key >= 0)
                addPivotTerm(params, work, accumulated, piece, pivot)
            else
                accumulated;
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(args.dq);
            const element = work.batch * args.dq_batch_stride +
                @as(u64, work.sequence.offset_q + query) * args.dq_row_stride +
                work.head * args.dq_head_stride + columnGroup() * half_columns + piece * 32;
            const scale = f32x2.splat(params.softmax_scale);
            var scaled: @Vector(32, f32) = undefined;
            inline for (0..16) |pair| {
                const value = f32x2.mul(.{ values[2 * pair], values[2 * pair + 1] }, scale);
                scaled[2 * pair] = value[0];
                scaled[2 * pair + 1] = value[1];
            }
            const packed_values = fragments.packBf16(32, scaled);
            inline for (0..4) |chunk| {
                if (columnGroup() * half_columns + piece * 32 + chunk * 8 < head_dim) {
                    out[element / 8 + chunk] = .{
                        packed_values[chunk * 4],     packed_values[chunk * 4 + 1],
                        packed_values[chunk * 4 + 2], packed_values[chunk * 4 + 3],
                    };
                }
            }
        }

        fn epilogueDq(
            params: BackwardParams,
            args: Args,
            work: Work,
            query: u32,
            count: u32,
            pivot: RowPivot,
        ) void {
            const valid = query < work.sequence.seqlen_q;
            if (count > 0) {
                mbarrier.wait(.cta, &dq_full, 0);
                tcgen05.fenceAfterSync();
            }
            inline for (0..dq_pieces) |piece_index| {
                const piece: u32 = piece_index;
                const values = if (count > 0) tcgen05.load(
                    tcgen05.piece_columns,
                    tcgen05.warpAddress(
                        tmem_base,
                        dq_column + columnGroup() * half_columns + piece * tcgen05.piece_columns,
                    ),
                ) else @as(@Vector(tcgen05.piece_columns, f32), @splat(0));
                if (count > 0) tcgen05.waitLoad();
                if (valid) storeDq(params, args, work, query, values, piece, pivot);
            }
        }

        fn computeWorkTile(params: BackwardParams, args: Args, work: Work) void {
            const query = work.first_row +
                cluster.rankIn2Cta() * query_rows + thread.warp() % 4 * 32 + thread.lane();
            const valid = query < work.sequence.seqlen_q;
            const vector = work.batch * params.lse_batch_stride +
                work.head * params.lse_head_stride + work.sequence.offset_q + query;
            const lse: [*]addrspace(.global) const f32 = @ptrFromInt(params.lse_log2);
            const dpsum: [*]addrspace(.global) const f32 = @ptrFromInt(params.dpsum);
            if (tiling.pdl) nvptx.griddepcontrol(.wait);
            const lse_negated = if (valid) lse[vector] else -std.math.inf(f32);
            const dpsum_negated = if (valid) dpsum[vector] else 0;
            const seqlen_k: i32 = @intCast(work.sequence.seqlen_k);
            const col_limit_left =
                if (valid) work.block_info.colLimitLeft(@intCast(query)) else seqlen_k;
            const col_limit_right = work.block_info.colLimitRight(@intCast(query));
            const count = work.n_block_max - work.n_block_min;
            var pivot_sum = f32x2.splat(0);
            var pivot_key: i32 = -1;
            const s_address = tcgen05.warpAddress(tmem_base, s_column + columnGroup() *
                keys_per_compute_group);
            const dp_address = tcgen05.warpAddress(tmem_base, dp_column + columnGroup() *
                keys_per_compute_group);
            var index: u32 = 0;
            while (index < count) : (index += 1) {
                const key0: i32 = @intCast((work.n_block_min + index) * key_rows +
                    columnGroup() * keys_per_compute_group);
                const lower = col_limit_left - key0;
                const upper = col_limit_right - key0;
                const interior =
                    nvptx.voteSync(.all, lower <= 0 and upper >= keys_per_compute_group, .all);
                mbarrier.wait(.cta, &s_full, index);
                tcgen05.fenceAfterSync();
                const scores = tcgen05.loadColumns(keys_per_compute_group, s_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &s_empty);
                var probabilities =
                    f32x2.exponentiate(scores, lse_negated, params.softmax_scale_log2);
                if (!interior) @import("mask.zig").maskR2p(&probabilities, lower, upper, 0);
                if (f32x2.warpHasPivot(probabilities)) takePivot(
                    &probabilities,
                    @intCast(work.sequence.offset_k + @as(u32, @intCast(key0))),
                    &pivot_key,
                );
                mbarrier.wait(.cta, &dp_full, index);
                tcgen05.fenceAfterSync();
                pivot_sum = f32x2.addKeepingSubnormals(
                    pivot_sum,
                    computeDs(probabilities, dp_address, dpsum_negated, interior, lower, upper),
                );
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &ds_ready);
            }
            const pivot = shareRowPivot(.{ .sum = pivot_sum[0] + pivot_sum[1], .key = pivot_key });
            if (valid and columnGroup() == 0) {
                const sums: [*]addrspace(.global) f32 = @ptrFromInt(params.pivot_sum);
                const keys: [*]addrspace(.global) i32 = @ptrFromInt(params.pivot_key);
                sums[vector] = pivot.sum * BackwardParams.pivot_sum_scale;
                keys[vector] = pivot.key;
            }
            epilogueDq(params, args, work, query, count, pivot);
        }

        pub fn runCompute(params: BackwardParams, args: Args) void {
            tcgen05.fenceAfterSync();
            if (Work.decode(params)) |work| computeWorkTile(params, args, work);
        }

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}
