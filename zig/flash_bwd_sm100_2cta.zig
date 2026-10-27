const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const tile_scheduler = @import("tile_scheduler.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
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

const Chunk = fragments.Chunk;
const Mbarrier = nvptx.Mbarrier;

pub fn FlashAttnBwdSm100TwoCta(
    comptime tiling: configurations.Sm100BwdTiling,
    comptime head_split: bool,
) type {
    const Sm100BwdTiling = configurations.Sm100BwdTiling;
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const swizzle_bytes = tiling.swizzleBytes();
    const swizzle = tcgen05.Swizzle.fromBytes(swizzle_bytes);
    const box_columns = tiling.boxColumns();
    const box_chunks = box_columns / 8;
    const boxes = padded / box_columns;
    const key_rows = tiling.block_n;
    const cluster_keys = 2 * key_rows;
    const half_queries = tiling.block_m;
    const query_rows = 2 * half_queries;
    const half_columns = padded / 2;
    const half_boxes = half_columns / box_columns;
    const compute_groups = Sm100BwdTiling.compute_groups;
    const queries_per_compute_group = query_rows / compute_groups;
    const group_steps = queries_per_compute_group / 16;
    const pieces = queries_per_compute_group / tcgen05.piece_columns;
    const ds_row_bytes = half_queries * 2;
    const ds_row_chunks = ds_row_bytes / 16;
    const Layout = tma.SwizzledLayout(padded / 8, swizzle_bytes);
    const box_rows = configurations.Sm100BwdTiling.dkv_box_rows;
    const vector_chunks = query_rows / 4;
    const dq_reduce_ncol = Sm100BwdTiling.dq_reduce_ncol;
    const reduce_chunks = dq_reduce_ncol / 4;
    const s_column = 0;
    const dq_column = queries_per_compute_group / 2;
    const dp_column = query_rows;
    const dv_column = 2 * query_rows;
    const dk_column = dv_column + padded;
    const reduce_warps = 4;
    const compute_warps = 4 * compute_groups;
    const other_warp = reduce_warps + compute_warps;
    const Work = @import("n_block_work.zig").NBlockWork(query_rows, cluster_keys, .{});
    const TileScheduler = tile_scheduler.TileScheduler(.{
        .block_m = cluster_keys,
        .row_bytes = 8 * head_dim,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = tiling.l2_swizzle,
    });
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(.single_tile, .{ .axis = .x, .ctas = 2 });
    const Scores = @Vector(queries_per_compute_group, f32);

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        pub const init_scope: mbarrier.Scope = .cluster;
        pub const initializer_warp: u32 = other_warp + 1;
        pub const tmem_warps: u32 = (2 << other_warp) - 1;
        pub const tmem_allocator_warp: u32 = other_warp;
        pub const tmem_barrier: nvptx.NamedBarrier = @fromBackingInt(2);
        pub const roles = .{
            .{ runReduce, reduce_warps, Sm100BwdTiling.num_regs.reduce },
            .{ runCompute, compute_warps, Sm100BwdTiling.num_regs.compute },
            .{ runOther, Sm100BwdTiling.other_warps, Sm100BwdTiling.num_regs.other },
        };
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            do: *const tma.TensorMap,
            dq_accum: *const tma.TensorMap,
            dk: if (tiling.use_tma_store) *const tma.TensorMap else void,
            dv: if (tiling.use_tma_store) *const tma.TensorMap else void,
        };

        const Tiles = struct {
            k: [key_rows * padded / 8]Chunk,
            v: [key_rows * padded / 8]Chunk,
            kt: [cluster_keys * half_columns / 8]Chunk,
            ds: [cluster_keys * ds_row_chunks]Chunk,
            staging: [reduce_warps][32 * reduce_chunks]Chunk,
            exchange: [key_rows * ds_row_chunks]Chunk,
            q: [half_queries * padded / 8]Chunk,
            do: [half_queries * padded / 8]Chunk,
            qt: [query_rows * half_columns / 8]Chunk,
            do_t: [query_rows * half_columns / 8]Chunk,
            lse_negated: [2][vector_chunks]Chunk,
            dpsum_negated: [2][vector_chunks]Chunk,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
            std.debug.assert(dk_column + padded <= tcgen05.capacity);
            std.debug.assert(other_warp + Sm100BwdTiling.other_warps == thread_count / 32);
            std.debug.assert(
                half_boxes == 1 and ds_row_bytes == swizzle_bytes and vector_chunks == 32,
            );
            std.debug.assert(ds_row_bytes == 128 and reduce_warps * 32 == query_rows);
            std.debug.assert(!tiling.use_tma_store or (padded == head_dim and
                @offsetOf(Tiles, "do") == @offsetOf(Tiles, "q") +
                    @sizeOf(@FieldType(Tiles, "q")) and
                compute_warps * box_rows * box_chunks * @sizeOf(
                    Chunk,
                ) <= @sizeOf(@FieldType(Tiles, "q")) + @sizeOf(@FieldType(Tiles, "do"))));
            for (.{
                "k", "v", "kt", "ds", "staging", "exchange", "q", "do", "qt", "do_t",
            }) |field| std.debug.assert(@offsetOf(Tiles, field) % smem.alignment == 0);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var k_full: Mbarrier addrspace(.shared) = undefined;
        var v_full: Mbarrier addrspace(.shared) = undefined;
        var kt_full: Mbarrier addrspace(.shared) = undefined;
        var q_full: Mbarrier addrspace(.shared) = undefined;
        var q_empty: Mbarrier addrspace(.shared) = undefined;
        var do_full: Mbarrier addrspace(.shared) = undefined;
        var do_empty: Mbarrier addrspace(.shared) = undefined;
        var qt_full: Mbarrier addrspace(.shared) = undefined;
        var qt_empty: Mbarrier addrspace(.shared) = undefined;
        var do_t_full: Mbarrier addrspace(.shared) = undefined;
        var do_t_empty: Mbarrier addrspace(.shared) = undefined;
        var s_full: Mbarrier addrspace(.shared) = undefined;
        var s_read: Mbarrier addrspace(.shared) = undefined;
        var p_ready: Mbarrier addrspace(.shared) = undefined;
        var dp_full: Mbarrier addrspace(.shared) = undefined;
        var ds_ready: Mbarrier addrspace(.shared) = undefined;
        var ds_stored: Mbarrier addrspace(.shared) = undefined;
        var ds_empty: Mbarrier addrspace(.shared) = undefined;
        var dq_full: Mbarrier addrspace(.shared) = undefined;
        var dq_empty: Mbarrier addrspace(.shared) = undefined;
        var dv_full: Mbarrier addrspace(.shared) = undefined;
        var dk_full: Mbarrier addrspace(.shared) = undefined;
        var exchange_full: Mbarrier addrspace(.shared) = undefined;
        var exchange_ready: Mbarrier addrspace(.shared) = undefined;
        var vectors_full: [2]Mbarrier addrspace(.shared) = undefined;
        var vectors_empty: [2]Mbarrier addrspace(.shared) = undefined;
        var tmem_base: u32 addrspace(.shared) = undefined;
        var kt_tma_full: Mbarrier addrspace(.shared) = undefined;
        var qt_tma_full: Mbarrier addrspace(.shared) = undefined;
        var do_t_tma_full: Mbarrier addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            tcgen05.allocate2Sm(&tmem_base, tcgen05.capacity);
        }

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                inline for (.{
                    &k_full,   &v_full,  &kt_full,   &q_full,     &q_empty, &do_full, &do_empty,
                    &qt_empty, &qt_full, &do_t_full, &do_t_empty, &s_full,  &dp_full, &dq_full,
                    &dv_full,  &dk_full,
                }) |barrier| barrier.init(1);
                ds_stored.init(compute_warps);
                ds_empty.init(1 + reduce_warps);
                inline for (.{
                    &kt_tma_full, &qt_tma_full, &do_t_tma_full,
                }) |barrier| barrier.init(1);
                inline for (
                    .{ &s_read, &p_ready, &ds_ready },
                ) |barrier| barrier.init(2 * compute_warps);
                dq_empty.init(2 * reduce_warps);
                exchange_full.init(4);
                exchange_ready.init(2);
                inline for (&vectors_full, &vectors_empty) |*full, *empty| {
                    full.init(1);
                    empty.init(compute_warps);
                }
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

        fn decodeWork(params: BackwardParams) ?Work {
            const persistent_scheduler: PersistentScheduler = .create();
            var grid = tile_scheduler.gridTile(.x, 2, persistent_scheduler.initialWorkTileInfo());
            var split: u32 = 0;
            var splits: u32 = 1;
            if (head_split and params.cu_seqlens_k == 0) {
                const key_blocks = (params.seqlen_k + cluster_keys - 1) / cluster_keys;
                if (grid.num_block > key_blocks) {
                    splits = grid.num_block / key_blocks;
                    split = grid.block / key_blocks;
                    grid.block -= split * key_blocks;
                    grid.num_block = key_blocks;
                }
            }
            const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
            return Work.init(params, scheduler.getGridWork(grid) orelse return null, split, splits);
        }

        fn ownKey(work: Work) u32 {
            return work.first_key + cluster.rankIn2Cta() * key_rows;
        }

        fn ownKeys(work: Work) u32 {
            const first = ownKey(work);
            return if (first < work.sequence.seqlen_k) @min(
                key_rows,
                work.sequence.seqlen_k - first,
            ) else 0;
        }

        pub fn runOther(params: BackwardParams, args: Args) void {
            switch (thread.warp() - other_warp) {
                0 => runMma(params),
                1 => runLoad(params, args),
                2 => runLoadLseDpsum(params),
                3 => runExchange(params),
                else => {},
            }
        }

        fn clearRows(tile: anytype, comptime rows: u32, valid: u32) void {
            tma.clearBoxRows(box_columns, rows, 1, tile, valid);
            nvptx.fenceProxyAsync(.shared);
        }

        fn copyColumns(
            tile: anytype,
            map: *const tma.TensorMap,
            comptime rows: u32,
            full: *addrspace(.shared) Mbarrier,
            head: u32,
            row: u32,
            batch: u32,
        ) void {
            if (elect.oneSync()) {
                if (cluster.rankIn2Cta() == 0) _ = full.arriveExpectTx(2 * rows * half_columns * 2);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = rows,
                    .box_rows = rows / 2,
                    .count = half_boxes,
                }, tile, map, cluster.leaderAddress(full), .{
                    cluster.rankIn2Cta() * half_columns, head, row, batch,
                });
            }
            elect.reconverge();
        }

        fn copyClearedColumns(
            tile: anytype,
            map: *const tma.TensorMap,
            comptime rows: u32,
            full: *addrspace(.shared) Mbarrier,
            tma_full: *addrspace(.shared) Mbarrier,
            phase: u32,
            valid: u32,
            head: u32,
            row: u32,
            batch: u32,
        ) void {
            @branchHint(.unlikely);
            const bytes = rows * half_columns * 2;
            if (elect.oneSync()) {
                if (cluster.rankIn2Cta() == 0) _ = full.arriveExpectTx(2 * bytes);
                _ = tma_full.arriveExpectTx(bytes);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = rows,
                    .box_rows = rows / 2,
                    .count = half_boxes,
                }, tile, map, tma_full, .{ cluster.rankIn2Cta() * half_columns, head, row, batch });
            }
            elect.reconverge();
            mbarrier.wait(.cta, tma_full, phase);
            clearRows(tile, rows, valid);
            nvptx.barWarpSync(.all);
            if (elect.oneSync()) cluster.completeTxReleaseCluster(
                cluster.leaderAddress(full),
                bytes,
            );
            elect.reconverge();
        }

        fn runLoad(params: BackwardParams, args: Args) void {
            const work = decodeWork(params) orelse return;
            const steps = work.iterations();
            if (steps == 0) return;
            const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
            const cluster_row = work.sequence.offset_k + work.first_key;
            const key_row = cluster_row + cluster.rankIn2Cta() * key_rows;
            if (elect.oneSync()) {
                if (cluster.rankIn2Cta() == 0) {
                    _ = k_full.arriveExpectTx(2 * key_rows * padded * 2);
                    _ = v_full.arriveExpectTx(2 * key_rows * padded * 2);
                }
                tma.copyBoxes(
                    .{ .columns = box_columns, .rows = key_rows, .count = boxes },
                    &tiles.k,
                    args.k,
                    cluster.leaderAddress(&k_full),
                    .{ 0, work.kv_head, key_row, batch },
                );
            }
            elect.reconverge();
            var cleared: u32 = 0;
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                const step = work.step(index);
                const row = work.sequence.offset_q + step.block_info.first_row;
                const own_row = row + cluster.rankIn2Cta() * half_queries;
                mbarrier.wait(.cta, &q_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        q_full.arriveExpectTx(2 * half_queries * padded * 2);
                    tma.copyBoxes(
                        .{ .columns = box_columns, .rows = half_queries, .count = boxes },
                        &tiles.q,
                        args.q,
                        cluster.leaderAddress(&q_full),
                        .{ 0, step.head, own_row, batch },
                    );
                    if (index == 0) tma.copyBoxes(
                        .{ .columns = box_columns, .rows = key_rows, .count = boxes },
                        &tiles.v,
                        args.v,
                        cluster.leaderAddress(&v_full),
                        .{ 0, work.kv_head, key_row, batch },
                    );
                }
                elect.reconverge();
                if (index == 0) {
                    const valid_keys = SeqlenInfoQK.rowsBeforeEnd(
                        params.cu_seqlens_k,
                        params.num_batch,
                        cluster_row,
                        cluster_keys,
                    );
                    if (valid_keys == cluster_keys)
                        copyColumns(
                            &tiles.kt,
                            args.k,
                            cluster_keys,
                            &kt_full,
                            work.kv_head,
                            cluster_row,
                            batch,
                        )
                    else
                        copyClearedColumns(
                            &tiles.kt,
                            args.k,
                            cluster_keys,
                            &kt_full,
                            &kt_tma_full,
                            0,
                            valid_keys,
                            work.kv_head,
                            cluster_row,
                            batch,
                        );
                }
                mbarrier.wait(.cta, &do_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        do_full.arriveExpectTx(2 * half_queries * padded * 2);
                    tma.copyBoxes(
                        .{ .columns = box_columns, .rows = half_queries, .count = boxes },
                        &tiles.do,
                        args.do,
                        cluster.leaderAddress(&do_full),
                        .{ 0, step.head, own_row, batch },
                    );
                }
                elect.reconverge();
                const valid = SeqlenInfoQK.rowsBeforeEnd(
                    params.cu_seqlens_q,
                    params.num_batch,
                    row,
                    query_rows,
                );
                if (valid == query_rows) {
                    mbarrier.wait(.cta, &do_t_empty, index ^ 1);
                    copyColumns(
                        &tiles.do_t,
                        args.do,
                        query_rows,
                        &do_t_full,
                        step.head,
                        row,
                        batch,
                    );
                    mbarrier.wait(.cta, &qt_empty, index ^ 1);
                    copyColumns(&tiles.qt, args.q, query_rows, &qt_full, step.head, row, batch);
                } else {
                    mbarrier.wait(.cta, &do_t_empty, index ^ 1);
                    copyClearedColumns(
                        &tiles.do_t,
                        args.do,
                        query_rows,
                        &do_t_full,
                        &do_t_tma_full,
                        cleared,
                        valid,
                        step.head,
                        row,
                        batch,
                    );
                    mbarrier.wait(.cta, &qt_empty, index ^ 1);
                    copyClearedColumns(
                        &tiles.qt,
                        args.q,
                        query_rows,
                        &qt_full,
                        &qt_tma_full,
                        cleared,
                        valid,
                        step.head,
                        row,
                        batch,
                    );
                    cleared += 1;
                }
            }
        }

        fn runExchange(params: BackwardParams) void {
            const work = decodeWork(params) orelse return;
            var index: u32 = 0;
            while (index < work.iterations()) : (index += 1) {
                mbarrier.wait(.cta, &exchange_full, index);
                mbarrier.arrive(.leader_cta, &exchange_ready);
            }
        }

        fn rowVector(params: BackwardParams, work: Work, step: Work.Step) u64 {
            return work.batch * params.lse_batch_stride + step.head * params.lse_head_stride +
                work.sequence.offset_q + step.block_info.first_row;
        }

        fn runLoadLseDpsum(params: BackwardParams) void {
            const work = decodeWork(params) orelse return;
            if (tiling.pdl) nvptx.griddepcontrol(.wait);
            var index: u32 = 0;
            while (index < work.iterations()) : (index += 1) {
                const step = work.step(index);
                const slot = index % 2;
                const vector = rowVector(params, work, step);
                const lse_negated =
                    fragments.gather(params.lse_log2, vector, step.block_info.row_count);
                const dpsum_negated =
                    fragments.gather(params.dpsum, vector, step.block_info.row_count);
                mbarrier.wait(.cta, &vectors_empty[slot], index / 2 ^ 1);
                tiles.lse_negated[slot][thread.lane()] = lse_negated;
                tiles.dpsum_negated[slot][thread.lane()] = dpsum_negated;
                mbarrier.arrive(.warp_synced, &vectors_full[slot]);
            }
        }

        const gradient_instruction = tcgen05.instructionDescriptor(cluster_keys, padded, .k, .mn);
        const dq_instruction = tcgen05.instructionDescriptor(query_rows, padded, .mn, .mn);

        fn groupOffsets(group_stride: u32) [query_rows / 16]u32 {
            var offsets: [query_rows / 16]u32 = undefined;
            for (&offsets, 0..) |*offset, step| offset.* =
                step / group_steps * group_stride + step % group_steps * 8;
            return offsets;
        }

        fn mmaTransposedScores(tmem: u32, comptime column: u32, a: anytype, b: anytype) void {
            tcgen05.mmaKMajor(2, padded, key_rows, half_queries, tmem + column, a, b);
        }

        fn mmaGradient(
            tmem: u32,
            comptime column: u32,
            comptime operand: u32,
            comptime offsets: [query_rows / 16]u32,
            b: anytype,
            accumulate: bool,
        ) void {
            tcgen05.mmaTsSteps2Sm(
                &offsets,
                &comptime tcgen05.rowOffsets(query_rows / 16, swizzle_bytes),
                tmem + column,
                tmem + operand,
                tcgen05.sharedDescriptor(
                    tcgen05.sharedAddress(b),
                    query_rows * swizzle_bytes,
                    8 * swizzle_bytes,
                    swizzle,
                ),
                gradient_instruction,
                accumulate,
            );
        }

        fn mmaDv(tmem: u32, index: u32) void {
            mbarrier.wait(.cta, &p_ready, index);
            mbarrier.wait(.cta, &do_t_full, index);
            tcgen05.fenceAfterSync();
            mmaGradient(tmem, dv_column, s_column, comptime groupOffsets(
                query_rows - queries_per_compute_group / 2,
            ), &tiles.do_t, index > 0);
            tcgen05.commit(.cta_group_2, &do_t_empty);
        }

        fn mmaDk(tmem: u32, index: u32) void {
            mbarrier.wait(.cta, &ds_ready, index);
            mbarrier.wait(.cta, &qt_full, index);
            tcgen05.fenceAfterSync();
            mmaGradient(
                tmem,
                dk_column,
                dp_column,
                comptime groupOffsets(queries_per_compute_group),
                &tiles.qt,
                index > 0,
            );
            tcgen05.commit(.cta_group_2, &qt_empty);
        }

        fn mmaDq(tmem: u32, index: u32) void {
            mbarrier.wait(.cta, &exchange_ready, index);
            tcgen05.fenceAfterSync();
            tcgen05.mmaSsSteps2Sm(
                &comptime tcgen05.rowOffsets(cluster_keys / 16, ds_row_bytes),
                &comptime tcgen05.rowOffsets(cluster_keys / 16, swizzle_bytes),
                tmem + dq_column,
                tcgen05.sharedDescriptor(
                    tcgen05.sharedAddress(&tiles.ds),
                    cluster_keys * ds_row_bytes,
                    8 * ds_row_bytes,
                    swizzle,
                ),
                tcgen05.sharedDescriptor(
                    tcgen05.sharedAddress(&tiles.kt),
                    cluster_keys * swizzle_bytes,
                    8 * swizzle_bytes,
                    swizzle,
                ),
                dq_instruction,
                false,
            );
            tcgen05.commit(.cta_group_2, &dq_full);
            tcgen05.commit(.cta_group_2, &ds_empty);
        }

        fn runMma(params: BackwardParams) void {
            const work = decodeWork(params) orelse return;
            const steps = work.iterations();
            if (cluster.rankIn2Cta() != 0 or steps == 0) return;
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            mbarrier.wait(.cta, &k_full, 0);
            mbarrier.wait(.cta, &q_full, 0);
            tcgen05.fenceAfterSync();
            mmaTransposedScores(tmem, s_column, &tiles.k, &tiles.q);
            tcgen05.commit(.cta_group_2, &s_full);
            tcgen05.commit(.cta_group_2, &q_empty);
            mbarrier.wait(.cta, &v_full, 0);
            mbarrier.wait(.cta, &do_full, 0);
            tcgen05.fenceAfterSync();
            mmaTransposedScores(tmem, dp_column, &tiles.v, &tiles.do);
            tcgen05.commit(.cta_group_2, &dp_full);
            tcgen05.commit(.cta_group_2, &do_empty);
            mmaDv(tmem, 0);
            mbarrier.wait(.cta, &kt_full, 0);
            var index: u32 = 0;
            while (index + 1 < steps) : (index += 1) {
                const next = index + 1;
                mbarrier.wait(.cta, &q_full, next);
                if (index > 0) mbarrier.wait(.cta, &dq_empty, index - 1);
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, s_column, &tiles.k, &tiles.q);
                tcgen05.commit(.cta_group_2, &s_full);
                tcgen05.commit(.cta_group_2, &q_empty);
                mmaDk(tmem, index);
                mbarrier.wait(.cta, &do_full, next);
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, dp_column, &tiles.v, &tiles.do);
                tcgen05.commit(.cta_group_2, &dp_full);
                tcgen05.commit(.cta_group_2, &do_empty);
                mbarrier.wait(.cta, &s_read, next);
                mmaDq(tmem, index);
                mmaDv(tmem, next);
            }
            tcgen05.commit(.cta_group_2, &dv_full);
            mmaDk(tmem, steps - 1);
            tcgen05.commit(.cta_group_2, &dk_full);
            mbarrier.wait(.cta, &s_read, steps - 1);
            if (steps > 1) mbarrier.wait(.cta, &dq_empty, steps - 2);
            mmaDq(tmem, steps - 1);
        }

        fn columnGroup() u32 {
            return (thread.warp() - reduce_warps) / 4;
        }

        const keyRow = tcgen05.datapathRow;

        fn computeDs(
            probabilities: Scores,
            dp_address: u32,
            dpsum_negated: *addrspace(.shared) const [queries_per_compute_group / 4]Chunk,
            interior: bool,
            lower: i32,
            upper: i32,
        ) [pieces]@Vector(16, u32) {
            const dp = tcgen05.loadPieces(pieces, dp_address);
            tcgen05.waitLoad();
            var words: [pieces]@Vector(16, u32) = undefined;
            inline for (0..pieces) |piece_index| {
                const piece: u32 = piece_index;
                words[piece] =
                    dsPiece(piece, probabilities, dp, dpsum_negated, interior, lower, upper);
                tcgen05.store(dp_address + piece * 16, words[piece]);
            }
            return words;
        }

        fn excludePivots(
            params: BackwardParams,
            work: Work,
            step: Work.Step,
            probabilities: *Scores,
        ) void {
            const keys: [*]addrspace(.global) i32 = @ptrFromInt(params.pivot_key);
            const first = rowVector(params, work, step) + columnGroup() * queries_per_compute_group;
            const key: i32 = @intCast(work.sequence.offset_k + ownKey(work) + keyRow());
            f32x2.excludePivots(probabilities, keys + first, key);
        }

        fn storeDs(words: [pieces]@Vector(16, u32)) void {
            const row = keyRow();
            const ds_tile: *addrspace(.shared) [key_rows * ds_row_chunks]Chunk =
                if (columnGroup() == cluster.rankIn2Cta()) @ptrCast(&tiles.ds[
                    cluster.rankIn2Cta() * key_rows * ds_row_chunks
                ]) else &tiles.exchange;
            inline for (0..pieces) |piece| {
                inline for (0..4) |part| {
                    const chunk = row * ds_row_chunks + ((piece * 4 + part) ^ row % 8);
                    ds_tile[chunk] = f32x2.quadAt(words[piece], part);
                }
            }
        }

        fn sendDsExchange() void {
            nvptx.barWarpSync(.all);
            if (elect.oneSync()) {
                const bytes = 32 * ds_row_bytes;
                const first =
                    (cluster.rankIn2Cta() * key_rows + thread.warp() % 4 * 32) * ds_row_chunks;
                const barrier = cluster.peerAddress(&exchange_full);
                cluster.arriveExpectTx(barrier, bytes);
                tma.copyPeer(cluster.peerAddress(&tiles.ds[first]), &tiles.exchange[
                    thread.warp() % 4 * 32 * ds_row_chunks
                ], bytes, barrier);
            }
            elect.reconverge();
        }

        fn reduceDkvPartial(
            pointer: u64,
            element: u64,
            column: u32,
            values: @Vector(32, f32),
            scale: f32,
        ) void {
            const out: [*]addrspace(.global) f32 = @ptrFromInt(pointer);
            inline for (0..8) |part| {
                if (column + part * 4 < head_dim) {
                    fragments.reduceAdd(&out[element + column + part * 4], @Vector(4, f32){
                        values[part * 4] * scale,
                        values[part * 4 + 1] * scale,
                        values[part * 4 + 2] * scale,
                        values[part * 4 + 3] * scale,
                    });
                }
            }
        }

        fn accumulatesDkv(params: BackwardParams, work: Work) bool {
            return head_split and work.qhead_per_kvhead != params.qhead_per_kvhead;
        }

        fn gradientElement(params: BackwardParams, work: Work) u64 {
            const row: u64 = work.sequence.offset_k + ownKey(work) + keyRow();
            return if (columnGroup() == 0) params.gradientElement(
                .dv,
                work.batch,
                work.kv_head,
                row,
            ) else params.gradientElement(.dk, work.batch, work.kv_head, row);
        }

        fn dkvStagingRegion() [*]addrspace(.shared) Chunk {
            const base: [*]addrspace(.shared) Chunk = @ptrCast(&tiles.q);
            return base + (thread.warp() - reduce_warps) * box_rows * box_chunks;
        }

        fn storeBox(params: BackwardParams, args: Args, work: Work, column: u32) void {
            nvptx.fenceProxyAsync(.shared);
            nvptx.barWarpSync(.all);
            if (elect.oneSync()) {
                const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
                const row = work.sequence.offset_k + ownKey(work) + thread.warp() % 4 * box_rows;
                tma.storeBox(
                    if (columnGroup() == 0) args.dv else args.dk,
                    &dkvStagingRegion()[0],
                    column,
                    work.kv_head,
                    row,
                    batch,
                );
                tma.commitBulkGroup();
            }
            elect.reconverge();
        }

        fn waitStoreReads() void {
            if (elect.oneSync()) tma.waitBulkReads();
            elect.reconverge();
        }

        fn epilogueDkv(params: BackwardParams, args: Args, work: Work) void {
            mbarrier.wait(.cta, if (columnGroup() == 0) &dv_full else &dk_full, 0);
            tcgen05.fenceAfterSync();
            const values_column: u32 = if (columnGroup() == 0) dv_column else dk_column;
            const pointer = if (columnGroup() == 0) params.dv else params.dk;
            const scale: f32 = if (columnGroup() == 0) 1 else params.softmax_scale;
            const element = gradientElement(params, work);
            const accumulate = accumulatesDkv(params, work);
            const boxed = tiling.use_tma_store and
                !accumulate and thread.warp() % 4 * box_rows + box_rows <= ownKeys(work);
            inline for (0..padded / tcgen05.piece_columns) |chunk_index| {
                const chunk: u32 = chunk_index;
                const piece = tcgen05.load(tcgen05.piece_columns, tcgen05.warpAddress(
                    tmem_base,
                    values_column + chunk * tcgen05.piece_columns,
                ));
                tcgen05.waitLoad();
                if (boxed) {
                    const box = chunk_index * tcgen05.piece_columns / box_columns;
                    const first_chunk = chunk_index * tcgen05.piece_columns % box_columns / 8;
                    if (first_chunk == 0 and box > 0) waitStoreReads();
                    inline for (0..4) |piece_part| {
                        dkvStagingRegion()[
                            Layout.chunkIndex(box_rows, thread.lane(), first_chunk + piece_part)
                        ] = fragments.scaledChunk(piece, piece_part, scale);
                    }
                    if (first_chunk + 4 == box_chunks) storeBox(
                        params,
                        args,
                        work,
                        @as(u32, box * box_columns),
                    );
                } else if (keyRow() < ownKeys(work)) {
                    if (accumulate)
                        reduceDkvPartial(
                            pointer,
                            element,
                            chunk * tcgen05.piece_columns,
                            piece,
                            scale,
                        )
                    else
                        fragments.storeScaledChunks(
                            head_dim,
                            pointer,
                            element,
                            chunk * tcgen05.piece_columns,
                            piece,
                            scale,
                        );
                }
            }
            if (boxed) waitStoreReads();
        }

        fn writeZeroDkv(params: BackwardParams, work: Work) void {
            if (keyRow() >= ownKeys(work) or accumulatesDkv(params, work)) return;
            const pointer = if (columnGroup() == 0) params.dv else params.dk;
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
            const element = gradientElement(params, work);
            inline for (0..head_dim / 8) |chunk| out[element / 8 + chunk] = @splat(0);
        }

        fn computeWorkTile(params: BackwardParams, args: Args, work: Work) void {
            const steps = work.iterations();
            const key_row = keyRow();
            const group = columnGroup();
            const key: i32 = @intCast(ownKey(work) + key_row);
            const seqlen_q: i32 = @intCast(work.sequence.seqlen_q);
            const row_limit_top =
                if (key_row < ownKeys(work)) work.block_info.rowLimitTop(key) else seqlen_q;
            const row_limit_bot = work.block_info.rowLimitBot(key);
            const s_address =
                tcgen05.warpAddress(tmem_base, s_column + group * queries_per_compute_group);
            const p_address = tcgen05.warpAddress(tmem_base, s_column + group *
                (query_rows - queries_per_compute_group / 2));
            const dp_address =
                tcgen05.warpAddress(tmem_base, dp_column + group * queries_per_compute_group);
            const vectors = group * queries_per_compute_group / 4;
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                const step = work.step(index);
                const slot = index % 2;
                const first: i32 =
                    @intCast(step.block_info.first_row + group * queries_per_compute_group);
                const lower = row_limit_top - first;
                const upper = row_limit_bot - first;
                const interior = nvptx.voteSync(.all, lower <= 0 and upper >=
                    queries_per_compute_group, .all);
                mbarrier.wait(.cta, &vectors_full[slot], index / 2);
                mbarrier.wait(.cta, &s_full, index);
                tcgen05.fenceAfterSync();
                const scores = tcgen05.loadColumns(queries_per_compute_group, s_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &s_read);
                const lse_negated =
                    tiles.lse_negated[slot][vectors..][0 .. queries_per_compute_group / 4];
                var probabilities =
                    f32x2.exponentiate(scores, lse_negated, params.softmax_scale_log2);
                if (!interior) @import("mask.zig").maskR2p(&probabilities, lower, upper, 0);
                tcgen05.storeColumns(p_address, fragments.packBf16(
                    queries_per_compute_group,
                    probabilities,
                ));
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &p_ready);
                if (f32x2.warpHasPivot(probabilities)) excludePivots(
                    params,
                    work,
                    step,
                    &probabilities,
                );
                mbarrier.wait(.cta, &dp_full, index);
                tcgen05.fenceAfterSync();
                const dpsum_negated =
                    tiles.dpsum_negated[slot][vectors..][0 .. queries_per_compute_group / 4];
                const words =
                    computeDs(probabilities, dp_address, dpsum_negated, interior, lower, upper);
                mbarrier.arrive(.warp_synced, &vectors_empty[slot]);
                if (index > 0) mbarrier.wait(.cta, &ds_empty, index - 1);
                storeDs(words);
                tcgen05.waitStore();
                nvptx.fenceProxyAsync(.shared);
                tcgen05.fenceBeforeSync();
                if (group != cluster.rankIn2Cta()) sendDsExchange();
                mbarrier.arrive(.warp_synced, &ds_stored);
                mbarrier.arrive(.leader_cta, &ds_ready);
            }
            epilogueDkv(params, args, work);
        }

        pub fn runCompute(params: BackwardParams, args: Args) void {
            tcgen05.fenceAfterSync();
            const work = decodeWork(params) orelse return;
            if (tiling.pdl) nvptx.griddepcontrol(.wait);
            if (work.iterations() == 0) {
                writeZeroDkv(params, work);
                return;
            }
            computeWorkTile(params, args, work);
        }

        fn sumChunk() u32 {
            return thread.warp() % 2 * 4 + thread.lane() % 4;
        }

        fn addPivotSum(params: BackwardParams, work: Work, step: Work.Step) void {
            const group = thread.warp() / 2;
            const ds_tile: *addrspace(.shared) const [key_rows * ds_row_chunks]Chunk =
                if (group == cluster.rankIn2Cta()) @ptrCast(&tiles.ds[
                    cluster.rankIn2Cta() * key_rows * ds_row_chunks
                ]) else &tiles.exchange;
            const sum = f32x2.swizzledColumnSum(ds_tile, sumChunk());
            mbarrier.arrive(.warp_synced, &ds_empty);
            const query = group * queries_per_compute_group + sumChunk() * 8 + thread.lane() / 4;
            if (query < step.block_info.row_count and sum != 0) {
                const sums: [*]addrspace(.global) f32 = @ptrFromInt(params.pivot_sum);
                const row = rowVector(params, work, step) + query;
                const scaled = sum * BackwardParams.pivot_sum_scale;
                nvptx.red(.gpu, .monotonic, .add, &sums[row], scaled);
            }
        }

        pub fn runReduce(params: BackwardParams, args: Args) void {
            tcgen05.fenceAfterSync();
            const work = decodeWork(params) orelse return;
            const steps = work.iterations();
            if (steps == 0) return;
            const row_base = work.sequence.offset_q +
                if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
            const dq_address = tmem_base + tcgen05.address(thread.warp() * 32, dq_column);
            const first_row =
                row_base + cluster.rankIn2Cta() * half_queries + thread.warp() % 2 * 32;
            const half = thread.warp() / 2;
            const staging = &tiles.staging[thread.warp()];
            if (tiling.pdl) nvptx.griddepcontrol(.wait);
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                const step = work.step(index);
                mbarrier.wait(.cta, &ds_stored, index);
                addPivotSum(params, work, step);
                mbarrier.wait(.cta, &dq_full, index);
                tcgen05.fenceAfterSync();
                const values = tcgen05.loadColumns(half_columns, dq_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &dq_empty);
                inline for (0..half_columns / dq_reduce_ncol) |chunk_index| {
                    const chunk: u32 = chunk_index;
                    if (elect.oneSync()) tma.waitBulkReads();
                    elect.reconverge();
                    tma.stageReduceRow(
                        reduce_chunks,
                        staging,
                        thread.lane,
                        &values,
                        chunk * dq_reduce_ncol,
                    );
                    nvptx.fenceProxyAsync(.shared);
                    nvptx.barWarpSync(.all);
                    if (elect.oneSync()) {
                        tma.reduceBox(
                            args.dq_accum,
                            &staging[0],
                            half * half_columns + chunk * dq_reduce_ncol,
                            step.head,
                            first_row + step.block_info.first_row,
                            0,
                        );
                        tma.commitBulkGroup();
                    }
                    elect.reconverge();
                }
            }
            if (elect.oneSync()) tma.waitBulkReads();
            elect.reconverge();
        }

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}
