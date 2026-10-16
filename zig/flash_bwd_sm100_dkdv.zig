const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const maskR2p = @import("mask.zig").maskR2p;
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

pub fn FlashAttnBwdSm100Dkdv(comptime tiling: configurations.Sm100BwdTiling) type {
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const swizzle_bytes = tiling.swizzleBytes();
    const swizzle = tcgen05.Swizzle.fromBytes(swizzle_bytes);
    const box_columns = tiling.boxColumns();
    const box_chunks = box_columns / 8;
    const boxes = padded / box_columns;
    const key_rows = tiling.block_n;
    const cluster_keys = 2 * key_rows;
    const query_rows = 2 * tiling.block_m;
    const half_queries = query_rows / 2;
    const half_columns = padded / 2;
    const half_boxes = half_columns / box_columns;
    const compute_groups = configurations.Sm100BwdTiling.compute_groups;
    const other_warps = configurations.Sm100BwdTiling.other_warps;
    const num_regs = configurations.Sm100BwdTiling.num_regs_separate_dq;
    const lane_queries = query_rows / 2;
    const queries_per_compute_group = lane_queries / compute_groups;
    const operand_row_chunks = query_rows * 2 / 16;
    const vector_chunks = query_rows / 4;
    const dk_column = 0;
    const dv_column = half_columns;
    const s_column = 2 * half_columns;
    const dp_column = s_column + lane_queries;
    const compute_warps = 4 * compute_groups;
    const empty_warps = 4;
    const other_warp = compute_warps + empty_warps;
    const Scores = @Vector(queries_per_compute_group, f32);
    const TileScheduler = tile_scheduler.TileScheduler(.{
        .block_m = cluster_keys,
        .row_bytes = 4 * head_dim,
        .pack_gqa = true,
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
        };

        const Tiles = struct {
            k: [key_rows * padded / 8]Chunk,
            v: [key_rows * padded / 8]Chunk,
            q: [half_queries * padded / 8]Chunk,
            do: [half_queries * padded / 8]Chunk,
            qt: [query_rows * half_columns / 8]Chunk,
            do_t: [query_rows * half_columns / 8]Chunk,
            p: [key_rows * operand_row_chunks]Chunk,
            ds: [key_rows * operand_row_chunks]Chunk,
            lse_negated: [vector_chunks]Chunk,
            dpsum_negated: [vector_chunks]Chunk,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == tiling.sharedBytes());
            std.debug.assert(vector_chunks == 32);
            std.debug.assert(dp_column + lane_queries <= tcgen05.capacity);
            std.debug.assert(other_warp + other_warps == thread_count / 32);
            for (.{
                "k", "v", "q", "do", "qt", "do_t", "p", "ds",
            }) |field| std.debug.assert(@offsetOf(Tiles, field) % smem.alignment == 0);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var k_full: Mbarrier addrspace(.shared) = undefined;
        var v_full: Mbarrier addrspace(.shared) = undefined;
        var q_full: Mbarrier addrspace(.shared) = undefined;
        var q_empty: Mbarrier addrspace(.shared) = undefined;
        var do_full: Mbarrier addrspace(.shared) = undefined;
        var do_empty: Mbarrier addrspace(.shared) = undefined;
        var qt_full: Mbarrier addrspace(.shared) = undefined;
        var qt_empty: Mbarrier addrspace(.shared) = undefined;
        var qt_tma_full: Mbarrier addrspace(.shared) = undefined;
        var do_t_full: Mbarrier addrspace(.shared) = undefined;
        var do_t_empty: Mbarrier addrspace(.shared) = undefined;
        var do_t_tma_full: Mbarrier addrspace(.shared) = undefined;
        var s_full: Mbarrier addrspace(.shared) = undefined;
        var s_empty: Mbarrier addrspace(.shared) = undefined;
        var dp_full: Mbarrier addrspace(.shared) = undefined;
        var dp_empty: Mbarrier addrspace(.shared) = undefined;
        var p_ready: Mbarrier addrspace(.shared) = undefined;
        var p_empty: Mbarrier addrspace(.shared) = undefined;
        var ds_ready: Mbarrier addrspace(.shared) = undefined;
        var ds_empty: Mbarrier addrspace(.shared) = undefined;
        var dv_full: Mbarrier addrspace(.shared) = undefined;
        var dk_full: Mbarrier addrspace(.shared) = undefined;
        var vectors_full: Mbarrier addrspace(.shared) = undefined;
        var vectors_empty: Mbarrier addrspace(.shared) = undefined;
        var tmem_base: u32 addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            tcgen05.allocate2Sm(&tmem_base, tcgen05.capacity);
        }

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                inline for (.{
                    &k_full,   &v_full,      &q_full,     &q_empty,       &do_full, &do_empty,
                    &qt_empty, &qt_tma_full, &do_t_empty, &do_t_tma_full, &s_full,  &dp_full,
                    &p_empty,  &ds_empty,    &dv_full,    &dk_full,
                }) |barrier| barrier.init(1);
                qt_full.init(2);
                do_t_full.init(2);
                inline for (.{
                    &s_empty, &dp_empty, &p_ready, &ds_ready,
                }) |barrier| barrier.init(2 * compute_warps);
                vectors_full.init(1);
                vectors_empty.init(compute_warps);
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
            kv_head: u32,
            qhead_per_kvhead: u32,
            sequence: SeqlenInfoQK,
            first_key: u32,
            key_count: u32,
            m_block_min: u32,
            m_block_count: u32,
            block_info: BlockInfo,

            const Step = struct {
                head: u32,
                first_row: u32,
                row_count: u32,
            };

            fn decode(params: BackwardParams) ?Work {
                const persistent_scheduler: PersistentScheduler = .create();
                const grid =
                    tile_scheduler.gridTile(.x, 2, persistent_scheduler.initialWorkTileInfo());
                const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
                const tile = scheduler.getGridWork(grid) orelse return null;
                const sequence = params.sequence(tile.batch);
                const cluster_key = tile.block * cluster_keys;
                const info: BlockInfo = .{
                    .first_row = 0,
                    .row_count = sequence.seqlen_q,
                    .seqlen_q = sequence.seqlen_q,
                    .seqlen_k = sequence.seqlen_k,
                    .window_size_left = params.window_size_left,
                    .window_size_right = params.window_size_right,
                };
                const span_keys = @min(cluster_keys, sequence.seqlen_k - cluster_key);
                const m_block_min, const m_block_max =
                    info.mBlockMinMax(cluster_key, span_keys, query_rows);
                const own_key = cluster_key + cluster.rankIn2Cta() * key_rows;
                return .{
                    .batch = tile.batch,
                    .kv_head = tile.head,
                    .qhead_per_kvhead = params.qhead_per_kvhead,
                    .sequence = sequence,
                    .first_key = own_key,
                    .key_count = if (own_key < sequence.seqlen_k) @min(
                        key_rows,
                        sequence.seqlen_k - own_key,
                    ) else 0,
                    .m_block_min = m_block_min,
                    .m_block_count = m_block_max - m_block_min,
                    .block_info = info,
                };
            }
            fn iterations(work: Work) u32 {
                return work.m_block_count * work.qhead_per_kvhead;
            }

            fn step(work: Work, iteration: u32) Step {
                const first_row = (work.m_block_min + iteration % work.m_block_count) * query_rows;
                return .{
                    .head = work.kv_head * work.qhead_per_kvhead + iteration / work.m_block_count,
                    .first_row = first_row,
                    .row_count = @min(query_rows, work.sequence.seqlen_q - first_row),
                };
            }
        };

        pub fn runOther(params: BackwardParams, args: Args) void {
            switch (thread.warp() - other_warp) {
                0 => runMma(params),
                1 => runLoad(params, args),
                2 => runLoadLseDpsum(params),
                else => {},
            }
        }

        pub fn runEmpty(_: BackwardParams, _: Args) void {}

        fn copyColumns(
            params: BackwardParams,
            tile: anytype,
            map: *const tma.TensorMap,
            empty: *addrspace(.shared) Mbarrier,
            tma_full: *addrspace(.shared) Mbarrier,
            full: *addrspace(.shared) Mbarrier,
            index: u32,
            head: u32,
            row: u32,
            batch: u32,
        ) void {
            mbarrier.wait(.cta, empty, index ^ 1);
            if (elect.oneSync()) {
                _ = tma_full.arriveExpectTx(query_rows * half_columns * 2);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = query_rows,
                    .box_rows = tiling.block_m,
                    .count = half_boxes,
                }, tile, map, tma_full, .{ cluster.rankIn2Cta() * half_columns, head, row, batch });
            }
            elect.reconverge();
            mbarrier.wait(.cta, tma_full, index);
            const valid = SeqlenInfoQK.rowsBeforeEnd(
                params.cu_seqlens_q,
                params.num_batch,
                row,
                query_rows,
            );
            if (valid < query_rows) {
                tma.clearBoxRows(box_columns, query_rows, half_boxes, tile, valid);
                nvptx.fenceProxyAsync(.shared);
            }
            mbarrier.arrive(.leader_cta, full);
        }

        fn runLoad(params: BackwardParams, args: Args) void {
            const work = Work.decode(params) orelse return;
            const steps = work.iterations();
            if (steps == 0) return;
            const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
            const key_row = work.sequence.offset_k + work.first_key;
            if (elect.oneSync()) {
                if (cluster.rankIn2Cta() == 0) _ = k_full.arriveExpectTx(2 * key_rows * padded * 2);
                tma.copyBoxes(.{
                    .columns = box_columns,
                    .rows = key_rows,
                    .box_rows = tiling.block_n,
                    .count = boxes,
                }, &tiles.k, args.k, cluster.leaderAddress(
                    &k_full,
                ), .{ 0, work.kv_head, key_row, batch });
            }
            elect.reconverge();
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                const step = work.step(index);
                const row = work.sequence.offset_q + step.first_row;
                const own_row = row + cluster.rankIn2Cta() * half_queries;
                mbarrier.wait(.cta, &q_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        q_full.arriveExpectTx(2 * half_queries * padded * 2);
                    tma.copyBoxes(.{
                        .columns = box_columns,
                        .rows = half_queries,
                        .box_rows = tiling.block_m,
                        .count = boxes,
                    }, &tiles.q, args.q, cluster.leaderAddress(
                        &q_full,
                    ), .{ 0, step.head, own_row, batch });
                    if (index == 0) {
                        if (cluster.rankIn2Cta() == 0) _ =
                            v_full.arriveExpectTx(2 * key_rows * padded * 2);
                        tma.copyBoxes(.{
                            .columns = box_columns,
                            .rows = key_rows,
                            .box_rows = tiling.block_n,
                            .count = boxes,
                        }, &tiles.v, args.v, cluster.leaderAddress(
                            &v_full,
                        ), .{ 0, work.kv_head, key_row, batch });
                    }
                }
                elect.reconverge();
                mbarrier.wait(.cta, &do_empty, index ^ 1);
                if (elect.oneSync()) {
                    if (cluster.rankIn2Cta() == 0) _ =
                        do_full.arriveExpectTx(2 * half_queries * padded * 2);
                    tma.copyBoxes(.{
                        .columns = box_columns,
                        .rows = half_queries,
                        .box_rows = tiling.block_m,
                        .count = boxes,
                    }, &tiles.do, args.do, cluster.leaderAddress(
                        &do_full,
                    ), .{ 0, step.head, own_row, batch });
                }
                elect.reconverge();
                copyColumns(
                    params,
                    &tiles.do_t,
                    args.do,
                    &do_t_empty,
                    &do_t_tma_full,
                    &do_t_full,
                    index,
                    step.head,
                    row,
                    batch,
                );
                copyColumns(
                    params,
                    &tiles.qt,
                    args.q,
                    &qt_empty,
                    &qt_tma_full,
                    &qt_full,
                    index,
                    step.head,
                    row,
                    batch,
                );
            }
        }

        fn runLoadLseDpsum(params: BackwardParams) void {
            const work = Work.decode(params) orelse return;
            if (tiling.pdl) nvptx.griddepcontrol(.wait);
            var index: u32 = 0;
            while (index < work.iterations()) : (index += 1) {
                const step = work.step(index);
                const vector =
                    work.batch * params.lse_batch_stride + step.head * params.lse_head_stride +
                    work.sequence.offset_q + step.first_row;
                const lse_negated = fragments.gather(params.lse_log2, vector, step.row_count);
                const dpsum_negated = fragments.gather(params.dpsum, vector, step.row_count);
                mbarrier.wait(.cta, &vectors_empty, index ^ 1);
                tiles.lse_negated[thread.lane()] = lse_negated;
                tiles.dpsum_negated[thread.lane()] = dpsum_negated;
                mbarrier.arrive(.warp_synced, &vectors_full);
            }
        }

        const gradient_instruction = tcgen05.instructionDescriptor(cluster_keys, padded, .k, .mn);

        fn mmaTransposedScores(tmem: u32, comptime column: u32, a: anytype, b: anytype) void {
            tcgen05.mmaKMajor(2, padded, key_rows, half_queries, tmem + column, a, b);
        }

        fn mmaGradient(
            tmem: u32,
            comptime column: u32,
            a: anytype,
            b: anytype,
            accumulate: bool,
        ) void {
            tcgen05.mmaSsSteps2Sm(
                &comptime tcgen05.kMajorOffsets(query_rows / 16, key_rows, swizzle_bytes),
                &comptime tcgen05.rowOffsets(query_rows / 16, swizzle_bytes),
                tmem + column,
                tcgen05.kMajorDescriptor(tcgen05.sharedAddress(a), swizzle_bytes),
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

        fn mmaDk(tmem: u32, index: u32, last: bool) void {
            mbarrier.wait(.cta, &ds_ready, index);
            mbarrier.wait(.cta, &qt_full, index);
            tcgen05.fenceAfterSync();
            mmaGradient(tmem, dk_column, &tiles.ds, &tiles.qt, index > 0);
            tcgen05.commit(.cta_group_2, &qt_empty);
            tcgen05.commit(.cta_group_2, &ds_empty);
            if (last) tcgen05.commit(.cta_group_2, &dk_full);
        }

        fn runMma(params: BackwardParams) void {
            const work = Work.decode(params) orelse return;
            const steps = work.iterations();
            if (cluster.rankIn2Cta() != 0 or steps == 0) return;
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            mbarrier.wait(.cta, &k_full, 0);
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                mbarrier.wait(.cta, &q_full, index);
                if (index > 0) mbarrier.wait(.cta, &s_empty, index - 1);
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, s_column, &tiles.k, &tiles.q);
                tcgen05.commit(.cta_group_2, &s_full);
                tcgen05.commit(.cta_group_2, &q_empty);
                if (index > 0) mmaDk(tmem, index - 1, false);
                if (index == 0) mbarrier.wait(.cta, &v_full, 0);
                mbarrier.wait(.cta, &do_full, index);
                if (index > 0) mbarrier.wait(.cta, &dp_empty, index - 1);
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, dp_column, &tiles.v, &tiles.do);
                tcgen05.commit(.cta_group_2, &dp_full);
                tcgen05.commit(.cta_group_2, &do_empty);
                mbarrier.wait(.cta, &p_ready, index);
                mbarrier.wait(.cta, &do_t_full, index);
                tcgen05.fenceAfterSync();
                mmaGradient(tmem, dv_column, &tiles.p, &tiles.do_t, index > 0);
                tcgen05.commit(.cta_group_2, &do_t_empty);
                tcgen05.commit(.cta_group_2, &p_empty);
            }
            tcgen05.commit(.cta_group_2, &dv_full);
            mmaDk(tmem, steps - 1, true);
        }

        const tensorRow = tcgen05.datapathRow;

        fn keyRow() u32 {
            return tensorRow() % key_rows;
        }

        fn queryHalf() u32 {
            return tensorRow() / key_rows;
        }

        fn columnGroup() u32 {
            return thread.warp() / 4;
        }

        fn storeMmaOperand(tile: anytype, words: @Vector(queries_per_compute_group / 2, u32)) void {
            const row = keyRow();
            const base = queryHalf() * key_rows * box_chunks + row * box_chunks;
            inline for (0..queries_per_compute_group / 8) |part_index| {
                const part: u32 = part_index;
                const chunk = columnGroup() * queries_per_compute_group / 8 + part;
                tile[base + (chunk ^ row % 8)] = .{
                    words[part * 4],     words[part * 4 + 1],
                    words[part * 4 + 2], words[part * 4 + 3],
                };
            }
        }

        fn rowVectorSlice(
            vectors: anytype,
        ) *addrspace(.shared) const [queries_per_compute_group / 4]Chunk {
            const first =
                (queryHalf() * lane_queries + columnGroup() * queries_per_compute_group) / 4;
            return @ptrCast(&vectors[first]);
        }

        fn computeDs(
            probabilities: Scores,
            dp: Scores,
            dpsum_negated: *addrspace(.shared) const [queries_per_compute_group / 4]Chunk,
        ) Scores {
            var ds: Scores = undefined;
            inline for (0..queries_per_compute_group / 4) |quad| {
                const sum: @Vector(4, f32) = @bitCast(dpsum_negated[quad]);
                inline for (0..2) |pair| {
                    const column = quad * 4 + pair * 2;
                    const value = f32x2.mul(
                        .{ probabilities[column], probabilities[column + 1] },
                        f32x2.add(
                            .{ dp[column], dp[column + 1] },
                            .{ sum[pair * 2], sum[pair * 2 + 1] },
                        ),
                    );
                    ds[column] = value[0];
                    ds[column + 1] = value[1];
                }
            }
            return ds;
        }

        fn gradientElement(params: BackwardParams, work: Work, comptime value_gradient: bool) u64 {
            const row: u64 = work.sequence.offset_k + work.first_key + keyRow();
            return params.gradientElement(
                if (value_gradient) .dv else .dk,
                work.batch,
                work.kv_head,
                row,
            );
        }

        fn storeDkv(
            params: BackwardParams,
            work: Work,
            comptime value_gradient: bool,
            steps: u32,
        ) void {
            const pointer = if (value_gradient) params.dv else params.dk;
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
            const scale: f32 = if (value_gradient) 1 else params.softmax_scale;
            const column: u32 = if (value_gradient) dv_column else dk_column;
            const element =
                gradientElement(params, work, value_gradient) + queryHalf() * half_columns;
            const valid = keyRow() < work.key_count;
            inline for (0..half_columns / tcgen05.piece_columns) |piece_index| {
                const piece: u32 = piece_index;
                const values = if (steps > 0) tcgen05.load(
                    tcgen05.piece_columns,
                    tcgen05.warpAddress(tmem_base, column + piece * tcgen05.piece_columns),
                ) else @as(@Vector(tcgen05.piece_columns, f32), @splat(0));
                if (steps > 0) tcgen05.waitLoad();
                if (valid) {
                    inline for (0..4) |part| {
                        const words = fragments.scaledChunk(values, part, scale);
                        if (queryHalf() * half_columns + piece * 32 +
                            part * 8 < head_dim) out[(element + piece * 32 + part * 8) / 8] =
                            words;
                    }
                }
            }
        }

        fn computeWorkTile(params: BackwardParams, work: Work) void {
            const steps = work.iterations();
            const key: i32 = @intCast(work.first_key + keyRow());
            const seqlen_q: i32 = @intCast(work.sequence.seqlen_q);
            const row_limit_top =
                if (keyRow() < work.key_count) work.block_info.rowLimitTop(key) else seqlen_q;
            const row_limit_bot = work.block_info.rowLimitBot(key);
            const column_offset =
                queryHalf() * lane_queries + columnGroup() * queries_per_compute_group;
            const s_address = tcgen05.warpAddress(tmem_base, s_column + columnGroup() *
                queries_per_compute_group);
            const dp_address = tcgen05.warpAddress(tmem_base, dp_column + columnGroup() *
                queries_per_compute_group);
            var index: u32 = 0;
            while (index < steps) : (index += 1) {
                const step = work.step(index);
                const first: i32 = @intCast(step.first_row + column_offset);
                const lower = row_limit_top - first;
                const upper = row_limit_bot - first;
                const interior = nvptx.voteSync(.all, lower <= 0 and upper >=
                    queries_per_compute_group, .all);
                mbarrier.wait(.cta, &vectors_full, index);
                mbarrier.wait(.cta, &s_full, index);
                tcgen05.fenceAfterSync();
                const scores = tcgen05.loadColumns(queries_per_compute_group, s_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &s_empty);
                const lse_negated = rowVectorSlice(&tiles.lse_negated);
                var probabilities =
                    f32x2.exponentiate(scores, lse_negated, params.softmax_scale_log2);
                if (!interior) maskR2p(&probabilities, lower, upper, 0);
                if (index > 0) mbarrier.wait(.cta, &p_empty, index - 1);
                storeMmaOperand(&tiles.p, fragments.packBf16(
                    queries_per_compute_group,
                    probabilities,
                ));
                nvptx.fenceProxyAsync(.shared);
                mbarrier.arrive(.leader_cta, &p_ready);
                mbarrier.wait(.cta, &dp_full, index);
                tcgen05.fenceAfterSync();
                const dp = tcgen05.loadColumns(queries_per_compute_group, dp_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.leader_cta, &dp_empty);
                var ds = computeDs(probabilities, dp, rowVectorSlice(&tiles.dpsum_negated));
                if (!interior) maskR2p(&ds, lower, upper, 0);
                mbarrier.arrive(.warp_synced, &vectors_empty);
                if (index > 0) mbarrier.wait(.cta, &ds_empty, index - 1);
                storeMmaOperand(&tiles.ds, fragments.packBf16(queries_per_compute_group, ds));
                nvptx.fenceProxyAsync(.shared);
                mbarrier.arrive(.leader_cta, &ds_ready);
            }
            if (columnGroup() == 0) {
                if (steps > 0) mbarrier.wait(.cta, &dv_full, 0);
                tcgen05.fenceAfterSync();
                storeDkv(params, work, true, steps);
            } else {
                if (steps > 0) mbarrier.wait(.cta, &dk_full, 0);
                tcgen05.fenceAfterSync();
                storeDkv(params, work, false, steps);
            }
        }

        pub fn runCompute(params: BackwardParams, _: Args) void {
            tcgen05.fenceAfterSync();
            if (Work.decode(params)) |work| computeWorkTile(params, work);
        }

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}
