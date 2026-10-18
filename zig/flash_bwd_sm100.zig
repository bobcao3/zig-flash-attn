const std = @import("std");
const nvptx = std.nvptx;
const config = @import("config");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const BackwardParams = @import("BackwardParams.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const dsPiece = @import("f32x2.zig").dsPiece;
const tma = @import("tma.zig");
const tcgen05 = @import("tcgen05.zig");
const f32x2 = @import("f32x2.zig");
const flash_bwd_preprocess = @import("flash_bwd_preprocess.zig");
const flash_bwd_postprocess = @import("flash_bwd_postprocess.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");
const cluster = @import("cluster.zig");
const mbarrier = @import("mbarrier.zig");
const two_cta_kernel = @import("flash_bwd_sm100_2cta.zig");

const Chunk = fragments.Chunk;
const Mbarrier = nvptx.Mbarrier;

const NamedBarrier = enum(u4) { staging = 1 };

pub fn FlashAttnBwdSm100(
    comptime tiling: configurations.Sm100BwdTiling,
    comptime specialization: configurations.Specialization,
) type {
    const Sm100BwdTiling = configurations.Sm100BwdTiling;
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const q_stage = tiling.q_stage;
    const do_stage = tiling.do_stage;
    const head_dim_parts = tiling.head_dim_parts;
    const part_columns = padded / head_dim_parts;
    const staging_buffers = tiling.staging_buffers;
    const swizzle_bytes = tiling.swizzleBytes();
    const swizzle = tcgen05.Swizzle.fromBytes(swizzle_bytes);
    const Layout = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const box_columns = tiling.boxColumns();
    const part_boxes = part_columns / box_columns;
    const dq_reduce_ncol = Sm100BwdTiling.dq_reduce_ncol;
    const reduce_chunks = dq_reduce_ncol / 4;
    const compute_groups = Sm100BwdTiling.compute_groups;
    const queries_per_compute_group = block_m / compute_groups;
    const group_steps = queries_per_compute_group / 16;
    const pieces = queries_per_compute_group / tcgen05.piece_columns;
    const ds_row_bytes = queries_per_compute_group * 2;
    const ds_row_chunks = ds_row_bytes / 16;
    const vector_chunks = block_m / 4;
    const rows_per_quarter = block_m / 4;
    const s_column = 0;
    const dp_column = block_m;
    const dv_column = 2 * block_m;
    const dk_column = dv_column + part_columns;
    const dq_shares_dp = dk_column + 2 * part_columns > tcgen05.capacity;
    const dq_column = if (dq_shares_dp) dp_column else dk_column + part_columns;
    const reduce_warps = 4;
    const compute_warps = 4 * compute_groups;
    const group_warps = compute_warps / compute_groups;
    const Work = @import("n_block_work.zig").NBlockWork(block_m, block_n, .{});
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
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .y, .ctas = null });
    const persistent = tiling.scheduling_mode != .single_tile;
    const staged_epilogue = tiling.staged_epilogue;
    const use_tma_store = tiling.use_tma_store;
    const box_rows = Sm100BwdTiling.dkv_box_rows;
    const box_chunks = box_columns / 8;
    const store_slots = ds_row_chunks / box_chunks;
    const Count = if (persistent) u32 else u0;
    const Scores = @Vector(queries_per_compute_group, f32);

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        const other_warp = reduce_warps + compute_warps;
        const control_roles = .{ runMma, runLoad, runLoadLseDpsum };
        const claiming_warps: u32 = other_warp + control_roles.len;
        pub const init_scope: mbarrier.Scope = .launch;
        const load_warp = other_warp + 1;
        pub const initializer_warp: u32 = other_warp + control_roles.len;
        pub const tmem_warps: u32 = (2 << other_warp) - 1;
        pub const tmem_allocator_warp: u32 = other_warp;
        pub const tmem_barrier: nvptx.NamedBarrier = @fromBackingInt(2);
        pub const roles = .{
            .{ runReduce, reduce_warps, Sm100BwdTiling.num_regs.reduce },
            .{ runCompute, compute_warps, Sm100BwdTiling.num_regs.compute },
            .{ runOther, Sm100BwdTiling.other_warps, Sm100BwdTiling.num_regs.other },
        };
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(
            scheduler_options.block_m,
            scheduler_options.block_order,
            specialization.sort_batches,
        );
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            do: *const tma.TensorMap,
            dq_accum: *const tma.TensorMap,
            dk: if (use_tma_store) *const tma.TensorMap else void,
            dv: if (use_tma_store) *const tma.TensorMap else void,
        };

        const Tiles = struct {
            k: [block_n * row_chunks]Chunk,
            v: [block_n * row_chunks]Chunk,
            q: [q_stage][block_m * row_chunks]Chunk,
            do: [do_stage][block_m * row_chunks]Chunk,
            ds: [compute_groups][block_n * ds_row_chunks]Chunk,
            staging: [staging_buffers][block_m * reduce_chunks]Chunk,
            lse_negated: [q_stage][vector_chunks]Chunk,
            dpsum_negated: [q_stage][vector_chunks]Chunk,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
            std.debug.assert(dq_column + part_columns <= tcgen05.capacity);
            std.debug.assert(other_warp + 4 == thread_count / 32);
            std.debug.assert(!dq_shares_dp or part_columns <= block_m);
            std.debug.assert(
                !staged_epilogue or (!persistent and head_dim_parts == 1 and block_m == block_n),
            );
            std.debug.assert(!use_tma_store or (!staged_epilogue and
                padded == head_dim and store_slots >= 1 and block_n == group_warps * box_rows));
            std.debug.assert(head_dim_parts > 1 or
                @sizeOf([part_columns / 4][block_n]Chunk) <= @offsetOf(Tiles, "q"));
            std.debug.assert(
                part_columns % box_columns == 0 and part_columns % dq_reduce_ncol == 0,
            );
            for (
                .{ "k", "v", "q", "do", "ds", "staging" },
            ) |field| std.debug.assert(@offsetOf(Tiles, field) % smem.alignment == 0);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var kv_full: Mbarrier addrspace(.shared) = undefined;
        var q_full: [q_stage]Mbarrier addrspace(.shared) = undefined;
        var q_empty: [q_stage]Mbarrier addrspace(.shared) = undefined;
        var vectors_full: [q_stage]Mbarrier addrspace(.shared) = undefined;
        var do_full: [do_stage]Mbarrier addrspace(.shared) = undefined;
        var do_empty: [do_stage]Mbarrier addrspace(.shared) = undefined;
        var s_full: Mbarrier addrspace(.shared) = undefined;
        var p_ready: Mbarrier addrspace(.shared) = undefined;
        var dp_full: Mbarrier addrspace(.shared) = undefined;
        var ds_ready: Mbarrier addrspace(.shared) = undefined;
        var ds_empty: Mbarrier addrspace(.shared) = undefined;
        var dq_full: Mbarrier addrspace(.shared) = undefined;
        var dq_empty: Mbarrier addrspace(.shared) = undefined;
        var dv_full: Mbarrier addrspace(.shared) = undefined;
        var dk_full: Mbarrier addrspace(.shared) = undefined;
        var dkv_empty: Mbarrier addrspace(.shared) = undefined;
        var tmem_dealloc: Mbarrier addrspace(.shared) = undefined;
        var partial_full: Mbarrier addrspace(.shared) = undefined;
        var partial_empty: Mbarrier addrspace(.shared) = undefined;
        var kv_empty: Mbarrier addrspace(.shared) = undefined;
        var tmem_base: u32 addrspace(.shared) = undefined;
        var prepared_work: ?Work addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            tcgen05.allocate(&tmem_base, tcgen05.capacity);
        }

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                if (persistent) kv_full.init(1);
                inline for (
                    &q_full,
                    &q_empty,
                    &vectors_full,
                    0..,
                ) |*full, *empty, *vectors, stage| {
                    if (persistent or stage > 0) full.init(1);
                    empty.init(1);
                    vectors.init(1);
                }
                inline for (&do_full, &do_empty, 0..) |*full, *empty, stage| {
                    if (persistent or stage > 0) full.init(1);
                    empty.init(1);
                }
                inline for (.{
                    &s_full, &dp_full, &ds_empty, &dq_full, &dv_full, &dk_full,
                }) |barrier| barrier.init(1);
                inline for (
                    .{ &p_ready, &ds_ready, &dkv_empty },
                ) |barrier| barrier.init(compute_warps);
                dq_empty.init(reduce_warps);
                tmem_dealloc.init(compute_warps + reduce_warps);
                partial_full.init(4);
                partial_empty.init(4);
                if (persistent) {
                    kv_empty.init(1 + (cluster.count() - 1) * group_warps);
                    PersistentScheduler.init(claiming_warps - 1);
                }
                mbarrier.pipelineInitFence();
            }
            elect.reconverge();
        }

        pub fn prepare(params: BackwardParams, args: Args) void {
            if (persistent or thread.warp() != load_warp) return;
            const work = decodeWork(params, cluster.count(), thread.ctaid());
            if (elect.oneSync()) {
                prepared_work = work;
                kv_full.init(1);
                q_full[0].init(1);
                do_full[0].init(1);
                mbarrier.pipelineInitFence();
                if (work) |tile| {
                    const steps = tile.iterations();
                    if (steps != 0) {
                        copyKeys(args, tile, tensorBatch(params, tile));
                        copyQueries(args, tile, tensorBatch(params, tile), steps, 0, 0, false);
                    }
                }
            }
            elect.reconverge();
        }

        fn currentWork(params: BackwardParams, splits: u32, ctaid: [3]u32) ?Work {
            if (persistent) return decodeWork(params, splits, ctaid);
            return prepared_work;
        }

        fn blockStep(index: u32, steps: u32) u32 {
            return if (head_dim_parts == 1) index else index % steps;
        }

        fn blockPart(index: u32, steps: u32) u32 {
            return if (head_dim_parts == 1) 0 else index / steps;
        }

        fn decodeWork(params: BackwardParams, splits: u32, ctaid: [3]u32) ?Work {
            const grid = tile_scheduler.gridTile(.y, splits, ctaid);
            const scheduler = TileScheduler.create(tile_scheduler.keyArguments(params, grid));
            return Work.init(
                params,
                scheduler.getGridWork(grid) orelse return null,
                grid.split_idx,
                grid.num_qhead_splits,
            );
        }

        pub fn runOther(params: BackwardParams, args: Args) void {
            switch (thread.warp() - other_warp) {
                inline 0...control_roles.len - 1 => |index| control_roles[index](params, args),
                else => {},
            }
        }

        fn tensorBatch(params: BackwardParams, work: Work) u32 {
            return if (params.cu_seqlens_q == 0) work.batch else 0;
        }

        fn copyKeys(args: Args, work: Work, batch: u32) void {
            const key_row = work.sequence.offset_k + work.first_key;
            _ = kv_full.arriveExpectTx(2 * block_n * padded * 2);
            Layout.copyTile(&tiles.k, args.k, &kv_full, work.kv_head, key_row, batch);
            Layout.copyTile(&tiles.v, args.v, &kv_full, work.kv_head, key_row, batch);
        }

        fn copyQueries(
            args: Args,
            work: Work,
            batch: u32,
            steps: u32,
            index: u32,
            block_index: u32,
            comptime waits: bool,
        ) void {
            const step = work.branchedStep(blockStep(index, steps));
            const row = work.sequence.offset_q + step.block_info.first_row;
            const q_slot = block_index % q_stage;
            if (waits) mbarrier.wait(.cta, &q_empty[q_slot], block_index / q_stage ^ 1);
            _ = q_full[q_slot].arriveExpectTx(block_m * padded * 2);
            Layout.copyTile(&tiles.q[q_slot], args.q, &q_full[q_slot], step.head, row, batch);
            const do_slot = block_index % do_stage;
            if (waits) mbarrier.wait(.cta, &do_empty[do_slot], block_index / do_stage ^ 1);
            _ = do_full[do_slot].arriveExpectTx(block_m * padded * 2);
            Layout.copyTile(&tiles.do[do_slot], args.do, &do_full[do_slot], step.head, row, batch);
        }

        fn runLoad(params: BackwardParams, args: Args) void {
            var persistent_scheduler: PersistentScheduler = .create();
            const splits = cluster.count();
            var ctaid = persistent_scheduler.initialWorkTileInfo();
            var loaded: Count = 0;
            var blocks: Count = 0;
            while (true) : ({
                persistent_scheduler.prefetchNextWork();
                ctaid = persistent_scheduler.advanceToNextWork() orelse break;
            }) {
                const work = currentWork(params, splits, ctaid) orelse continue;
                const steps = work.iterations();
                if (steps == 0 or (!persistent and !elect.oneSync())) continue;
                if (!persistent or elect.oneSync()) {
                    const batch = tensorBatch(params, work);
                    if (persistent) {
                        mbarrier.wait(.cta, &kv_empty, loaded ^ 1);
                        copyKeys(args, work, batch);
                    }
                    var index: u32 = @intFromBool(!persistent);
                    while (index < steps * head_dim_parts) : (index += 1) {
                        copyQueries(args, work, batch, steps, index, blocks + index, true);
                    }
                }
                if (persistent) {
                    elect.reconverge();
                    blocks += steps * head_dim_parts;
                    loaded += 1;
                }
            }
            persistent_scheduler.producerTail();
        }

        fn runLoadLseDpsum(params: BackwardParams, _: Args) void {
            if (persistent and tiling.pdl) nvptx.griddepcontrol(.wait);
            var persistent_scheduler: PersistentScheduler = .create();
            const splits = cluster.count();
            var ctaid = persistent_scheduler.initialWorkTileInfo();
            var blocks: Count = 0;
            while (true) : (ctaid = persistent_scheduler.advanceToNextWork() orelse break) {
                const work = currentWork(params, splits, ctaid) orelse continue;
                const steps = work.iterations();
                if (!persistent and tiling.pdl) nvptx.griddepcontrol(.wait);
                var index: u32 = 0;
                while (index < steps * head_dim_parts) : (index += 1) {
                    const step = work.branchedStep(blockStep(index, steps));
                    const block_index = blocks + index;
                    const slot = block_index % q_stage;
                    const vector = work.batch * params.lse_batch_stride +
                        step.head * params.lse_head_stride +
                        work.sequence.offset_q + step.block_info.first_row;
                    const lse_negated =
                        fragments.gather(params.lse_log2, vector, step.block_info.row_count);
                    const dpsum_negated =
                        fragments.gather(params.dpsum, vector, step.block_info.row_count);
                    mbarrier.wait(.cta, &q_empty[slot], block_index / q_stage ^ 1);
                    if (thread.lane() < vector_chunks) {
                        tiles.lse_negated[slot][thread.lane()] = lse_negated;
                        tiles.dpsum_negated[slot][thread.lane()] = dpsum_negated;
                    }
                    mbarrier.arrive(.warp_synced, &vectors_full[slot]);
                }
                if (persistent) blocks += steps * head_dim_parts;
            }
        }

        const kv_instruction = tcgen05.instructionDescriptor(block_n, part_columns, .k, .mn);
        const dq_instruction = tcgen05.instructionDescriptor(block_m, part_columns, .mn, .mn);

        fn operandOffsets() [block_m / 16]u32 {
            var offsets: [block_m / 16]u32 = undefined;
            for (&offsets, 0..) |*offset, step| offset.* =
                step / group_steps * queries_per_compute_group + step % group_steps * 8;
            return offsets;
        }

        fn tilePartAddress(tile: anytype, comptime rows: u32, part: u32) u32 {
            return tcgen05.sharedAddress(tile) + part * part_boxes * rows * swizzle_bytes;
        }

        fn mmaTransposedScores(tmem: u32, comptime column: u32, a: anytype, b: anytype) void {
            tcgen05.mmaKMajor(1, padded, block_n, block_m, tmem + column, a, b);
        }

        fn mmaDkv(
            tmem: u32,
            comptime column: u32,
            comptime operand: u32,
            b_start: u32,
            accumulate: bool,
        ) void {
            tcgen05.mmaTsSteps(
                &comptime operandOffsets(),
                &comptime tcgen05.rowOffsets(block_m / 16, swizzle_bytes),
                tmem + column,
                tmem + operand,
                tcgen05.sharedDescriptor(
                    b_start,
                    block_m * swizzle_bytes,
                    8 * swizzle_bytes,
                    swizzle,
                ),
                kv_instruction,
                accumulate,
            );
        }

        fn mmaDkDq(tmem: u32, blocks: Count, index: u32, steps: u32) void {
            const block_index = blocks + index;
            const q_slot = block_index % q_stage;
            const part = blockPart(index, steps);
            const local = blockStep(index, steps);
            mbarrier.wait(.cta, &ds_ready, block_index);
            if (!dq_shares_dp and block_index > 0) mbarrier.wait(.cta, &dq_empty, block_index - 1);
            tcgen05.fenceAfterSync();
            mmaDkv(
                tmem,
                dk_column,
                dp_column,
                tilePartAddress(&tiles.q[q_slot], block_m, part),
                local > 0,
            );
            tcgen05.commit(.cta_group_1, &q_empty[q_slot]);
            tcgen05.mmaSsSteps(
                &comptime tcgen05.rowOffsets(block_n / 16, ds_row_bytes),
                &comptime tcgen05.rowOffsets(block_n / 16, swizzle_bytes),
                tmem + dq_column,
                tcgen05.sharedDescriptor(
                    tcgen05.sharedAddress(&tiles.ds),
                    block_n * ds_row_bytes,
                    8 * ds_row_bytes,
                    comptime tcgen05.Swizzle.fromBytes(ds_row_bytes),
                ),
                tcgen05.sharedDescriptor(
                    tilePartAddress(&tiles.k, block_n, part),
                    block_n * swizzle_bytes,
                    8 * swizzle_bytes,
                    swizzle,
                ),
                dq_instruction,
                false,
            );
            tcgen05.commit(.cta_group_1, &dq_full);
            tcgen05.commit(.cta_group_1, &ds_empty);
            if (local + 1 == steps) tcgen05.commit(.cta_group_1, &dk_full);
        }

        fn clearRows(tile: anytype, comptime rows: u32, first_cleared: u32) void {
            if (first_cleared >= rows) return;
            Layout.clearRows(tile, first_cleared, 32, thread.lane());
            nvptx.fenceProxyAsync(.shared);
            nvptx.barWarpSync(.all);
        }

        fn mmaWorkTile(
            params: BackwardParams,
            work: Work,
            tmem: u32,
            blocks: Count,
            finished: Count,
        ) void {
            const steps = work.iterations();
            mbarrier.wait(.cta, &kv_full, finished);
            const key_start = work.sequence.offset_k + work.first_key;
            clearRows(&tiles.k, block_n, SeqlenInfoQK.rowsBeforeEnd(
                params.cu_seqlens_k,
                params.num_batch,
                key_start,
                block_n,
            ));
            const total = steps * head_dim_parts;
            var index: u32 = 0;
            while (index < total) : (index += 1) {
                const step = work.branchedStep(blockStep(index, steps));
                const row_start = work.sequence.offset_q + step.block_info.first_row;
                const valid_rows = SeqlenInfoQK.rowsBeforeEnd(
                    params.cu_seqlens_q,
                    params.num_batch,
                    row_start,
                    block_m,
                );
                const block_index = blocks + index;
                const q_slot = block_index % q_stage;
                const do_slot = block_index % do_stage;
                const part = blockPart(index, steps);
                const accumulation = finished * head_dim_parts + part;
                if (q_stage == 1 and index > 0) mmaDkDq(tmem, blocks, index - 1, steps);
                mbarrier.wait(.cta, &q_full[q_slot], block_index / q_stage);
                clearRows(&tiles.q[q_slot], block_m, valid_rows);
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, s_column, &tiles.k, &tiles.q[q_slot]);
                tcgen05.commit(.cta_group_1, &s_full);
                if (q_stage > 1 and index > 0) mmaDkDq(tmem, blocks, index - 1, steps);
                mbarrier.wait(.cta, &do_full[do_slot], block_index / do_stage);
                clearRows(&tiles.do[do_slot], block_m, valid_rows);
                if (dq_shares_dp and block_index > 0) mbarrier.wait(
                    .cta,
                    &dq_empty,
                    block_index - 1,
                );
                tcgen05.fenceAfterSync();
                mmaTransposedScores(tmem, dp_column, &tiles.v, &tiles.do[do_slot]);
                tcgen05.commit(.cta_group_1, &dp_full);
                mbarrier.wait(.cta, &p_ready, block_index);
                if (blockStep(index, steps) == 0 and
                    accumulation > 0) mbarrier.wait(.cta, &dkv_empty, accumulation - 1);
                tcgen05.fenceAfterSync();
                mmaDkv(
                    tmem,
                    dv_column,
                    s_column,
                    tilePartAddress(&tiles.do[do_slot], block_m, part),
                    blockStep(index, steps) > 0,
                );
                tcgen05.commit(.cta_group_1, &do_empty[do_slot]);
                if (blockStep(index, steps) + 1 == steps) tcgen05.commit(.cta_group_1, &dv_full);
            }
            mmaDkDq(tmem, blocks, total - 1, steps);
            if (persistent) tcgen05.commit(.cta_group_1, &kv_empty);
        }

        fn runMma(params: BackwardParams, _: Args) void {
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            var persistent_scheduler: PersistentScheduler = .create();
            const splits = cluster.count();
            var ctaid = persistent_scheduler.initialWorkTileInfo();
            var finished: Count = 0;
            var blocks: Count = 0;
            while (true) : (ctaid = persistent_scheduler.advanceToNextWork() orelse break) {
                const work = currentWork(params, splits, ctaid) orelse continue;
                const steps = work.iterations();
                if (steps == 0) continue;
                mmaWorkTile(params, work, tmem, blocks, finished);
                if (persistent) {
                    blocks += steps * head_dim_parts;
                    finished += 1;
                }
            }
            mbarrier.wait(.cta, &tmem_dealloc, 0);
            tcgen05.deallocate(tmem, tcgen05.capacity);
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
        ) void {
            const dp = tcgen05.loadPieces(pieces, dp_address);
            tcgen05.waitLoad();
            const row = keyRow();
            const row_swizzle = switch (ds_row_bytes) {
                128 => row % 8,
                64 => (row >> 1) % 4,
                else => @compileError("unsupported dS row width"),
            };
            const ds_tile = &tiles.ds[columnGroup()];
            inline for (0..pieces) |piece_index| {
                const piece: u32 = piece_index;
                const words =
                    dsPiece(piece, probabilities, dp, dpsum_negated, interior, lower, upper);
                tcgen05.store(dp_address + piece * 16, words);
                inline for (0..4) |part_index| {
                    const part: u32 = part_index;
                    ds_tile[row * ds_row_chunks + ((piece * 4 + part) ^ row_swizzle)] =
                        f32x2.quadAt(words, part);
                }
            }
        }

        fn dkvStagingRows() *addrspace(.shared) [32 * row_chunks]Chunk {
            const tile = if (columnGroup() == 0) &tiles.do[0] else &tiles.q[0];
            return tile[thread.warp() % 4 * 32 * row_chunks ..][0 .. 32 * row_chunks];
        }

        fn storeStagedDkv(params: BackwardParams, work: Work, pointer: u64) void {
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
            const first_row = thread.warp() % 4 * 32;
            const element = gradientElement(params, work, first_row);
            const row_stride: u64 =
                if (columnGroup() == 0) params.dv_row_stride else params.dk_row_stride;
            inline for (0..row_chunks) |pass| {
                const index = pass * 32 + thread.lane();
                const row = index / row_chunks;
                const column = index % row_chunks;
                if (column * 8 < head_dim and first_row + row < work.key_count) {
                    out[(element + row * row_stride) / 8 + column] =
                        dkvStagingRows()[column * 32 + (row ^ column)];
                }
            }
        }

        fn gradientElement(params: BackwardParams, work: Work, key: u32) u64 {
            const row: u64 = work.sequence.offset_k + work.first_key + key;
            return if (columnGroup() == 0) params.gradientElement(
                .dv,
                work.batch,
                work.kv_head,
                row,
            ) else params.gradientElement(.dk, work.batch, work.kv_head, row);
        }

        fn dkvPartialBuffer() *addrspace(.shared) [part_columns / 4][block_n]Chunk {
            return @ptrCast(&tiles.k);
        }

        fn addPeerPartial(piece: @Vector(32, f32), remote: u32, chunk: u32) @Vector(32, f32) {
            var sum = piece;
            inline for (0..8) |quad_index| {
                const quad: u32 = quad_index;
                const slot = (chunk * 8 + quad) * block_n;
                const partial = cluster.load(remote + (slot + keyRow()) * @sizeOf(Chunk));
                inline for (0..2) |pair| {
                    const value = f32x2.add(
                        .{ sum[quad * 4 + pair * 2], sum[quad * 4 + pair * 2 + 1] },
                        .{ partial[pair * 2], partial[pair * 2 + 1] },
                    );
                    sum[quad * 4 + pair * 2] = value[0];
                    sum[quad * 4 + pair * 2 + 1] = value[1];
                }
            }
            return sum;
        }

        fn dkvStagingRegion(comptime slot: u32) [*]addrspace(.shared) Chunk {
            const base: [*]addrspace(.shared) Chunk = @ptrCast(&tiles.ds[columnGroup()]);
            return base + thread.warp() % group_warps * box_rows * ds_row_chunks +
                slot * box_rows * box_chunks;
        }

        fn waitStoreReads(comptime pending: u32) void {
            if (elect.oneSync()) tma.waitBulkReadsPending(pending);
            elect.reconverge();
        }

        fn storeBox(
            params: BackwardParams,
            args: Args,
            work: Work,
            comptime slot: u32,
            column: u32,
        ) void {
            nvptx.fenceProxyAsync(.shared);
            nvptx.barWarpSync(.all);
            if (elect.oneSync()) {
                const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
                const row = work.sequence.offset_k +
                    work.first_key + thread.warp() % group_warps * box_rows;
                tma.storeBox(
                    if (columnGroup() == 0) args.dv else args.dk,
                    &dkvStagingRegion(slot)[0],
                    column,
                    work.kv_head,
                    row,
                    batch,
                );
                tma.commitBulkGroup();
            }
            elect.reconverge();
        }

        fn epilogueDkv(
            params: BackwardParams,
            args: Args,
            work: Work,
            part: u32,
            accumulation: u32,
        ) void {
            mbarrier.wait(.cta, if (columnGroup() == 0) &dv_full else &dk_full, accumulation);
            const exchange = cluster.count() > 1;
            const sending = exchange and columnGroup() != cluster.rankIn2Cta();
            if (sending or (use_tma_store and columnGroup() == 0)) mbarrier.wait(
                .cta,
                &dk_full,
                accumulation,
            );
            if (exchange and !sending) mbarrier.wait(
                .cluster,
                &partial_full,
                if (persistent) accumulation else 0,
            );
            tcgen05.fenceAfterSync();
            const values_column: u32 = if (columnGroup() == 0) dv_column else dk_column;
            const pointer = if (columnGroup() == 0) params.dv else params.dk;
            const scale: f32 = if (columnGroup() == 0) 1 else params.softmax_scale;
            const element = gradientElement(params, work, keyRow());
            const remote = if (exchange) cluster.peerAddress(dkvPartialBuffer()) else 0;
            const staged = staged_epilogue and !exchange;
            const boxed = use_tma_store and !sending and
                thread.warp() % group_warps * box_rows + box_rows <= work.key_count;
            inline for (0..part_columns / tcgen05.piece_columns) |chunk_index| {
                const chunk: u32 = chunk_index;
                var piece = tcgen05.load(tcgen05.piece_columns, tcgen05.warpAddress(
                    tmem_base,
                    values_column + chunk * tcgen05.piece_columns,
                ));
                tcgen05.waitLoad();
                if (sending) {
                    inline for (0..8) |quad| {
                        dkvPartialBuffer()[chunk * 8 + quad][keyRow()] =
                            @bitCast(f32x2.quadAt(piece, quad));
                    }
                } else {
                    if (exchange) piece = addPeerPartial(piece, remote, chunk);
                    const column = part * part_columns + chunk * tcgen05.piece_columns;
                    if (staged) {
                        inline for (0..4) |piece_part| {
                            const index = column / 8 + piece_part;
                            if (index * 8 < head_dim)
                                dkvStagingRows()[index * 32 + (thread.lane() ^ index)] =
                                    fragments.scaledChunk(piece, piece_part, scale);
                        }
                    } else if (boxed) {
                        const box = chunk_index * tcgen05.piece_columns / box_columns;
                        const slot = box % store_slots;
                        const first_chunk = chunk_index * tcgen05.piece_columns % box_columns / 8;
                        if (first_chunk == 0 and box >= store_slots) waitStoreReads(
                            store_slots - 1,
                        );
                        inline for (0..4) |piece_part| {
                            dkvStagingRegion(slot)[
                                Layout.chunkIndex(box_rows, thread.lane(), first_chunk + piece_part)
                            ] = fragments.scaledChunk(piece, piece_part, scale);
                        }
                        if (first_chunk + 4 == box_chunks) storeBox(
                            params,
                            args,
                            work,
                            slot,
                            part * part_columns + @as(u32, box * box_columns),
                        );
                    } else if (keyRow() < work.key_count) {
                        fragments.storeScaledChunks(
                            head_dim,
                            pointer,
                            element,
                            column,
                            piece,
                            scale,
                        );
                    }
                }
            }
            if (staged) {
                nvptx.barWarpSync(.all);
                storeStagedDkv(params, work, pointer);
            }
            if (boxed) waitStoreReads(0);
            if (exchange) {
                mbarrier.arrive(
                    .peer_release_cluster,
                    if (sending) &partial_full else &partial_empty,
                );
                if (sending) {
                    if (persistent) {
                        mbarrier.wait(.cluster, &partial_empty, accumulation);
                        if (part + 1 == head_dim_parts) {
                            nvptx.fenceProxyAsync(.shared);
                            mbarrier.arrive(.warp_synced, &kv_empty);
                        }
                    } else {
                        mbarrier.wait(.cta, &partial_empty, 0);
                    }
                }
            }
            if (persistent or part + 1 < head_dim_parts) {
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.warp_synced, &dkv_empty);
            }
        }

        fn writeZeroDkv(params: BackwardParams, work: Work) void {
            if (keyRow() >= work.key_count) return;
            const pointer = if (columnGroup() == 0) params.dv else params.dk;
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(pointer);
            const element = gradientElement(params, work, keyRow());
            inline for (0..head_dim / 8) |chunk| out[element / 8 + chunk] = @splat(0);
        }

        fn computeWorkTile(
            params: BackwardParams,
            args: Args,
            work: Work,
            blocks: Count,
            finished: Count,
        ) void {
            const steps = work.iterations();
            const key_row = keyRow();
            const group = columnGroup();
            const key: i32 = @intCast(work.first_key + key_row);
            const seqlen_q: i32 = @intCast(work.sequence.seqlen_q);
            const row_limit_top = if (key_row < work.key_count)
                work.block_info.rowLimitTop(key)
            else
                seqlen_q;
            const row_limit_bot = work.block_info.rowLimitBot(key);
            const s_address =
                tcgen05.warpAddress(tmem_base, s_column + group * queries_per_compute_group);
            const dp_address =
                tcgen05.warpAddress(tmem_base, dp_column + group * queries_per_compute_group);
            const vectors = group * queries_per_compute_group / 4;
            var index: u32 = 0;
            while (index < steps * head_dim_parts) : (index += 1) {
                const step = work.branchedStep(blockStep(index, steps));
                const block_index = blocks + index;
                const q_slot = block_index % q_stage;
                const first: i32 =
                    @intCast(step.block_info.first_row + group * queries_per_compute_group);
                const lower = row_limit_top - first;
                const upper = row_limit_bot - first;
                const interior = nvptx.voteSync(.all, lower <= 0 and upper >=
                    queries_per_compute_group, .all);
                mbarrier.wait(.cta, &vectors_full[q_slot], block_index / q_stage);
                mbarrier.wait(.cta, &s_full, block_index);
                tcgen05.fenceAfterSync();
                const scores = tcgen05.loadColumns(queries_per_compute_group, s_address);
                tcgen05.waitLoad();
                const lse_negated =
                    tiles.lse_negated[q_slot][vectors..][0 .. queries_per_compute_group / 4];
                var probabilities =
                    f32x2.exponentiate(scores, lse_negated, params.softmax_scale_log2);
                if (!interior) @import("mask.zig").maskR2p(&probabilities, lower, upper, 0);
                tcgen05.storeColumns(s_address, fragments.packBf16(
                    queries_per_compute_group,
                    probabilities,
                ));
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.warp_synced, &p_ready);
                mbarrier.wait(.cta, &dp_full, block_index);
                if (block_index > 0) mbarrier.wait(.cta, &ds_empty, block_index - 1);
                tcgen05.fenceAfterSync();
                const dpsum_negated =
                    tiles.dpsum_negated[q_slot][vectors..][0 .. queries_per_compute_group / 4];
                computeDs(probabilities, dp_address, dpsum_negated, interior, lower, upper);
                tcgen05.waitStore();
                nvptx.fenceProxyAsync(.shared);
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.warp_synced, &ds_ready);
                if (blockStep(index, steps) == steps - 1) {
                    const part = blockPart(index, steps);
                    epilogueDkv(params, args, work, part, finished * head_dim_parts + part);
                }
            }
        }

        pub fn runCompute(params: BackwardParams, args: Args) void {
            tcgen05.fenceAfterSync();
            var persistent_scheduler: PersistentScheduler = .create();
            const splits = cluster.count();
            var ctaid = persistent_scheduler.initialWorkTileInfo();
            var finished: Count = 0;
            var blocks: Count = 0;
            while (true) : (ctaid = persistent_scheduler.advanceToNextWork() orelse break) {
                const work = currentWork(params, splits, ctaid) orelse continue;
                const steps = work.iterations();
                if (steps == 0) {
                    writeZeroDkv(params, work);
                    continue;
                }
                computeWorkTile(params, args, work, blocks, finished);
                if (persistent) {
                    blocks += steps * head_dim_parts;
                    finished += 1;
                }
            }
            tcgen05.fenceBeforeSync();
            mbarrier.arrive(.warp_synced, &tmem_dealloc);
        }

        fn dqStagingRow() u32 {
            return thread.warp() * rows_per_quarter + thread.lane();
        }

        fn waitDqReduceReads() void {
            if (thread.warp() == 0 and elect.oneSync()) tma.waitBulkReads();
            elect.reconverge();
        }

        fn reduceWorkTile(
            params: BackwardParams,
            args: Args,
            work: Work,
            blocks: Count,
            carried_stores: *Count,
        ) void {
            const steps = work.iterations();
            const row_base = work.sequence.offset_q +
                if (params.cu_seqlens_q == 0) work.batch * params.seqlen_q else 0;
            const dq_address = tmem_base + tcgen05.address(thread.warp() * 32, dq_column);
            if (!persistent and tiling.pdl) nvptx.griddepcontrol(.wait);
            var stores: u32 = carried_stores.*;
            var index: u32 = 0;
            while (index < steps * head_dim_parts) : (index += 1) {
                const step = work.branchedStep(blockStep(index, steps));
                const part = blockPart(index, steps);
                mbarrier.wait(.cta, &dq_full, blocks + index);
                tcgen05.fenceAfterSync();
                const values = tcgen05.loadColumns(part_columns, dq_address);
                tcgen05.waitLoad();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.warp_synced, &dq_empty);
                inline for (0..part_columns / dq_reduce_ncol) |chunk_index| {
                    const chunk: u32 = chunk_index;
                    const buffer = stores % staging_buffers;
                    if (stores >= staging_buffers and thread.warp() == 0 and elect.oneSync()) {
                        tma.waitBulkReadsPending(staging_buffers - 1);
                    }
                    elect.reconverge();
                    nvptx.barrierSyncAligned(NamedBarrier.staging, reduce_warps * 32);
                    if (thread.lane() < rows_per_quarter) tma.stageReduceRow(
                        reduce_chunks,
                        &tiles.staging[buffer],
                        dqStagingRow,
                        &values,
                        chunk * dq_reduce_ncol,
                    );
                    nvptx.fenceProxyAsync(.shared);
                    nvptx.barrierSyncAligned(NamedBarrier.staging, reduce_warps * 32);
                    if (thread.warp() == 0 and elect.oneSync()) {
                        tma.reduceBox(
                            args.dq_accum,
                            &tiles.staging[buffer][0],
                            part * part_columns + chunk * dq_reduce_ncol,
                            step.head,
                            row_base + step.block_info.first_row,
                            0,
                        );
                        tma.commitBulkGroup();
                    }
                    elect.reconverge();
                    stores += 1;
                }
            }
            if (persistent) carried_stores.* = stores else waitDqReduceReads();
        }

        pub fn runReduce(params: BackwardParams, args: Args) void {
            tcgen05.fenceAfterSync();
            if (persistent and tiling.pdl) nvptx.griddepcontrol(.wait);
            var persistent_scheduler: PersistentScheduler = .create();
            const splits = cluster.count();
            var ctaid = persistent_scheduler.initialWorkTileInfo();
            var blocks: Count = 0;
            var stores: Count = 0;
            while (true) : (ctaid = persistent_scheduler.advanceToNextWork() orelse break) {
                const work = currentWork(params, splits, ctaid) orelse continue;
                reduceWorkTile(params, args, work, blocks, &stores);
                if (persistent) blocks += work.iterations() * head_dim_parts;
            }
            if (persistent) waitDqReduceReads();
            tcgen05.fenceBeforeSync();
            mbarrier.arrive(.warp_synced, &tmem_dealloc);
        }

        pub const Kernel = WarpSpecializedKernel(BackwardParams, @This());
    };
}

const variant = configurations.variants(.flash_bwd_sm100, config.head_dim)[config.variant];
const backward_tiling = variant.tiling;
const Backward = if (backward_tiling.use_2cta_instrs) two_cta_kernel.FlashAttnBwdSm100TwoCta(
    backward_tiling,
    variant.specialization.head_split,
) else FlashAttnBwdSm100(backward_tiling, variant.specialization);

fn launch(params: BackwardParams, args: Backward.Args) void {
    if (backward_tiling.pdl) nvptx.griddepcontrol(.launch_dependents);
    if (variant.specialization.scheduler_metadata) nvptx.griddepcontrol(.wait);
    Backward.Kernel.run(params, args);
    if (backward_tiling.use_2cta_instrs) Backward.finish();
}

const Entry = entry.Entry(BackwardParams, Backward.Args, .{
    "q_map", "k_map", "v_map", "do_map", "dq_accum_map", "lse_log2", "dpsum", "dk", "dv",
} ++ entry.backward_sequence ++ entry.strides("dk") ++ entry.strides("dv") ++
    .{ "lse_batch_stride", "lse_head_stride", "num_batch" } ++
    (if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}) ++
    if (backward_tiling.use_tma_store) .{ "dk_map", "dv_map" } else .{}, .{
    .required_work_group_size = .{ .x = Backward.thread_count },
    .max_registers_per_work_item = Backward.max_registers,
    .cluster_size = if (backward_tiling.use_2cta_instrs) .{ .x = 2 } else null,
}, launch);

fn SplitEntry(comptime Split: type, comptime gradients: anytype, comptime strides: anytype) type {
    const Launch = struct {
        fn launch(params: BackwardParams, args: Split.Args) void {
            if (backward_tiling.pdl) nvptx.griddepcontrol(.launch_dependents);
            Split.Kernel.run(params, args);
            Split.finish();
        }
    };
    return entry.Entry(BackwardParams, Split.Args, .{
        "q_map", "k_map", "v_map", "do_map", "lse_log2", "dpsum",
    } ++ gradients ++ entry.backward_sequence ++ strides ++
        .{ "lse_batch_stride", "lse_head_stride", "num_batch" }, .{
        .required_work_group_size = .{ .x = Split.thread_count },
        .max_registers_per_work_item = Split.max_registers,
        .cluster_size = .{ .x = 2 },
    }, Launch.launch);
}

const EntryDkdv = SplitEntry(
    @import("flash_bwd_sm100_dkdv.zig").FlashAttnBwdSm100Dkdv(backward_tiling),
    .{ "dq_accum", "dk", "dv" },
    entry.strides("dk") ++ entry.strides("dv") ++ entry.strides("dq_accum"),
);
const EntryDq = SplitEntry(
    @import("flash_bwd_sm100_dq.zig").FlashAttnBwdSm100Dq(backward_tiling),
    .{"dq"},
    entry.strides("dq"),
);

comptime {
    if (@import("builtin").cpu.arch.isNvptx() and @import("root") == @This()) {
        const auxiliary: flash_bwd_preprocess.Options = .{
            .clear_dq_accum = backward_tiling.dq_block_m == 0,
            .pdl = backward_tiling.pdl,
            .varlen_as_one_sequence = variant.specialization.scheduler_metadata,
        };
        @export(
            &flash_bwd_preprocess.FlashAttnBwdPreprocess(auxiliary).flashAttnBwdPreprocess,
            .{ .name = "flash_attn_bwd_preprocess" },
        );
        if (backward_tiling.dq_block_m != 0) {
            @export(&EntryDkdv.entry, .{ .name = "flash_attn_bwd" });
            @export(&EntryDq.entry, .{ .name = "flash_attn_bwd_dq" });
        } else {
            @export(
                &flash_bwd_postprocess.FlashAttnBwdPostprocessConvertDq(auxiliary)
                    .flashAttnBwdPostprocess,
                .{ .name = "flash_attn_bwd_postprocess" },
            );
            if (variant.specialization.head_split) @export(
                &flash_bwd_preprocess.flashAttnBwdClear,
                .{ .name = "flash_attn_bwd_clear" },
            );
            @export(&Entry.entry, .{ .name = "flash_attn_bwd" });
        }
        if (variant.specialization.scheduler_metadata) {
            if (backward_tiling.dq_block_m != 0) @compileError(
                "the backward with a separate query gradient kernel has no block table",
            );
            @export(&Backward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_bwd_prepare_varlen_num_blocks",
            });
        }
    }
}
