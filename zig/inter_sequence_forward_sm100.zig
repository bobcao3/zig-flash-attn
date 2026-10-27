const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const forward = @import("inter_sequence_forward.zig");
const sm120 = @import("inter_sequence_forward_sm120.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const tile_scheduler = @import("tile_scheduler.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const maskR2p = @import("mask.zig").maskR2p;
const tma = @import("tma.zig");
const tcgen05 = @import("tcgen05.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");
const cluster = @import("cluster.zig");
const mbarrier = @import("mbarrier.zig");
const f32x2 = @import("f32x2.zig");

const Chunk = fragments.Chunk;
const Params = forward.Params;
const Trace = forward.Trace;
const Mbarrier = nvptx.Mbarrier;
const minus_infinity = -std.math.inf(f32);

const NamedBarrier = enum(u4) { correction = 1 };

/// Returns 1 in each element where `a != b` or either is NaN, and 0 elsewhere (PTX ISA,
/// comparison and selection instructions, set).
fn differs(a: f32x2.Pair, b: f32x2.Pair) f32x2.Pair {
    var result: f32x2.Pair = undefined;
    inline for (0..2) |element| result[element] = asm ("set.neu.f32.f32 %[d], %[a], %[b];"
        : [d] "=f" (-> f32),
        : [a] "f" (a[element]),
          [b] "f" (b[element]),
    );
    return result;
}

pub fn InterSequenceFwdSm100(comptime tiling: configurations.Sm100FwdTiling, comptime packs: bool) type {
    const head_dim = tiling.head_dim;
    const padded = tiling.headDimPadded();
    const mma_head_dim = std.mem.alignForward(u32, head_dim, 16);
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.clusterTilerM();
    const block_n = tiling.block_n;
    const m_block_size = tiling.m_block_size;
    const kv_stage = tiling.kv_stage;
    const staging_buffers = tiling.staging_buffers;
    const swizzle_bytes = tiling.swizzleBytes();
    const swizzle = tcgen05.Swizzle.fromBytes(swizzle_bytes);
    const Layout = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const box_columns = tiling.boxColumns();
    const box_chunks = box_columns / 8;
    const boxes = padded / box_columns;
    const q_stage = tiling.q_stage;
    const score_buffers = 2;
    const stage_warps = m_block_size / 32;
    const softmax_warps = q_stage * stage_warps;
    const correction_warps = stage_warps;
    const other_warps = 4;
    const correction_warp = softmax_warps;
    const other_warp = correction_warp + correction_warps;
    const probability_offset = block_n / 2;
    const output_column = score_buffers * block_n;
    const ex2_emulated_pairs = tiling.ex2_emulated_pairs;
    const Cursor = forward.NBlockCursor(block_n);
    const QueryRows = forward.QueryRows(packs);
    const Staging = tma.Layout(box_chunks);
    const Scores = @Vector(block_n, f32);
    const two_cta = tiling.use_2cta_instrs;
    const cluster_ctas = tiling.clusterCtas();
    const release_arrival: mbarrier.Arrival = if (two_cta) .leader_cta else .thread;
    const commit_group: tcgen05.Commit = if (two_cta) .cta_group_2 else .cta_group_1;
    const PersistentScheduler = tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{
        .axis = null,
        .ctas = cluster_ctas,
        .release_scope = .cta,
        .producer_consumes = two_cta,
    });
    const slot_rows = block_n / cluster_ctas;
    const mma_rows = cluster_ctas * m_block_size;
    const p_split_pairs = tiling.p_split_pairs;
    const split_steps = p_split_pairs * 2 / 16;
    const prologue_loads = tiling.prologue_loads;

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = configurations.maxRegistersPerThread(thread_count);
        const num_regs = configurations.Sm100FwdTiling.num_regs_by_q_stage[q_stage - 1];
        pub const init_scope: mbarrier.Scope = if (two_cta) .cluster else .cta;
        pub const initializer_warp: u32 = other_warp + 1;
        pub const tmem_warps: u32 = (2 << other_warp) - 1;
        pub const tmem_allocator_warp: u32 = other_warp;
        pub const tmem_barrier: nvptx.NamedBarrier = @fromBackingInt(2);
        pub const roles = .{
            .{ runSoftmax, softmax_warps, num_regs.softmax },
            .{ runCorrection, correction_warps, num_regs.correction },
            .{ runOther, other_warps, num_regs.other },
        };
        pub const Args = sm120.Args;

        comptime {
            std.debug.assert(!two_cta and !prologue_loads and !tiling.divmod_decode);
            std.debug.assert(tiling.scheduling_mode == .clc);
        }

        /// One work tile: its query rows, the KV head they read and the number of key blocks
        /// that the cursor walks over the visible key runs.
        const Work = struct {
            located: forward.Work,
            rows: QueryRows,
            head: u32,
            kv_head: u32,
            count: u32,

            fn init(params: Params, located: forward.Work) Work {
                const rows: QueryRows = .init(located, block_m);
                var work: Work = .{
                    .located = located,
                    .rows = rows,
                    .head = rows.head,
                    .kv_head = forward.tile_scheduler.qhead_per_kvhead.divide(rows.head),
                    .count = 0,
                };
                var walk = work.cursor(params);
                while (walk.valid()) {
                    work.count += walk.n_block_max - walk.n_block;
                    walk.n_block = walk.n_block_max - 1;
                    walk.advance(params);
                }
                return work;
            }

            fn cursor(work: Work, params: Params) Cursor {
                return .init(params, work.located, work.rows.token(0), work.rows.tokenCount());
            }
        };

        /// The CLC grid holds one CTA per tile up to a bound that the host computes; a tile
        /// index past the tile count that the device computes holds no work.
        const Tiles = struct {
            fn totalBlocks(_: Tiles) u32 {
                return nvptx.nctaid(0) / cluster_ctas;
            }

            fn getCurrentWork(_: Tiles, tile_id: u32) ?forward.Work {
                const scheduler = &forward.tile_scheduler;
                if (tile_id >= scheduler.tileCount(.cta)) return null;
                return scheduler.locate(shared_params, .cta, tile_id);
            }
        };

        const TileStorage = struct {
            q: [q_stage][m_block_size * row_chunks]Chunk,
            kv: [kv_stage][block_n * row_chunks / cluster_ctas]Chunk,
            staging: [staging_buffers][m_block_size * box_chunks]Chunk,
        };

        comptime {
            std.debug.assert(@sizeOf(TileStorage) == shared_bytes);
            std.debug.assert(!two_cta or q_stage == 1);
            std.debug.assert(!two_cta or !prologue_loads);
            std.debug.assert(p_split_pairs == 0 or q_stage == 2 and p_split_pairs * 2 % 16 == 0);
            std.debug.assert(q_stage == 1 or split_steps > 0);
            std.debug.assert(!tiling.fragment_stores or p_split_pairs > 0);
            std.debug.assert(tiling.ex2_emulation_placement != .inner_fragments or block_n == 128);
            std.debug.assert(score_buffers * block_n + q_stage * mma_head_dim <= tcgen05.capacity);
            std.debug.assert(other_warp + other_warps == thread_count / 32);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) TileStorage = @ptrCast(&shared_storage);
        var q_full: [q_stage]Mbarrier addrspace(.shared) = undefined;
        var q_empty: Mbarrier addrspace(.shared) = undefined;
        var kv_full: [kv_stage]Mbarrier addrspace(.shared) = undefined;
        var kv_empty: [kv_stage]Mbarrier addrspace(.shared) = undefined;
        var s_full: [score_buffers]Mbarrier addrspace(.shared) = undefined;
        var o_full: [q_stage]Mbarrier addrspace(.shared) = undefined;
        var p_full_o_rescaled: [score_buffers]Mbarrier addrspace(.shared) = undefined;
        var p_full_2: [score_buffers]Mbarrier addrspace(.shared) = undefined;
        var softmax_corr_full: [score_buffers]Mbarrier addrspace(.shared) = undefined;
        var softmax_corr_empty: [score_buffers]Mbarrier addrspace(.shared) = undefined;
        var tmem_dealloc: Mbarrier addrspace(.shared) = undefined;
        var tmem_base: u32 addrspace(.shared) = undefined;
        var row_scale: [score_buffers][m_block_size]f32 addrspace(.shared) = undefined;
        var row_sum: [score_buffers][m_block_size]f32 addrspace(.shared) = undefined;
        var shared_params: Params addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            if (two_cta) tcgen05.allocate2Sm(
                &tmem_base,
                tcgen05.capacity,
            ) else tcgen05.allocate(&tmem_base, tcgen05.capacity);
        }

        pub fn prepare(params: Params, _: Args) void {
            if (thread.warp() != initializer_warp) return;
            const scheduler: forward.TileScheduler =
                .initForCtas(params, packs, block_m, false, params.num_sm);
            if (thread.lane() == 0) {
                forward.tile_scheduler = scheduler;
                shared_params = params;
            }
        }

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                q_empty.init(1);
                tmem_dealloc.init((q_stage + 1) * m_block_size);
                PersistentScheduler.init(softmax_warps + correction_warps + 2);
                inline for (&q_full, &o_full) |*full, *output| {
                    if (!prologue_loads) full.init(1);
                    output.init(1);
                }
                inline for (0..score_buffers) |buffer_index| {
                    const buffer: u32 = buffer_index;
                    s_full[buffer].init(1);
                    p_full_o_rescaled[buffer].init(comptime mbarrier.arrivals(
                        release_arrival,
                        stage_warps + correction_warps,
                        cluster_ctas,
                    ));
                    softmax_corr_full[buffer].init(m_block_size);
                    p_full_2[buffer].init(m_block_size);
                    softmax_corr_empty[buffer].init(m_block_size);
                }
                inline for (&kv_full, &kv_empty, 0..) |*full, *empty, slot| {
                    if (!prologue_loads or slot > 0) full.init(1);
                    empty.init(1);
                }
                mbarrier.pipelineInitFence();
            }
            elect.reconverge();
        }

        fn rowOffset() u32 {
            return if (two_cta) cluster.rankIn2Cta() * m_block_size else 0;
        }

        fn tileScheduler(_: Params) Tiles {
            return .{};
        }

        fn nextTile(
            scheduler: Tiles,
            persistent_scheduler: *PersistentScheduler,
            tile_id: u32,
            trace: *Trace,
        ) u32 {
            trace.begin(.get_next_work);
            defer trace.end(.get_next_work);
            const clc = nvptx.nctaid(0) == cluster_ctas * scheduler.totalBlocks();
            if (!clc) return tile_id + nvptx.nctaid(0) / cluster_ctas;
            return persistent_scheduler.advanceToNextWork();
        }

        fn scoreBuffer(sequence: u32, stage: u32) u32 {
            return if (q_stage == 2) stage else sequence % score_buffers;
        }

        fn scoreRound(sequence: u32) u32 {
            return if (q_stage == 2) sequence else sequence / score_buffers;
        }

        fn tensorRow() u32 {
            return thread.id() % m_block_size;
        }

        pub fn runOther(params: Params, args: Args) void {
            switch (thread.warp() - other_warp) {
                0 => runMma(params),
                1 => runLoad(params, args),
                2 => runScheduler(params),
                else => {},
            }
        }

        fn runScheduler(params: Params) void {
            const scheduler = tileScheduler(params);
            if (nvptx.nctaid(0) != cluster_ctas * scheduler.totalBlocks()) return;
            if (two_cta and cluster.rankIn2Cta() != 0) return produceTileIds(scheduler, .follower);
            produceTileIds(scheduler, .leader);
        }

        fn produceTileIds(scheduler: Tiles, comptime role: tile_scheduler.CtaRole) void {
            var persistent_scheduler: PersistentScheduler = .create();
            while (true) : (persistent_scheduler.advanceConsumerState()) {
                persistent_scheduler.prefetchNextWorkAs(role);
                persistent_scheduler.consumerWait();
                const tile_id = persistent_scheduler.getCurrentWork();
                if (PersistentScheduler.producer_consumes)
                    persistent_scheduler.consumerRelease();
                if (tile_id >= scheduler.totalBlocks()) break;
            }
            persistent_scheduler.producerTail();
        }

        const Operand = enum { keys, values };

        fn copyQuery(args: Args, work: Work, stage: u32) void {
            const packing = if (packs) forward.tile_scheduler.packing.divisor else 1;
            const token = work.rows.first_query + (work.rows.first_row + stage * m_block_size) / packing;
            _ = q_full[stage].arriveExpectTx(m_block_size * padded * 2);
            Layout.copyTile(&tiles.q[stage], args.q, &q_full[stage], work.head, token, 0);
        }

        fn copyKv(args: Args, comptime operand: Operand, work: Work, kv_index: u32, first_key: u32) void {
            const slot = kv_index % kv_stage;
            mbarrier.waitBackoff(&kv_empty[slot], kv_index / kv_stage ^ 1, tiling.load_backoff_ns);
            const map = if (operand == .keys) args.k else args.v;
            _ = kv_full[slot].arriveExpectTx(block_n * padded * 2);
            Layout.copyTile(&tiles.kv[slot], map, &kv_full[slot], work.kv_head, first_key, 0);
        }

        /// Loads the tiles in the order in which the MMA warp takes them: the first key tile, the
        /// query tiles, and then the other tiles. With two query stages, the value tile of each
        /// key block follows its key tile. With one query stage, each key tile comes before the
        /// value tile of the key block before it.
        fn copyKvTile(params: Params, args: Args, work: Work, first_index: u32, loaded: u32) void {
            var cursor = work.cursor(params);
            var kv_index = first_index;
            var previous_key = cursor.firstKey();
            copyKv(args, .keys, work, kv_index, previous_key);
            kv_index += 1;
            cursor.advance(params);
            mbarrier.waitBackoff(&q_empty, loaded ^ 1, tiling.load_backoff_ns);
            inline for (0..q_stage) |stage| copyQuery(args, work, stage);
            if (q_stage == 2) {
                copyKv(args, .values, work, kv_index, previous_key);
                kv_index += 1;
                while (cursor.valid()) : (cursor.advance(params)) {
                    copyKv(args, .keys, work, kv_index, cursor.firstKey());
                    copyKv(args, .values, work, kv_index + 1, cursor.firstKey());
                    kv_index += 2;
                }
            } else {
                while (cursor.valid()) : (cursor.advance(params)) {
                    copyKv(args, .keys, work, kv_index, cursor.firstKey());
                    copyKv(args, .values, work, kv_index + 1, previous_key);
                    previous_key = cursor.firstKey();
                    kv_index += 2;
                }
                copyKv(args, .values, work, kv_index, previous_key);
            }
        }

        fn runLoad(params: Params, args: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            var persistent_scheduler: PersistentScheduler = .create();
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var loaded: u32 = 0;
            var kv_index: u32 = 0;
            const scheduler = tileScheduler(params);
            while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(
                scheduler,
                &persistent_scheduler,
                tile_id,
                &trace,
            )) {
                const work: Work = .init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                const count = work.count;
                if (count == 0) continue;
                trace.begin(.tile);
                if (elect.oneSync()) copyKvTile(params, args, work, kv_index, loaded);
                elect.reconverge();
                trace.end(.tile);
                kv_index += 2 * count;
                loaded += 1;
            }
        }

        const score_instruction = tcgen05.instructionDescriptor(mma_rows, block_n, .k, .k);
        const value_instruction = tcgen05.instructionDescriptor(mma_rows, mma_head_dim, .k, .mn);

        fn valueOffsets(comptime unit: u32) [block_n / 16]u32 {
            var offsets: [block_n / 16]u32 = undefined;
            for (&offsets, 0..) |*offset, step| offset.* = step * unit;
            return offsets;
        }

        fn mmaScores(tmem: u32, stage: u32, buffer: u32, slot: u32) void {
            const mma = if (two_cta) tcgen05.mmaSsSteps2Sm else tcgen05.mmaSsSteps;
            mma(
                &comptime tcgen05.kMajorOffsets(mma_head_dim / 16, m_block_size, swizzle_bytes),
                &comptime tcgen05.kMajorOffsets(mma_head_dim / 16, slot_rows, swizzle_bytes),
                tmem + buffer * block_n,
                tcgen05.kMajorDescriptor(tcgen05.sharedAddress(&tiles.q[stage]), swizzle_bytes),
                tcgen05.kMajorDescriptor(tcgen05.sharedAddress(&tiles.kv[slot]), swizzle_bytes),
                score_instruction,
                false,
            );
        }

        fn mmaValues(
            tmem: u32,
            stage: u32,
            buffer: u32,
            slot: u32,
            accumulate: bool,
            round: u32,
        ) void {
            const mma = if (two_cta) tcgen05.mmaTsSteps2Sm else tcgen05.mmaTsSteps;
            const a_offsets = comptime valueOffsets(8);
            const b_offsets = comptime valueOffsets(16 * swizzle_bytes >> 4);
            const output = tmem + output_column + stage * mma_head_dim;
            const probabilities = tmem + buffer * block_n + probability_offset;
            const descriptor = tcgen05.sharedDescriptor(
                tcgen05.sharedAddress(&tiles.kv[slot]),
                block_n * swizzle_bytes,
                8 * swizzle_bytes,
                swizzle,
            );
            if (split_steps == 0) {
                mma(
                    &a_offsets,
                    &b_offsets,
                    output,
                    probabilities,
                    descriptor,
                    value_instruction,
                    accumulate,
                );
                return;
            }
            mma(
                a_offsets[0..split_steps],
                b_offsets[0..split_steps],
                output,
                probabilities,
                descriptor,
                value_instruction,
                accumulate,
            );
            mbarrier.wait(.cta, &p_full_2[buffer], round);
            tcgen05.fenceAfterSync();
            mma(
                a_offsets[split_steps..],
                b_offsets[split_steps..],
                output,
                probabilities,
                descriptor,
                value_instruction,
                true,
            );
        }

        const score_steps = mma_head_dim / 16;
        const tail_steps = block_n / 16 - split_steps;

        const StageOperands = struct {
            q: [score_steps]u64,
            k: [score_steps]u64,
            p: [tail_steps]u32,
            v: [tail_steps]u64,
        };

        fn mmaStageOperands(
            tmem: u32,
            comptime stage: u32,
            key_slot: u32,
            value_slot: u32,
        ) StageOperands {
            const q_offsets =
                comptime tcgen05.kMajorOffsets(mma_head_dim / 16, m_block_size, swizzle_bytes);
            const k_offsets =
                comptime tcgen05.kMajorOffsets(mma_head_dim / 16, slot_rows, swizzle_bytes);
            const p_offsets = comptime valueOffsets(8);
            const v_offsets = comptime valueOffsets(16 * swizzle_bytes >> 4);
            const q =
                tcgen05.kMajorDescriptor(tcgen05.sharedAddress(&tiles.q[stage]), swizzle_bytes);
            const k = tcgen05.kMajorDescriptor(
                tcgen05.sharedAddress(&tiles.kv[key_slot]),
                swizzle_bytes,
            );
            const v = tcgen05.sharedDescriptor(
                tcgen05.sharedAddress(&tiles.kv[value_slot]),
                block_n * swizzle_bytes,
                8 * swizzle_bytes,
                swizzle,
            );
            const probabilities = tmem + stage * block_n + probability_offset;
            var operands: StageOperands = undefined;
            inline for (0..score_steps) |step| {
                operands.q[step] = q + q_offsets[step];
                operands.k[step] = k + k_offsets[step];
            }
            inline for (0..tail_steps) |step| {
                operands.p[step] = probabilities + p_offsets[split_steps + step];
                operands.v[step] = v + v_offsets[split_steps + step];
            }
            return operands;
        }

        fn mmaValuesHead(tmem: u32, stage: u32, slot: u32, accumulate: bool) void {
            const p_offsets = comptime valueOffsets(8);
            const v_offsets = comptime valueOffsets(16 * swizzle_bytes >> 4);
            const descriptor = tcgen05.sharedDescriptor(
                tcgen05.sharedAddress(&tiles.kv[slot]),
                block_n * swizzle_bytes,
                8 * swizzle_bytes,
                swizzle,
            );
            const output = tmem + output_column + stage * mma_head_dim;
            const probabilities = tmem + stage * block_n + probability_offset;
            tcgen05.mmaTsSteps(
                p_offsets[0..split_steps],
                v_offsets[0..split_steps],
                output,
                probabilities,
                descriptor,
                value_instruction,
                accumulate,
            );
        }

        fn mmaTileQStage2(
            tmem: u32,
            count: u32,
            finished: u32,
            kv_index: *u32,
            blocks: *u32,
            trace: *Trace,
        ) void {
            var key_slot = kv_index.* % kv_stage;
            trace.begin(.wait_k);
            mbarrier.wait(.cta, &kv_full[key_slot], kv_index.* / kv_stage);
            trace.end(.wait_k);
            inline for (0..2) |stage_index| {
                const stage: u32 = stage_index;
                trace.begin(.load_q);
                mbarrier.wait(.cta, &q_full[stage], finished);
                trace.end(.load_q);
                tcgen05.fenceAfterSync();
                mmaScores(tmem, stage, stage, key_slot);
                tcgen05.commit(commit_group, &s_full[stage]);
            }
            tcgen05.commit(commit_group, &kv_empty[key_slot]);
            kv_index.* += 1;
            var index: u32 = 1;
            while (index < count) : (index += 1) {
                const value_slot = kv_index.* % kv_stage;
                trace.begin(.wait_v);
                mbarrier.wait(.cta, &kv_full[value_slot], kv_index.* / kv_stage);
                trace.end(.wait_v);
                key_slot = (kv_index.* + 1) % kv_stage;
                inline for (0..2) |stage_index| {
                    const stage: u32 = stage_index;
                    const operands = mmaStageOperands(tmem, stage, key_slot, value_slot);
                    trace.begin(.wait_probabilities);
                    mbarrier.wait(.cta, &p_full_o_rescaled[stage], blocks.*);
                    trace.end(.wait_probabilities);
                    tcgen05.fenceAfterSync();
                    mmaValuesHead(tmem, stage, value_slot, index > 1);
                    trace.begin(.wait_probabilities);
                    mbarrier.wait(.cta, &p_full_2[stage], blocks.*);
                    trace.end(.wait_probabilities);
                    tcgen05.fenceAfterSync();
                    if (elect.oneSync()) {
                        const output = tmem + output_column + stage * mma_head_dim;
                        inline for (operands.p, operands.v) |p, v| {
                            tcgen05.mmaTs(output, p, v, value_instruction, true);
                        }
                    }
                    elect.reconverge();
                    if (stage == 0) {
                        trace.begin(.wait_k);
                        mbarrier.wait(.cta, &kv_full[key_slot], (kv_index.* + 1) / kv_stage);
                        trace.end(.wait_k);
                    }
                    if (stage == 1) tcgen05.commit(commit_group, &kv_empty[value_slot]);
                    if (elect.oneSync()) {
                        const score_column = tmem + stage * block_n;
                        inline for (operands.q, operands.k, 0..) |q, k, step| {
                            tcgen05.mmaSs(score_column, q, k, score_instruction, step != 0);
                        }
                    }
                    elect.reconverge();
                    tcgen05.commit(commit_group, &s_full[stage]);
                }
                tcgen05.commit(commit_group, &kv_empty[key_slot]);
                kv_index.* += 2;
                blocks.* += 1;
            }
            tcgen05.commit(commit_group, &q_empty);
        }

        fn issueScores(tmem: u32, buffer: u32, kv_index: *u32, last: bool) void {
            const slot = kv_index.* % kv_stage;
            mbarrier.wait(.cta, &kv_full[slot], kv_index.* / kv_stage);
            mmaScores(tmem, 0, buffer, slot);
            tcgen05.commit(commit_group, &s_full[buffer]);
            tcgen05.commit(commit_group, &kv_empty[slot]);
            if (last) tcgen05.commit(commit_group, &q_empty);
            kv_index.* += 1;
        }

        fn mmaTileQStage1(tmem: u32, count: u32, finished: u32, kv_index: *u32, blocks: *u32) void {
            const first = blocks.*;
            mbarrier.wait(.cta, &q_full[0], finished);
            tcgen05.fenceAfterSync();
            issueScores(tmem, first % score_buffers, kv_index, count == 1);
            if (count > 1) issueScores(tmem, (first + 1) % score_buffers, kv_index, count == 2);
            var index: u32 = 0;
            while (index < count) : (index += 1) {
                const buffer = (first + index) % score_buffers;
                const value_slot = kv_index.* % kv_stage;
                mbarrier.wait(.cta, &kv_full[value_slot], kv_index.* / kv_stage);
                mbarrier.wait(.cta, &p_full_o_rescaled[buffer], (first + index) / score_buffers);
                tcgen05.fenceAfterSync();
                mmaValues(tmem, 0, buffer, value_slot, index > 0, 0);
                tcgen05.commit(commit_group, &o_full[0]);
                tcgen05.commit(commit_group, &kv_empty[value_slot]);
                kv_index.* += 1;
                if (index + 2 < count) issueScores(tmem, buffer, kv_index, index + 3 == count);
            }
            blocks.* += count;
        }

        fn idleUntilLastTile(params: Params) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            var persistent_scheduler: PersistentScheduler = .create();
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            const scheduler = tileScheduler(params);
            while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(
                scheduler,
                &persistent_scheduler,
                tile_id,
                &trace,
            )) {}
        }

        fn decodeTileWork(params: Params, scheduler: Tiles, tile_id: u32) ?Work {
            if (tile_id >= scheduler.totalBlocks()) return null;
            const work: Work = .init(params, scheduler.getCurrentWork(tile_id) orelse return null);
            return if (work.count > 0) work else null;
        }

        fn nextVisibleWork(
            params: Params,
            scheduler: Tiles,
            persistent_scheduler: *PersistentScheduler,
            tile_id: *u32,
            trace: *Trace,
        ) ?Work {
            while (tile_id.* < scheduler.totalBlocks()) : (tile_id.* = nextTile(
                scheduler,
                persistent_scheduler,
                tile_id.*,
                trace,
            )) {
                if (decodeTileWork(params, scheduler, tile_id.*)) |work| return work;
            }
            return null;
        }

        fn runMma(params: Params) void {
            if (two_cta and cluster.rankIn2Cta() != 0) return idleUntilLastTile(params);
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            var persistent_scheduler: PersistentScheduler = .create();
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var finished: u32 = 0;
            var kv_index: u32 = 0;
            var blocks: u32 = 0;
            const scheduler = tileScheduler(params);
            var current = nextVisibleWork(params, scheduler, &persistent_scheduler, &tile_id, &trace);
            while (current) |work| : (finished += 1) {
                const count = work.count;
                trace.begin(.tile);
                defer trace.end(.tile);
                if (q_stage == 2) {
                    mmaTileQStage2(tmem, count, finished, &kv_index, &blocks, &trace);
                } else {
                    mmaTileQStage1(tmem, count, finished, &kv_index, &blocks);
                }
                if (tiling.prefetch_next_work_early) {
                    tile_id = nextTile(scheduler, &persistent_scheduler, tile_id, &trace);
                    current = decodeTileWork(params, scheduler, tile_id);
                }
                if (q_stage == 2) {
                    const value_slot = kv_index % kv_stage;
                    trace.begin(.wait_v);
                    mbarrier.wait(.cta, &kv_full[value_slot], kv_index / kv_stage);
                    trace.end(.wait_v);
                    inline for (0..2) |stage_index| {
                        const stage: u32 = stage_index;
                        trace.begin(.wait_probabilities);
                        mbarrier.wait(.cta, &p_full_o_rescaled[stage], blocks);
                        trace.end(.wait_probabilities);
                        tcgen05.fenceAfterSync();
                        mmaValues(tmem, stage, stage, value_slot, count > 1, blocks);
                        tcgen05.commit(commit_group, &o_full[stage]);
                    }
                    tcgen05.commit(commit_group, &kv_empty[value_slot]);
                    kv_index += 1;
                    blocks += 1;
                }
                if (!tiling.prefetch_next_work_early) {
                    tile_id = nextTile(scheduler, &persistent_scheduler, tile_id, &trace);
                    current = decodeTileWork(params, scheduler, tile_id);
                }
                if (current == null) current =
                    nextVisibleWork(params, scheduler, &persistent_scheduler, &tile_id, &trace);
            }
            if (two_cta) return;
            mbarrier.wait(.cta, &tmem_dealloc, 0);
            tcgen05.deallocate(tmem, tcgen05.capacity);
        }

        const fragment_pairs = 16;
        const frg_cnt = block_n / 2 / fragment_pairs;

        fn exponentiateScores(
            comptime counts_ones: bool,
            scores: Scores,
            softmax_scale_log2: f32,
            max: f32,
            offset: f32,
            address: u32,
            sum: *f32,
            ones: *f32,
            buffer: u32,
        ) Scores {
            const scale = f32x2.splat(softmax_scale_log2);
            const shift = f32x2.splat(-offset);
            var sums: [4]f32x2.Pair = @splat(f32x2.splat(0));
            var kept = f32x2.splat(0);
            var values: Scores = undefined;
            var probabilities: @Vector(block_n / 2, u32) = undefined;
            inline for (0..block_n / 2) |pair| {
                const pair_scores: f32x2.Pair = .{ scores[2 * pair], scores[2 * pair + 1] };
                const exponent = f32x2.fma(pair_scores, scale, shift);
                const value = if (comptime ex2Emulated(pair))
                    f32x2.ex2Emulation(exponent)
                else
                    f32x2.exp2f(exponent);
                values[2 * pair] = value[0];
                values[2 * pair + 1] = value[1];
                if (!tiling.late_row_sum and counts_ones) {
                    const keep = differs(pair_scores, f32x2.splat(max));
                    kept = f32x2.add(kept, keep);
                    sums[pair % 4] = f32x2.fma(value, keep, sums[pair % 4]);
                } else if (!tiling.late_row_sum) {
                    sums[pair % 4] = f32x2.add(sums[pair % 4], value);
                }
                probabilities[pair] = @backingInt(nvptx.Bf16x2.fromF32(value[0], value[1]));
                if (tiling.fragment_stores and (pair + 1) % fragment_pairs == 0) {
                    const first: u32 = pair + 1 - fragment_pairs;
                    var fragment: @Vector(fragment_pairs, u32) = undefined;
                    inline for (0..fragment_pairs) |column| fragment[column] =
                        probabilities[first + column];
                    tcgen05.storeColumns(address + probability_offset + first, fragment);
                    if (first < p_split_pairs and p_split_pairs <= pair + 1) {
                        tcgen05.waitStore();
                        tcgen05.fenceBeforeSync();
                        mbarrier.arrive(release_arrival, &p_full_o_rescaled[buffer]);
                    }
                }
                if (!tiling.fragment_stores and p_split_pairs > 0 and pair + 1 == p_split_pairs) {
                    var head: @Vector(p_split_pairs, u32) = undefined;
                    inline for (0..p_split_pairs) |column| head[column] = probabilities[column];
                    tcgen05.storeColumns(address + probability_offset, head);
                    tcgen05.waitStore();
                    tcgen05.fenceBeforeSync();
                    mbarrier.arrive(release_arrival, &p_full_o_rescaled[buffer]);
                }
            }
            if (tiling.fragment_stores) {
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.thread, &p_full_2[buffer]);
            } else if (p_split_pairs > 0) {
                var tail: @Vector(block_n / 2 - p_split_pairs, u32) = undefined;
                inline for (p_split_pairs..block_n / 2) |column| tail[column - p_split_pairs] =
                    probabilities[column];
                tcgen05.storeColumns(address + probability_offset + p_split_pairs, tail);
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.thread, &p_full_2[buffer]);
            } else {
                tcgen05.storeColumns(address + probability_offset, probabilities);
                tcgen05.waitStore();
                tcgen05.fenceBeforeSync();
            }
            if (!tiling.late_row_sum) {
                sum.* = softmaxTotal(sums);
                ones.* = if (counts_ones) tileOnes(max, kept) else 0;
            }
            return values;
        }

        fn ex2Emulated(comptime pair: u32) bool {
            return switch (tiling.ex2_emulation_placement) {
                .interleaved => pair % 8 < ex2_emulated_pairs,
                .leading => pair < ex2_emulated_pairs * 8,
                .inner_fragments => pair / fragment_pairs != 0 and pair / fragment_pairs !=
                    frg_cnt - 1 and pair % 8 >= 8 - 2 * ex2_emulated_pairs,
            };
        }

        fn tileOnes(max: f32, kept: f32x2.Pair) f32 {
            return if (max == minus_infinity) 0 else block_n - (kept[0] + kept[1]);
        }

        fn softmaxTotal(sums: [4]f32x2.Pair) f32 {
            const pair = f32x2.add(f32x2.add(sums[0], sums[1]), f32x2.add(sums[2], sums[3]));
            return pair[0] + pair[1];
        }

        /// Returns the base 2 log-sum-exp of a row and the sum of the exponentials of its keys
        /// relative to `max * softmax_scale_log2`, given the stored maximum score `max`, the number
        /// `ones` of keys that `runSoftmax` counts as scoring `max` and the sum `sum` of the
        /// exponentials of the other keys. A key that scores `max` has the exponential 2^e, where
        /// e is the exact product `max * softmax_scale_log2` minus its f32 rounding. The
        /// exponentials far below 1 keep their accuracy, except that the columns that use
        /// `f32x2.ex2Emulation` carry its relative error of up to 8.8e-5 and that the `.ftz`
        /// instructions flush values below 2^-126 to zero.
        fn rowTotals(max: f32, ones: f32, sum: f32, softmax_scale_log2: f32) struct { f32, f32 } {
            if (max == minus_infinity) return .{ minus_infinity, 0 };
            const high = max * softmax_scale_log2;
            const low = @mulAdd(f32, max, softmax_scale_log2, -high);
            const top = nvptx.math.ex2ApproxF32(true, low);
            const rest = @mulAdd(f32, ones - 1, top, sum);
            const total = top + rest;
            return .{ high + fragments.log2Sum(low, rest, total), total };
        }

        pub fn runSoftmax(params: Params, _: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            const stage = thread.id() / m_block_size;
            tcgen05.fenceAfterSync();
            const tmem = tmem_base;
            const softmax_scale_log2 = params.softmax_scale_log2;
            var persistent_scheduler: PersistentScheduler = .create();
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var blocks: u32 = 0;
            var softmax_corr_index: u32 = 0;
            const scheduler = tileScheduler(params);
            while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(
                scheduler,
                &persistent_scheduler,
                tile_id,
                &trace,
            )) {
                const work: Work = .init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                const count = work.count;
                if (count == 0) continue;
                trace.begin(.tile);
                defer trace.end(.tile);
                const query: i32 = @intCast(work.rows.token(stage * m_block_size + tensorRow()));
                var cursor = work.cursor(params);
                var run_first_key = cursor.run_first_key;
                var col_limit_left = cursor.block_info.colLimitLeft(query);
                var col_limit_right = cursor.block_info.colLimitRight(query);
                var max: f32 = minus_infinity;
                var sum: f32 = 0;
                var ones: f32 = 0;
                var index: u32 = 0;
                while (index < count) : ({
                    index += 1;
                    cursor.advance(params);
                }) {
                    if (cursor.run_first_key != run_first_key) {
                        run_first_key = cursor.run_first_key;
                        col_limit_left = cursor.block_info.colLimitLeft(query);
                        col_limit_right = cursor.block_info.colLimitRight(query);
                    }
                    const buffer = scoreBuffer(blocks, stage);
                    const scores_address = tcgen05.warpAddress(tmem, buffer * block_n);
                    trace.begin(.wait_scores);
                    mbarrier.wait(.cta, &s_full[buffer], scoreRound(blocks));
                    trace.end(.wait_scores);
                    trace.begin(.softmax);
                    tcgen05.fenceAfterSync();
                    var scores = tcgen05.loadColumns(block_n, scores_address);
                    tcgen05.waitLoad();
                    if (cursor.needsMask()) {
                        const first_column: i32 = @intCast(cursor.n_block * block_n);
                        maskR2p(
                            &scores,
                            col_limit_left - first_column,
                            col_limit_right - first_column,
                            minus_infinity,
                        );
                    }
                    var partial_max = [_]f32{ max, minus_infinity, minus_infinity, minus_infinity };
                    inline for (0..block_n / 2) |pair| {
                        partial_max[pair % 4] = f32x2.max3(
                            partial_max[pair % 4],
                            scores[2 * pair],
                            scores[2 * pair + 1],
                        );
                    }
                    const tile_max = f32x2.max3(
                        partial_max[0],
                        partial_max[1],
                        @max(partial_max[2], partial_max[3]),
                    );
                    var scale: f32 = 1;
                    var sets_max = index == 0;
                    if (index == 0) {
                        max = tile_max;
                    } else {
                        const reference = if (tile_max == minus_infinity) 0 else tile_max;
                        const exponent = (max - reference) * softmax_scale_log2;
                        if (exponent < -fragments.rescale_threshold) {
                            scale = nvptx.math.ex2ApproxF32(true, exponent);
                            max = tile_max;
                            sum = @mulAdd(f32, sum, scale, ones * scale);
                            ones = 0;
                            sets_max = true;
                        }
                        const softmax_corr_buffer = scoreBuffer(softmax_corr_index, stage);
                        row_scale[softmax_corr_buffer][tensorRow()] = scale;
                        mbarrier.arrive(.thread, &softmax_corr_full[softmax_corr_buffer]);
                        softmax_corr_index += 1;
                    }
                    const offset = if (max == minus_infinity) 0 else max * softmax_scale_log2;
                    var tile_sum: f32 = undefined;
                    var tile_ones: f32 = undefined;
                    // A block counts the keys that score `max` only when a row of the warp sets
                    // its `max` in the block, unless the tiling counts them in every block. With a
                    // late row sum, the block counts the keys whose exponential equals that of
                    // `max` instead. The other blocks add the exponential of such a key, about 1,
                    // to `sum`, which is then at least about 1 and no longer needs the terms that
                    // the addition rounds away. In the columns that use `f32x2.ex2Emulation`, that
                    // exponential carries the relative error of the emulation, up to 8.8e-5.
                    const counts_ones = tiling.counts_ties_every_block or
                        nvptx.voteSync(.any, sets_max, .all);
                    const values = if (!tiling.late_row_sum and counts_ones) exponentiateScores(
                        true,
                        scores,
                        softmax_scale_log2,
                        max,
                        offset,
                        scores_address,
                        &tile_sum,
                        &tile_ones,
                        buffer,
                    ) else exponentiateScores(
                        false,
                        scores,
                        softmax_scale_log2,
                        max,
                        offset,
                        scores_address,
                        &tile_sum,
                        &tile_ones,
                        buffer,
                    );
                    if (p_split_pairs == 0) mbarrier.arrive(
                        release_arrival,
                        &p_full_o_rescaled[buffer],
                    );
                    trace.end(.softmax);
                    trace.begin(.wait_correction);
                    mbarrier.wait(
                        .cta,
                        &softmax_corr_empty[scoreBuffer(softmax_corr_index, stage)],
                        scoreRound(softmax_corr_index) ^ 1,
                    );
                    trace.end(.wait_correction);
                    if (tiling.late_row_sum and !counts_ones) {
                        var sums: [4]f32x2.Pair = @splat(f32x2.splat(0));
                        inline for (0..block_n / 2) |pair| {
                            sums[pair % 4] = f32x2.add(
                                sums[pair % 4],
                                .{ values[2 * pair], values[2 * pair + 1] },
                            );
                        }
                        tile_sum = softmaxTotal(sums);
                        tile_ones = 0;
                    } else if (tiling.late_row_sum) {
                        // A key that scores `max` has the exponential `top_exact` or
                        // `top_emulated`, depending on its column.
                        const top_exponent = f32x2.fma(
                            f32x2.splat(max),
                            f32x2.splat(softmax_scale_log2),
                            f32x2.splat(-offset),
                        );
                        const top_exact = f32x2.exp2f(top_exponent);
                        const top_emulated = f32x2.ex2Emulation(top_exponent);
                        var sums: [4]f32x2.Pair = @splat(f32x2.splat(0));
                        var kept = f32x2.splat(0);
                        inline for (0..block_n / 2) |pair| {
                            const value: f32x2.Pair = .{ values[2 * pair], values[2 * pair + 1] };
                            const keep = differs(
                                value,
                                if (comptime ex2Emulated(pair)) top_emulated else top_exact,
                            );
                            kept = f32x2.add(kept, keep);
                            sums[pair % 4] = f32x2.fma(value, keep, sums[pair % 4]);
                        }
                        tile_sum = softmaxTotal(sums);
                        tile_ones = tileOnes(max, kept);
                    }
                    sum += tile_sum;
                    ones += tile_ones;
                    blocks += 1;
                }
                const softmax_corr_buffer = scoreBuffer(softmax_corr_index, stage);
                const lse_log2, const total = rowTotals(max, ones, sum, softmax_scale_log2);
                row_scale[softmax_corr_buffer][tensorRow()] = lse_log2;
                row_sum[softmax_corr_buffer][tensorRow()] = total;
                mbarrier.arrive(.thread, &softmax_corr_full[softmax_corr_buffer]);
                softmax_corr_index += 1;
            }
            tcgen05.fenceBeforeSync();
            mbarrier.arrive(.thread, &tmem_dealloc);
        }

        fn rescale(output: u32, scale: f32) void {
            const width = 16;
            const chunks = mma_head_dim / width;
            const factor = f32x2.splat(scale);
            var values = tcgen05.load(width, output);
            inline for (0..chunks) |chunk_index| {
                const chunk: u32 = chunk_index;
                tcgen05.waitLoad();
                const current = values;
                if (chunk + 1 < chunks) values = tcgen05.load(width, output + (chunk + 1) * width);
                var scaled: @Vector(width, f32) = undefined;
                inline for (0..width / 2) |pair| {
                    const product =
                        f32x2.mul(.{ current[2 * pair], current[2 * pair + 1] }, factor);
                    scaled[2 * pair] = product[0];
                    scaled[2 * pair + 1] = product[1];
                }
                tcgen05.store(output + chunk * width, scaled);
            }
            tcgen05.waitStore();
        }

        fn outputElement(params: Params, work: Work, row: u32) u64 {
            return work.rows.offset(row, params.o_row_stride, params.o_head_stride);
        }

        fn storeLse(params: Params, work: Work, row: u32, value: f32) void {
            const lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
            const token = work.rows.first_query + work.rows.token(row);
            lse[work.rows.headOf(row) * params.lse_head_stride + token] = value;
        }

        fn writeEmptyOutput(params: Params, work: Work) void {
            const row = tensorRow();
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
            const real_chunks = head_dim / 8;
            inline for (0..q_stage) |stage_index| {
                const first: u32 = stage_index * m_block_size;
                if (first < work.rows.count) {
                    const valid_rows = @min(m_block_size, work.rows.count - first);
                    var index = row;
                    while (index < valid_rows * real_chunks) : (index += m_block_size) {
                        const element = outputElement(params, work, first + index / real_chunks);
                        out[element / 8 + index % real_chunks] = @splat(0);
                    }
                    if (row < valid_rows) storeLse(params, work, first + row, minus_infinity);
                }
            }
        }

        fn storeOutput(
            params: Params,
            _: Args,
            work: Work,
            stage: u32,
            output: u32,
            lse_log2: f32,
            sum: f32,
            stores: *u32,
        ) void {
            const row = tensorRow();
            const first = stage * m_block_size;
            if (first >= work.rows.count) return;
            const valid_rows = @min(m_block_size, work.rows.count - first);
            const bulk = false;
            const empty_row = !(sum > 0);
            const inverse: f32 = if (empty_row) 0 else 1 / sum;
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
            const lse = if (empty_row) minus_infinity else lse_log2 * std.math.ln2;
            inline for (0..boxes) |box_index| {
                const box: u32 = box_index;
                const buffer = stores.* % staging_buffers;
                const staging = &tiles.staging[buffer];
                if (thread.warp() == correction_warp and
                    stores.* >= staging_buffers and elect.oneSync())
                {
                    tma.waitBulkReadsPending(staging_buffers - 1);
                }
                elect.reconverge();
                nvptx.barrierSyncAligned(NamedBarrier.correction, m_block_size);
                const box_end = comptime @min(box_columns, mma_head_dim - box_index * box_columns);
                comptime var loaded: u32 = 0;
                inline while (loaded < box_end) {
                    const width = comptime @min(tcgen05.piece_columns, box_end - loaded);
                    const values = tcgen05.load(width, output + box * box_columns + loaded);
                    tcgen05.waitLoad();
                    inline for (0..width / 8) |chunk| {
                        Staging.chunk(staging, row, loaded / 8 + chunk, 0).* =
                            fragments.scaledChunk(values, chunk, inverse);
                    }
                    loaded += width;
                }
                nvptx.fenceProxyAsync(.shared);
                nvptx.barrierSyncAligned(NamedBarrier.correction, m_block_size);
                if (!bulk) {
                    var index = row;
                    while (index < valid_rows * box_chunks) : (index += m_block_size) {
                        const chunk = index % box_chunks;
                        const column = box * box_columns + chunk * 8;
                        if (column < head_dim) {
                            const element =
                                outputElement(params, work, first + index / box_chunks) +
                                column;
                            out[element / 8] =
                                Staging.chunk(staging, index / box_chunks, chunk, 0).*;
                        }
                    }
                }
                if (thread.warp() == correction_warp and elect.oneSync()) {
                    tma.commitBulkGroup();
                }
                elect.reconverge();
                stores.* += 1;
            }
            if (row < valid_rows) storeLse(params, work, first + row, lse);
        }

        fn takeRowScale(buffer: u32, round: u32) f32 {
            mbarrier.wait(.cta, &softmax_corr_full[buffer], round);
            const scale = row_scale[buffer][tensorRow()];
            mbarrier.arrive(.thread, &softmax_corr_empty[buffer]);
            return scale;
        }

        fn rescaleOutputIfNeeded(output: u32, scale: f32) void {
            if (nvptx.voteSync(.any, scale != 1, .all)) {
                tcgen05.fenceAfterSync();
                rescale(output, scale);
            }
            tcgen05.fenceBeforeSync();
        }

        fn correctionEpilogue(
            params: Params,
            args: Args,
            work: Work,
            stage: u32,
            buffer: u32,
            round: u32,
            output: u32,
            stores: *u32,
        ) void {
            mbarrier.wait(.cta, &softmax_corr_full[buffer], round);
            const lse_log2 = row_scale[buffer][tensorRow()];
            const sum = row_sum[buffer][tensorRow()];
            mbarrier.arrive(.thread, &softmax_corr_empty[buffer]);
            tcgen05.fenceAfterSync();
            storeOutput(params, args, work, stage, output, lse_log2, sum, stores);
            tcgen05.fenceBeforeSync();
        }

        pub fn runCorrection(params: Params, args: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            tcgen05.fenceAfterSync();
            const output_address = tcgen05.warpAddress(tmem_base, output_column);
            inline for (
                0..q_stage,
            ) |buffer| mbarrier.arrive(release_arrival, &p_full_o_rescaled[buffer]);
            var persistent_scheduler: PersistentScheduler = .create();
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var softmax_corr_index: u32 = 0;
            var finished: u32 = 0;
            var blocks: u32 = 0;
            var stores: u32 = 0;
            const scheduler = tileScheduler(params);
            while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(
                scheduler,
                &persistent_scheduler,
                tile_id,
                &trace,
            )) {
                const work: Work = .init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                const count = work.count;
                if (count == 0) {
                    writeEmptyOutput(params, work);
                    continue;
                }
                trace.begin(.tile);
                defer trace.end(.tile);
                if (q_stage == 2) {
                    var index: u32 = 1;
                    while (index < count) : (index += 1) {
                        inline for (0..2) |stage_index| {
                            const stage: u32 = stage_index;
                            trace.begin(.wait_row_scale);
                            const scale = takeRowScale(stage, softmax_corr_index);
                            trace.end(.wait_row_scale);
                            trace.begin(.rescale);
                            rescaleOutputIfNeeded(output_address + stage * mma_head_dim, scale);
                            trace.end(.rescale);
                            mbarrier.arrive(release_arrival, &p_full_o_rescaled[stage]);
                        }
                        softmax_corr_index += 1;
                    }
                    inline for (0..2) |stage_index| {
                        const stage: u32 = stage_index;
                        trace.begin(.wait_output);
                        mbarrier.wait(.cta, &o_full[stage], finished);
                        trace.end(.wait_output);
                        trace.begin(.store);
                        defer trace.end(.store);
                        correctionEpilogue(
                            params,
                            args,
                            work,
                            stage,
                            stage,
                            softmax_corr_index,
                            output_address + stage * mma_head_dim,
                            &stores,
                        );
                        mbarrier.arrive(release_arrival, &p_full_o_rescaled[stage]);
                    }
                    softmax_corr_index += 1;
                } else {
                    var index: u32 = 1;
                    while (index < count) : (index += 1) {
                        const scale = takeRowScale(
                            softmax_corr_index % score_buffers,
                            softmax_corr_index / score_buffers,
                        );
                        softmax_corr_index += 1;
                        mbarrier.wait(.cta, &o_full[0], blocks + index - 1);
                        rescaleOutputIfNeeded(output_address, scale);
                        mbarrier.arrive(
                            release_arrival,
                            &p_full_o_rescaled[(blocks + index) % score_buffers],
                        );
                    }
                    mbarrier.wait(.cta, &o_full[0], blocks + count - 1);
                    correctionEpilogue(
                        params,
                        args,
                        work,
                        0,
                        softmax_corr_index % score_buffers,
                        softmax_corr_index / score_buffers,
                        output_address,
                        &stores,
                    );
                    softmax_corr_index += 1;
                    blocks += count;
                    mbarrier.arrive(release_arrival, &p_full_o_rescaled[blocks % score_buffers]);
                }
                finished += 1;
            }
            if (thread.warp() == correction_warp and elect.oneSync()) tma.waitBulkReads();
            elect.reconverge();
            tcgen05.fenceBeforeSync();
            mbarrier.arrive(.thread, &tmem_dealloc);
        }

        pub const Kernel = WarpSpecializedKernel(Params, @This());
    };
}
