const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const configurations = @import("configurations.zig");
const ForwardParams = @import("ForwardParams.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const MBlockWork = @import("m_block_work.zig").MBlockWork;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const grid_dependency = @import("grid_dependency.zig");
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
const Mbarrier = nvptx.Mbarrier;
const minus_infinity = -std.math.inf(f32);

const NamedBarrier = enum(u4) { correction = 1 };

pub fn FlashAttnFwdSm100(comptime tiling: configurations.Sm100FwdTiling, comptime specialization: configurations.Specialization) type {
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
    const Work = MBlockWork(block_m, block_n, .{ .pack_gqa = specialization.pack_gqa });
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_m,
        .row_bytes = 4 * padded,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const Staging = tma.Layout(box_chunks);
    const Scores = @Vector(block_n, f32);
    const two_cta = tiling.use_2cta_instrs;
    const cluster_ctas = tiling.clusterCtas();
    const release_arrival: mbarrier.Arrival = if (two_cta) .leader_cta else .thread;
    const commit_group: tcgen05.Commit = if (two_cta) .cta_group_2 else .cta_group_1;
    const PersistentScheduler = tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = null, .ctas = cluster_ctas, .release_scope = .cta, .producer_consumes = two_cta });
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
            .{ .first_warp = 0, .warps = softmax_warps, .registers = num_regs.softmax, .Role = Softmax },
            .{ .first_warp = correction_warp, .warps = correction_warps, .registers = num_regs.correction, .Role = Correction },
            .{ .first_warp = other_warp, .warps = other_warps, .registers = num_regs.other, .Role = Other },
        };
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
            o: *const tma.TensorMap,
        };
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(scheduler_options.block_m, scheduler_options.block_order, specialization.sort_batches);

        const Tiles = struct {
            q: [q_stage][m_block_size * row_chunks]Chunk,
            kv: [kv_stage][block_n * row_chunks / cluster_ctas]Chunk,
            staging: [staging_buffers][m_block_size * box_chunks]Chunk,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
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
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
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
        var tile_divisors: tile_scheduler.SectionDivisors addrspace(.shared) = undefined;

        pub fn allocateTmem() void {
            if (two_cta) tcgen05.allocate2Sm(&tmem_base, tcgen05.capacity) else tcgen05.allocate(&tmem_base, tcgen05.capacity);
        }

        pub fn prepare(params: ForwardParams, args: Args) void {
            if (tiling.divmod_decode) {
                if (thread.warp() == initializer_warp and elect.oneSync()) tile_divisors = tileScheduler(params).sectionDivisors();
                elect.reconverge();
            }
            if (!prologue_loads or thread.warp() != initializer_warp) return;
            const first = Mma.workAt(params, tileScheduler(params), firstTile(.create()));
            if (elect.oneSync()) {
                kv_full[0].init(1);
                inline for (&q_full) |*full| full.init(1);
                mbarrier.pipelineInitFence();
                if (first) |work| {
                    Load.issueKv(args, .keys, work, tensorBatch(params, work), 0, nBlock(work, 0));
                    Load.copyQueries(args, work, tensorBatch(params, work));
                }
            }
            elect.reconverge();
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
                    p_full_o_rescaled[buffer].init(comptime mbarrier.arrivals(release_arrival, stage_warps + correction_warps, cluster_ctas));
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

        pub fn teardown() void {
            if (!two_cta) return;
            tcgen05.fenceBeforeSync();
            cluster.sync();
            if (thread.warp() == other_warp) {
                tcgen05.fenceAfterSync();
                tcgen05.deallocate2Sm(tmem_base, tcgen05.capacity);
            }
        }

        fn rowOffset() u32 {
            return if (two_cta) cluster.rankIn2Cta() * m_block_size else 0;
        }

        fn firstTile(persistent_scheduler: PersistentScheduler) u32 {
            return persistent_scheduler.initialWorkTileInfo();
        }

        fn tileScheduler(params: ForwardParams) TileScheduler {
            return .create(tile_scheduler.forwardArguments(params));
        }

        fn usesClc(scheduler: TileScheduler) bool {
            return nvptx.nctaid(0) == cluster_ctas * scheduler.totalBlocks();
        }

        fn nextTile(scheduler: TileScheduler, persistent_scheduler: *PersistentScheduler, tile_id: u32) u32 {
            if (!usesClc(scheduler)) return tile_id + nvptx.nctaid(0) / cluster_ctas;
            return persistent_scheduler.advanceToNextWork();
        }

        fn tileWork(params: ForwardParams, scheduler: TileScheduler, tile_id: u32) ?Work {
            if (params.cu_seqlens_q == 0) {
                const divisors = tile_divisors;
                const tile = scheduler.divmodWork(divisors, tile_id);
                if (specialization.pack_gqa) return Work.init(params, tile);
                var ungrouped = params;
                ungrouped.qhead_per_kvhead = 1;
                var work = Work.init(ungrouped, tile);
                work.kv_head = divisors.qhead_per_kvhead_divmod.divide(tile.head);
                return work;
            }
            const tile = scheduler.getCurrentWork(tile_id) orelse return null;
            return Work.init(params, tile);
        }

        fn bufferOf(sequence: u32, stage: u32) u32 {
            return if (q_stage == 2) stage else sequence % score_buffers;
        }

        fn roundOf(sequence: u32) u32 {
            return if (q_stage == 2) sequence else sequence / score_buffers;
        }

        fn tensorRow() u32 {
            return thread.id() % m_block_size;
        }

        fn nBlock(work: Work, index: u32) u32 {
            return work.n_block_max - 1 - index;
        }

        fn tensorBatch(params: ForwardParams, work: Work) u32 {
            return if (params.cu_seqlens_q == 0) work.batch else 0;
        }

        fn tmaRow(work: Work, row: u32) u32 {
            return work.sequence.offset_q + if (specialization.pack_gqa) work.pack_gqa.mIdx(row) else row;
        }

        fn rowLimit(work: Work) u32 {
            return if (specialization.pack_gqa) work.pack_gqa.end_row else work.sequence.seqlen_q;
        }

        pub const Other = struct {
            pub fn run(params: ForwardParams, args: Args) void {
                switch (thread.warp() - other_warp) {
                    0 => Mma.run(params),
                    1 => Load.run(params, args),
                    2 => Scheduler.run(params),
                    else => {},
                }
            }
        };

        const Scheduler = struct {
            fn run(params: ForwardParams) void {
                const scheduler = tileScheduler(params);
                if (!usesClc(scheduler)) return;
                if (two_cta and cluster.rankIn2Cta() != 0) return produce(scheduler, .follower);
                produce(scheduler, .leader);
            }

            fn produce(scheduler: TileScheduler, comptime role: tile_scheduler.CtaRole) void {
                var persistent_scheduler: PersistentScheduler = .create();
                while (true) : (persistent_scheduler.advanceConsumerState()) {
                    persistent_scheduler.prefetchNextWorkAs(role);
                    persistent_scheduler.consumerWait();
                    const tile_id = persistent_scheduler.getCurrentWork();
                    if (PersistentScheduler.producer_consumes) persistent_scheduler.consumerRelease();
                    if (tile_id >= scheduler.totalBlocks()) break;
                }
                persistent_scheduler.producerTail();
            }
        };

        const Load = struct {
            const Operand = enum { keys, values };

            fn copyCtaGroup(
                tile: anytype,
                map: *const tma.TensorMap,
                barrier: u32,
                comptime rows: u32,
                comptime box_rows: u32,
                first_box: u32,
                comptime box_count: u32,
                head: u32,
                row: u32,
                batch: u32,
            ) void {
                inline for (0..box_count) |box_index| {
                    inline for (0..rows / box_rows) |part_index| {
                        const box: u32 = box_index;
                        const part: u32 = part_index;
                        const column = (first_box + box) * box_columns;
                        const destination = &tile[(box * rows + part * box_rows) * box_chunks];
                        tma.copyBoxCtaGroup(map, destination, barrier, column, head, row + part * box_rows, batch);
                    }
                }
            }

            fn copyQuery(args: Args, work: Work, batch: u32, stage: u32) void {
                const row = if (specialization.pack_gqa)
                    tmaRow(work, work.first_row + rowOffset() + stage * m_block_size)
                else
                    work.sequence.offset_q + work.first_row + rowOffset() + stage * m_block_size;
                if (two_cta) {
                    if (cluster.rankIn2Cta() == 0) _ = q_full[stage].arriveExpectTx(cluster_ctas * m_block_size * padded * 2);
                    const barrier = cluster.leaderAddress(&q_full[stage]);
                    copyCtaGroup(&tiles.q[stage], args.q, barrier, m_block_size, m_block_size, 0, boxes, work.head, row, batch);
                } else {
                    _ = q_full[stage].arriveExpectTx(m_block_size * padded * 2);
                    Layout.copyTile(&tiles.q[stage], args.q, &q_full[stage], work.head, row, batch);
                }
            }

            fn copyKv(args: Args, comptime operand: Operand, work: Work, batch: u32, kv_index: u32, n_block: u32) void {
                const slot = kv_index % kv_stage;
                mbarrier.waitBackoff(&kv_empty[slot], kv_index / kv_stage ^ 1, tiling.load_backoff_ns);
                issueKv(args, operand, work, batch, slot, n_block);
            }

            fn issueKv(args: Args, comptime operand: Operand, work: Work, batch: u32, slot: u32, n_block: u32) void {
                const map = if (operand == .keys) args.k else args.v;
                const first_key = work.sequence.offset_k + n_block * block_n;
                if (!two_cta) {
                    _ = kv_full[slot].arriveExpectTx(block_n * padded * 2);
                    Layout.copyTile(&tiles.kv[slot], map, &kv_full[slot], work.kv_head, first_key, batch);
                    return;
                }
                if (cluster.rankIn2Cta() == 0) _ = kv_full[slot].arriveExpectTx(block_n * padded * 2);
                const barrier = cluster.leaderAddress(&kv_full[slot]);
                const rank = cluster.rankIn2Cta();
                switch (operand) {
                    .keys => copyCtaGroup(&tiles.kv[slot], map, barrier, slot_rows, slot_rows, 0, boxes, work.kv_head, first_key + rank * slot_rows, batch),
                    .values => copyCtaGroup(&tiles.kv[slot], map, barrier, block_n, slot_rows, rank * boxes / 2, boxes / 2, work.kv_head, first_key, batch),
                }
            }

            fn copyQueries(args: Args, work: Work, batch: u32) void {
                copyQuery(args, work, batch, 0);
                if (q_stage == 2) copyQuery(args, work, batch, 1);
            }

            fn copyTile(args: Args, work: Work, batch: u32, count: u32, first_index: u32, loaded: u32, head_loaded: bool) void {
                const loads_head = !prologue_loads or !head_loaded;
                var kv_index = first_index;
                if (loads_head) copyKv(args, .keys, work, batch, kv_index, nBlock(work, 0));
                kv_index += 1;
                if (loads_head) {
                    mbarrier.waitBackoff(&q_empty, loaded ^ 1, tiling.load_backoff_ns);
                    copyQueries(args, work, batch);
                }
                if (q_stage == 2) {
                    var index: u32 = 0;
                    while (index < count) : (index += 1) {
                        if (index > 0) {
                            copyKv(args, .keys, work, batch, kv_index, nBlock(work, index));
                            kv_index += 1;
                        }
                        copyKv(args, .values, work, batch, kv_index, nBlock(work, index));
                        kv_index += 1;
                    }
                } else {
                    var index: u32 = 1;
                    while (index < count) : (index += 1) {
                        copyKv(args, .keys, work, batch, kv_index, nBlock(work, index));
                        copyKv(args, .values, work, batch, kv_index + 1, nBlock(work, index - 1));
                        kv_index += 2;
                    }
                    copyKv(args, .values, work, batch, kv_index, nBlock(work, count - 1));
                }
            }

            fn run(params: ForwardParams, args: Args) void {
                var persistent_scheduler: PersistentScheduler = .create();
                var tile_id = firstTile(persistent_scheduler);
                const first_tile = tile_id;
                var loaded: u32 = 0;
                var kv_index: u32 = 0;
                const scheduler = tileScheduler(params);
                while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(scheduler, &persistent_scheduler, tile_id)) {
                    const work = if (tiling.divmod_decode) tileWork(params, scheduler, tile_id) orelse continue else Work.init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                    const count = work.n_block_max - work.n_block_min;
                    if (count == 0) continue;
                    if (elect.oneSync()) copyTile(args, work, tensorBatch(params, work), count, kv_index, loaded, prologue_loads and tile_id == first_tile);
                    elect.reconverge();
                    kv_index += 2 * count;
                    loaded += 1;
                }
            }
        };

        const Mma = struct {
            const score_instruction = tcgen05.instructionDescriptor(mma_rows, block_n, .k, .k);
            const value_instruction = tcgen05.instructionDescriptor(mma_rows, mma_head_dim, .k, .mn);

            fn valueOffsets(comptime unit: u32) [block_n / 16]u32 {
                var offsets: [block_n / 16]u32 = undefined;
                for (&offsets, 0..) |*offset, step| offset.* = step * unit;
                return offsets;
            }

            fn scores(tmem: u32, stage: u32, buffer: u32, slot: u32) void {
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

            fn values(tmem: u32, stage: u32, buffer: u32, slot: u32, accumulate: bool, round: u32) void {
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
                    mma(&a_offsets, &b_offsets, output, probabilities, descriptor, value_instruction, accumulate);
                    return;
                }
                mma(a_offsets[0..split_steps], b_offsets[0..split_steps], output, probabilities, descriptor, value_instruction, accumulate);
                mbarrier.wait(.cta, &p_full_2[buffer], round);
                tcgen05.fenceAfterSync();
                mma(a_offsets[split_steps..], b_offsets[split_steps..], output, probabilities, descriptor, value_instruction, true);
            }

            const score_steps = mma_head_dim / 16;
            const tail_steps = block_n / 16 - split_steps;

            const StageOperands = struct {
                q: [score_steps]u64,
                k: [score_steps]u64,
                p: [tail_steps]u32,
                v: [tail_steps]u64,
            };

            fn stageOperands(tmem: u32, comptime stage: u32, key_slot: u32, value_slot: u32) StageOperands {
                const q_offsets = comptime tcgen05.kMajorOffsets(mma_head_dim / 16, m_block_size, swizzle_bytes);
                const k_offsets = comptime tcgen05.kMajorOffsets(mma_head_dim / 16, slot_rows, swizzle_bytes);
                const p_offsets = comptime valueOffsets(8);
                const v_offsets = comptime valueOffsets(16 * swizzle_bytes >> 4);
                const q = tcgen05.kMajorDescriptor(tcgen05.sharedAddress(&tiles.q[stage]), swizzle_bytes);
                const k = tcgen05.kMajorDescriptor(tcgen05.sharedAddress(&tiles.kv[key_slot]), swizzle_bytes);
                const v = tcgen05.sharedDescriptor(tcgen05.sharedAddress(&tiles.kv[value_slot]), block_n * swizzle_bytes, 8 * swizzle_bytes, swizzle);
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

            fn valuesHead(tmem: u32, stage: u32, slot: u32, accumulate: bool) void {
                const p_offsets = comptime valueOffsets(8);
                const v_offsets = comptime valueOffsets(16 * swizzle_bytes >> 4);
                const descriptor = tcgen05.sharedDescriptor(tcgen05.sharedAddress(&tiles.kv[slot]), block_n * swizzle_bytes, 8 * swizzle_bytes, swizzle);
                const output = tmem + output_column + stage * mma_head_dim;
                const probabilities = tmem + stage * block_n + probability_offset;
                tcgen05.mmaTsSteps(p_offsets[0..split_steps], v_offsets[0..split_steps], output, probabilities, descriptor, value_instruction, accumulate);
            }

            fn valuesTail(tmem: u32, comptime stage: u32, operands: StageOperands) void {
                if (elect.oneSync()) {
                    const output = tmem + output_column + stage * mma_head_dim;
                    inline for (operands.p, operands.v) |p, v| tcgen05.mmaTs(output, p, v, value_instruction, true);
                }
                elect.reconverge();
            }

            fn stageScores(tmem: u32, comptime stage: u32, operands: StageOperands) void {
                if (elect.oneSync()) {
                    inline for (operands.q, operands.k, 0..) |q, k, step| tcgen05.mmaSs(tmem + stage * block_n, q, k, score_instruction, step != 0);
                }
                elect.reconverge();
            }

            fn mmaTileQStage2(tmem: u32, count: u32, finished: u32, kv_index: *u32, blocks: *u32) void {
                var key_slot = kv_index.* % kv_stage;
                mbarrier.wait(.cta, &kv_full[key_slot], kv_index.* / kv_stage);
                inline for (0..2) |stage_index| {
                    const stage: u32 = stage_index;
                    mbarrier.wait(.cta, &q_full[stage], finished);
                    tcgen05.fenceAfterSync();
                    scores(tmem, stage, stage, key_slot);
                    tcgen05.commit(commit_group, &s_full[stage]);
                }
                tcgen05.commit(commit_group, &kv_empty[key_slot]);
                kv_index.* += 1;
                var index: u32 = 1;
                while (index < count) : (index += 1) {
                    const value_slot = kv_index.* % kv_stage;
                    mbarrier.wait(.cta, &kv_full[value_slot], kv_index.* / kv_stage);
                    key_slot = (kv_index.* + 1) % kv_stage;
                    inline for (0..2) |stage_index| {
                        const stage: u32 = stage_index;
                        const operands = stageOperands(tmem, stage, key_slot, value_slot);
                        mbarrier.wait(.cta, &p_full_o_rescaled[stage], blocks.*);
                        tcgen05.fenceAfterSync();
                        valuesHead(tmem, stage, value_slot, index > 1);
                        mbarrier.wait(.cta, &p_full_2[stage], blocks.*);
                        tcgen05.fenceAfterSync();
                        valuesTail(tmem, stage, operands);
                        if (stage == 0) mbarrier.wait(.cta, &kv_full[key_slot], (kv_index.* + 1) / kv_stage);
                        if (stage == 1) tcgen05.commit(commit_group, &kv_empty[value_slot]);
                        stageScores(tmem, stage, operands);
                        tcgen05.commit(commit_group, &s_full[stage]);
                    }
                    tcgen05.commit(commit_group, &kv_empty[key_slot]);
                    kv_index.* += 2;
                    blocks.* += 1;
                }
                tcgen05.commit(commit_group, &q_empty);
            }

            fn lastValues(tmem: u32, count: u32, kv_index: *u32, blocks: *u32) void {
                const value_slot = kv_index.* % kv_stage;
                mbarrier.wait(.cta, &kv_full[value_slot], kv_index.* / kv_stage);
                inline for (0..2) |stage_index| {
                    const stage: u32 = stage_index;
                    mbarrier.wait(.cta, &p_full_o_rescaled[stage], blocks.*);
                    tcgen05.fenceAfterSync();
                    values(tmem, stage, stage, value_slot, count > 1, blocks.*);
                    tcgen05.commit(commit_group, &o_full[stage]);
                }
                tcgen05.commit(commit_group, &kv_empty[value_slot]);
                kv_index.* += 1;
                blocks.* += 1;
            }

            fn issueScores(tmem: u32, buffer: u32, kv_index: *u32, last: bool) void {
                const slot = kv_index.* % kv_stage;
                mbarrier.wait(.cta, &kv_full[slot], kv_index.* / kv_stage);
                scores(tmem, 0, buffer, slot);
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
                    values(tmem, 0, buffer, value_slot, index > 0, 0);
                    tcgen05.commit(commit_group, &o_full[0]);
                    tcgen05.commit(commit_group, &kv_empty[value_slot]);
                    kv_index.* += 1;
                    if (index + 2 < count) issueScores(tmem, buffer, kv_index, index + 3 == count);
                }
                blocks.* += count;
            }

            fn idle(params: ForwardParams) void {
                var persistent_scheduler: PersistentScheduler = .create();
                var tile_id = firstTile(persistent_scheduler);
                const scheduler = tileScheduler(params);
                while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(scheduler, &persistent_scheduler, tile_id)) {}
            }

            fn workAt(params: ForwardParams, scheduler: TileScheduler, tile_id: u32) ?Work {
                if (tile_id >= scheduler.totalBlocks()) return null;
                const work = if (tiling.divmod_decode) tileWork(params, scheduler, tile_id) orelse return null else Work.init(params, scheduler.getCurrentWork(tile_id) orelse return null);
                return if (work.n_block_max > work.n_block_min) work else null;
            }

            fn findWork(params: ForwardParams, scheduler: TileScheduler, persistent_scheduler: *PersistentScheduler, tile_id: *u32) ?Work {
                while (tile_id.* < scheduler.totalBlocks()) : (tile_id.* = nextTile(scheduler, persistent_scheduler, tile_id.*)) {
                    if (workAt(params, scheduler, tile_id.*)) |work| return work;
                }
                return null;
            }

            fn run(params: ForwardParams) void {
                if (two_cta and cluster.rankIn2Cta() != 0) return idle(params);
                tcgen05.fenceAfterSync();
                const tmem = tmem_base;
                var persistent_scheduler: PersistentScheduler = .create();
                var tile_id = firstTile(persistent_scheduler);
                var finished: u32 = 0;
                var kv_index: u32 = 0;
                var blocks: u32 = 0;
                const scheduler = tileScheduler(params);
                var current = findWork(params, scheduler, &persistent_scheduler, &tile_id);
                while (current) |work| : (finished += 1) {
                    const count = work.n_block_max - work.n_block_min;
                    if (q_stage == 2) {
                        mmaTileQStage2(tmem, count, finished, &kv_index, &blocks);
                    } else {
                        mmaTileQStage1(tmem, count, finished, &kv_index, &blocks);
                    }
                    if (tiling.prefetch_next_work_early) {
                        tile_id = nextTile(scheduler, &persistent_scheduler, tile_id);
                        current = workAt(params, scheduler, tile_id);
                    }
                    if (q_stage == 2) lastValues(tmem, count, &kv_index, &blocks);
                    if (!tiling.prefetch_next_work_early) {
                        tile_id = nextTile(scheduler, &persistent_scheduler, tile_id);
                        current = workAt(params, scheduler, tile_id);
                    }
                    if (current == null) current = findWork(params, scheduler, &persistent_scheduler, &tile_id);
                }
                if (two_cta) return;
                mbarrier.wait(.cta, &tmem_dealloc, 0);
                tcgen05.deallocate(tmem, tcgen05.capacity);
            }
        };

        pub const Softmax = struct {
            const fragment_pairs = 16;
            const frg_cnt = block_n / 2 / fragment_pairs;

            fn exponentiate(scores: Scores, softmax_scale_log2: f32, offset: f32, address: u32, sum: *f32, buffer: u32) Scores {
                const scale = f32x2.splat(softmax_scale_log2);
                const shift = f32x2.splat(-offset);
                var sums: [4]f32x2.Pair = @splat(f32x2.splat(0));
                var values: Scores = undefined;
                var probabilities: @Vector(block_n / 2, u32) = undefined;
                inline for (0..block_n / 2) |pair| {
                    const exponent = f32x2.fma(.{ scores[2 * pair], scores[2 * pair + 1] }, scale, shift);
                    const emulated = switch (tiling.ex2_emulation_placement) {
                        .interleaved => pair % 8 < ex2_emulated_pairs,
                        .leading => pair < ex2_emulated_pairs * 8,
                        .inner_fragments => pair / fragment_pairs != 0 and pair / fragment_pairs != frg_cnt - 1 and pair % 8 >= 8 - 2 * ex2_emulated_pairs,
                    };
                    const value = if (emulated)
                        f32x2.ex2Emulation(exponent)
                    else
                        f32x2.exp2f(exponent);
                    values[2 * pair] = value[0];
                    values[2 * pair + 1] = value[1];
                    if (!tiling.late_row_sum) sums[pair % 4] = f32x2.add(sums[pair % 4], value);
                    probabilities[pair] = @backingInt(nvptx.Bf16x2.fromF32(value[0], value[1]));
                    if (tiling.fragment_stores and (pair + 1) % fragment_pairs == 0) {
                        const first: u32 = pair + 1 - fragment_pairs;
                        var fragment: @Vector(fragment_pairs, u32) = undefined;
                        inline for (0..fragment_pairs) |column| fragment[column] = probabilities[first + column];
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
                    inline for (p_split_pairs..block_n / 2) |column| tail[column - p_split_pairs] = probabilities[column];
                    tcgen05.storeColumns(address + probability_offset + p_split_pairs, tail);
                    tcgen05.waitStore();
                    tcgen05.fenceBeforeSync();
                    mbarrier.arrive(.thread, &p_full_2[buffer]);
                } else {
                    tcgen05.storeColumns(address + probability_offset, probabilities);
                    tcgen05.waitStore();
                    tcgen05.fenceBeforeSync();
                }
                if (!tiling.late_row_sum) sum.* = total(sums);
                return values;
            }

            fn total(sums: [4]f32x2.Pair) f32 {
                const pair = f32x2.add(f32x2.add(sums[0], sums[1]), f32x2.add(sums[2], sums[3]));
                return pair[0] + pair[1];
            }

            pub fn run(params: ForwardParams, args: Args) void {
                _ = args;
                const stage = thread.id() / m_block_size;
                tcgen05.fenceAfterSync();
                const tmem = tmem_base;
                const softmax_scale_log2 = params.softmax_scale_log2;
                var persistent_scheduler: PersistentScheduler = .create();
                var tile_id = firstTile(persistent_scheduler);
                var blocks: u32 = 0;
                var softmax_corr_index: u32 = 0;
                const scheduler = tileScheduler(params);
                while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(scheduler, &persistent_scheduler, tile_id)) {
                    const work = if (tiling.divmod_decode) tileWork(params, scheduler, tile_id) orelse continue else Work.init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                    const count = work.n_block_max - work.n_block_min;
                    if (count == 0) continue;
                    const row = work.first_row + rowOffset() + stage * m_block_size + tensorRow();
                    const query: i32 = @intCast(if (specialization.pack_gqa) work.pack_gqa.mIdx(row) else row);
                    const col_limit_left = work.block_info.colLimitLeft(query);
                    const col_limit_right = work.block_info.colLimitRight(query);
                    var max: f32 = minus_infinity;
                    var sum: f32 = 0;
                    var index: u32 = 0;
                    while (index < count) : (index += 1) {
                        const n_block = nBlock(work, index);
                        const buffer = bufferOf(blocks, stage);
                        const scores_address = tcgen05.warpAddress(tmem, buffer * block_n);
                        mbarrier.wait(.cta, &s_full[buffer], roundOf(blocks));
                        tcgen05.fenceAfterSync();
                        var scores = tcgen05.loadColumns(block_n, scores_address);
                        tcgen05.waitLoad();
                        if (work.needsMask(n_block)) {
                            const first_column: i32 = @intCast(n_block * block_n);
                            maskR2p(&scores, col_limit_left - first_column, col_limit_right - first_column, minus_infinity);
                        }
                        var partial_max = [_]f32{ max, minus_infinity, minus_infinity, minus_infinity };
                        inline for (0..block_n / 2) |pair| {
                            partial_max[pair % 4] = f32x2.max3(partial_max[pair % 4], scores[2 * pair], scores[2 * pair + 1]);
                        }
                        const tile_max = f32x2.max3(partial_max[0], partial_max[1], @max(partial_max[2], partial_max[3]));
                        var scale: f32 = 1;
                        if (index == 0) {
                            max = tile_max;
                        } else {
                            const reference = if (tile_max == minus_infinity) 0 else tile_max;
                            const exponent = (max - reference) * softmax_scale_log2;
                            if (exponent < -fragments.rescale_threshold) {
                                scale = nvptx.math.ex2ApproxF32(true, exponent);
                                max = tile_max;
                            }
                            const softmax_corr_buffer = bufferOf(softmax_corr_index, stage);
                            row_scale[softmax_corr_buffer][tensorRow()] = scale;
                            mbarrier.arrive(.thread, &softmax_corr_full[softmax_corr_buffer]);
                            softmax_corr_index += 1;
                        }
                        const offset = if (max == minus_infinity) 0 else max * softmax_scale_log2;
                        var tile_sum: f32 = undefined;
                        const values = exponentiate(scores, softmax_scale_log2, offset, scores_address, &tile_sum, buffer);
                        if (p_split_pairs == 0) mbarrier.arrive(release_arrival, &p_full_o_rescaled[buffer]);
                        mbarrier.wait(.cta, &softmax_corr_empty[bufferOf(softmax_corr_index, stage)], roundOf(softmax_corr_index) ^ 1);
                        if (tiling.late_row_sum) {
                            var sums: [4]f32x2.Pair = @splat(f32x2.splat(0));
                            inline for (0..block_n / 2) |pair| {
                                sums[pair % 4] = f32x2.add(sums[pair % 4], .{ values[2 * pair], values[2 * pair + 1] });
                            }
                            tile_sum = total(sums);
                        }
                        sum = sum * scale + tile_sum;
                        blocks += 1;
                    }
                    const softmax_corr_buffer = bufferOf(softmax_corr_index, stage);
                    row_scale[softmax_corr_buffer][tensorRow()] = max;
                    row_sum[softmax_corr_buffer][tensorRow()] = sum;
                    mbarrier.arrive(.thread, &softmax_corr_full[softmax_corr_buffer]);
                    softmax_corr_index += 1;
                }
                tcgen05.fenceBeforeSync();
                mbarrier.arrive(.thread, &tmem_dealloc);
            }
        };

        pub const Correction = struct {
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
                        const product = f32x2.mul(.{ current[2 * pair], current[2 * pair + 1] }, factor);
                        scaled[2 * pair] = product[0];
                        scaled[2 * pair + 1] = product[1];
                    }
                    tcgen05.store(output + chunk * width, scaled);
                }
                tcgen05.waitStore();
            }

            fn outputElement(params: ForwardParams, work: Work, sequence_row: u32) u64 {
                if (specialization.pack_gqa) {
                    return work.batch * params.o_batch_stride + @as(u64, work.sequence.offset_q) * params.o_row_stride +
                        work.head * params.o_head_stride + work.pack_gqa.rowOffset(sequence_row, params.o_row_stride, params.o_head_stride);
                }
                return work.batch * params.o_batch_stride +
                    @as(u64, work.sequence.offset_q + sequence_row) * params.o_row_stride +
                    work.head * params.o_head_stride;
            }

            fn lseIndex(params: ForwardParams, work: Work, sequence_row: u32) u64 {
                const row_offset = if (specialization.pack_gqa) work.pack_gqa.rowOffset(sequence_row, 1, params.lse_head_stride) else sequence_row;
                return work.batch * params.lse_batch_stride + work.head * params.lse_head_stride +
                    work.sequence.offset_q + row_offset;
            }

            fn writeEmpty(params: ForwardParams, work: Work) void {
                const row = tensorRow();
                const out: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
                const lse: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                const real_chunks = head_dim / 8;
                inline for (0..q_stage) |stage_index| {
                    const stage: u32 = stage_index;
                    const first = work.first_row + rowOffset() + stage * m_block_size;
                    if (first < rowLimit(work)) {
                        const valid_rows = @min(m_block_size, rowLimit(work) - first);
                        var index = row;
                        while (index < valid_rows * real_chunks) : (index += m_block_size) {
                            const element = outputElement(params, work, first + index / real_chunks);
                            out[element / 8 + index % real_chunks] = @splat(0);
                        }
                        if (row < valid_rows) lse[lseIndex(params, work, first + row)] = minus_infinity;
                    }
                }
            }

            fn epilogue(
                params: ForwardParams,
                args: Args,
                work: Work,
                stage: u32,
                output: u32,
                max: f32,
                sum: f32,
                stores: *u32,
            ) void {
                const row = tensorRow();
                const first = work.first_row + rowOffset() + stage * m_block_size;
                if (first >= rowLimit(work)) return;
                const valid_rows = @min(m_block_size, rowLimit(work) - first);
                const bulk = params.cu_seqlens_q == 0 or valid_rows == m_block_size;
                const empty_row = !(sum > 0);
                const inverse: f32 = if (empty_row) 0 else 1 / sum;
                const out: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
                const total = max * params.softmax_scale_log2 + nvptx.math.lg2ApproxF32(true, sum);
                const lse = if (empty_row) minus_infinity else total * std.math.ln2;
                inline for (0..boxes) |box_index| {
                    const box: u32 = box_index;
                    const buffer = stores.* % staging_buffers;
                    const staging = &tiles.staging[buffer];
                    if (thread.warp() == correction_warp and stores.* >= staging_buffers and elect.oneSync()) {
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
                            var words: Chunk = undefined;
                            inline for (0..4) |word| {
                                const index = chunk * 8 + word * 2;
                                words[word] = @backingInt(nvptx.Bf16x2.fromF32(values[index] * inverse, values[index + 1] * inverse));
                            }
                            Staging.chunk(staging, row, loaded / 8 + chunk, 0).* = words;
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
                                const element = outputElement(params, work, first + index / box_chunks) + column;
                                out[element / 8] = Staging.chunk(staging, index / box_chunks, chunk, 0).*;
                            }
                        }
                    }
                    if (thread.warp() == correction_warp and elect.oneSync()) {
                        if (bulk) {
                            tma.storeBox(
                                args.o,
                                &staging[0],
                                box * box_columns,
                                work.head,
                                tmaRow(work, first),
                                tensorBatch(params, work),
                            );
                        }
                        tma.commitBulkGroup();
                    }
                    elect.reconverge();
                    stores.* += 1;
                }
                storeLse(params, work, first, valid_rows, lse);
            }

            fn storeLse(params: ForwardParams, work: Work, first: u32, valid_rows: u32, lse: f32) void {
                const row = tensorRow();
                if (specialization.pack_gqa) {
                    const out: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                    if (row < valid_rows) out[lseIndex(params, work, first + row)] = lse;
                    return;
                }
                const lane = thread.lane();
                const warp_first = row - lane;
                const clamp = nvptx.ShuffleClamp.fromWidth(.idx, 32);
                var quad: @Vector(4, f32) = undefined;
                inline for (0..4) |element_index| {
                    const element: u32 = element_index;
                    quad[element] = nvptx.shflSync(.idx, f32, lse, (lane * 4 + element) % 32, clamp, .all);
                }
                const address = params.lse + 4 * lseIndex(params, work, first + warp_first);
                if (address % 16 == 0 and warp_first + 32 <= valid_rows) {
                    const out: [*]addrspace(.global) @Vector(4, f32) = @ptrFromInt(address);
                    if (lane < 8) out[lane] = quad;
                } else if (row < valid_rows) {
                    const out: [*]addrspace(.global) f32 = @ptrFromInt(params.lse);
                    out[lseIndex(params, work, first + row)] = lse;
                }
            }

            fn takeScale(buffer: u32, round: u32) f32 {
                mbarrier.wait(.cta, &softmax_corr_full[buffer], round);
                const scale = row_scale[buffer][tensorRow()];
                mbarrier.arrive(.thread, &softmax_corr_empty[buffer]);
                return scale;
            }

            fn correct(output: u32, scale: f32) void {
                if (nvptx.voteSync(.any, scale != 1, .all)) {
                    tcgen05.fenceAfterSync();
                    rescale(output, scale);
                }
                tcgen05.fenceBeforeSync();
            }

            fn finish(
                params: ForwardParams,
                args: Args,
                work: Work,
                stage: u32,
                buffer: u32,
                round: u32,
                output: u32,
                stores: *u32,
            ) void {
                mbarrier.wait(.cta, &softmax_corr_full[buffer], round);
                const max = row_scale[buffer][tensorRow()];
                const sum = row_sum[buffer][tensorRow()];
                mbarrier.arrive(.thread, &softmax_corr_empty[buffer]);
                tcgen05.fenceAfterSync();
                epilogue(params, args, work, stage, output, max, sum, stores);
                tcgen05.fenceBeforeSync();
            }

            pub fn run(params: ForwardParams, args: Args) void {
                tcgen05.fenceAfterSync();
                const output_address = tcgen05.warpAddress(tmem_base, output_column);
                inline for (0..q_stage) |buffer| mbarrier.arrive(release_arrival, &p_full_o_rescaled[buffer]);
                var persistent_scheduler: PersistentScheduler = .create();
                var tile_id = firstTile(persistent_scheduler);
                var softmax_corr_index: u32 = 0;
                var finished: u32 = 0;
                var blocks: u32 = 0;
                var stores: u32 = 0;
                const scheduler = tileScheduler(params);
                while (tile_id < scheduler.totalBlocks()) : (tile_id = nextTile(scheduler, &persistent_scheduler, tile_id)) {
                    const work = if (tiling.divmod_decode) tileWork(params, scheduler, tile_id) orelse continue else Work.init(params, scheduler.getCurrentWork(tile_id) orelse continue);
                    const count = work.n_block_max - work.n_block_min;
                    if (count == 0) {
                        writeEmpty(params, work);
                        continue;
                    }
                    if (q_stage == 2) {
                        var index: u32 = 1;
                        while (index < count) : (index += 1) {
                            inline for (0..2) |stage_index| {
                                const stage: u32 = stage_index;
                                correct(output_address + stage * mma_head_dim, takeScale(stage, softmax_corr_index));
                                mbarrier.arrive(release_arrival, &p_full_o_rescaled[stage]);
                            }
                            softmax_corr_index += 1;
                        }
                        inline for (0..2) |stage_index| {
                            const stage: u32 = stage_index;
                            mbarrier.wait(.cta, &o_full[stage], finished);
                            finish(params, args, work, stage, stage, softmax_corr_index, output_address + stage * mma_head_dim, &stores);
                            mbarrier.arrive(release_arrival, &p_full_o_rescaled[stage]);
                        }
                        softmax_corr_index += 1;
                    } else {
                        var index: u32 = 1;
                        while (index < count) : (index += 1) {
                            const scale = takeScale(softmax_corr_index % score_buffers, softmax_corr_index / score_buffers);
                            softmax_corr_index += 1;
                            mbarrier.wait(.cta, &o_full[0], blocks + index - 1);
                            correct(output_address, scale);
                            mbarrier.arrive(release_arrival, &p_full_o_rescaled[(blocks + index) % score_buffers]);
                        }
                        mbarrier.wait(.cta, &o_full[0], blocks + count - 1);
                        finish(params, args, work, 0, softmax_corr_index % score_buffers, softmax_corr_index / score_buffers, output_address, &stores);
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
        };

        pub const Kernel = WarpSpecializedKernel(ForwardParams, @This());
    };
}

const variant = configurations.forwardSm100Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const forward_tiling = variant.tiling;
const Forward = FlashAttnFwdSm100(forward_tiling, variant.specialization);

fn flashAttnFwd(
    q_map: tma.TensorMap,
    k_map: tma.TensorMap,
    v_map: tma.TensorMap,
    o_map: tma.TensorMap,
    o: u64,
    lse: u64,
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    seqlen_q: u32,
    seqlen_k: u32,
    qhead_per_kvhead: u32,
    softmax_scale_log2: f32,
    window_size_left: i32,
    window_size_right: i32,
    num_m_blocks: u32,
    num_head: u32,
    num_batch: u32,
    o_batch_stride: u64,
    o_row_stride: u64,
    o_head_stride: u64,
    lse_batch_stride: u64,
    lse_head_stride: u64,
    scheduler_metadata: if (variant.specialization.scheduler_metadata) u64 else void,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
    .cluster_size = if (forward_tiling.use_2cta_instrs) .{ .x = forward_tiling.clusterCtas() } else null,
} }) void {
    @setEvalBranchQuota(100_000);
    if (variant.specialization.scheduler_metadata) grid_dependency.wait();
    Forward.Kernel.run(.{
        .q = 0,
        .k = 0,
        .v = 0,
        .o = o,
        .lse = lse,
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_k = cu_seqlens_k,
        .seqlen_q = seqlen_q,
        .seqlen_k = seqlen_k,
        .qhead_per_kvhead = qhead_per_kvhead,
        .softmax_scale_log2 = softmax_scale_log2,
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
        .o_batch_stride = o_batch_stride,
        .o_row_stride = o_row_stride,
        .o_head_stride = o_head_stride,
        .lse_batch_stride = lse_batch_stride,
        .lse_head_stride = lse_head_stride,
        .num_m_blocks = num_m_blocks,
        .num_head = num_head,
        .num_batch = num_batch,
        .scheduler_metadata = if (variant.specialization.scheduler_metadata) scheduler_metadata else 0,
    }, .{ .q = &q_map, .k = &k_map, .v = &v_map, .o = &o_map });
    Forward.teardown();
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&flashAttnFwd, .{ .name = "flash_attn_fwd" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_fwd_prepare_varlen_num_blocks" });
        }
    }
}
