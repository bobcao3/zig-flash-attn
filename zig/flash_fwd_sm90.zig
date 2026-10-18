const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const elect = @import("elect.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const ForwardParams = @import("ForwardParams.zig");
const WarpSpecializedKernel = @import("warp_specialized_kernel.zig").WarpSpecializedKernel;
const MBlockWork = @import("m_block_work.zig").MBlockWork;
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const applyMask = @import("mask.zig").applyMask;
const tma = @import("tma.zig");
const wgmma = @import("wgmma.zig");
const ColLimits = @import("BlockInfo.zig").ColLimits;

const Accumulator = fragments.Accumulator;
const RowPair = fragments.RowPair;

pub fn FlashAttnFwdSm90(
    comptime tiling: configurations.Sm90FwdTiling,
    comptime specialization: configurations.Specialization,
) type {
    const head_dim = tiling.head_dim;
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const producer_warps = configurations.Sm90FwdTiling.producer_warps;
    const consumer_warps = tiling.consumerWarps();
    const warps_per_group = configurations.warps_per_warp_group;
    const swizzle_bytes = tiling.swizzleBytes();
    const Operands = wgmma.Operands(swizzle_bytes);
    const columns = tiling.mmaHeadDim();
    const k_steps = columns / 16;
    const Layout = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const Online = fragments.Softmax(block_n);
    const Work = MBlockWork(block_m, block_n, .{
        .bottom_aligned = tiling.bottom_aligned,
        .pack_gqa = specialization.pack_gqa,
        .divmod_pack = true,
    });
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_m,
        .row_bytes = 4 * head_dim,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const PersistentScheduler =
        tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 1 });
    const rescale_o_before_gemm = tiling.rescale_o_before_gemm;
    const use_scheduler_barrier = tiling.use_scheduler_barrier;
    const intra_wg_overlap = tiling.intra_wg_overlap;
    const producer_registers: u32, const consumer_registers: u32 =
        configurations.registerRequirements(tiling.consumerGroups());

    return struct {
        pub const thread_count = tiling.threads();
        pub const shared_bytes = tiling.sharedBytes();
        pub const max_registers = @min(configurations.maxRegistersPerThread(thread_count), 248);
        pub const init_scope: mbarrier.Scope = .cta;
        pub const initializer_warp: u32 = 0;
        pub const roles = .{
            .{ runProducer, producer_warps, producer_registers },
            .{ runConsumer, consumer_warps, consumer_registers },
        };
        pub const Args = struct {
            q: *const tma.TensorMap,
            k: *const tma.TensorMap,
            v: *const tma.TensorMap,
        };
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(
            scheduler_options.block_m,
            scheduler_options.block_order,
            specialization.sort_batches,
        );

        const Stage = struct {
            k: Layout.Tile(block_n),
            v: Layout.Tile(block_n),
        };
        const staging_columns = tiling.stagingColumns();
        const StagingLayout = tma.SwizzledLayout(
            staging_columns / 8,
            configurations.wgmmaSwizzleBytes(staging_columns),
        );
        const Staging = StagingLayout.Tile(block_m);
        const Tiles = struct {
            q: Layout.Tile(block_m),
            stages: [stages]Stage,
            staging: if (tiling.overlap_so_sv) void else Staging,
        };
        const staging_barrier: nvptx.NamedBarrier =
            @fromBackingInt(@intCast(1 + tiling.consumerGroups()));

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
            std.debug.assert(
                !tiling.overlap_so_sv or @sizeOf(Staging) <= @sizeOf(Layout.Tile(block_n)),
            );
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var q_full: nvptx.Mbarrier addrspace(.shared) = undefined;
        var q_empty: nvptx.Mbarrier addrspace(.shared) = undefined;
        var k_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var k_empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var v_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var v_empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub fn initialize() void {
            if (thread.warp() == initializer_warp and elect.oneSync()) {
                q_full.init(1);
                q_empty.init(consumer_warps * 32);
                PersistentScheduler.init(consumer_warps);
                inline for (0..stages) |stage| {
                    k_full[stage].init(1);
                    v_full[stage].init(1);
                    k_empty[stage].init(consumer_warps * 32);
                    v_empty[stage].init(consumer_warps * 32);
                }
                mbarrier.pipelineInitFence();
            }
            elect.reconverge();
        }

        fn copyQuery(args: Args, work: Work, batch: u32, processed: u32) void {
            if (processed > 0) mbarrier.waitSuspendHint(
                &q_empty,
                processed - 1,
                tiling.suspend_time_hint_ns,
            );
            const row = work.sequence.offset_q +
                if (specialization.pack_gqa) work.block_info.first_row else work.first_row;
            _ = q_full.arriveExpectTx(comptime tiling.tileBytes(block_m));
            Layout.copyTile(&tiles.q, args.q, &q_full, work.head, row, batch);
        }

        pub fn runProducer(params: ForwardParams, args: Args) void {
            if (thread.warp() != 0) return;
            var persistent_scheduler: PersistentScheduler = .create(params.tile_count_semaphore);
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var processed: u32 = 0;
            var kv_index: u32 = 0;
            const scheduler = TileScheduler.create(tile_scheduler.forwardArguments(params));
            while (tile_id < scheduler.totalBlocks()) : ({
                persistent_scheduler.prefetchNextWork();
                tile_id = persistent_scheduler.getCurrentWork();
            }) {
                const tile = scheduler.getCurrentWork(tile_id) orelse continue;
                const work = Work.init(params, tile);
                const batch = if (params.cu_seqlens_q == 0) work.batch else 0;
                const count = work.n_block_max - work.n_block_min;
                var index: u32 = 0;
                while (index < count) : (index += 1) {
                    const state: mbarrier.PipelineState(stages) = .{ .count = kv_index };
                    const stage = state.index();
                    if (elect.oneSync()) {
                        const row = work.sequence.offset_k + (work.n_block_min + index) * block_n;
                        if (state.count >= stages) mbarrier.waitSuspendHint(
                            &k_empty[stage],
                            state.phase() ^ 1,
                            tiling.suspend_time_hint_ns,
                        );
                        _ = k_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_n));
                        Layout.copyTile(
                            &tiles.stages[stage].k,
                            args.k,
                            &k_full[stage],
                            work.kv_head,
                            row,
                            batch,
                        );
                        if (index == 0) copyQuery(args, work, batch, processed);
                        if (state.count >= stages) mbarrier.waitSuspendHint(
                            &v_empty[stage],
                            state.phase() ^ 1,
                            tiling.suspend_time_hint_ns,
                        );
                        _ = v_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_n));
                        Layout.copyTile(
                            &tiles.stages[stage].v,
                            args.v,
                            &v_full[stage],
                            work.kv_head,
                            row,
                            batch,
                        );
                    }
                    elect.reconverge();
                    kv_index += 1;
                }
                if (count == 0 and elect.oneSync()) copyQuery(args, work, batch, processed);
                elect.reconverge();
                processed += 1;
            }
            persistent_scheduler.producerTail();
        }

        fn mmaScores(scores: *wgmma.Accumulator(block_n), query: u64, stage: u32) void {
            const key = Operands.kMajor(wgmma.address(&tiles.stages[stage].k));
            wgmma.fence();
            Operands.mmaKSteps(block_n, scores, k_steps, query, block_m, key, block_n);
            wgmma.commit();
        }

        fn mmaValues(
            output: *wgmma.Accumulator(columns),
            probabilities: Online.Probabilities,
            stage: u32,
        ) void {
            const value = Operands.mnMajor(wgmma.address(&tiles.stages[stage].v), block_n);
            wgmma.fence();
            inline for (0..block_n / 16) |key_step| {
                wgmma.mmaRegisters(
                    columns,
                    output,
                    probabilities[key_step],
                    wgmma.advance(value, key_step * 16 * swizzle_bytes),
                    true,
                    true,
                );
            }
            wgmma.commit();
        }

        fn exponentiate(
            softmax: *Online,
            scores: wgmma.Accumulator(block_n),
            masked: bool,
            n_block: u32,
            limits: ColLimits,
            softmax_scale_log2: f32,
        ) Online.Exponentials {
            var score_fragments: [block_n / 8]Accumulator = undefined;
            inline for (&score_fragments, 0..) |*fragment, tile| {
                fragment.* = .{
                    scores[tile * 4],     scores[tile * 4 + 1],
                    scores[tile * 4 + 2], scores[tile * 4 + 3],
                };
            }
            if (intra_wg_overlap and !masked) return Online.exponentials(
                softmax,
                score_fragments,
                softmax_scale_log2,
            );
            if (masked) applyMask(&score_fragments, n_block * block_n, limits, -std.math.inf(f32));
            return Online.exponentials(softmax, score_fragments, softmax_scale_log2);
        }

        fn rescaleOutput(
            output: *wgmma.Accumulator(columns),
            exponentials: Online.Exponentials,
        ) void {
            if (!exponentials.should_rescale) return;
            inline for (0..columns / 8) |tile| {
                inline for (
                    0..4,
                ) |element| output[tile * 4 + element] *= exponentials.row_scale[element];
            }
        }

        fn consumerWarp() u32 {
            return thread.warp() - producer_warps;
        }

        pub fn runConsumer(params: ForwardParams, _: Args) void {
            const group = nvptx.shflSync(
                .idx,
                u32,
                consumerWarp() / warps_per_group,
                0,
                nvptx.ShuffleClamp.fromWidth(.idx, 32),
                .all,
            );
            const query = Operands.kMajor(wgmma.address(&tiles.q) + group * 64 * swizzle_bytes);
            const turn: nvptx.NamedBarrier = @fromBackingInt(@intCast(1 + group));
            const successor: nvptx.NamedBarrier =
                @fromBackingInt(@intCast(1 + (group + 1) % tiling.consumerGroups()));
            const scheduler_barrier_threads = 2 * warps_per_group * 32;
            if (use_scheduler_barrier and group == 0) nvptx.barrierArriveAligned(
                turn,
                scheduler_barrier_threads,
            );
            var persistent_scheduler: PersistentScheduler = .create(params.tile_count_semaphore);
            var tile_id = persistent_scheduler.initialWorkTileInfo();
            var q_phase: u1 = 0;
            var kv_index: u32 = 0;
            const scheduler = TileScheduler.create(tile_scheduler.forwardArguments(params));
            while (tile_id < scheduler.totalBlocks()) : (tile_id =
                persistent_scheduler.advanceToNextWork())
            {
                const tile = scheduler.getCurrentWork(tile_id) orelse continue;
                const work = Work.init(params, tile);
                mbarrier.waitSuspendHint(&q_full, q_phase, tiling.suspend_time_hint_ns);
                q_phase ^= 1;
                const row = work.first_row + consumerWarp() * 16 + thread.lane() / 4;
                const limits = if (specialization.pack_gqa) work.colLimits(
                    row,
                ) else ColLimits.init(work.block_info, row);
                var softmax = Online.init();
                var output: wgmma.Accumulator(columns) = @splat(0);
                const count = work.n_block_max - work.n_block_min;
                if (count > 0) {
                    var stage = kv_index % stages;
                    var phase: u1 = @intCast((kv_index / stages) & 1);
                    var scores: wgmma.Accumulator(block_n) = @splat(0);
                    if (tiling.suspend_time_hint_ns == 0) {
                        while (!k_full[stage].tryWaitParity(phase)) {}
                    } else mbarrier.waitSuspendHint(
                        &k_full[stage],
                        phase,
                        tiling.suspend_time_hint_ns,
                    );
                    mmaScores(&scores, query, stage);
                    wgmma.wait(0);
                    wgmma.fenceOperand(&scores);
                    mbarrier.arrive(.thread, &k_empty[stage]);
                    if (count == 1) mbarrier.arrive(.thread, &q_empty);
                    var exponentials = exponentiate(
                        &softmax,
                        scores,
                        work.needsMask(work.n_block_min),
                        work.n_block_min,
                        limits,
                        params.softmax_scale_log2,
                    );
                    var probabilities = Online.pack(exponentials.values);
                    var previous_stage = stage;
                    var previous_phase = phase;
                    const phase_masks: []const ?bool =
                        if (tiling.separate_masking_iterations) &.{
                            true, false, true,
                        } else &.{null};
                    const phase_ends: [phase_masks.len]u32 =
                        if (tiling.separate_masking_iterations) ends: {
                            const unmasked_first, const unmasked_end = work.unmaskedBlocks();
                            const unmasked_begin =
                                std.math.clamp(unmasked_first -| work.n_block_min, 1, count);
                            break :ends .{ unmasked_begin, std.math.clamp(
                                unmasked_end -| work.n_block_min,
                                unmasked_begin,
                                count,
                            ), count };
                        } else .{count};
                    var index: u32 = 1;
                    inline for (phase_ends, phase_masks) |phase_end, phase_mask| {
                        while (index < phase_end) : (index += 1) {
                            const n_block = work.n_block_min + index;
                            const masked_block = phase_mask orelse work.needsMask(n_block);
                            const kv = kv_index + index;
                            stage = kv % stages;
                            phase = @intCast((kv / stages) & 1);
                            if (tiling.suspend_time_hint_ns == 0) {
                                while (!k_full[stage].tryWaitParity(phase)) {}
                            } else mbarrier.waitSuspendHint(
                                &k_full[stage],
                                phase,
                                tiling.suspend_time_hint_ns,
                            );
                            if (use_scheduler_barrier) nvptx.barrierSyncAligned(
                                turn,
                                scheduler_barrier_threads,
                            );
                            mmaScores(&scores, query, stage);
                            if (rescale_o_before_gemm) rescaleOutput(&output, exponentials);
                            if (tiling.suspend_time_hint_ns == 0) {
                                while (!v_full[previous_stage].tryWaitParity(previous_phase)) {}
                            } else mbarrier.waitSuspendHint(
                                &v_full[previous_stage],
                                previous_phase,
                                tiling.suspend_time_hint_ns,
                            );
                            mmaValues(&output, probabilities, previous_stage);
                            if (use_scheduler_barrier) nvptx.barrierArriveAligned(
                                successor,
                                scheduler_barrier_threads,
                            );
                            wgmma.wait(1);
                            wgmma.fenceOperand(&scores);
                            mbarrier.arrive(.thread, &k_empty[stage]);
                            if (index + 1 == count) mbarrier.arrive(.thread, &q_empty);
                            exponentials = exponentiate(
                                &softmax,
                                scores,
                                masked_block,
                                n_block,
                                limits,
                                params.softmax_scale_log2,
                            );
                            wgmma.wait(0);
                            wgmma.fenceOperand(&output);
                            mbarrier.arrive(.thread, &v_empty[previous_stage]);
                            probabilities = Online.pack(exponentials.values);
                            if (!rescale_o_before_gemm) rescaleOutput(&output, exponentials);
                            previous_stage = stage;
                            previous_phase = phase;
                        }
                    }
                    if (rescale_o_before_gemm and count > 1) rescaleOutput(&output, exponentials);
                    if (tiling.suspend_time_hint_ns == 0) {
                        while (!v_full[previous_stage].tryWaitParity(previous_phase)) {}
                    } else mbarrier.waitSuspendHint(
                        &v_full[previous_stage],
                        previous_phase,
                        tiling.suspend_time_hint_ns,
                    );
                    mmaValues(&output, probabilities, previous_stage);
                    wgmma.wait(0);
                    wgmma.fenceOperand(&output);
                    if (!tiling.overlap_so_sv) mbarrier.arrive(.thread, &v_empty[previous_stage]);
                    kv_index += count;
                }
                if (tiling.overlap_so_sv) {
                    const last_stage = (kv_index + stages - 1) % stages;
                    const staging: *addrspace(.shared) Staging = if (count == 0) @ptrCast(
                        &tiles.q,
                    ) else @ptrCast(&tiles.stages[last_stage].v);
                    nvptx.barrierSyncAligned(staging_barrier, consumer_warps * 32);
                    storeOutput(params, work, softmax, output, staging);
                    nvptx.fenceProxyAsync(.shared);
                    mbarrier.arrive(.thread, if (count == 0) &q_empty else &v_empty[last_stage]);
                } else {
                    if (count == 0) mbarrier.arrive(.thread, &q_empty);
                    storeOutput(params, work, softmax, output, {});
                }
            }
        }

        fn OutputCopier(comptime first_column: u32, comptime StagingPointer: type) type {
            return struct {
                o: [*]addrspace(.global) fragments.Chunk,
                origin: u64,
                row_stride: u64,
                head_stride: u64,
                first_row: u32,
                row_limit: u32,
                pack_gqa: @FieldType(Work, "pack_gqa"),
                staging: StagingPointer,

                pub fn copy(copier: @This(), tile_row: u32, chunk: u32) void {
                    const column = first_column + chunk;
                    const query_row = copier.first_row + consumerWarp() * 16 + tile_row;
                    if (query_row < copier.row_limit and column * 8 < head_dim) {
                        const row_offset = if (specialization.pack_gqa)
                            copier.pack_gqa.rowOffset(
                                query_row,
                                copier.row_stride,
                                copier.head_stride / 8,
                            )
                        else
                            @as(u64, query_row) * copier.row_stride;
                        copier.o[copier.origin / 8 + row_offset + column] = StagingLayout.chunk(
                            copier.staging,
                            consumerWarp() * 16 + tile_row,
                            chunk,
                            0,
                        ).*;
                    }
                }
            };
        }

        fn storeOutput(
            params: ForwardParams,
            work: Work,
            softmax: Online,
            output: wgmma.Accumulator(columns),
            staging_in_values: if (tiling.overlap_so_sv) *addrspace(.shared) Staging else void,
        ) void {
            const staging = if (tiling.overlap_so_sv) staging_in_values else &tiles.staging;
            const finalized = Online.finalize(softmax, params.softmax_scale_log2);
            const sequence = work.sequence;
            const row_limit = if (specialization.pack_gqa)
                work.pack_gqa.end_row
            else
                work.block_info.first_row + work.block_info.row_count;
            const o: [*]addrspace(.global) fragments.Chunk = @ptrFromInt(params.o);
            const origin = work.batch * params.o_batch_stride +
                @as(u64, sequence.offset_q) * params.o_row_stride +
                work.head * params.o_head_stride;
            const row_stride = params.o_row_stride / 8;
            const row = work.first_row + consumerWarp() * 16 + thread.lane() / 4;
            const chunks_per_pass = staging_columns / 8;
            inline for (0..columns / staging_columns) |pass| {
                inline for (0..chunks_per_pass) |chunk_index| {
                    inline for (.{ 0, 1 }) |half| {
                        const tile = pass * chunks_per_pass + chunk_index;
                        const words: [*]addrspace(.shared) u32 = @ptrCast(StagingLayout.chunk(
                            staging,
                            consumerWarp() * 16 + thread.lane() / 4 + half * 8,
                            chunk_index,
                            0,
                        ));
                        words[thread.lane() % 4] = @backingInt(nvptx.Bf16x2.fromF32(
                            output[tile * 4 + half * 2] * finalized.row_scale[half],
                            output[tile * 4 + half * 2 + 1] * finalized.row_scale[half],
                        ));
                    }
                }
                nvptx.barWarpSync(.all);
                const copier: OutputCopier(pass * chunks_per_pass, @TypeOf(staging)) = .{
                    .o = o,
                    .origin = origin,
                    .row_stride = row_stride,
                    .head_stride = params.o_head_stride,
                    .first_row = work.first_row,
                    .row_limit = row_limit,
                    .pack_gqa = work.pack_gqa,
                    .staging = staging,
                };
                smem.copyStagedRows(16, chunks_per_pass, 32, thread.lane, copier);
                nvptx.barWarpSync(.all);
            }
            if (elect.groupLeader(4)) fragments.storeLse(
                params,
                work,
                row,
                row_limit,
                finalized.lse,
            );
            elect.reconverge();
        }

        pub const Kernel = WarpSpecializedKernel(ForwardParams, @This());
    };
}

const variant = configurations.variants(.flash_fwd_sm90, config.head_dim)[config.variant];
const forward_tiling = variant.tiling;
const Forward = FlashAttnFwdSm90(forward_tiling, variant.specialization);

fn launch(params: ForwardParams, args: Forward.Args) void {
    if (variant.specialization.scheduler_metadata) nvptx.griddepcontrol(.wait);
    var divmod_params = params;
    if (variant.specialization.pack_gqa) {
        divmod_params.qhead_per_kvhead_divmod = .init(params.qhead_per_kvhead);
    }
    Forward.Kernel.run(divmod_params, args);
}

const Entry = entry.Entry(ForwardParams, Forward.Args, .{
    "q_map",             "k_map",            "v_map",              "o",
    "lse",               "cu_seqlens_q",     "cu_seqlens_k",       "seqlen_q",
    "seqlen_k",          "qhead_per_kvhead", "softmax_scale_log2", "window_size_left",
    "window_size_right", "num_m_blocks",     "num_head",           "num_batch",
} ++ entry.strides("o") ++ .{ "lse_batch_stride", "lse_head_stride", "tile_count_semaphore" } ++
    if (variant.specialization.scheduler_metadata) .{"scheduler_metadata"} else .{}, .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
}, launch);

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_fwd" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{
                .name = "flash_attn_fwd_prepare_varlen_num_blocks",
            });
        }
    }
}
