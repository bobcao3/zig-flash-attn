const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const config = @import("config");
const configurations = @import("configurations.zig");
const Sm80FwdTiling = configurations.Sm80FwdTiling;
const ForwardParams = @import("ForwardParams.zig");
const FlashAttnFwdSm80Kernel = @import("flash_fwd_kernel_sm80.zig").FlashAttnFwdSm80Kernel;
const m_block_work = @import("m_block_work.zig");
const tile_scheduler = @import("tile_scheduler.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");
const fragments = @import("fragments.zig");
const applyMask = @import("mask.zig").applyMask;
const ColLimits = @import("BlockInfo.zig").ColLimits;
const smem = @import("smem.zig");
const elect = @import("elect.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const chunk_elements = Sm80FwdTiling.chunk_elements;

const Span = struct {
    first: u32,
    end: u32,
};

pub fn SharedStorage(comptime tiling: Sm80FwdTiling, comptime specialization: configurations.Specialization) type {
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const threads = tiling.threads();
    const warps = tiling.warps();
    const query_in_stage = tiling.shareQvSmem();
    const Rows = smem.Layout(row_chunks);
    const PackGqaRows = if (specialization.pack_gqa) m_block_work.PackGqa else void;

    return struct {
        pub const Layout = Rows;
        pub const shared_bytes = tiling.sharedBytes();
        pub const Args = struct {};

        const Stage = struct {
            k: Rows.Tile(block_n),
            v: Rows.Tile(block_n),
        };
        const Tiles = struct {
            stages: [stages]Stage,
            q: if (query_in_stage) void else Rows.Tile(block_m),
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var k_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var v_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var v_empty: [if (tiling.separate_v_empty) stages else 0]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub fn queryTile() *addrspace(.shared) Rows.Tile(block_m) {
            return if (query_in_stage) @ptrCast(&tiles.stages[stages - 1]) else &tiles.q;
        }

        pub fn kTile(stage: u32) *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.stages[stage].k;
        }

        pub fn vTile(stage: u32) *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.stages[stage].v;
        }

        pub const Load = struct {
            q: [*]addrspace(.global) const Chunk,
            k: [*]addrspace(.global) const Chunk,
            v: [*]addrspace(.global) const Chunk,
            q_stride: u64,
            k_stride: u64,
            v_stride: u64,
            first_tile: u32,
            tile_count: u32,
            first_row: u32,
            seqlen_q: u32,
            seqlen_k: u32,
            q_head_stride: if (specialization.pack_gqa) u64 else void,
            pack_gqa: PackGqaRows,

            pub fn init(params: ForwardParams, args: Args, work: anytype) Load {
                _ = args;
                const tile = work.tile;
                const sequence = tile.sequence;
                const q: [*]addrspace(.global) const Chunk = @ptrFromInt(params.q);
                const k: [*]addrspace(.global) const Chunk = @ptrFromInt(params.k);
                const v: [*]addrspace(.global) const Chunk = @ptrFromInt(params.v);
                const q_origin = tile.batch * params.q_batch_stride +
                    @as(u64, sequence.offset_q) * params.q_row_stride + tile.head * params.q_head_stride;
                const k_origin = tile.batch * params.k_batch_stride +
                    @as(u64, sequence.offset_k) * params.k_row_stride + tile.kv_head * params.k_head_stride;
                const v_origin = tile.batch * params.v_batch_stride +
                    @as(u64, sequence.offset_k) * params.v_row_stride + tile.kv_head * params.v_head_stride;
                return .{
                    .q = q + q_origin / chunk_elements,
                    .k = k + k_origin / chunk_elements,
                    .v = v + v_origin / chunk_elements,
                    .q_stride = params.q_row_stride / chunk_elements,
                    .k_stride = params.k_row_stride / chunk_elements,
                    .v_stride = params.v_row_stride / chunk_elements,
                    .first_tile = tile.n_block_min,
                    .tile_count = tile.n_block_max - tile.n_block_min,
                    .first_row = tile.first_row,
                    .seqlen_q = sequence.seqlen_q,
                    .seqlen_k = sequence.seqlen_k,
                    .q_head_stride = if (specialization.pack_gqa) params.q_head_stride / chunk_elements else {},
                    .pack_gqa = tile.pack_gqa,
                };
            }

            fn copyRows(
                comptime count: u32,
                destination: *addrspace(.shared) Rows.Tile(count),
                source: [*]addrspace(.global) const Chunk,
                stride: u64,
                first_row: u32,
                limit: u32,
                packing: anytype,
            ) void {
                const chunks = count * row_chunks;
                inline for (0..(chunks + threads - 1) / threads) |pass| {
                    const linear: u32 = @as(u32, pass) * threads + thread.id();
                    if ((pass + 1) * threads <= chunks or linear < chunks) {
                        const row = linear / row_chunks;
                        const column = linear % row_chunks;
                        const source_row = @min(first_row + row, limit - 1);
                        const offset = if (@TypeOf(packing) == void)
                            source_row * stride
                        else
                            packing.pack_gqa.rowOffset(source_row, stride, packing.head_stride);
                        cp_async.copy(
                            .cache_global_level,
                            &destination[Rows.slot(row, column)],
                            &source[offset + column],
                        );
                    }
                }
            }

            fn copyKeys(load: *const Load, n_block: u32, stage: u32) void {
                copyRows(block_n, kTile(stage), load.k, load.k_stride, n_block * block_n, load.seqlen_k, {});
            }

            fn copyValues(load: *const Load, n_block: u32, stage: u32) void {
                copyRows(block_n, vTile(stage), load.v, load.v_stride, n_block * block_n, load.seqlen_k, {});
            }

            fn copyTile(load: *const Load, n_block: u32, stage: u32) void {
                load.copyKeys(n_block, stage);
                mbarrier.arrive(.cp_async, &k_full[stage]);
                load.copyValues(n_block, stage);
                mbarrier.arrive(.cp_async, &v_full[stage]);
            }

            pub fn start(load: *const Load) void {
                if (thread.warp() == 0 and elect.oneSync()) {
                    inline for (&k_full, &v_full, &empty) |*k_stage, *v_stage, *empty_stage| {
                        k_stage.init(threads);
                        v_stage.init(threads);
                        empty_stage.init(warps);
                    }
                    inline for (&v_empty) |*v_empty_stage| v_empty_stage.init(warps);
                    mbarrier.pipelineInitFence();
                }
                elect.reconverge();
                if (specialization.pack_gqa) {
                    const packing = .{ .pack_gqa = load.pack_gqa, .head_stride = load.q_head_stride };
                    copyRows(block_m, queryTile(), load.q, load.q_stride, load.first_row, load.pack_gqa.end_row, packing);
                } else {
                    copyRows(block_m, queryTile(), load.q, load.q_stride, load.first_row, load.seqlen_q, {});
                }
                cp_async.commitGroup();
                comptime std.debug.assert(tiling.prefetch_stages >= 1);
                if (0 < load.tile_count) load.copyKeys(load.first_tile, 0);
                mbarrier.pipelineInitWait(.cta, threads, .other);
                if (0 < load.tile_count) {
                    mbarrier.arrive(.cp_async, &k_full[0]);
                    load.copyValues(load.first_tile, 0);
                    mbarrier.arrive(.cp_async, &v_full[0]);
                }
                inline for (1..tiling.prefetch_stages) |prefetched| {
                    const first: u32 = prefetched;
                    if (first < load.tile_count) load.copyTile(load.first_tile + first, first);
                }
                cp_async.waitGroup(0);
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn queryConsumed(load: *const Load) void {
                _ = load;
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn acquireK(load: *const Load, index: u32) u32 {
                const refill = index + tiling.prefetch_stages;
                if (!tiling.separate_v_empty and refill < load.tile_count) {
                    const refill_stage = refill % stages;
                    if (refill >= stages) {
                        const uses = refill / stages;
                        while (!empty[refill_stage].testWaitParity(@intCast((uses - 1) & 1))) {}
                    }
                    load.copyTile(load.first_tile + refill, refill_stage);
                }
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&k_full);
                return state.index();
            }

            pub fn acquireV(load: *const Load, index: u32) void {
                _ = load;
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&v_full);
            }

            pub fn release(load: *const Load, index: u32) void {
                _ = load;
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerRelease(.warp, &empty);
            }

            fn copyReleased(load: *const Load, index: u32, released: *addrspace(.shared) [stages]nvptx.Mbarrier, full: *addrspace(.shared) [stages]nvptx.Mbarrier, copy: fn (*const Load, u32, u32) void) void {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerRelease(.warp, released);
                const next = index + stages;
                if (next < load.tile_count) {
                    mbarrier.wait(.cta, &released[state.index()], state.phase());
                    copy(load, load.first_tile + next, state.index());
                    mbarrier.arrive(.cp_async, &full[state.index()]);
                }
            }

            pub fn releaseKeys(load: *const Load, index: u32) void {
                load.copyReleased(index, &empty, &k_full, copyKeys);
            }

            pub fn releaseValues(load: *const Load, index: u32) void {
                load.copyReleased(index, &v_empty, &v_full, copyValues);
            }
        };
    };
}

pub fn FlashAttnFwdSm80(comptime tiling: Sm80FwdTiling, comptime specialization: configurations.Specialization, comptime Storage: type) type {
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const threads = tiling.threads();
    const k_steps = tiling.head_dim / 16;
    const narrow_step = tiling.head_dim % 16 == 8;
    const mma_m_per_warp = tiling.mma_m_per_warp;
    const warp_rows = 16 * mma_m_per_warp;
    const Layout = Storage.Layout;
    const Tile = m_block_work.MBlockWork(block_m, block_n, .{ .pack_gqa = specialization.pack_gqa });
    const scheduler_options: tile_scheduler.Options = .{
        .block_m = block_m,
        .row_bytes = 4 * tiling.head_dim,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = tiling.l2_swizzle,
        .scheduler_metadata = specialization.scheduler_metadata,
        .batch_major = tiling.batch_major,
    };
    const TileScheduler = tile_scheduler.TileScheduler(scheduler_options);
    const PersistentScheduler = tile_scheduler.PersistentScheduler(tiling.scheduling_mode, .{ .axis = .x, .ctas = 1 });
    const Online = fragments.Softmax(block_n);

    return struct {
        pub const thread_count = threads;
        pub const max_registers = tiling.max_registers;
        pub const separate_v_empty = tiling.separate_v_empty;
        pub const shared_bytes = Storage.shared_bytes;
        pub const Args = Storage.Args;
        pub const Load = Storage.Load;
        pub const PrepareScheduler = prepare_scheduler.PrepareScheduler(scheduler_options.block_m, scheduler_options.block_order, specialization.sort_batches);

        pub const Work = struct {
            tile: Tile,

            pub const all_groups: Span = .{ .first = 0, .end = block_n / 16 };

            pub fn init(params: ForwardParams) ?Work {
                const scheduler = TileScheduler.create(tile_scheduler.forwardArguments(params));
                const persistent_scheduler: PersistentScheduler = .create();
                const tile = scheduler.getGridWork(tile_scheduler.gridTile(.x, 1, persistent_scheduler.initialWorkTileInfo())) orelse return null;
                return .{ .tile = Tile.init(params, tile) };
            }

            pub fn visibleGroups(work: *const Work, n_block: u32) Span {
                const first_key: i32 = @intCast(n_block * block_n);
                const info = work.tile.block_info;
                const row = work.tile.first_row + thread.warp() * warp_rows;
                const first_query: i32 = @intCast(if (specialization.pack_gqa) work.tile.pack_gqa.mIdx(row) else row);
                const last_query: i32 = if (specialization.pack_gqa) @intCast(work.tile.pack_gqa.mIdx(row + warp_rows - 1)) else first_query + warp_rows - 1;
                const col_limit_left = info.colLimitLeft(first_query);
                const col_limit_right = info.colLimitRight(last_query);
                const first = @max(0, @divFloor(col_limit_left - first_key, 16));
                const end = @min(block_n / 16, @divFloor(col_limit_right - first_key + 15, 16));
                return .{ .first = @intCast(first), .end = @intCast(@max(first, end)) };
            }
        };

        pub const Qk = struct {
            pub const Query = if (tiling.q_in_regs) [mma_m_per_warp]struct {
                wide: [k_steps]@Vector(4, u32),
                narrow: if (narrow_step) @Vector(2, u32) else void,
            } else void;
            pub const Scores = [mma_m_per_warp]Online.Scores;

            fn queryAddresses() Layout.Addresses(block_m) {
                const lane = thread.lane();
                return .init(thread.warp() * warp_rows + lane % 16, lane / 16);
            }

            pub fn loadQuery() Query {
                if (!tiling.q_in_regs) return;
                const query_addresses = queryAddresses();
                var query: Query = undefined;
                inline for (&query, 0..) |*rows, row_tile| {
                    inline for (&rows.wide, 0..) |*fragment, k_step| {
                        fragment.* = nvptx.ldmatrix(.x4, false, query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_step * 2));
                    }
                    if (narrow_step) rows.narrow = nvptx.ldmatrix(
                        .x2,
                        false,
                        query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_steps * 2),
                    );
                }
                return query;
            }

            pub fn scores(query: Query, stage: u32, visible: Span) Scores {
                @setEvalBranchQuota(100_000);
                const lane = thread.lane();
                const query_addresses = queryAddresses();
                const keys = Layout.Addresses(block_n).init(lane % 8 + lane / 16 * 8, lane / 8 % 2);
                const tile = Storage.kTile(stage);
                var result: Scores = @splat(@splat(@splat(0)));
                inline for (0..k_steps) |k_step_index| {
                    const k_step: u32 = k_step_index;
                    var a: [mma_m_per_warp]@Vector(4, u32) = undefined;
                    inline for (&a, 0..) |*fragment, row_tile| {
                        fragment.* = if (tiling.q_in_regs) query[row_tile].wide[k_step] else nvptx.ldmatrix(
                            .x4,
                            false,
                            query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_step * 2),
                        );
                    }
                    inline for (0..block_n / 16) |j| {
                        if (j >= visible.first and j < visible.end) {
                            const b = nvptx.ldmatrix(.x4, false, keys.chunk(tile, j * 16, k_step * 2));
                            const low: @Vector(2, u32) = .{ b[0], b[1] };
                            const high: @Vector(2, u32) = .{ b[2], b[3] };
                            inline for (&result, a) |*rows, fragment| {
                                rows[2 * j] = nvptx.mmaSync(fragments.mma_m16n8k16, fragment, low, rows[2 * j]);
                                rows[2 * j + 1] = nvptx.mmaSync(fragments.mma_m16n8k16, fragment, high, rows[2 * j + 1]);
                            }
                        }
                    }
                }
                if (narrow_step) {
                    var a: [mma_m_per_warp]@Vector(2, u32) = undefined;
                    inline for (&a, 0..) |*fragment, row_tile| {
                        fragment.* = if (tiling.q_in_regs) query[row_tile].narrow else nvptx.ldmatrix(
                            .x2,
                            false,
                            query_addresses.chunk(Storage.queryTile(), row_tile * 16, k_steps * 2),
                        );
                    }
                    const narrow_keys = Layout.Addresses(block_n).init(lane % 16, 0);
                    inline for (0..block_n / 16) |j| {
                        if (j >= visible.first and j < visible.end) {
                            const b = nvptx.ldmatrix(.x2, false, narrow_keys.chunk(tile, j * 16, k_steps * 2));
                            inline for (&result, a) |*rows, fragment| {
                                rows[2 * j] = nvptx.mmaSync(fragments.mma_m16n8k8, fragment, @Vector(1, u32){b[0]}, rows[2 * j]);
                                rows[2 * j + 1] = nvptx.mmaSync(fragments.mma_m16n8k8, fragment, @Vector(1, u32){b[1]}, rows[2 * j + 1]);
                            }
                        }
                    }
                }
                return result;
            }
        };

        pub const Softmax = struct {
            pub const Running = [mma_m_per_warp]Online;
            pub const Step = [mma_m_per_warp]Online.Step;

            pub fn init() Running {
                var running: Running = undefined;
                inline for (&running) |*rows| rows.* = Online.init();
                return running;
            }

            pub fn step(running: *Running, scores: Qk.Scores, softmax_scale_log2: f32) Step {
                var result: Step = undefined;
                inline for (&result, running, scores) |*rows, *rows_running, rows_scores| {
                    rows.* = Online.step(rows_running, rows_scores, softmax_scale_log2);
                }
                return result;
            }

            pub fn mask(scores: *Qk.Scores, n_block: u32, work: *const Work) void {
                inline for (scores, 0..) |*rows, row_tile| {
                    const row = work.tile.first_row + thread.warp() * warp_rows + @as(u32, row_tile) * 16 + thread.lane() / 4;
                    const limits = if (specialization.pack_gqa) work.tile.colLimits(row) else ColLimits.init(work.tile.block_info, row);
                    applyMask(rows, n_block * block_n, limits, -std.math.inf(f32));
                }
            }
        };

        pub const Pv = struct {
            pub const Output = [mma_m_per_warp][row_chunks]Accumulator;

            pub fn zero() Output {
                return @splat(@splat(@splat(0)));
            }

            pub fn accumulate(output: *Output, step: Softmax.Step, stage: u32, visible: Span) void {
                @setEvalBranchQuota(100_000);
                const lane = thread.lane();
                const pairs = Layout.Addresses(block_n).init(lane % 8 + lane / 8 % 2 * 8, lane / 16);
                inline for (output, step) |*rows, rows_step| {
                    if (rows_step.should_rescale) {
                        inline for (rows) |*fragment| fragment.* *= rows_step.row_scale;
                    }
                }
                const tile = Storage.vTile(stage);
                inline for (0..block_n / 16) |k_step| {
                    if (k_step >= visible.first and k_step < visible.end) {
                        inline for (0..row_chunks / 2) |pair| {
                            const j: u32 = pair;
                            const b = nvptx.ldmatrix(.x4, true, pairs.chunk(tile, k_step * 16, j * 2));
                            const low: @Vector(2, u32) = .{ b[0], b[1] };
                            const high: @Vector(2, u32) = .{ b[2], b[3] };
                            inline for (output, step) |*rows, rows_step| {
                                const a = rows_step.probabilities[k_step];
                                rows[2 * j] = nvptx.mmaSync(fragments.mma_m16n8k16, a, low, rows[2 * j]);
                                rows[2 * j + 1] = nvptx.mmaSync(fragments.mma_m16n8k16, a, high, rows[2 * j + 1]);
                            }
                        }
                        if (row_chunks % 2 == 1) {
                            const tail = Layout.Addresses(block_n).init(lane % 16, 0);
                            const b = nvptx.ldmatrix(.x2, true, tail.chunk(tile, k_step * 16, row_chunks - 1));
                            inline for (output, step) |*rows, rows_step| {
                                rows[row_chunks - 1] = nvptx.mmaSync(
                                    fragments.mma_m16n8k16,
                                    rows_step.probabilities[k_step],
                                    b,
                                    rows[row_chunks - 1],
                                );
                            }
                        }
                    }
                }
            }
        };

        pub const Epilogue = struct {
            pub fn store(
                params: ForwardParams,
                work: *const Work,
                softmax: Softmax.Running,
                output: Pv.Output,
            ) void {
                var finalized: [mma_m_per_warp]Online.Finalized = undefined;
                inline for (&finalized, softmax) |*rows, rows_softmax| rows.* = Online.finalize(rows_softmax, params.softmax_scale_log2);
                const tile = work.tile;
                const sequence = tile.sequence;
                const warp = thread.warp();
                const lane = thread.lane();
                const lane_word = lane % 4;
                const staging = Storage.queryTile();
                nvptx.barrierSyncAligned(.work_group, null);
                inline for (output, finalized, 0..) |rows, rows_finalized, row_tile| {
                    inline for (rows, 0..) |fragment, n_tile| {
                        inline for (.{ 0, 1 }) |half| {
                            const words: [*]addrspace(.shared) u32 = @ptrCast(Layout.chunk(
                                staging,
                                warp * warp_rows + @as(u32, row_tile) * 16 + lane / 4 + half * 8,
                                n_tile,
                                lane / 4,
                            ));
                            words[lane_word] = @backingInt(nvptx.Bf16x2.fromF32(
                                fragment[half * 2] * rows_finalized.row_scale[half],
                                fragment[half * 2 + 1] * rows_finalized.row_scale[half],
                            ));
                        }
                    }
                }
                nvptx.barWarpSync(.all);
                const o: [*]addrspace(.global) Chunk = @ptrFromInt(params.o);
                const origin = tile.batch * params.o_batch_stride +
                    @as(u64, sequence.offset_q) * params.o_row_stride + tile.head * params.o_head_stride;
                const row_stride = params.o_row_stride / chunk_elements;
                const row_limit = if (specialization.pack_gqa) tile.pack_gqa.end_row else sequence.seqlen_q;
                inline for (0..(warp_rows * row_chunks + 31) / 32) |pass| {
                    const linear: u32 = @as(u32, pass) * 32 + lane;
                    if ((pass + 1) * 32 <= warp_rows * row_chunks or linear < warp_rows * row_chunks) {
                        const tile_row = linear / row_chunks;
                        const column = linear % row_chunks;
                        const query_row = tile.first_row + warp * warp_rows + tile_row;
                        if (query_row < row_limit) {
                            const row_offset = if (specialization.pack_gqa)
                                tile.pack_gqa.rowOffset(query_row, row_stride, params.o_head_stride / chunk_elements)
                            else
                                @as(u64, query_row) * row_stride;
                            o[origin / chunk_elements + row_offset + column] =
                                Layout.chunk(staging, warp * warp_rows + tile_row, column, tile_row).*;
                        }
                    }
                }
                if (elect.groupLeader(4)) {
                    inline for (finalized, 0..) |rows_finalized, row_tile| {
                        fragments.storeLse(params, tile, tile.first_row + warp * warp_rows + @as(u32, row_tile) * 16 + lane / 4, row_limit, rows_finalized.lse);
                    }
                }
            }
        };

        pub const Kernel = FlashAttnFwdSm80Kernel(@This());
    };
}

const variant = configurations.forwardSm80Variant(config.head_dim, @field(configurations.Mode, config.mode)).?;
const forward_tiling = variant.tiling;
const Forward = FlashAttnFwdSm80(forward_tiling, variant.specialization, SharedStorage(forward_tiling, variant.specialization));

fn flashAttnFwd(
    q: u64,
    k: u64,
    v: u64,
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
    q_batch_stride: u64,
    q_row_stride: u64,
    q_head_stride: u64,
    k_batch_stride: u64,
    k_row_stride: u64,
    k_head_stride: u64,
    v_batch_stride: u64,
    v_row_stride: u64,
    v_head_stride: u64,
    o_batch_stride: u64,
    o_row_stride: u64,
    o_head_stride: u64,
    lse_batch_stride: u64,
    lse_head_stride: u64,
    num_batch: u32,
    scheduler_metadata: if (variant.specialization.scheduler_metadata) u64 else void,
) callconv(.{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
} }) void {
    @setEvalBranchQuota(100_000);
    Forward.Kernel.run(.{
        .q = q,
        .k = k,
        .v = v,
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
        .q_batch_stride = q_batch_stride,
        .q_row_stride = q_row_stride,
        .q_head_stride = q_head_stride,
        .k_batch_stride = k_batch_stride,
        .k_row_stride = k_row_stride,
        .k_head_stride = k_head_stride,
        .v_batch_stride = v_batch_stride,
        .v_row_stride = v_row_stride,
        .v_head_stride = v_head_stride,
        .o_batch_stride = o_batch_stride,
        .o_row_stride = o_row_stride,
        .o_head_stride = o_head_stride,
        .lse_batch_stride = lse_batch_stride,
        .lse_head_stride = lse_head_stride,
        .num_m_blocks = nvptx.nctaid(0),
        .num_head = nvptx.nctaid(1),
        .num_batch = num_batch,
        .scheduler_metadata = if (variant.specialization.scheduler_metadata) scheduler_metadata else 0,
    }, .{});
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&flashAttnFwd, .{ .name = "flash_attn_fwd" });
        if (variant.specialization.scheduler_metadata) {
            @export(&Forward.PrepareScheduler.prepareVarlenNumBlocks, .{ .name = "flash_attn_fwd_prepare_varlen_num_blocks" });
        }
    }
}
