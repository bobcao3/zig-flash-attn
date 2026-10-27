const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const tma = @import("tma.zig");
const smem = @import("smem.zig");
const mbarrier = @import("mbarrier.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");
const forward = @import("inter_sequence_forward.zig");
const BlockInfo = @import("BlockInfo.zig");
const cluster = @import("cluster.zig");
const SchedulingMode = @import("tile_scheduler.zig").SchedulingMode;

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const Mbarrier = nvptx.Mbarrier;
const Params = forward.Params;
const Work = forward.Work;
const Trace = forward.Trace;
const TileScheduler = forward.TileScheduler;
const layout = configurations.inter_sequence_sm120;
const block_m = layout.block_m;
const block_n = layout.block_n;
const slot_rows = layout.slot_rows;
const consumer_warps = layout.consumer_warps;
const consumer_threads = consumer_warps * nvptx.warp_size;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const consumer_barrier: nvptx.NamedBarrier = @fromBackingInt(1);

pub const Args = struct {
    q: *const tma.TensorMap,
    k: *const tma.TensorMap,
    v: *const tma.TensorMap,
};

pub const names = names: {
    const maps = [_][:0]const u8{ "q_map", "k_map", "v_map" };
    var result: [maps.len + forward.names.len - maps.len][:0]const u8 = undefined;
    result[0..maps.len].* = maps;
    var count = maps.len;
    for (forward.names) |name| {
        if (std.mem.eql(u8, name, "q") or std.mem.eql(u8, name, "k") or
            std.mem.eql(u8, name, "v")) continue;
        result[count] = name;
        count += 1;
    }
    const final = result;
    break :names &final;
};

extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
var producer_tile_idx: u32 addrspace(.shared) = undefined;
var arrived_splits: u32 addrspace(.shared) = undefined;
var cancel_response: @Vector(4, u32) align(16) addrspace(.shared) = undefined;
var cancel_full: Mbarrier addrspace(.shared) = undefined;

pub fn Ring(comptime count: u32) type {
    return struct {
        index: u32 = 0,
        phase: u32 = 0,

        const State = @This();

        pub fn advance(state: *State) void {
            const wraps = state.index + 1 == count;
            state.index = if (wraps) 0 else state.index + 1;
            state.phase ^= @intFromBool(wraps);
        }

        pub fn next(state: State) State {
            var copy = state;
            copy.advance();
            return copy;
        }
    };
}

pub const Published = struct {
    work: Work,
    valid: bool,
};

pub fn Kernel(comptime head_dim: u32, comptime packs: bool, comptime scheduling: SchedulingMode) type {
    const row_chunks = head_dim / chunk_elements;
    const Rows = tma.Layout(row_chunks);
    const Slot = Rows.Tile(slot_rows);
    const slot_bytes = @sizeOf(Slot);
    const slots = layout.slots(head_dim);
    const Online = fragments.Softmax(block_n);
    const Query = [head_dim / 16]@Vector(4, u32);
    const Output = [row_chunks]Accumulator;
    const Cursor = forward.NBlockCursor(block_n);
    const Rows64 = forward.QueryRows(packs);
    const SlotRing = Ring(slots);
    const WorkRing = Ring(layout.work_slots);

    comptime {
        std.debug.assert(slot_bytes == layout.slotBytes(head_dim));
        std.debug.assert(block_m == 2 * slot_rows and block_n == slot_rows);
        std.debug.assert(Rows.swizzle_columns == layout.box_columns / chunk_elements);
    }

    return struct {
        pub const threads = layout.threads();
        pub const shared_bytes = slots * slot_bytes;
        pub const max_registers = configurations.maxRegistersPerThread(threads);
        const clc = scheduling == .clc;
        const no_tile = std.math.maxInt(u32);

        comptime {
            const producer_registers = layout.producer_registers * layout.producer_warps;
            const consumer_registers = layout.consumer_registers * consumer_warps;
            std.debug.assert((producer_registers + consumer_registers) * nvptx.warp_size <=
                configurations.registers_per_cta);
        }

        var full: [slots]Mbarrier addrspace(.shared) = undefined;
        var empty: [slots]Mbarrier addrspace(.shared) = undefined;
        var work_full: [layout.work_slots]Mbarrier addrspace(.shared) = undefined;
        var work_empty: [layout.work_slots]Mbarrier addrspace(.shared) = undefined;
        var published: [layout.work_slots]Published addrspace(.shared) = undefined;

        fn slot(index: u32) *addrspace(.shared) Slot {
            const tiles: [*]addrspace(.shared) Slot = @ptrCast(@alignCast(&shared_storage));
            return &tiles[index];
        }

        pub fn run(params: Params, args: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            trace.begin(.setup);
            const producer_warp = consumer_warps;
            const producer = thread.warp() >= producer_warp;
            if (thread.warp() == producer_warp and elect.oneSync()) {
                for (&full, &empty) |*full_slot, *empty_slot| {
                    full_slot.init(1);
                    empty_slot.init(consumer_warps);
                }
                for (&work_full, &work_empty) |*full_slot, *empty_slot| {
                    full_slot.init(1);
                    empty_slot.init(consumer_warps);
                }
                cancel_full.init(1);
                mbarrier.pipelineInitFence();
                tma.prefetchMaps(args);
            }
            elect.reconverge();
            nvptx.griddepcontrol(.wait);
            if (thread.warp() == producer_warp) {
                const ctas = if (clc) params.num_sm else nvptx.nctaid(0);
                const scheduler: TileScheduler = .initForCtas(params, packs, block_m, false, ctas);
                if (thread.lane() == 0) forward.tile_scheduler = scheduler;
            }
            nvptx.barrierSyncAligned(.work_group, null);
            trace.end(.setup);
            if (producer) {
                asm volatile (std.fmt.comptimePrint(
                        "setmaxnreg.dec.sync.aligned.u32 {d};",
                        .{layout.producer_registers},
                    ));
                if (thread.warp() == producer_warp and thread.lane() == 0) produce(params, args);
                nvptx.barWarpSync(.all);
            } else {
                asm volatile (std.fmt.comptimePrint(
                        "setmaxnreg.inc.sync.aligned.u32 {d};",
                        .{layout.consumer_registers},
                    ));
                consume(params, &trace);
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (!clc and thread.id() == 0) forward.TileCountSemaphore.finishCta(params);
        }

        fn issue(ring: *SlotRing, map: *const tma.TensorMap, head: u32, row: u32) void {
            const index = ring.index;
            mbarrier.wait(.cta, &empty[index], ring.phase ^ 1);
            _ = full[index].arriveExpectTx(slot_bytes);
            Rows.copyTile(slot(index), map, &full[index], head, row, 0);
            ring.advance();
        }

        fn publish(works: *WorkRing, located: ?Work) void {
            mbarrier.wait(.cta, &work_empty[works.index], works.phase ^ 1);
            published[works.index] = .{ .work = located orelse undefined, .valid = located != null };
            _ = work_full[works.index].arrive();
            works.advance();
        }

        /// Issues the loads of the consumers in the order in which they take them: for each
        /// work tile, the two halves of the query tile, then the key and value tile of each key
        /// block. One thread runs it.
        fn produce(params: Params, args: Args) void {
            const scheduler = &forward.tile_scheduler;
            const tile_count = scheduler.tileCount(.cta);
            const packing = if (packs) scheduler.packing.divisor else 1;
            var ring: SlotRing = .{};
            var works: WorkRing = .{};
            var tiles: TileSource = .{};
            while (true) {
                const tile_idx = tiles.next(params, tile_count);
                const located = if (tile_idx < tile_count)
                    scheduler.locate(params, .cta, tile_idx)
                else
                    null;
                publish(&works, located);
                const work = located orelse return;
                const rows: Rows64 = .init(work, block_m);
                const kv_head = scheduler.qhead_per_kvhead.divide(rows.head);
                const q_token = rows.first_query + rows.first_row / packing;
                issue(&ring, args.q, rows.head, q_token);
                issue(&ring, args.q, rows.head, q_token + slot_rows / packing);
                var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
                if (work.num_splits > 1) {
                    const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                    cursor.skip(params, first_n_block);
                    cursor.limit(n_blocks);
                }
                while (cursor.valid()) {
                    issue(&ring, args.k, kv_head, cursor.firstKey());
                    issue(&ring, args.v, kv_head, cursor.firstKey());
                    cursor.advance(params);
                }
                tiles.finishTile();
            }
        }

        /// Hands out the tile indices of one CTA. With CLC, the CTA first takes its own ctaid and
        /// then the ctaid of each CTA of the grid that it cancels before that CTA starts (PTX ISA
        /// 9.7.15.18). The request for a tile leaves when the loads of the previous tile are
        /// issued, and no request may follow one that failed.
        const TileSource = struct {
            requests: u32 = 0,

            fn request(tiles: *TileSource) void {
                if (tiles.requests > 0) nvptx.fenceProxyAsync(.shared);
                _ = cancel_full.arriveExpectTx(@sizeOf(@TypeOf(cancel_response)));
                cluster.tryCancel(&cancel_response, &cancel_full, false);
                tiles.requests += 1;
            }

            fn take(tiles: *TileSource, params: Params) u32 {
                if (!clc) {
                    const fetch: forward.NextWork = .{
                        .leader = true,
                        .tile_idx = &producer_tile_idx,
                    };
                    fetch.prefetch(params, .cta);
                    return producer_tile_idx;
                }
                if (tiles.requests == 0) return @workGroupId(0);
                mbarrier.wait(.cta, &cancel_full, tiles.requests - 1);
                return cluster.firstCtaid(&cancel_response, 0);
            }

            /// Returns the next tile index below `tile_count`, or an index at or past it when no
            /// tile is left. The grid launches CTAs past the tile count, which only the device
            /// knows, and their indices are skipped.
            fn next(tiles: *TileSource, params: Params, tile_count: u32) u32 {
                var tile_idx = tiles.take(params);
                while (clc and tile_idx >= tile_count and tile_idx != no_tile) {
                    tiles.request();
                    tile_idx = tiles.take(params);
                }
                return tile_idx;
            }

            fn finishTile(tiles: *TileSource) void {
                if (clc) tiles.request();
            }
        };

        fn release(ring: SlotRing) void {
            if (thread.lane() == 0) _ = empty[ring.index].arrive();
        }

        fn consume(params: Params, trace: *Trace) void {
            var position: SlotRing = .{};
            var works: WorkRing = .{};
            while (true) {
                trace.begin(.get_next_work);
                mbarrier.wait(.cta, &work_full[works.index], works.phase);
                const entry = published[works.index];
                nvptx.barWarpSync(.all);
                if (thread.lane() == 0) _ = work_empty[works.index].arrive();
                works.advance();
                trace.end(.get_next_work);
                if (!entry.valid) break;
                trace.begin(.tile);
                process(params, entry.work, &position, trace);
                trace.end(.tile);
            }
        }

        const Lanes = struct {
            k: XorAddresses,
            v: XorAddresses,

            fn init() Lanes {
                const lane = thread.lane();
                return .{
                    .k = .init(lane % 8 + lane / 16 * 8, lane / 8 % 2),
                    .v = .init(lane % 8 + lane / 8 % 2 * 8, lane / 16),
                };
            }
        };

        /// Locates the chunks of a slot for a thread whose own column is 0 or 1. Within a box of
        /// `Rows.swizzle_columns` chunks, the swizzle stores chunk c of row r at c ^ (r % 8), so
        /// the even column offsets of a row follow from the thread's own chunk by an XOR.
        const XorAddresses = struct {
            base: u32,

            fn init(row: u32, column: u32) XorAddresses {
                return .{ .base = Rows.chunkIndex(slot_rows, row, column) * @sizeOf(Chunk) };
            }

            fn chunk(
                addresses: XorAddresses,
                tile: *addrspace(.shared) Slot,
                comptime row: u32,
                comptime column: u32,
            ) *addrspace(.shared) Chunk {
                comptime std.debug.assert(row % 8 == 0 and column % 2 == 0);
                const key = column % Rows.swizzle_columns;
                const offset = comptime Rows.chunkIndex(slot_rows, row, column - key) *
                    @sizeOf(Chunk);
                const bytes: [*]addrspace(.shared) u8 = @ptrCast(tile);
                return @ptrCast(@alignCast(bytes + (addresses.base ^ key * @sizeOf(Chunk)) +
                    offset));
            }
        };

        fn loadQuery(tile: *addrspace(.shared) Slot, first_row: u32) Query {
            const lane = thread.lane();
            const addresses: XorAddresses = .init(first_row + lane % 16, lane / 16);
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                fragment.* = nvptx.ldmatrix(.x4, false, addresses.chunk(tile, 0, k_step * 2));
            }
            return query;
        }

        inline fn qkScores(
            query: Query,
            k_tile: *addrspace(.shared) Slot,
            k_addresses: XorAddresses,
        ) Online.Scores {
            @setEvalBranchQuota(100_000);
            var scores: Online.Scores = @splat(@splat(0));
            inline for (query, 0..) |fragment, k_step| {
                inline for (0..block_n / 16) |j| {
                    const b = nvptx.ldmatrix(
                        .x4,
                        false,
                        k_addresses.chunk(k_tile, j * 16, k_step * 2),
                    );
                    scores[2 * j] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[0], b[1] },
                        scores[2 * j],
                    );
                    scores[2 * j + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        fragment,
                        @Vector(2, u32){ b[2], b[3] },
                        scores[2 * j + 1],
                    );
                }
            }
            return scores;
        }

        inline fn accumulateOutput(
            output: *Output,
            step: Online.Step,
            v_tile: *addrspace(.shared) Slot,
            v_addresses: XorAddresses,
        ) void {
            @setEvalBranchQuota(100_000);
            if (step.should_rescale) {
                inline for (output) |*fragment| fragment.* *= step.row_scale;
            }
            inline for (0..block_n / 16) |k_step| {
                inline for (0..row_chunks / 2) |column_pair| {
                    const b = nvptx.ldmatrix(
                        .x4,
                        true,
                        v_addresses.chunk(v_tile, k_step * 16, column_pair * 2),
                    );
                    const probabilities = step.probabilities[k_step];
                    output[2 * column_pair] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[0], b[1] },
                        output[2 * column_pair],
                    );
                    output[2 * column_pair + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[2], b[3] },
                        output[2 * column_pair + 1],
                    );
                }
            }
        }

        /// The scores of one key tile before masking, so that a warp multiplies the next key tile
        /// before it runs the softmax of this one.
        const Scored = struct {
            scores: Online.Scores,
            computes: bool,
            masked: bool,
            block_info: BlockInfo,
            first_column: u32,

            fn init(
                cursor: *const Cursor,
                rows: Rows64,
                query: Query,
                k_slot: SlotRing,
                lanes: Lanes,
                has_rows: bool,
            ) Scored {
                mbarrier.wait(.cta, &full[k_slot.index], k_slot.phase);
                const masked = cursor.needsMask();
                var scored: Scored = .{
                    .scores = undefined,
                    .computes = has_rows and (!masked or sees(cursor, rows)),
                    .masked = masked,
                    .block_info = cursor.block_info,
                    .first_column = cursor.n_block * block_n,
                };
                if (scored.computes) scored.scores = qkScores(query, slot(k_slot.index), lanes.k);
                release(k_slot);
                return scored;
            }
        };

        fn sees(cursor: *const Cursor, rows: Rows64) bool {
            const row = thread.warp() * 16;
            const first_key: i32 = @intCast(cursor.n_block * block_n);
            const last_row = @min(row + 15, rows.count - 1);
            const col_limit_left = cursor.block_info.colLimitLeft(@intCast(rows.token(row)));
            const col_limit_right =
                cursor.block_info.colLimitRight(@intCast(rows.token(last_row)));
            return col_limit_left < first_key + block_n and col_limit_right > first_key and
                col_limit_left < col_limit_right;
        }

        fn process(
            params: Params,
            work: Work,
            position: *SlotRing,
            trace: *Trace,
        ) void {
            const rows: Rows64 = .init(work, block_m);
            const warp = thread.warp();
            var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
            if (work.num_splits > 1) {
                const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                cursor.skip(params, first_n_block);
                cursor.limit(n_blocks);
            }
            const has_rows = warp * 16 < rows.count;
            trace.begin(.load_q);
            const query_slots = [2]SlotRing{ position.*, position.next() };
            for (query_slots) |query_slot| {
                mbarrier.wait(.cta, &full[query_slot.index], query_slot.phase);
            }
            const query: Query = if (has_rows and cursor.valid()) loadQuery(
                slot(query_slots[warp / (consumer_warps / 2)].index),
                warp % (consumer_warps / 2) * 16,
            ) else undefined;
            nvptx.barWarpSync(.all);
            for (query_slots) |query_slot| release(query_slot);
            position.advance();
            position.advance();
            trace.end(.load_q);
            const lanes: Lanes = .init();
            var softmax = Online.init();
            var output: Output = @splat(@splat(0));
            if (comptime layout.overlapsScores(head_dim)) {
                if (cursor.valid()) {
                    var current: Scored = .init(&cursor, rows, query, position.*, lanes, has_rows);
                    while (true) {
                        const v_slot = position.next();
                        position.advance();
                        position.advance();
                        cursor.advance(params);
                        const has_next = cursor.valid();
                        const upcoming: Scored = if (has_next)
                            .init(&cursor, rows, query, position.*, lanes, has_rows)
                        else
                            undefined;
                        if (current.computes) {
                            if (current.masked) {
                                const row = warp * 16 + thread.lane() / 4;
                                forward.maskScores(
                                    &current.scores,
                                    current.block_info,
                                    current.first_column,
                                    rows,
                                    row,
                                );
                            }
                            const step = Online.step(&softmax, current.scores, params.softmax_scale_log2);
                            mbarrier.wait(.cta, &full[v_slot.index], v_slot.phase);
                            accumulateOutput(&output, step, slot(v_slot.index), lanes.v);
                        } else mbarrier.wait(.cta, &full[v_slot.index], v_slot.phase);
                        release(v_slot);
                        if (!has_next) break;
                        current = upcoming;
                    }
                }
            } else while (cursor.valid()) {
                const k_slot = position.*;
                const v_slot = position.next();
                trace.begin(.scores);
                trace.begin(.wait_k);
                mbarrier.wait(.cta, &full[k_slot.index], k_slot.phase);
                trace.end(.wait_k);
                const masked = cursor.needsMask();
                if (has_rows and (!masked or sees(&cursor, rows))) {
                    trace.begin(.qk_mma);
                    var scores = qkScores(query, slot(k_slot.index), lanes.k);
                    trace.end(.qk_mma);
                    release(k_slot);
                    if (masked) {
                        const first_column = cursor.n_block * block_n;
                        const row = warp * 16 + thread.lane() / 4;
                        forward.maskScores(&scores, cursor.block_info, first_column, rows, row);
                    }
                    trace.end(.scores);
                    trace.begin(.softmax);
                    const step = Online.step(&softmax, scores, params.softmax_scale_log2);
                    trace.end(.softmax);
                    trace.begin(.output);
                    trace.begin(.wait_v);
                    mbarrier.wait(.cta, &full[v_slot.index], v_slot.phase);
                    trace.end(.wait_v);
                    trace.begin(.pv_mma);
                    accumulateOutput(&output, step, slot(v_slot.index), lanes.v);
                    trace.end(.pv_mma);
                    trace.end(.output);
                } else {
                    release(k_slot);
                    trace.end(.scores);
                    trace.begin(.wait_v);
                    mbarrier.wait(.cta, &full[v_slot.index], v_slot.phase);
                    trace.end(.wait_v);
                }
                release(v_slot);
                position.advance();
                position.advance();
                cursor.advance(params);
            }
            if (work.num_splits > 1) {
                if (!combineSplits(params, &softmax, &output, work, trace)) return;
            }
            trace.begin(.store);
            forward.storeHalfRows(head_dim, params, softmax, output, rows, warp * 16, 0);
            trace.end(.store);
        }

        fn combineSplits(
            params: Params,
            softmax: *Online,
            output: *Output,
            work: Work,
            trace: *Trace,
        ) bool {
            trace.begin(.combine);
            defer trace.end(.combine);
            const OAccum = [output.len + 1][threads]Accumulator;
            const o_accum: [*]addrspace(.global) OAccum = @ptrFromInt(params.o_accum);
            const first_split = work.split_block * work.num_splits;
            const own = &o_accum[first_split + work.split_idx];
            const id = thread.id();
            const caching: nvptx.cache.StoreCaching = .{ .operator = .cache_global_level };
            inline for (output.*, 0..) |fragment, chunk| {
                nvptx.cache.store(caching, &own[chunk][id], fragment);
            }
            const state: Accumulator = .{
                softmax.row_max[0], softmax.row_max[1], softmax.rest_sum[0], softmax.rest_sum[1],
            };
            nvptx.cache.store(caching, &own[output.len][id], state);
            nvptx.fence(.seq_cst, .gpu);
            nvptx.barrierSyncAligned(consumer_barrier, consumer_threads);
            if (id == 0) {
                const split_counters: [*]addrspace(.global) u32 = @ptrFromInt(params.o_accum +
                    @as(u64, params.o_accum_splits) * @sizeOf(OAccum));
                arrived_splits =
                    nvptx.atom(.gpu, .monotonic, .add, &split_counters[work.split_block], 1);
            }
            nvptx.barrierSyncAligned(consumer_barrier, consumer_threads);
            if (arrived_splits + 1 < work.num_splits) return false;
            nvptx.fence(.seq_cst, .gpu);
            const loading: nvptx.cache.LoadCaching = .{ .operator = .cache_global_level };
            softmax.* = .init();
            output.* = @splat(@splat(0));
            for (0..work.num_splits) |split_idx| {
                const other = &o_accum[first_split + split_idx];
                const other_state = nvptx.cache.load(loading, &other[output.len][id]);
                const own_scale, const other_scale = softmax.combine(.{
                    .row_max = .{ other_state[0], other_state[1] },
                    .rest_sum = .{ other_state[2], other_state[3] },
                }, params.softmax_scale_log2);
                inline for (output, 0..) |*fragment, chunk| {
                    const vector = nvptx.cache.load(loading, &other[chunk][id]);
                    fragment.* = fragment.* * own_scale + vector * other_scale;
                }
            }
            return true;
        }
    };
}
