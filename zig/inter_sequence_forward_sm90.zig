const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const tma = @import("tma.zig");
const smem = @import("smem.zig");
const mbarrier = @import("mbarrier.zig");
const elect = @import("elect.zig");
const thread = @import("thread.zig");
const wgmma = @import("wgmma.zig");
const forward = @import("inter_sequence_forward.zig");
const sm120 = @import("inter_sequence_forward_sm120.zig");

const Accumulator = fragments.Accumulator;
const Mbarrier = nvptx.Mbarrier;
const Params = forward.Params;
const Work = forward.Work;
const Trace = forward.Trace;
const TileScheduler = forward.TileScheduler;
const layout = configurations.inter_sequence_sm90;
const block_n = layout.block_n;
const query_rows = layout.query_box_rows;
const group_rows = 64;
const group_threads = configurations.warps_per_warp_group * nvptx.warp_size;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;

pub const Args = sm120.Args;
pub const names = sm120.names;

extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
var producer_tile_idx: u32 addrspace(.shared) = undefined;
var arrived_splits: u32 addrspace(.shared) = undefined;

pub fn Kernel(comptime head_dim: u32, comptime packs: bool) type {
    const groups = layout.groups(head_dim);
    const block_m = layout.blockM(head_dim);
    const consumer_warps = layout.consumerWarps(head_dim);
    const consumer_threads = consumer_warps * nvptx.warp_size;
    const consumer_barrier: nvptx.NamedBarrier = @fromBackingInt(1 + groups);
    const producer_registers = layout.producerRegisters(head_dim);
    const consumer_registers = layout.consumerRegisters(head_dim);
    const row_chunks = head_dim / chunk_elements;
    const swizzle_bytes = layout.swizzle_bytes;
    const Rows = tma.SwizzledLayout(row_chunks, swizzle_bytes);
    const Operands = wgmma.Operands(swizzle_bytes);
    const Slot = Rows.Tile(block_n);
    const QueryHalf = Rows.Tile(query_rows);
    const slot_bytes = @sizeOf(Slot);
    const slots = layout.slots(head_dim);
    const Online = fragments.Softmax(block_n);
    const Scores = wgmma.Accumulator(block_n);
    const Output = wgmma.Accumulator(head_dim);
    const Cursor = forward.NBlockCursor(block_n);
    const QueryRows = forward.QueryRows(packs);
    const SlotRing = sm120.Ring(slots);
    const WorkRing = sm120.Ring(layout.work_slots);
    const QueryRing = sm120.Ring(layout.query_buffers);
    const k_steps = head_dim / 16;

    comptime {
        std.debug.assert(slot_bytes == layout.slotBytes(head_dim));
        std.debug.assert(block_m == groups * group_rows and group_rows == query_rows);
    }

    return struct {
        pub const threads = layout.threads(head_dim);
        pub const shared_bytes = layout.sharedBytes(head_dim);
        pub const max_registers = configurations.maxRegistersPerThread(threads);

        const Tiles = struct {
            queries: [layout.query_buffers][groups]QueryHalf,
            ring: [slots]Slot,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
            const producer_total = producer_registers * layout.producer_warps;
            const consumer_total = consumer_registers * consumer_warps;
            std.debug.assert((producer_total + consumer_total) * nvptx.warp_size <=
                configurations.registers_per_cta);
        }

        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var full: [slots]Mbarrier addrspace(.shared) = undefined;
        var empty: [slots]Mbarrier addrspace(.shared) = undefined;
        var query_full: [layout.query_buffers]Mbarrier addrspace(.shared) = undefined;
        var query_empty: [layout.query_buffers]Mbarrier addrspace(.shared) = undefined;
        var work_full: [layout.work_slots]Mbarrier addrspace(.shared) = undefined;
        var work_empty: [layout.work_slots]Mbarrier addrspace(.shared) = undefined;
        var published: [layout.work_slots]sm120.Published addrspace(.shared) = undefined;

        pub fn run(params: Params, args: Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            const producer_warp = consumer_warps;
            if (thread.warp() == producer_warp and elect.oneSync()) {
                for (&full, &empty) |*full_slot, *empty_slot| {
                    full_slot.init(1);
                    empty_slot.init(consumer_threads);
                }
                for (&query_full, &query_empty) |*full_buffer, *empty_buffer| {
                    full_buffer.init(1);
                    empty_buffer.init(consumer_threads);
                }
                for (&work_full, &work_empty) |*full_slot, *empty_slot| {
                    full_slot.init(1);
                    empty_slot.init(consumer_warps);
                }
                mbarrier.pipelineInitFence();
                tma.prefetchMaps(args);
            }
            elect.reconverge();
            nvptx.griddepcontrol(.wait);
            if (thread.warp() == producer_warp) {
                const scheduler: TileScheduler = .init(params, packs, block_m, false);
                if (thread.lane() == 0) forward.tile_scheduler = scheduler;
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (thread.warp() >= producer_warp) {
                asm volatile (std.fmt.comptimePrint(
                        "setmaxnreg.dec.sync.aligned.u32 {d};",
                        .{producer_registers},
                    ));
                if (thread.warp() == producer_warp and thread.lane() == 0) produce(params, args);
                nvptx.barWarpSync(.all);
            } else {
                asm volatile (std.fmt.comptimePrint(
                        "setmaxnreg.inc.sync.aligned.u32 {d};",
                        .{consumer_registers},
                    ));
                consume(params, &trace);
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (thread.id() == 0) forward.TileCountSemaphore.finishCta(params);
        }

        fn issue(ring: *SlotRing, map: *const tma.TensorMap, head: u32, row: u32) void {
            const index = ring.index;
            mbarrier.wait(.cta, &empty[index], ring.phase ^ 1);
            _ = full[index].arriveExpectTx(slot_bytes);
            Rows.copyTile(&tiles.ring[index], map, &full[index], head, row, 0);
            ring.advance();
        }

        fn issueQuery(
            queries: *QueryRing,
            map: *const tma.TensorMap,
            head: u32,
            first_token: u32,
            packing: u32,
        ) void {
            const index = queries.index;
            mbarrier.wait(.cta, &query_empty[index], queries.phase ^ 1);
            _ = query_full[index].arriveExpectTx(groups * @sizeOf(QueryHalf));
            inline for (0..groups) |group| {
                const token = first_token + @as(u32, group) * query_rows / packing;
                Rows.copyTile(&tiles.queries[index][group], map, &query_full[index], head, token, 0);
            }
            queries.advance();
        }

        fn publish(works: *WorkRing, located: ?Work) void {
            mbarrier.wait(.cta, &work_empty[works.index], works.phase ^ 1);
            published[works.index] = .{ .work = located orelse undefined, .valid = located != null };
            _ = work_full[works.index].arrive();
            works.advance();
        }

        /// Issues the loads in the order in which the consumers take them: for each work tile,
        /// the first key tile, the query tile and then the value tile of each key block followed
        /// by the next key tile. One thread runs it.
        fn produce(params: Params, args: Args) void {
            const scheduler = &forward.tile_scheduler;
            const tile_count = scheduler.tileCount(.cta);
            const packing = if (packs) scheduler.packing.divisor else 1;
            var ring: SlotRing = .{};
            var queries: QueryRing = .{};
            var works: WorkRing = .{};
            while (true) {
                const fetch: forward.NextWork = .{ .leader = true, .tile_idx = &producer_tile_idx };
                fetch.prefetch(params, .cta);
                const tile_idx = producer_tile_idx;
                const located = if (tile_idx < tile_count)
                    scheduler.locate(params, .cta, tile_idx)
                else
                    null;
                publish(&works, located);
                const work = located orelse return;
                const rows: QueryRows = .init(work, block_m);
                const kv_head = scheduler.qhead_per_kvhead.divide(rows.head);
                const first_token = rows.first_query + rows.first_row / packing;
                var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
                if (work.num_splits > 1) {
                    const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                    cursor.skip(params, first_n_block);
                    cursor.limit(n_blocks);
                }
                var previous_key: ?u32 = null;
                if (cursor.valid()) {
                    issue(&ring, args.k, kv_head, cursor.firstKey());
                    previous_key = cursor.firstKey();
                    cursor.advance(params);
                }
                issueQuery(&queries, args.q, rows.head, first_token, packing);
                while (cursor.valid()) {
                    issue(&ring, args.k, kv_head, cursor.firstKey());
                    issue(&ring, args.v, kv_head, previous_key.?);
                    previous_key = cursor.firstKey();
                    cursor.advance(params);
                }
                if (previous_key) |key| issue(&ring, args.v, kv_head, key);
            }
        }

        fn release(ring: SlotRing) void {
            _ = empty[ring.index].arrive();
        }

        fn consume(params: Params, trace: *Trace) void {
            var position: SlotRing = .{};
            var works: WorkRing = .{};
            var queries: QueryRing = .{};
            const group = thread.warp() / configurations.warps_per_warp_group;
            if (group == 0) nvptx.barrierArriveAligned(turn(0), 2 * group_threads);
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
                process(params, entry.work, group, &position, &queries, trace);
                trace.end(.tile);
            }
        }

        fn turn(group: u32) nvptx.NamedBarrier {
            return @fromBackingInt(@intCast(1 + group));
        }

        fn mmaScores(scores: *Scores, query: u64, k_slot: u32) void {
            const key = Operands.kMajor(wgmma.address(&tiles.ring[k_slot]));
            wgmma.fence();
            Operands.mmaKSteps(block_n, scores, k_steps, query, query_rows, key, block_n);
            wgmma.commit();
        }

        fn mmaValues(output: *Output, probabilities: Online.Probabilities, v_slot: u32) void {
            const value = Operands.mnMajor(wgmma.address(&tiles.ring[v_slot]), block_n);
            wgmma.fence();
            inline for (0..block_n / 16) |key_step| {
                wgmma.mmaRegisters(
                    head_dim,
                    output,
                    probabilities[key_step],
                    wgmma.advance(value, key_step * 16 * swizzle_bytes),
                    true,
                    true,
                );
            }
            wgmma.commit();
        }

        /// What masking the scores of one key block needs, taken before the cursor moves on.
        const Block = struct {
            masked: bool,
            block_info: @FieldType(Cursor, "block_info"),
            first_column: u32,

            fn take(cursor: *Cursor, params: Params) Block {
                const block: Block = .{
                    .masked = cursor.needsMask(),
                    .block_info = cursor.block_info,
                    .first_column = cursor.n_block * block_n,
                };
                cursor.advance(params);
                return block;
            }
        };

        fn exponentiate(
            comptime masked: bool,
            softmax: *Online,
            scores: Scores,
            block: Block,
            rows: QueryRows,
            scale_log2: f32,
        ) Online.Exponentials {
            var score_fragments: Online.Scores = @bitCast(scores);
            if (masked) {
                const row = thread.warp() * 16 + thread.lane() / 4;
                forward.maskScores(&score_fragments, block.block_info, block.first_column, rows, row);
            }
            const decision = softmax.raise(score_fragments, scale_log2);
            return if (decision.raised)
                softmax.exponentialsAfter(true, decision, score_fragments, scale_log2)
            else
                softmax.exponentialsAfter(false, decision, score_fragments, scale_log2);
        }

        fn rescale(output: *Output, exponentials: Online.Exponentials) void {
            if (!exponentials.should_rescale) return;
            inline for (0..head_dim / 8) |tile| {
                inline for (0..4) |element| output[tile * 4 + element] *= exponentials.row_scale[element];
            }
        }

        fn waitFull(ring: SlotRing) void {
            mbarrier.wait(.cta, &full[ring.index], ring.phase);
        }

        /// The key blocks of one work tile for one warp group. A warp group whose rows all lie
        /// past the end of the tile issues no wgmma and only keeps the barriers in step.
        const Tile = struct {
            rows: QueryRows,
            group: u32,
            query: u64,
            query_buffer: QueryRing,
            scale_log2: f32,

            fn releaseQuery(tile: Tile) void {
                _ = query_empty[tile.query_buffer.index].arrive();
            }

            const Step = struct {
                k_slot: SlotRing,
                v_slot: SlotRing,
                block: Block,
                last: bool,
            };

            /// Multiplies the scores of the next key block and the values of the previous one,
            /// then runs the softmax of the next key block while the values multiply.
            inline fn step(
                tile: Tile,
                comptime computes: bool,
                comptime masked: bool,
                pipeline: Step,
                scores: *Scores,
                probabilities: *Online.Probabilities,
                softmax: *Online,
                output: *Output,
                trace: *Trace,
            ) void {
                trace.begin(.wait_turn);
                nvptx.barrierSyncAligned(turn(tile.group), 2 * group_threads);
                trace.end(.wait_turn);
                if (computes) mmaScores(scores, tile.query, pipeline.k_slot.index);
                trace.begin(.wait_v);
                waitFull(pipeline.v_slot);
                trace.end(.wait_v);
                if (computes) mmaValues(output, probabilities.*, pipeline.v_slot.index);
                nvptx.barrierArriveAligned(turn((tile.group + 1) % groups), 2 * group_threads);
                if (computes) {
                    trace.begin(.qk_mma);
                    wgmma.wait(1);
                    wgmma.fenceOperand(scores);
                    trace.end(.qk_mma);
                }
                release(pipeline.k_slot);
                if (pipeline.last) tile.releaseQuery();
                if (computes) {
                    trace.begin(.softmax);
                    const exponentials = exponentiate(
                        masked,
                        softmax,
                        scores.*,
                        pipeline.block,
                        tile.rows,
                        tile.scale_log2,
                    );
                    trace.end(.softmax);
                    trace.begin(.pv_mma);
                    wgmma.wait(0);
                    wgmma.fenceOperand(output);
                    trace.end(.pv_mma);
                    release(pipeline.v_slot);
                    rescale(output, exponentials);
                    probabilities.* = Online.pack(exponentials.values);
                } else release(pipeline.v_slot);
            }

            fn blocks(
                tile: Tile,
                comptime computes: bool,
                params: Params,
                cursor: *Cursor,
                position: *SlotRing,
                softmax: *Online,
                output: *Output,
                trace: *Trace,
            ) void {
                if (!cursor.valid()) return tile.releaseQuery();
                var scores: Scores = undefined;
                const first_k_slot = position.*;
                position.advance();
                trace.begin(.wait_k);
                waitFull(first_k_slot);
                trace.end(.wait_k);
                const first = Block.take(cursor, params);
                if (computes) {
                    mmaScores(&scores, tile.query, first_k_slot.index);
                    wgmma.wait(0);
                    wgmma.fenceOperand(&scores);
                }
                release(first_k_slot);
                if (!cursor.valid()) tile.releaseQuery();
                var probabilities: Online.Probabilities = undefined;
                if (computes) {
                    const exponentials = if (first.masked)
                        exponentiate(true, softmax, scores, first, tile.rows, tile.scale_log2)
                    else
                        exponentiate(false, softmax, scores, first, tile.rows, tile.scale_log2);
                    probabilities = Online.pack(exponentials.values);
                }
                while (cursor.valid()) {
                    const k_slot = position.*;
                    position.advance();
                    const v_slot = position.*;
                    position.advance();
                    trace.begin(.wait_k);
                    waitFull(k_slot);
                    trace.end(.wait_k);
                    const block = Block.take(cursor, params);
                    const pipeline: Step = .{
                        .k_slot = k_slot,
                        .v_slot = v_slot,
                        .block = block,
                        .last = !cursor.valid(),
                    };
                    if (block.masked) {
                        tile.step(
                            computes,
                            true,
                            pipeline,
                            &scores,
                            &probabilities,
                            softmax,
                            output,
                            trace,
                        );
                    } else {
                        tile.step(
                            computes,
                            false,
                            pipeline,
                            &scores,
                            &probabilities,
                            softmax,
                            output,
                            trace,
                        );
                    }
                }
                const v_slot = position.*;
                position.advance();
                waitFull(v_slot);
                if (computes) {
                    mmaValues(output, probabilities, v_slot.index);
                    wgmma.wait(0);
                    wgmma.fenceOperand(output);
                }
                release(v_slot);
            }
        };

        fn process(
            params: Params,
            work: Work,
            group: u32,
            position: *SlotRing,
            queries: *QueryRing,
            trace: *Trace,
        ) void {
            const rows: QueryRows = .init(work, block_m);
            var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
            if (work.num_splits > 1) {
                const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                cursor.skip(params, first_n_block);
                cursor.limit(n_blocks);
            }
            const computes = group * group_rows < rows.count;
            const query_buffer = queries.*;
            queries.advance();
            const query = Operands.kMajor(
                wgmma.address(&tiles.queries[query_buffer.index][group]),
            );
            trace.begin(.load_q);
            mbarrier.wait(.cta, &query_full[query_buffer.index], query_buffer.phase);
            trace.end(.load_q);
            var softmax = Online.init();
            var output: Output = @splat(0);
            const tile: Tile = .{
                .rows = rows,
                .group = group,
                .query = query,
                .query_buffer = query_buffer,
                .scale_log2 = params.softmax_scale_log2,
            };
            if (computes) {
                tile.blocks(true, params, &cursor, position, &softmax, &output, trace);
            } else tile.blocks(false, params, &cursor, position, &softmax, &output, trace);
            if (work.num_splits > 1) {
                if (!combineSplits(params, &softmax, &output, work, trace)) return;
            }
            trace.begin(.store);
            const output_fragments: [head_dim / 8]Accumulator = @bitCast(output);
            forward.storeHalfRows(
                head_dim,
                params,
                softmax,
                output_fragments,
                rows,
                thread.warp() * 16,
                0,
            );
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
            const chunks = head_dim / 8;
            const OAccum = [chunks + 1][threads]Accumulator;
            const o_accum: [*]addrspace(.global) OAccum = @ptrFromInt(params.o_accum);
            const first_split = work.split_block * work.num_splits;
            const own = &o_accum[first_split + work.split_idx];
            const id = thread.id();
            const caching: nvptx.cache.StoreCaching = .{ .operator = .cache_global_level };
            var fragments_out: [chunks]Accumulator = @bitCast(output.*);
            inline for (fragments_out, 0..) |fragment, chunk| {
                nvptx.cache.store(caching, &own[chunk][id], fragment);
            }
            const state: Accumulator = .{
                softmax.row_max[0], softmax.row_max[1], softmax.rest_sum[0], softmax.rest_sum[1],
            };
            nvptx.cache.store(caching, &own[chunks][id], state);
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
            fragments_out = @splat(@splat(0));
            for (0..work.num_splits) |split_idx| {
                const other = &o_accum[first_split + split_idx];
                const other_state = nvptx.cache.load(loading, &other[chunks][id]);
                const own_scale, const other_scale = softmax.combine(.{
                    .row_max = .{ other_state[0], other_state[1] },
                    .rest_sum = .{ other_state[2], other_state[3] },
                }, params.softmax_scale_log2);
                inline for (&fragments_out, 0..) |*fragment, chunk| {
                    const vector = nvptx.cache.load(loading, &other[chunk][id]);
                    fragment.* = fragment.* * own_scale + vector * other_scale;
                }
            }
            output.* = @bitCast(fragments_out);
            return true;
        }
    };
}
