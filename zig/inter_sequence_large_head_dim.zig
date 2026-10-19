const std = @import("std");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const smem = @import("smem.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");
const forward = @import("inter_sequence_forward.zig");

const Chunk = fragments.Chunk;
const Accumulator = fragments.Accumulator;
const Params = forward.Params;
const Trace = forward.Trace;
const Work = forward.Work;
const NextWork = forward.NextWork;
const TileScheduler = forward.TileScheduler;
const persistent = configurations.inter_sequence_persistent;
const chunk_elements = configurations.Sm80FwdTiling.chunk_elements;
const element_bytes = @sizeOf(Chunk) / chunk_elements;
const cta_warps = persistent.cta_warps;
const cta_threads = cta_warps * nvptx.warp_size;
const pair_threads = 2 * nvptx.warp_size;
const block_m = cta_warps / 2 * 16;
const block_n = persistent.large_head_dim_block_n;
const stages = persistent.large_head_dim_stages;
const KvPipeline = forward.Pipeline(stages);

extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);

pub fn Kernel(comptime head_dim: u32, comptime packs: bool) type {
    const row_chunks = head_dim / chunk_elements;
    const half_chunks = row_chunks / 2;
    const Rows = smem.Layout(row_chunks);
    const swizzle_columns: u32 = Rows.swizzle_columns;
    const KvTile = Rows.Tile(block_n);
    const KvAddresses = Rows.PairAddresses(block_n);
    const Online = fragments.Softmax(block_n);
    const Query = [half_chunks / 2]@Vector(4, u32);
    const Output = [half_chunks]Accumulator;
    const Cursor = forward.NBlockCursor(block_n);
    const QueryRows = forward.QueryRows(packs);
    const Exchange = [2][cta_warps][block_n / 8][nvptx.warp_size]Accumulator;
    const copy_row_threads = cta_threads / block_n;
    const copy_columns = row_chunks / copy_row_threads;
    return struct {
        pub const threads = cta_threads;
        pub const shared_bytes = persistent.sharedBytes(head_dim);

        comptime {
            std.debug.assert(shared_bytes == 2 * stages * @sizeOf(KvTile) + @sizeOf(Exchange));
            std.debug.assert(copy_row_threads % swizzle_columns == 0);
            std.debug.assert(half_chunks % swizzle_columns == 0);
        }

        pub fn run(params: Params, _: forward.Args) void {
            var trace: Trace = .init(params.profile);
            defer trace.finish();
            trace.begin(.setup);
            if (thread.warp() == 0) {
                const scheduler: TileScheduler = .init(params, packs, block_m, false);
                if (thread.lane() == 0) forward.tile_scheduler = scheduler;
            }
            KvPipeline.initBarriers(cta_threads, cta_warps);
            trace.end(.setup);
            processTiles(params, &trace);
            nvptx.barrierSyncAligned(.work_group, null);
            if (thread.id() == 0) forward.TileCountSemaphore.finishCta(params);
        }

        fn kTile(stage: u32) *addrspace(.shared) KvTile {
            const tiles: [*]addrspace(.shared) KvTile = @ptrCast(@alignCast(&shared_storage));
            return &tiles[2 * stage];
        }

        fn vTile(stage: u32) *addrspace(.shared) KvTile {
            const tiles: [*]addrspace(.shared) KvTile = @ptrCast(@alignCast(&shared_storage));
            return &tiles[2 * stage + 1];
        }

        fn exchange() *addrspace(.shared) Exchange {
            const bytes: [*]addrspace(.shared) u8 = @ptrCast(&shared_storage);
            return @ptrCast(@alignCast(bytes + 2 * stages * @sizeOf(KvTile)));
        }

        const RowCopy = struct {
            slot: u32,
            row: u32,
            column: u32,

            fn init() RowCopy {
                const row = thread.id() / copy_row_threads;
                const column = thread.id() % copy_row_threads;
                return .{ .slot = Rows.slot(row, column), .row = row, .column = column };
            }

            fn copy(
                copier: RowCopy,
                destination: *addrspace(.shared) KvTile,
                source: RowSource,
                first_row: u32,
                valid_rows: u32,
            ) void {
                const size: u32 = if (copier.row < valid_rows) @sizeOf(Chunk) else 0;
                const address = source.first + first_row * source.row_bytes;
                inline for (0..copy_columns) |thread_column| {
                    const column: u32 = thread_column * copy_row_threads;
                    const chunk: *addrspace(.global) const Chunk =
                        @ptrFromInt(address + column * @sizeOf(Chunk));
                    const slot = copier.slot + comptime Rows.slot(0, column);
                    cp_async.copyZeroFill(.cache_global_level, &destination[slot], chunk, size);
                }
            }
        };

        const RowSource = struct {
            first: u64,
            row_bytes: u64,

            fn init(copier: RowCopy, address: u64, offset: u64, stride: u64) RowSource {
                const row_bytes = stride * element_bytes;
                return .{
                    .first = address + offset * element_bytes + copier.row * row_bytes +
                        copier.column * @sizeOf(Chunk),
                    .row_bytes = row_bytes,
                };
            }
        };

        const Load = struct {
            cursor: Cursor,
            copier: RowCopy,
            keys: RowSource,
            values: RowSource,
            next: NextWork,

            fn advance(load: *Load, params: Params) void {
                load.cursor.advance(params);
                if (!load.cursor.valid()) load.next.prefetch(params, .cta);
            }

            fn copyTile(load: *Load, params: Params, write: *KvPipeline.State) void {
                if (!load.cursor.valid()) return;
                const first_key = load.cursor.firstKey();
                const keys_left = load.cursor.keysLeft();
                const stage = write.index;
                mbarrier.wait(.cta, &KvPipeline.empty[stage], write.phase ^ 1);
                load.copier.copy(kTile(stage), load.keys, first_key, keys_left);
                mbarrier.arrive(.cp_async, &KvPipeline.k_full[stage]);
                load.copier.copy(vTile(stage), load.values, first_key, keys_left);
                mbarrier.arrive(.cp_async, &KvPipeline.v_full[stage]);
                load.advance(params);
                write.advance();
            }
        };

        const Lanes = struct {
            k: KvAddresses,
            v: KvAddresses,

            fn init(hdim_half: u32) Lanes {
                const lane = thread.lane();
                const first_column = hdim_half * half_chunks;
                return .{
                    .k = .init(lane % 8 + lane / 16 * 8, first_column + lane / 8 % 2),
                    .v = .init(lane % 8 + lane / 8 % 2 * 8, first_column + lane / 16),
                };
            }
        };

        fn loadQuery(params: Params, rows: QueryRows, warp_pair: u32, hdim_half: u32) Query {
            const lane = thread.lane();
            const q: [*]addrspace(.global) const u32 = @ptrFromInt(params.q);
            const column = hdim_half * (head_dim / 2) + lane % 4 * 2;
            var valid: [2]bool = undefined;
            var bases: [2]u64 = undefined;
            inline for (.{ 0, 1 }) |row_half| {
                const row = warp_pair * 16 + lane / 4 + row_half * 8;
                valid[row_half] = row < rows.count;
                const source_row = if (valid[row_half]) row else 0;
                bases[row_half] =
                    rows.offset(source_row, params.q_row_stride, params.q_head_stride) + column;
            }
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                inline for (0..4) |word| {
                    const row_half = word % 2;
                    const offset = bases[row_half] + k_step * 16 + word / 2 * 8;
                    fragment[word] = if (valid[row_half]) q[offset / 2] else 0;
                }
            }
            return query;
        }

        inline fn partialScores(
            query: Query,
            k_tile: *addrspace(.shared) KvTile,
            k_addresses: KvAddresses,
        ) Online.Scores {
            @setEvalBranchQuota(100_000);
            var sums: [2]Online.Scores = @splat(@splat(@splat(0)));
            inline for (query, 0..) |fragment, k_step| {
                const scores = &sums[k_step % 2];
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
            var scores = sums[0];
            inline for (&scores, sums[1]) |*fragment, odd| fragment.* += odd;
            return scores;
        }

        inline fn accumulateOutput(
            output: *Output,
            step: Online.Step,
            v_tile: *addrspace(.shared) KvTile,
            v_addresses: KvAddresses,
        ) void {
            @setEvalBranchQuota(100_000);
            if (step.should_rescale) {
                inline for (output) |*fragment| fragment.* *= step.row_scale;
            }
            inline for (0..block_n / 16) |k_step| {
                inline for (0..half_chunks / 2) |column_pair| {
                    const column = column_pair * 2;
                    const b = nvptx.ldmatrix(
                        .x4,
                        true,
                        v_addresses.chunk(v_tile, k_step * 16, column),
                    );
                    const probabilities = step.probabilities[k_step];
                    output[column] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[0], b[1] },
                        output[column],
                    );
                    output[column + 1] = nvptx.mmaSync(
                        fragments.mma_m16n8k16,
                        probabilities,
                        @Vector(2, u32){ b[2], b[3] },
                        output[column + 1],
                    );
                }
            }
        }

        fn process(
            params: Params,
            work: Work,
            pipeline: *KvPipeline,
            next: NextWork,
            trace: *Trace,
        ) void {
            const rows: QueryRows = .init(work, block_m);
            const kv_head = forward.tile_scheduler.qhead_per_kvhead.divide(rows.head);
            const copier: RowCopy = .init();
            trace.begin(.load_q);
            trace.begin(.locate);
            var cursor: Cursor = .init(params, work, rows.token(0), rows.tokenCount());
            trace.end(.locate);
            if (work.num_splits > 1) {
                const first_n_block, const n_blocks = work.splitNBlocks(block_m, block_n);
                cursor.skip(params, first_n_block);
                cursor.limit(n_blocks);
            }
            var load: Load = .{
                .cursor = cursor,
                .copier = copier,
                .keys = .init(
                    copier,
                    params.k,
                    kv_head * params.k_head_stride,
                    params.k_row_stride,
                ),
                .values = .init(
                    copier,
                    params.v,
                    kv_head * params.v_head_stride,
                    params.v_row_stride,
                ),
                .next = next,
            };
            if (!load.cursor.valid()) next.prefetch(params, .cta);
            load.copyTile(params, &pipeline.write);
            const warp = thread.warp();
            const lane = thread.lane();
            const warp_pair = warp / 2;
            const hdim_half = warp % 2;
            const query = loadQuery(params, rows, warp_pair, hdim_half);
            trace.end(.load_q);
            const lanes: Lanes = .init(hdim_half);
            const has_rows = warp_pair * 16 < rows.count;
            var softmax = Online.init();
            var output: Output = @splat(@splat(0));
            var parity: u1 = 0;
            while (cursor.valid()) {
                load.copyTile(params, &pipeline.write);
                const stage = pipeline.read.index;
                const phase = pipeline.read.phase;
                trace.begin(.scores);
                trace.begin(.wait_k);
                mbarrier.wait(.cta, &KvPipeline.k_full[stage], phase);
                trace.end(.wait_k);
                const masked = cursor.needsMask();
                if (has_rows and (!masked or sees: {
                    const first_key: i32 = @intCast(cursor.n_block * block_n);
                    const first_row = warp_pair * 16;
                    const last_row = @min(first_row + 15, rows.count - 1);
                    const info = cursor.block_info;
                    const col_limit_left = info.colLimitLeft(@intCast(rows.token(first_row)));
                    const col_limit_right = info.colLimitRight(@intCast(rows.token(last_row)));
                    break :sees col_limit_left < first_key + block_n and
                        col_limit_right > first_key;
                })) {
                    trace.begin(.qk_mma);
                    var scores = partialScores(query, kTile(stage), lanes.k);
                    trace.end(.qk_mma);
                    const buffers = &exchange()[parity];
                    inline for (scores, 0..) |fragment, n_tile| {
                        buffers[warp][n_tile][lane] = fragment;
                    }
                    const pair_barrier: nvptx.NamedBarrier =
                        @fromBackingInt(@as(u4, @intCast(1 + warp_pair)));
                    nvptx.barrierSyncAligned(pair_barrier, pair_threads);
                    const partner = &buffers[warp ^ 1];
                    inline for (&scores, 0..) |*fragment, n_tile| {
                        fragment.* += partner[n_tile][lane];
                    }
                    parity +%= 1;
                    if (masked) {
                        const first_column = cursor.n_block * block_n;
                        const row = warp_pair * 16 + lane / 4;
                        forward.maskScores(&scores, cursor.block_info, first_column, rows, row);
                    }
                    trace.end(.scores);
                    trace.begin(.softmax);
                    const step = Online.step(&softmax, scores, params.softmax_scale_log2);
                    trace.end(.softmax);
                    trace.begin(.output);
                    trace.begin(.wait_v);
                    mbarrier.wait(.cta, &KvPipeline.v_full[stage], phase);
                    trace.end(.wait_v);
                    trace.begin(.pv_mma);
                    accumulateOutput(&output, step, vTile(stage), lanes.v);
                    trace.end(.pv_mma);
                    trace.end(.output);
                } else {
                    trace.end(.scores);
                    trace.begin(.wait_v);
                    mbarrier.wait(.cta, &KvPipeline.v_full[stage], phase);
                    trace.end(.wait_v);
                }
                if (lane == 0) _ = KvPipeline.empty[stage].arrive();
                pipeline.read.advance();
                cursor.advance(params);
            }
            trace.begin(.get_next_work);
            nvptx.barrierSyncAligned(.work_group, null);
            trace.end(.get_next_work);
            if (work.num_splits > 1) {
                if (!forward.combineSplits(params, &softmax, &output, work, trace)) return;
            }
            trace.begin(.store);
            if (has_rows) {
                forward.storeHalfRows(
                    head_dim,
                    params,
                    softmax,
                    output,
                    rows,
                    warp_pair * 16,
                    hdim_half,
                );
            }
            trace.end(.store);
        }

        fn processTiles(params: Params, trace: *Trace) void {
            const tile_count = forward.tile_scheduler.tileCount(.cta);
            const leader = thread.id() == 0;
            var parity: u1 = 0;
            var pipeline: KvPipeline = .{};
            trace.begin(.get_next_work);
            const next_tile_idx = &forward.next_tile_idx[0];
            const first: NextWork = .{ .leader = leader, .tile_idx = &next_tile_idx[parity] };
            first.prefetch(params, .cta);
            nvptx.barrierSyncAligned(.work_group, null);
            var tile_idx = first.tile_idx.*;
            trace.end(.get_next_work);
            while (tile_idx < tile_count) {
                parity +%= 1;
                const next: NextWork = .{ .leader = leader, .tile_idx = &next_tile_idx[parity] };
                trace.begin(.tile);
                trace.begin(.locate);
                const work = forward.tile_scheduler.locate(params, .cta, tile_idx).?;
                trace.end(.locate);
                process(params, work, &pipeline, next, trace);
                trace.end(.tile);
                tile_idx = next.tile_idx.*;
            }
        }
    };
}
