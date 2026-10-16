const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const fragments = @import("fragments.zig");
const ForwardParams = @import("ForwardParams.zig");
const BackwardParams = @import("BackwardParams.zig");
const smem = @import("smem.zig");
const elect = @import("elect.zig");
const mbarrier = @import("mbarrier.zig");
const thread = @import("thread.zig");

const Chunk = fragments.Chunk;
const Sm80FwdTiling = configurations.Sm80FwdTiling;

pub const TensorMap = extern struct {
    bytes: [128]u8 align(64),
};

pub fn Layout(comptime row_chunks: comptime_int) type {
    return SwizzledLayout(row_chunks, configurations.swizzleBytes(row_chunks));
}

pub fn SwizzledLayout(
    comptime row_chunks: comptime_int,
    comptime swizzle_bytes: comptime_int,
) type {
    const box_chunk_count = configurations.boxChunks(row_chunks, swizzle_bytes);
    return struct {
        pub const swizzle_columns = if (swizzle_bytes == 0) 1 else box_chunk_count;

        pub fn Tile(comptime rows: comptime_int) type {
            return [rows * row_chunks]Chunk;
        }

        pub fn Addresses(comptime rows: comptime_int) type {
            return smem.TileAddresses(@This(), rows);
        }

        fn rowCount(comptime Pointer: type) comptime_int {
            return @typeInfo(@typeInfo(Pointer).pointer.child).array.len / row_chunks;
        }

        pub fn chunkIndex(comptime rows: comptime_int, row: u32, column: u32) u32 {
            if (swizzle_bytes == 0) return row * row_chunks + column;
            const box = column / box_chunk_count * rows * box_chunk_count;
            const swizzled = smem.swizzledColumn(swizzle_bytes, column, row) % box_chunk_count;
            return box + row * box_chunk_count + swizzled;
        }

        pub fn chunk(tile: anytype, row: u32, column: u32, _: u32) *addrspace(.shared) Chunk {
            return &tile[chunkIndex(rowCount(@TypeOf(tile)), row, column)];
        }

        pub fn clearRows(
            tile: anytype,
            first_cleared: u32,
            comptime participants: u32,
            participant: u32,
        ) void {
            const chunks = rowCount(@TypeOf(tile)) * row_chunks;
            var index = first_cleared * row_chunks + participant;
            while (index < chunks) : (index += participants) {
                chunk(tile, index / row_chunks, index % row_chunks, 0).* = @splat(0);
            }
        }

        pub fn copyTile(
            destination: anytype,
            map: *const TensorMap,
            barrier: *addrspace(.shared) nvptx.Mbarrier,
            head: u32,
            row: u32,
            batch: u32,
        ) void {
            const rows = rowCount(@TypeOf(destination));
            const box_columns = box_chunk_count * Sm80FwdTiling.chunk_elements;
            copyBoxes(.{ .columns = box_columns, .rows = rows, .count = row_chunks /
                box_chunk_count }, destination, map, barrier, .{ 0, head, row, batch });
        }
    };
}

pub fn stageReduceRow(
    comptime chunks: u32,
    staging: anytype,
    row: anytype,
    values: anytype,
    comptime first_column: u32,
) void {
    inline for (0..chunks) |quad| {
        const column = first_column + quad * 4;
        const staged_row: u32 = if (@typeInfo(@TypeOf(row)) == .@"fn") row() else row;
        staging[staged_row * chunks + (quad ^ staged_row % 8)] = @bitCast(@Vector(4, f32){
            values.*[column],
            values.*[column + 1],
            values.*[column + 2],
            values.*[column + 3],
        });
    }
}

pub fn clearBoxRows(
    comptime box_columns: u32,
    comptime rows: u32,
    comptime box_count: u32,
    tile: anytype,
    first_cleared: u32,
) void {
    const box_chunks = box_columns / Sm80FwdTiling.chunk_elements;
    var chunk = first_cleared * box_chunks + thread.lane();
    while (chunk < rows * box_chunks) : (chunk += 32) {
        inline for (0..box_count) |box| tile[box * rows * box_chunks + chunk] = @splat(0);
    }
}

pub const Boxes = struct {
    columns: u32,
    rows: u32,
    box_rows: ?u32 = null,
    count: u32,
};

pub fn copyBoxes(
    comptime boxes: Boxes,
    tile: anytype,
    map: *const TensorMap,
    barrier: anytype,
    coordinates: [4]u32,
) void {
    const box_columns = boxes.columns;
    const rows = boxes.rows;
    const box_rows = boxes.box_rows orelse rows;
    const box_count = boxes.count;
    const first_column, const head, const row, const batch = coordinates;
    inline for (0..box_count) |box_index| {
        inline for (0..rows / box_rows) |part_index| {
            const box: u32 = box_index;
            const part: u32 = part_index;
            const destination = &tile[
                (box * rows + part * box_rows) *
                    (box_columns / Sm80FwdTiling.chunk_elements)
            ];
            const column = first_column + box * box_columns;
            if (@TypeOf(barrier) == u32) {
                copyBoxCtaGroup(
                    map,
                    destination,
                    barrier,
                    column,
                    head,
                    row + part * box_rows,
                    batch,
                );
            } else {
                copyBox(map, destination, barrier, column, head, row + part * box_rows, batch);
            }
        }
    }
}

pub fn copyBox(
    map: *const TensorMap,
    destination: *addrspace(.shared) Chunk,
    barrier: *addrspace(.shared) nvptx.Mbarrier,
    column: u32,
    head: u32,
    row: u32,
    batch: u32,
) void {
    asm volatile ("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes " ++
            "[%[destination]], [%[map], {%[column], %[head], %[row], %[batch]}], [%[barrier]];"
        :
        : [destination] "r" (destination),
          [map] "l" (map),
          [column] "r" (column),
          [head] "r" (head),
          [row] "r" (row),
          [barrier] "r" (barrier),
          [batch] "r" (batch),
        : .{ .memory = true });
}

pub fn prefetchMaps(maps: anytype) void {
    const Maps = @TypeOf(maps);
    inline for (@typeInfo(Maps).@"struct".field_names) |name| {
        if (@FieldType(Maps, name) == *const TensorMap) {
            nvptx.cache.prefetchTensormap(@field(maps, name));
        }
    }
}

pub fn reduceBox(
    map: *const TensorMap,
    source: *addrspace(.shared) Chunk,
    column: u32,
    head: u32,
    row: u32,
    batch: u32,
) void {
    asm volatile ("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group " ++
            "[%[map], {%[column], %[head], %[row], %[batch]}], [%[source]];"
        :
        : [source] "r" (source),
          [map] "l" (map),
          [column] "r" (column),
          [head] "r" (head),
          [row] "r" (row),
          [batch] "r" (batch),
        : .{ .memory = true });
}

pub fn reduceInterleavedBox(
    map: *const TensorMap,
    source: *addrspace(.shared) Chunk,
    column: u32,
    row: u32,
    head: u32,
) void {
    asm volatile ("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group " ++
            "[%[map], {%[column], %[zero], %[row], %[zero], %[head]}], [%[source]];"
        :
        : [source] "r" (source),
          [map] "l" (map),
          [column] "r" (column),
          [zero] "r" (@as(u32, 0)),
          [row] "r" (row),
          [head] "r" (head),
        : .{ .memory = true });
}

pub fn InterleavedReduceStaging(comptime rows: u32, comptime box_columns: u32) type {
    return struct {
        const box_bytes = rows * box_columns * 4;

        pub fn chunks(base: u32, row: u32, lane: u32) [4]u32 {
            const row_address = base + row * box_columns * 4 + lane % 2 * 8;
            const swizzle_offset = (lane % 4 / 2 ^ row % 8) * 16;
            var addresses: [4]u32 = undefined;
            inline for (0..4) |chunk| addresses[chunk] =
                row_address + (@as(u32, chunk * 32) ^ swizzle_offset);
            return addresses;
        }

        pub fn store(
            comptime columns: u32,
            addresses: [4]u32,
            offset: u32,
            accumulator: anytype,
        ) void {
            inline for (0..columns / 8) |tile| {
                const box_offset = tile * 8 / box_columns * box_bytes;
                inline for (0..2) |half| {
                    const destination: *addrspace(.shared) @Vector(2, f32) = @ptrFromInt(
                        addresses[tile % 4] + offset + box_offset + half * 8 * box_columns * 4,
                    );
                    destination.* = .{
                        accumulator[tile * 4 + half * 2], accumulator[tile * 4 + half * 2 + 1],
                    };
                }
            }
        }

        pub fn reduce(
            map: *const TensorMap,
            staging: anytype,
            comptime first_box: u32,
            comptime count: u32,
            comptime column_box: u32,
            row: u32,
            head: u32,
        ) void {
            inline for (first_box..first_box + count) |box| {
                reduceInterleavedBox(
                    map,
                    &staging[box][0],
                    (column_box + box) * box_columns,
                    row,
                    head,
                );
            }
            commitBulkGroup();
        }
    };
}

pub fn storeBox(
    map: *const TensorMap,
    source: *addrspace(.shared) Chunk,
    column: u32,
    head: u32,
    row: u32,
    batch: u32,
) void {
    asm volatile ("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group " ++
            "[%[map], {%[column], %[head], %[row], %[batch]}], [%[source]];"
        :
        : [source] "r" (source),
          [map] "l" (map),
          [column] "r" (column),
          [head] "r" (head),
          [row] "r" (row),
          [batch] "r" (batch),
        : .{ .memory = true });
}

pub fn copyPeer(destination: u32, source: anytype, comptime bytes: u32, barrier: u32) void {
    asm volatile (std.fmt.comptimePrint(
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes " ++
                "[%[destination]], [%[source]], {d}, [%[barrier]];",
            .{bytes},
        )
        :
        : [destination] "r" (destination),
          [source] "r" (source),
          [barrier] "r" (barrier),
        : .{ .memory = true });
}

pub fn waitBulkReadsPending(comptime pending: u32) void {
    nvptx.cp_async_bulk.waitGroup(pending, true);
}

pub const commitBulkGroup = nvptx.cp_async_bulk.commitGroup;

pub fn waitBulkReads() void {
    nvptx.cp_async_bulk.waitGroup(0, true);
}

pub fn SharedStorage(
    comptime tiling: Sm80FwdTiling,
    comptime specialization: configurations.Specialization,
) type {
    const row_chunks = tiling.rowChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const warps = tiling.warps();
    const query_in_stage = tiling.shareQvSmem();
    const Rows = Layout(row_chunks);
    const initial_tiles = if (query_in_stage) stages - 1 else stages;

    return struct {
        pub const Layout = Rows;
        pub const shared_bytes = tiling.sharedBytes();

        pub const Args = struct {
            q: *const TensorMap,
            k: *const TensorMap,
            v: *const TensorMap,
        };

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
        var v_empty: [if (tiling.separate_v_empty) stages else 0]nvptx.Mbarrier addrspace(.shared) =
            undefined;
        var query_full: nvptx.Mbarrier addrspace(.shared) = undefined;

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
            maps: Args,
            head: u32,
            kv_head: u32,
            batch: u32,
            offset_q: u32,
            offset_k: u32,
            first_tile: u32,
            tile_count: u32,
            first_row: u32,

            pub fn init(params: ForwardParams, args: Args, work: anytype) Load {
                const tile = work.tile;
                return .{
                    .maps = args,
                    .head = tile.head,
                    .kv_head = tile.kv_head,
                    .batch = if (params.cu_seqlens_q == 0) tile.batch else 0,
                    .offset_q = tile.sequence.offset_q,
                    .offset_k = tile.sequence.offset_k,
                    .first_tile = tile.n_block_min,
                    .tile_count = tile.n_block_max - tile.n_block_min,
                    .first_row = if (specialization.pack_gqa)
                        tile.block_info.first_row
                    else
                        tile.first_row,
                };
            }

            fn copyTile(load: *const Load, n_block: u32, stage: u32) void {
                const row = load.offset_k + n_block * block_n;
                _ = k_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_n));
                Rows.copyTile(
                    kTile(stage),
                    load.maps.k,
                    &k_full[stage],
                    load.kv_head,
                    row,
                    load.batch,
                );
                _ = v_full[stage].arriveExpectTx(comptime tiling.tileBytes(block_n));
                Rows.copyTile(
                    vTile(stage),
                    load.maps.v,
                    &v_full[stage],
                    load.kv_head,
                    row,
                    load.batch,
                );
            }

            pub fn start(load: *const Load) void {
                if (thread.warp() == 0 and elect.oneSync()) {
                    inline for (&k_full, &v_full, &empty) |*k_stage, *v_stage, *empty_stage| {
                        k_stage.init(1);
                        v_stage.init(1);
                        empty_stage.init(warps);
                    }
                    inline for (&v_empty) |*v_empty_stage| v_empty_stage.init(warps);
                    query_full.init(1);
                    mbarrier.pipelineInitFence();
                    prefetchMaps(load.maps);
                    _ = query_full.arriveExpectTx(comptime tiling.queryBytes());
                    const query_row = load.offset_q + load.first_row;
                    Rows.copyTile(
                        queryTile(),
                        load.maps.q,
                        &query_full,
                        load.head,
                        query_row,
                        load.batch,
                    );
                    inline for (0..initial_tiles) |prefetched| {
                        const first: u32 = prefetched;
                        if (first < load.tile_count) load.copyTile(load.first_tile + first, first);
                    }
                }
                elect.reconverge();
                mbarrier.pipelineInitWait(.cta, warps * 32, .other);
                mbarrier.wait(.cta, &query_full, 0);
            }

            pub fn queryConsumed(load: *const Load) void {
                nvptx.barrierSyncAligned(.work_group, null);
                if (query_in_stage) {
                    if (thread.warp() == 0 and stages - 1 < load.tile_count and elect.oneSync()) {
                        load.copyTile(load.first_tile + stages - 1, stages - 1);
                    }
                    elect.reconverge();
                }
            }

            pub fn acquireK(_: *const Load, index: u32) u32 {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&k_full);
                return state.index();
            }

            pub fn acquireV(_: *const Load, index: u32) void {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerWait(&v_full);
            }

            fn copyReleased(
                load: *const Load,
                index: u32,
                released: *addrspace(.shared) [stages]nvptx.Mbarrier,
                full: *addrspace(.shared) [stages]nvptx.Mbarrier,
                map: *const TensorMap,
                tile: fn (u32) *addrspace(.shared) Rows.Tile(block_n),
            ) void {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerRelease(.warp, released);
                const next = index + stages;
                if (thread.warp() == 0 and next < load.tile_count and elect.oneSync()) {
                    mbarrier.wait(.cta, &released[state.index()], state.phase());
                    _ = full[state.index()].arriveExpectTx(comptime tiling.tileBytes(block_n));
                    Rows.copyTile(
                        tile(state.index()),
                        map,
                        &full[state.index()],
                        load.kv_head,
                        load.offset_k + (load.first_tile + next) * block_n,
                        load.batch,
                    );
                }
                elect.reconverge();
            }

            pub fn releaseKeys(load: *const Load, index: u32) void {
                load.copyReleased(index, &empty, &k_full, load.maps.k, kTile);
            }

            pub fn releaseValues(load: *const Load, index: u32) void {
                load.copyReleased(index, &v_empty, &v_full, load.maps.v, vTile);
            }

            pub fn release(load: *const Load, index: u32) void {
                const state: mbarrier.PipelineState(stages) = .{ .count = index };
                state.consumerRelease(.warp, &empty);
                const refill = index + stages;
                if (thread.warp() == 0 and refill < load.tile_count and elect.oneSync()) {
                    mbarrier.wait(.cta, &empty[state.index()], state.phase());
                    load.copyTile(load.first_tile + refill, state.index());
                }
                elect.reconverge();
            }
        };
    };
}

pub fn BackwardStorage(comptime tiling: configurations.Sm80BwdTiling) type {
    const row_chunks = tiling.rowPitchChunks();
    const block_m = tiling.block_m;
    const block_n = tiling.block_n;
    const stages = tiling.stages;
    const threads = tiling.threads();
    const warps = tiling.warps;
    const dq_accum_warps = if (tiling.bulk_reduce) warps else 0;
    const dq_accum_box_columns = tiling.dqAccumBoxColumns();
    const Rows = Layout(row_chunks);
    const PdS = smem.Layout(block_n / Sm80FwdTiling.chunk_elements);

    return struct {
        pub const Layout = Rows;
        pub const PdSLayout = PdS;
        pub const shared_bytes = tiling.sharedBytes();

        pub const Args = struct {
            q: *const TensorMap,
            k: *const TensorMap,
            v: *const TensorMap,
            do: *const TensorMap,
            dq_accum: if (tiling.bulk_reduce) *const TensorMap else void,
        };

        const DqAccumBox = struct { values: [
            configurations.Sm80BwdTiling.dq_accum_box_rows *
                dq_accum_box_columns
        ]f32 align(smem.alignment) };

        const Tiles = struct {
            k: Rows.Tile(block_n),
            v: Rows.Tile(block_n),
            q: [stages]Rows.Tile(block_m),
            do: [stages]Rows.Tile(block_m),
            p: PdS.Tile(block_m),
            ds: PdS.Tile(block_m),
            dq_accum: [dq_accum_warps]DqAccumBox,
        };

        comptime {
            std.debug.assert(@sizeOf(Tiles) == shared_bytes);
        }

        extern var shared_storage: [0]u8 align(smem.alignment) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var q_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var do_full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub fn kTile() *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.k;
        }

        pub fn vTile() *addrspace(.shared) Rows.Tile(block_n) {
            return &tiles.v;
        }

        pub fn qTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.q[buffer];
        }

        pub fn doTile(buffer: u32) *addrspace(.shared) Rows.Tile(block_m) {
            return &tiles.do[buffer];
        }

        pub fn pTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &tiles.p;
        }

        pub fn dsTile() *addrspace(.shared) PdS.Tile(block_m) {
            return &tiles.ds;
        }

        pub const DqAccum = struct {
            pub const box_columns = dq_accum_box_columns;

            fn box() *addrspace(.shared) DqAccumBox {
                return &tiles.dq_accum[thread.warp()];
            }

            pub fn waitFree() void {
                if (elect.oneSync()) waitBulkReads();
                elect.reconverge();
            }

            pub fn store(row: u32, column: u32, pair: @Vector(2, f32)) void {
                const chunk = smem.swizzledColumn(box_columns * 4, column / 4, row);
                const index = row * box_columns + chunk % (box_columns / 4) * 4 + column % 4;
                const words: *align(8) addrspace(.shared) @Vector(2, f32) =
                    @ptrCast(@alignCast(&box().values[index]));
                words.* = pair;
            }

            pub fn publish(load: *const Load, column: u32, head: u32, row: u32) void {
                nvptx.fenceProxyAsync(.shared);
                elect.reconverge();
                if (elect.oneSync()) {
                    reduceBox(load.maps.dq_accum, @ptrCast(&box().values[0]), column, head, row, 0);
                    commitBulkGroup();
                }
                elect.reconverge();
            }
        };

        pub const Load = struct {
            maps: Args,
            batch: u32,
            varlen: bool,

            pub fn init(params: BackwardParams, args: Args, work: anytype) Load {
                const varlen = params.cu_seqlens_k != 0;
                return .{
                    .maps = args,
                    .batch = if (varlen) 0 else work.batch,
                    .varlen = varlen,
                };
            }

            fn issues() bool {
                return thread.warp() == 0 and elect.oneSync();
            }

            fn copyQuery(
                load: *const Load,
                work: anytype,
                iteration: u32,
                comptime key_rows: u32,
            ) void {
                const step = work.step(iteration);
                const buffer = iteration % stages;
                const row = work.sequence.offset_q + step.block_info.first_row;
                const key_row = work.sequence.offset_k + work.first_key;
                const bytes = comptime tiling.tileBytes(key_rows) + tiling.tileBytes(block_m);
                _ = q_full[buffer].arriveExpectTx(bytes);
                if (key_rows > 0) Rows.copyTile(
                    kTile(),
                    load.maps.k,
                    &q_full[buffer],
                    work.kv_head,
                    key_row,
                    load.batch,
                );
                Rows.copyTile(
                    qTile(buffer),
                    load.maps.q,
                    &q_full[buffer],
                    step.head,
                    row,
                    load.batch,
                );
                _ = do_full[buffer].arriveExpectTx(bytes);
                if (key_rows > 0) Rows.copyTile(
                    vTile(),
                    load.maps.v,
                    &do_full[buffer],
                    work.kv_head,
                    key_row,
                    load.batch,
                );
                Rows.copyTile(
                    doTile(buffer),
                    load.maps.do,
                    &do_full[buffer],
                    step.head,
                    row,
                    load.batch,
                );
            }

            pub fn start(load: *const Load, work: anytype) void {
                if (issues()) {
                    inline for (&q_full, &do_full) |*q_stage, *do_stage| {
                        q_stage.init(1);
                        do_stage.init(1);
                    }
                    mbarrier.pipelineInitFence();
                    prefetchMaps(load.maps);
                    if (work.iterations() > 0) load.copyQuery(work, 0, block_n);
                }
                elect.reconverge();
                mbarrier.pipelineInitWait(.cta, threads, .other);
                if (work.iterations() == 0) return;
                mbarrier.wait(.cta, &q_full[0], 0);
                if (load.varlen) {
                    Rows.clearRows(kTile(), work.key_count, threads, thread.id());
                    Rows.clearRows(
                        qTile(0),
                        work.step(0).block_info.row_count,
                        threads,
                        thread.id(),
                    );
                    nvptx.fenceProxyAsync(.shared);
                }
                nvptx.barrierSyncAligned(.work_group, null);
            }

            pub fn acquireDo(load: *const Load, iteration: u32, work: anytype) void {
                if (iteration > 0) return;
                mbarrier.wait(.cta, &do_full[0], 0);
                if (load.varlen) {
                    Rows.clearRows(vTile(), work.key_count, threads, thread.id());
                    Rows.clearRows(
                        doTile(0),
                        work.step(0).block_info.row_count,
                        threads,
                        thread.id(),
                    );
                    nvptx.fenceProxyAsync(.shared);
                    nvptx.barrierSyncAligned(.work_group, null);
                }
            }

            pub fn prefetch(load: *const Load, next: u32, count: u32, work: anytype) void {
                if (stages == 2 and next < count and issues()) load.copyQuery(work, next, 0);
                elect.reconverge();
            }

            pub fn release(load: *const Load, iteration: u32, count: u32, work: anytype) void {
                const next = iteration + 1;
                if (stages == 1 and next < count) {
                    nvptx.barrierSyncAligned(.work_group, null);
                    if (issues()) load.copyQuery(work, next, 0);
                    elect.reconverge();
                }
            }

            pub fn finish(load: *const Load, iteration: u32, count: u32, work: anytype) void {
                const next = iteration + 1;
                if (next < count) {
                    const state: mbarrier.PipelineState(stages) = .{ .count = next };
                    state.consumerWait(&q_full);
                    state.consumerWait(&do_full);
                    if (load.varlen) {
                        const valid_rows = work.step(next).block_info.row_count;
                        Rows.clearRows(qTile(state.index()), valid_rows, threads, thread.id());
                        Rows.clearRows(doTile(state.index()), valid_rows, threads, thread.id());
                        nvptx.fenceProxyAsync(.shared);
                    }
                }
                nvptx.barrierSyncAligned(.work_group, null);
            }
        };
    };
}

pub fn copyBoxCtaGroup(
    map: *const TensorMap,
    destination: *addrspace(.shared) Chunk,
    barrier: u32,
    column: u32,
    head: u32,
    row: u32,
    batch: u32,
) void {
    asm volatile ("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global" ++
            ".mbarrier::complete_tx::bytes " ++
            "[%[destination]], [%[map], {%[column], %[head], %[row], %[batch]}], [%[barrier]];"
        :
        : [destination] "r" (destination),
          [map] "l" (map),
          [column] "r" (column),
          [head] "r" (head),
          [row] "r" (row),
          [barrier] "r" (barrier),
          [batch] "r" (batch),
        : .{ .memory = true });
}
