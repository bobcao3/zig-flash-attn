const std = @import("std");
const nvptx = std.nvptx;
const configurations = @import("configurations.zig");
const ForwardParams = @import("ForwardParams.zig");
const BackwardParams = @import("BackwardParams.zig");
const thread = @import("thread.zig");
const elect = @import("elect.zig");
const cluster = @import("cluster.zig");
const mbarrier = @import("mbarrier.zig");
const prepare_scheduler = @import("prepare_scheduler.zig");

pub const SchedulingMode = enum { single_tile, static, dynamic, clc };

pub const WorkTileInfo = struct {
    block: u32,
    head: u32,
    batch: u32,
};

pub const GridTile = struct {
    tile_idx: u32,
    split_idx: u32,
    num_block: u32,
    num_head: u32,
    num_qhead_splits: u32,
    block: u32,
    head: u32,
    batch: u32,
};

pub const SplitAxis = enum { x, y };

pub const ReleaseScope = enum { cta, cluster };

pub const CtaRole = enum { leader, follower };

pub const ClcOptions = struct {
    axis: ?SplitAxis,
    ctas: ?u32,
    release_scope: ReleaseScope = .cluster,
    producer_consumes: bool = true,
};

pub const Options = struct {
    block_m: u32,
    row_bytes: u32,
    pack_gqa: bool,
    block_order: configurations.BlockOrder,
    l2_swizzle: configurations.L2SwizzlePolicy,
    scheduler_metadata: bool = false,
    batch_major: bool = false,
};

pub const TileSchedulerArguments = struct {
    num_block: u32,
    num_head: u32,
    num_batch: u32,
    num_qhead_splits: u32 = 1,
    seqlen_other: u32,
    qhead_per_kvhead: u32,
    window_size_left: i32,
    window_size_right: i32,
    cu_seqlens: u64,
    cu_seqlens_other: u64,
    scheduler_metadata: u64 = 0,

    pub fn numHb(args: TileSchedulerArguments) u32 {
        return if (args.cu_seqlens != 0) args.num_head else args.num_head * args.num_batch;
    }
};

pub const Location = struct {
    batch: u32,
    block: u32,
    head: u32 = 0,
};

pub const Section = struct {
    block: u32,
    bidhb: u32,
};

pub const FastDivmodDivisor = struct {
    divisor: u32,
    multiplier: u32,
    shift: u32,

    pub fn init(divisor: u32) FastDivmodDivisor {
        if (divisor <= 1) return .{ .divisor = 1, .multiplier = 0, .shift = 0 };
        const p: u6 = @intCast(31 + 32 - @clz(divisor - 1));
        const multiplier = ((@as(u64, 1) << p) + divisor - 1) / divisor;
        return .{ .divisor = divisor, .multiplier = @intCast(multiplier), .shift = p - 32 };
    }

    pub fn divide(fast: FastDivmodDivisor, dividend: u32) u32 {
        const high: u32 = @intCast(std.math.mulWide(u32, dividend, fast.multiplier) >> 32);
        return if (fast.multiplier == 0) dividend else high >> @intCast(fast.shift);
    }
};

pub const SectionDivisors = struct {
    l2_major_divmod: FastDivmodDivisor,
    l2_minor_divmod: FastDivmodDivisor,
    last_section_divmod: FastDivmodDivisor,
    full_sections: u32,
    num_head_divmod: FastDivmodDivisor,
    qhead_per_kvhead_divmod: FastDivmodDivisor,

    pub fn init(num_block: u32, num_hb: u32, l2_minor: u32, num_head: u32, qhead_per_kvhead: u32) SectionDivisors {
        return .{
            .l2_major_divmod = .init(l2_minor * num_block),
            .l2_minor_divmod = .init(l2_minor),
            .last_section_divmod = .init(num_hb % l2_minor),
            .full_sections = num_hb / l2_minor,
            .num_head_divmod = .init(num_head),
            .qhead_per_kvhead_divmod = .init(qhead_per_kvhead),
        };
    }

    pub fn section(divisors: SectionDivisors, tile_idx: u32) Section {
        const section_idx = divisors.l2_major_divmod.divide(tile_idx);
        const l2_mod = tile_idx - section_idx * divisors.l2_major_divmod.divisor;
        const heads = if (section_idx < divisors.full_sections) divisors.l2_minor_divmod else divisors.last_section_divmod;
        const block = heads.divide(l2_mod);
        return .{ .block = block, .bidhb = section_idx * divisors.l2_minor_divmod.divisor + l2_mod - block * heads.divisor };
    }
};

pub fn gridTile(comptime axis: SplitAxis, num_qhead_splits: u32, ctaid: [3]u32) GridTile {
    const x = ctaid[0];
    const y = ctaid[1];
    const z = ctaid[2];
    const num_x = nvptx.nctaid(0);
    const num_y = nvptx.nctaid(1);
    return switch (axis) {
        .x => .{
            .tile_idx = x / num_qhead_splits + num_x / num_qhead_splits * (y + num_y * z),
            .split_idx = x % num_qhead_splits,
            .num_block = num_x / num_qhead_splits,
            .num_head = num_y,
            .num_qhead_splits = num_qhead_splits,
            .block = x / num_qhead_splits,
            .head = y,
            .batch = z,
        },
        .y => .{
            .tile_idx = x + num_x * (y / num_qhead_splits + num_y / num_qhead_splits * z),
            .split_idx = y % num_qhead_splits,
            .num_block = num_x,
            .num_head = num_y / num_qhead_splits,
            .num_qhead_splits = num_qhead_splits,
            .block = x,
            .head = y / num_qhead_splits,
            .batch = z,
        },
    };
}

pub fn swizzle(size_l2: u32, row_bytes: u32, rows: u32) u32 {
    const bytes = @max(@as(u64, row_bytes) * rows, 1);
    if (size_l2 < bytes) return 1;
    const fit = size_l2 / @as(u32, @intCast(bytes));
    return @as(u32, 1) << @intCast(31 - @clz(fit));
}

pub fn section(tile_idx: u32, num_block: u32, num_hb: u32, l2_minor: u32) Section {
    const l2_major = l2_minor * num_block;
    const section_idx = tile_idx / l2_major;
    const l2_mod = tile_idx % l2_major;
    const num_hb_quotient = num_hb / l2_minor;
    const nheads_in_this_section = if (section_idx < num_hb_quotient) l2_minor else num_hb % l2_minor;
    return .{
        .block = l2_mod / nheads_in_this_section,
        .bidhb = section_idx * l2_minor + l2_mod % nheads_in_this_section,
    };
}

pub fn locate(
    cu_seqlens: u64,
    num_batch: u32,
    block: u32,
    heads: u32,
    comptime block_m: u32,
    comptime block_order: configurations.BlockOrder,
    comptime batch_major: bool,
) ?Location {
    if (num_batch == 0) return null;
    const m_cu_seqlens: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens);
    const lane = thread.lane();
    const up = nvptx.ShuffleClamp.fromWidth(.up, 32);
    const idx = nvptx.ShuffleClamp.fromWidth(.idx, 32);
    var first_block: u32 = 0;
    var first_batch: u32 = 0;
    while (first_batch < num_batch) : (first_batch += 32) {
        const index = first_batch + lane;
        const blocks: u32 = if (index < num_batch)
            (@as(u32, @intCast(m_cu_seqlens[index + 1] - m_cu_seqlens[index])) + block_m - 1) / block_m
        else
            0;
        var inclusive = if (batch_major) blocks * heads else blocks;
        inline for (.{ 1, 2, 4, 8, 16 }) |distance| {
            const previous_sum = nvptx.shflSync(.up, u32, inclusive, distance, up, .all);
            if (lane >= distance) inclusive += previous_sum;
        }
        const window_blocks = nvptx.shflSync(.idx, u32, inclusive, 31, idx, .all);
        if (block < first_block + window_blocks) {
            const local = block - first_block;
            const owner: u32 = @ctz(nvptx.voteBallotSync(inclusive > local, .all));
            const owner_inclusive = nvptx.shflSync(.idx, u32, inclusive, owner, idx, .all);
            const owner_blocks = nvptx.shflSync(.idx, u32, blocks, owner, idx, .all);
            if (batch_major) {
                const offset = local - (owner_inclusive - owner_blocks * heads);
                const sequence_block = offset % owner_blocks;
                return .{
                    .batch = first_batch + owner,
                    .block = if (block_order == .descending) owner_blocks - 1 - sequence_block else sequence_block,
                    .head = offset / owner_blocks,
                };
            }
            return .{
                .batch = first_batch + owner,
                .block = if (block_order == .descending) owner_inclusive - 1 - local else local - (owner_inclusive - owner_blocks),
            };
        }
        first_block += window_blocks;
    }
    return null;
}

pub fn PersistentScheduler(comptime mode: SchedulingMode, comptime options: ClcOptions) type {
    return switch (mode) {
        .single_tile => struct {
            const Scheduler = @This();

            pub fn create() Scheduler {
                return .{};
            }

            pub fn initialWorkTileInfo(_: Scheduler) [3]u32 {
                return thread.ctaid();
            }

            pub fn advanceToNextWork(_: *Scheduler) ?noreturn {
                return null;
            }

            pub fn prefetchNextWork(_: Scheduler) void {}

            pub fn producerTail(_: Scheduler) void {}
        },
        .static => @compileError("a static persistent grid strides over the tiles without shared scheduler state"),
        .dynamic => DynamicPersistentTileFetch(2),
        .clc => ClcTileFetch(options),
    };
}

pub fn ClcTileFetch(comptime options: ClcOptions) type {
    const clustered = options.ctas != 1;
    const cluster_scope = clustered and options.release_scope == .cluster;
    const NextWork = if (options.axis == null) u32 else ?[3]u32;
    const order: mbarrier.Order = if (cluster_scope) .cluster else .cta;
    const release: mbarrier.Arrival = if (!clustered) .thread else if (cluster_scope) .leader_release_cluster else .leader_cta;
    return struct {
        count: u32,

        const Fetch = @This();

        pub const producer_consumes = options.producer_consumes;

        var response: @Vector(4, u32) align(16) addrspace(.shared) = undefined;
        var full: nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: nvptx.Mbarrier addrspace(.shared) = undefined;

        pub fn init(comptime consumer_warps: u32) void {
            const warps = consumer_warps + @intFromBool(producer_consumes);
            full.init(1);
            empty.init(mbarrier.arrivals(release, warps, options.ctas orelse cluster.count()));
        }

        pub fn create() Fetch {
            return .{ .count = 0 };
        }

        pub fn initialWorkTileInfo(_: Fetch) if (options.axis == null) u32 else [3]u32 {
            if (options.axis == null) return @workGroupId(0) / options.ctas.?;
            return thread.ctaid();
        }

        pub fn prefetchNextWork(fetch: Fetch) void {
            fetch.prefetchNextWorkAs(null);
        }

        pub fn prefetchNextWorkAs(fetch: Fetch, comptime role: ?CtaRole) void {
            const leader_cta = if (role) |known| known == .leader else !clustered or cluster.rankIn2Cta() == 0;
            if (leader_cta and fetch.count > 0) mbarrier.wait(order, &empty, fetch.count - 1);
            if (elect.oneSync()) {
                if (!cluster_scope and leader_cta) nvptx.fenceProxyAsync(if (clustered) null else .shared);
                _ = full.arriveExpectTx(@sizeOf(@TypeOf(response)));
                if (leader_cta) {
                    if (cluster_scope) {
                        asm volatile ("fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;" ::: .{ .memory = true });
                    }
                    if (options.ctas) |ctas| {
                        cluster.tryCancel(&response, &full, ctas > 1);
                    } else if (cluster.count() > 1) {
                        cluster.tryCancel(&response, &full, true);
                    } else {
                        cluster.tryCancel(&response, &full, false);
                    }
                }
            }
            elect.reconverge();
        }

        pub fn consumerWait(fetch: Fetch) void {
            mbarrier.wait(order, &full, fetch.count);
        }

        pub fn getCurrentWork(_: Fetch) u32 {
            return cluster.firstCtaid(&response, 0) / options.ctas.?;
        }

        pub fn consumerRelease(_: Fetch) void {
            if (cluster_scope) {
                asm volatile ("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::: .{ .memory = true });
            }
            mbarrier.arrive(release, &empty);
        }

        pub fn advanceConsumerState(fetch: *Fetch) void {
            fetch.count += 1;
        }

        pub fn advanceToNextWork(fetch: *Fetch) NextWork {
            if (options.axis == null) {
                fetch.consumerWait();
                const tile = fetch.getCurrentWork();
                fetch.consumerRelease();
                fetch.advanceConsumerState();
                return tile;
            }
            mbarrier.wait(order, &full, fetch.count);
            fetch.advanceConsumerState();
            var ctaid: [3]u32 = undefined;
            inline for (&ctaid, 0..) |*coordinate, dimension| coordinate.* = cluster.firstCtaid(&response, dimension);
            if (ctaid[0] == std.math.maxInt(u32)) return null;
            fetch.consumerRelease();
            if (clustered) ctaid[@backingInt(options.axis.?)] += cluster.rankIn2Cta();
            return ctaid;
        }

        pub fn producerTail(_: Fetch) void {}
    };
}

pub fn DynamicPersistentTileFetch(comptime stages: u32) type {
    return struct {
        tile_count_semaphore: u64,
        count: u32,

        const Fetch = @This();

        var work_info: [stages]u32 addrspace(.shared) = undefined;
        var full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub fn init(comptime consumer_warps: u32) void {
            inline for (&full, &empty) |*published, *released| {
                published.init(1);
                released.init(32 * consumer_warps);
            }
        }

        pub fn create(tile_count_semaphore: u64) Fetch {
            return .{ .tile_count_semaphore = tile_count_semaphore, .count = 0 };
        }

        pub fn initialWorkTileInfo(_: Fetch) u32 {
            return @workGroupId(0);
        }

        pub fn prefetchNextWork(fetch: *Fetch) void {
            if (elect.oneSync()) {
                const slot = (fetch.count + 1) % stages;
                const counter: *addrspace(.global) u32 = @ptrFromInt(fetch.tile_count_semaphore);
                const tile_idx = nvptx.nctaid(0) + nvptx.atom(.gpu, .monotonic, .add, counter, 1);
                if (fetch.count >= stages) {
                    mbarrier.wait(.cta, &empty[slot], fetch.count / stages - 1);
                }
                work_info[slot] = tile_idx;
                mbarrier.arrive(.thread, &full[slot]);
            }
            elect.reconverge();
            fetch.count += 1;
        }

        pub fn getCurrentWork(fetch: Fetch) u32 {
            return work_info[fetch.count % stages];
        }

        pub fn advanceToNextWork(fetch: *Fetch) u32 {
            const slot = (fetch.count + 1) % stages;
            mbarrier.wait(.cta, &full[slot], fetch.count / stages);
            const tile_idx = work_info[slot];
            mbarrier.arrive(.thread, &empty[slot]);
            fetch.count += 1;
            return tile_idx;
        }

        pub fn producerTail(_: Fetch) void {}
    };
}

pub fn SingleTileScheduler(comptime options: Options) type {
    return struct {
        args: TileSchedulerArguments,

        const Scheduler = @This();

        pub fn create(args: TileSchedulerArguments) Scheduler {
            return .{ .args = args };
        }

        pub fn totalBlocks(scheduler: Scheduler) u32 {
            return scheduler.args.num_block * scheduler.args.numHb();
        }

        pub fn getCurrentWork(scheduler: Scheduler, tile_idx: u32) ?WorkTileInfo {
            const args = scheduler.args;
            const block = tile_idx % args.num_block;
            const bidhb = tile_idx / args.num_block;
            if (options.batch_major and args.cu_seqlens != 0) return scheduler.getGridWork(.{
                .tile_idx = tile_idx,
                .split_idx = 0,
                .num_block = args.num_block,
                .num_head = args.num_head,
                .num_qhead_splits = 1,
                .block = block,
                .head = bidhb,
                .batch = 0,
            });
            if (args.cu_seqlens != 0) return varlenWork(args, block, bidhb);
            return .{
                .block = orderBlock(block, args.num_block),
                .head = bidhb % args.num_head,
                .batch = bidhb / args.num_head,
            };
        }

        pub fn getGridWork(scheduler: Scheduler, grid: GridTile) ?WorkTileInfo {
            const args = scheduler.args;
            if (options.batch_major and !options.scheduler_metadata and args.cu_seqlens != 0) {
                const tile = grid.block + grid.num_block * grid.head;
                const location = locate(args.cu_seqlens, args.num_batch, tile, grid.num_head, options.block_m, options.block_order, true) orelse return null;
                return .{ .block = location.block, .head = location.head, .batch = location.batch };
            }
            if (args.cu_seqlens != 0) return varlenWork(args, grid.block, grid.head);
            return .{ .block = orderBlock(grid.block, args.num_block), .head = grid.head, .batch = grid.batch };
        }

        fn varlenWork(args: TileSchedulerArguments, block: u32, head: u32) ?WorkTileInfo {
            if (options.scheduler_metadata) {
                const entry = prepare_scheduler.lookup(args.scheduler_metadata, args.num_batch, block) orelse return null;
                return .{ .block = entry[1], .head = head, .batch = entry[0] };
            }
            const location = locate(args.cu_seqlens, args.num_batch, block, 1, options.block_m, options.block_order, false) orelse return null;
            return .{ .block = location.block, .head = head, .batch = location.batch };
        }

        fn orderBlock(block: u32, num_block: u32) u32 {
            return if (options.block_order == .descending) num_block - 1 - block else block;
        }
    };
}

pub fn SingleTileLPTScheduler(comptime options: Options) type {
    return struct {
        single: Single,
        l2_minor: u32,
        block_order: configurations.BlockOrder,

        const Scheduler = @This();
        const Single = SingleTileScheduler(options);
        const default_block_order = options.block_order;
        const varlen_sections = for (options.l2_swizzle) |rule| {
            if (rule.varlen) break true;
        } else false;

        pub fn create(args: TileSchedulerArguments) Scheduler {
            const single: Single = .create(args);
            const seqlen_other = sectionSeqlenOther(args);
            inline for (options.l2_swizzle) |rule| {
                if (matches(rule, args, seqlen_other)) {
                    const row_bytes = if (options.pack_gqa) options.row_bytes * args.qhead_per_kvhead else options.row_bytes;
                    const l2_minor = if (rule.size_l2 == configurations.L2SwizzleRule.unlimited) sectionNumHb(args) else @min(swizzle(rule.size_l2, row_bytes, seqlen_other), sectionNumHb(args));
                    return .{
                        .single = single,
                        .l2_minor = if (rule.max_l2_major == configurations.L2SwizzleRule.unlimited) l2_minor else @min(l2_minor, @max(rule.max_l2_major / @max(args.num_block, 1), 1)),
                        .block_order = rule.block_order orelse default_block_order,
                    };
                }
            }
            return .{ .single = single, .l2_minor = 1, .block_order = default_block_order };
        }

        fn swizzles(args: TileSchedulerArguments) bool {
            const seqlen_other = sectionSeqlenOther(args);
            var any = false;
            inline for (options.l2_swizzle) |rule| {
                const match = matches(rule, args, seqlen_other);
                any = any or match;
            }
            return any;
        }

        fn matches(comptime rule: configurations.L2SwizzleRule, args: TileSchedulerArguments, seqlen_other: u32) bool {
            comptime std.debug.assert(!rule.varlen or rule.block_order == null);
            var window_matches = false;
            inline for (comptime std.enums.values(configurations.Mask)) |kind| {
                if (comptime rule.masks.contains(kind)) {
                    const is_kind = switch (kind) {
                        .dense => args.window_size_left < 0 and args.window_size_right < 0,
                        .causal => args.window_size_left < 0 and args.window_size_right == 0,
                        .local_right => args.window_size_left < 0 and args.window_size_right > 0,
                        .local_left => args.window_size_left >= 0 and args.window_size_right < 0,
                        .local_two_sided => args.window_size_left >= 0 and args.window_size_right >= 0,
                    };
                    window_matches = window_matches or is_kind;
                }
            }
            const ctas = @as(u64, args.num_block) * args.num_qhead_splits * sectionNumHb(args);
            return (rule.varlen or args.cu_seqlens == 0) and window_matches and
                seqlen_other >= rule.min_seqlen_other and seqlen_other <= rule.max_seqlen_other and
                args.num_block <= rule.max_num_block and (rule.max_ctas == configurations.L2SwizzleRule.unlimited or ctas <= rule.max_ctas);
        }

        fn sectionSeqlenOther(args: TileSchedulerArguments) u32 {
            if (!varlen_sections or args.cu_seqlens_other == 0) return args.seqlen_other;
            const m_cu_seqlens: [*]addrspace(.global) const i32 = @ptrFromInt(args.cu_seqlens_other);
            return @as(u32, @intCast(m_cu_seqlens[args.num_batch])) / @max(args.num_batch, 1);
        }

        pub fn totalBlocks(scheduler: Scheduler) u32 {
            return scheduler.single.totalBlocks();
        }

        pub fn getCurrentWork(scheduler: Scheduler, tile_idx: u32) ?WorkTileInfo {
            if ((options.batch_major or !varlen_sections) and scheduler.single.args.cu_seqlens != 0) return scheduler.single.getCurrentWork(tile_idx);
            return scheduler.sectionWork(tile_idx);
        }

        pub fn getGridWork(scheduler: Scheduler, grid: GridTile) ?WorkTileInfo {
            if (!swizzles(scheduler.single.args)) return scheduler.single.getGridWork(grid);
            return create(scheduler.single.args).sectionWork(grid.tile_idx);
        }

        fn sectionNumHb(args: TileSchedulerArguments) u32 {
            return if (varlen_sections) args.numHb() else args.num_head * args.num_batch;
        }

        pub fn sectionDivisors(scheduler: Scheduler) SectionDivisors {
            const args = scheduler.single.args;
            return .init(args.num_block, sectionNumHb(args), scheduler.l2_minor, args.num_head, args.qhead_per_kvhead);
        }

        pub fn divmodWork(scheduler: Scheduler, divisors: SectionDivisors, tile_idx: u32) WorkTileInfo {
            const args = scheduler.single.args;
            const coordinates = divisors.section(tile_idx);
            const batch = divisors.num_head_divmod.divide(coordinates.bidhb);
            return .{
                .block = switch (scheduler.block_order) {
                    .ascending => coordinates.block,
                    .descending => args.num_block - 1 - coordinates.block,
                },
                .head = coordinates.bidhb - batch * args.num_head,
                .batch = batch,
            };
        }

        fn sectionWork(scheduler: Scheduler, tile_idx: u32) ?WorkTileInfo {
            const args = scheduler.single.args;
            const coordinates = section(tile_idx, args.num_block, sectionNumHb(args), scheduler.l2_minor);
            if (varlen_sections and args.cu_seqlens != 0) return Single.varlenWork(args, coordinates.block, coordinates.bidhb);
            return .{
                .block = switch (scheduler.block_order) {
                    .ascending => coordinates.block,
                    .descending => args.num_block - 1 - coordinates.block,
                },
                .head = coordinates.bidhb % args.num_head,
                .batch = coordinates.bidhb / args.num_head,
            };
        }
    };
}

pub fn TileScheduler(comptime options: Options) type {
    if (options.l2_swizzle.len == 0) return SingleTileScheduler(options);
    return SingleTileLPTScheduler(options);
}

pub fn forwardArguments(params: ForwardParams) TileSchedulerArguments {
    return .{
        .num_block = params.num_m_blocks,
        .num_head = params.num_head,
        .num_batch = params.num_batch,
        .seqlen_other = params.seqlen_k,
        .qhead_per_kvhead = params.qhead_per_kvhead,
        .window_size_left = params.window_size_left,
        .window_size_right = params.window_size_right,
        .cu_seqlens = params.cu_seqlens_q,
        .cu_seqlens_other = params.cu_seqlens_k,
        .scheduler_metadata = params.scheduler_metadata,
    };
}

pub fn keyArguments(params: BackwardParams, grid: GridTile) TileSchedulerArguments {
    return .{
        .num_block = grid.num_block,
        .num_head = grid.num_head,
        .num_batch = params.num_batch,
        .num_qhead_splits = grid.num_qhead_splits,
        .seqlen_other = params.seqlen_q,
        .qhead_per_kvhead = params.qhead_per_kvhead,
        .window_size_left = params.window_size_left,
        .window_size_right = params.window_size_right,
        .cu_seqlens = params.cu_seqlens_k,
        .cu_seqlens_other = params.cu_seqlens_q,
        .scheduler_metadata = params.scheduler_metadata,
    };
}

test "section visits every block and pair once" {
    var seen: [8 * 12]bool = undefined;
    for (1..9) |num_block_index| {
        for (1..13) |num_hb_index| {
            for (1..num_hb_index + 1) |l2_minor_index| {
                const num_block: u32 = @intCast(num_block_index);
                const num_hb: u32 = @intCast(num_hb_index);
                const l2_minor: u32 = @intCast(l2_minor_index);
                @memset(&seen, false);
                var tile_idx: u32 = 0;
                while (tile_idx < num_block * num_hb) : (tile_idx += 1) {
                    const coordinates = section(tile_idx, num_block, num_hb, l2_minor);
                    try std.testing.expect(coordinates.block < num_block and coordinates.bidhb < num_hb);
                    const index = coordinates.bidhb * num_block + coordinates.block;
                    try std.testing.expect(!seen[index]);
                    seen[index] = true;
                    if (l2_minor == 1) try std.testing.expect(coordinates.block == tile_idx % num_block and coordinates.bidhb == tile_idx / num_block);
                    if (l2_minor == num_hb) try std.testing.expect(coordinates.block == tile_idx / num_hb and coordinates.bidhb == tile_idx % num_hb);
                }
            }
        }
    }
}

test "the multiply-shift division matches the integer division below 2^31" {
    const divisors = [_]u32{ 1, 2, 3, 5, 7, 12, 21, 32, 63, 64, 65, 96, 128, 1000, 4097, 65535, 65536, 1 << 20, 3_000_017, (1 << 31) - 1, 1 << 31 };
    const dividends = [_]u32{ 0, 1, 2, 63, 64, 65, 1023, 4095, 65535, 65537, 1_000_003, 3_000_016, 3_000_017, (1 << 30) + 7, (1 << 31) - 2, (1 << 31) - 1 };
    for (divisors) |divisor| {
        const fast: FastDivmodDivisor = .init(divisor);
        for (dividends) |dividend| try std.testing.expectEqual(dividend / divisor, fast.divide(dividend));
        var dividend: u32 = 0;
        while (dividend < 20_000) : (dividend += 1) try std.testing.expectEqual(dividend / divisor, fast.divide(dividend));
    }
}

test "the precomputed section divisors decode every tile as section does" {
    for (1..9) |num_block_index| {
        for (1..13) |num_hb_index| {
            for (1..num_hb_index + 1) |l2_minor_index| {
                const num_block: u32 = @intCast(num_block_index);
                const num_hb: u32 = @intCast(num_hb_index);
                const l2_minor: u32 = @intCast(l2_minor_index);
                const divisors: SectionDivisors = .init(num_block, num_hb, l2_minor, 3, 2);
                var tile_idx: u32 = 0;
                while (tile_idx < num_block * num_hb) : (tile_idx += 1) {
                    try std.testing.expectEqual(section(tile_idx, num_block, num_hb, l2_minor), divisors.section(tile_idx));
                }
            }
        }
    }
}

test "the swizzle is the largest power of two of reused rows that fit the budget" {
    for ([_]u32{ 0, 1, 255, 256, 4096, 16 << 20, 50 << 20, std.math.maxInt(u32) - 1 }) |size_l2| {
        for ([_]u32{ 1, 256, 288, 512, 1024 }) |row_bytes| {
            for ([_]u32{ 0, 1, 512, 1000, 8192, 65536 }) |rows| {
                const fit = swizzle(size_l2, row_bytes, rows);
                const bytes = @max(@as(u64, row_bytes) * rows, 1);
                try std.testing.expect(std.math.isPowerOfTwo(fit));
                try std.testing.expect(fit == 1 or fit * bytes <= size_l2);
                try std.testing.expect(2 * bytes * fit > size_l2);
            }
        }
    }
}

test "a policy without rules selects the single tile scheduler" {
    const options: Options = .{ .block_m = 128, .row_bytes = 512, .pack_gqa = false, .block_order = .descending, .l2_swizzle = &.{} };
    try std.testing.expect(TileScheduler(options) == SingleTileScheduler(options));
    const swizzled: Options = comptime .{
        .block_m = 128,
        .row_bytes = 512,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = &.{.{ .masks = .initOne(.local_two_sided), .size_l2 = 1 }},
    };
    try std.testing.expect(TileScheduler(swizzled) == SingleTileLPTScheduler(swizzled));
}

test "the first rule that matches the mask kind, sequence length and grid selects the order" {
    const Scheduler = SingleTileLPTScheduler(.{
        .block_m = 128,
        .row_bytes = 512,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = &.{
            .{ .masks = .initOne(.causal), .size_l2 = configurations.L2SwizzleRule.unlimited, .max_num_block = 8, .max_ctas = 512, .block_order = .ascending },
            .{ .masks = .initOne(.causal), .size_l2 = 16 << 20, .min_seqlen_other = 1024 },
            .{ .masks = .initMany(&.{ .local_right, .local_two_sided }), .size_l2 = 16 << 20, .max_seqlen_other = 2048 },
        },
    });
    const causal: TileSchedulerArguments = .{
        .num_block = 16,
        .num_head = 16,
        .num_batch = 8,
        .seqlen_other = 2048,
        .qhead_per_kvhead = 1,
        .window_size_left = -1,
        .window_size_right = 0,
        .cu_seqlens = 0,
        .cu_seqlens_other = 0,
    };
    try std.testing.expectEqual(@as(u32, 16), Scheduler.create(causal).l2_minor);
    try std.testing.expectEqual(.descending, Scheduler.create(causal).block_order);
    var grouped = causal;
    grouped.qhead_per_kvhead = 4;
    try std.testing.expectEqual(@as(u32, 16), Scheduler.create(grouped).l2_minor);
    var small = causal;
    small.num_block = 8;
    small.num_head = 4;
    small.num_batch = 16;
    try std.testing.expectEqual(@as(u32, 64), Scheduler.create(small).l2_minor);
    try std.testing.expectEqual(.ascending, Scheduler.create(small).block_order);
    small.num_qhead_splits = 2;
    try std.testing.expectEqual(@as(u32, 16), Scheduler.create(small).l2_minor);
    try std.testing.expectEqual(.descending, Scheduler.create(small).block_order);
    var short = causal;
    short.seqlen_other = 512;
    try std.testing.expectEqual(@as(u32, 1), Scheduler.create(short).l2_minor);
    var varlen = small;
    varlen.num_qhead_splits = 1;
    varlen.cu_seqlens = 64;
    try std.testing.expectEqual(@as(u32, 1), Scheduler.create(varlen).l2_minor);
    try std.testing.expectEqual(.descending, Scheduler.create(varlen).block_order);
    var dense = causal;
    dense.window_size_right = -1;
    try std.testing.expectEqual(@as(u32, 1), Scheduler.create(dense).l2_minor);
    var local_right = causal;
    local_right.window_size_right = 64;
    try std.testing.expectEqual(@as(u32, 16), Scheduler.create(local_right).l2_minor);
    var local_two_sided = causal;
    local_two_sided.window_size_left = 256;
    try std.testing.expectEqual(@as(u32, 16), Scheduler.create(local_two_sided).l2_minor);
    local_two_sided.seqlen_other = 4096;
    try std.testing.expectEqual(@as(u32, 1), Scheduler.create(local_two_sided).l2_minor);
    var local_left = causal;
    local_left.window_size_left = 256;
    local_left.window_size_right = -1;
    try std.testing.expectEqual(@as(u32, 1), Scheduler.create(local_left).l2_minor);
}

test "the heavy-first rules visit the key blocks of a small one-sided grid in sections of 128 tiles" {
    const Scheduler = SingleTileLPTScheduler(.{
        .block_m = 128,
        .row_bytes = 1024,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = &configurations.Sm100BwdTiling.lpt_l2_swizzle,
    });
    const windows = [_][2]i32{ .{ -1, -1 }, .{ -1, 0 }, .{ -1, 32 }, .{ 32, -1 }, .{ 32, 0 } };
    for (windows) |window_sizes| for ([_]u32{ 1, 3, 4, 31, 32, 33 }) |num_block| for ([_]u32{ 1, 5, 16 }) |num_head| for ([_]u32{ 1, 2 }) |num_qhead_splits| for ([_]u32{ 1, 3, 8 }) |num_batch| {
        const scheduler: Scheduler = .create(.{
            .num_block = num_block,
            .num_head = num_head,
            .num_batch = num_batch,
            .num_qhead_splits = num_qhead_splits,
            .seqlen_other = 4096,
            .qhead_per_kvhead = 8,
            .window_size_left = window_sizes[0],
            .window_size_right = window_sizes[1],
            .cu_seqlens = 0,
            .cu_seqlens_other = 0,
        });
        const one_sided = (window_sizes[0] < 0) != (window_sizes[1] < 0);
        if (!one_sided or num_block > 32 or num_block * num_head * num_qhead_splits * num_batch > 1024) {
            try std.testing.expectEqual(@as(u32, 1), scheduler.l2_minor);
            try std.testing.expectEqual(.ascending, scheduler.block_order);
            continue;
        }
        const num_hb = num_head * num_batch;
        const width = @max(1, 128 / num_block);
        var tile_idx: u32 = 0;
        while (tile_idx < num_block * num_hb) : (tile_idx += 1) {
            const first_bidhb = tile_idx / (width * num_block) * width;
            const offset = tile_idx % (width * num_block);
            const section_width = @min(width, num_hb - first_bidhb);
            const position = offset / section_width;
            const bidhb = first_bidhb + offset % section_width;
            const tile = scheduler.sectionWork(tile_idx).?;
            try std.testing.expectEqual(if (window_sizes[0] >= 0) num_block - 1 - position else position, tile.block);
            try std.testing.expectEqual(bidhb % num_head, tile.head);
            try std.testing.expectEqual(bidhb / num_head, tile.batch);
        }
    };
}

test "an unlimited L2 budget puts every pair in one section" {
    const Scheduler = SingleTileLPTScheduler(.{
        .block_m = 256,
        .row_bytes = 256,
        .pack_gqa = false,
        .block_order = .descending,
        .l2_swizzle = &.{.{ .masks = .initOne(.causal), .size_l2 = configurations.L2SwizzleRule.unlimited }},
    });
    var args: TileSchedulerArguments = .{
        .num_block = 16,
        .num_head = 48,
        .num_batch = 64,
        .seqlen_other = 4096,
        .qhead_per_kvhead = 1,
        .window_size_left = -1,
        .window_size_right = 0,
        .cu_seqlens = 0,
        .cu_seqlens_other = 0,
    };
    try std.testing.expectEqual(@as(u32, 48 * 64), Scheduler.create(args).l2_minor);
    args.qhead_per_kvhead = 4;
    try std.testing.expectEqual(@as(u32, 48 * 64), Scheduler.create(args).l2_minor);
}

test "the swizzle gate is open whenever the selected rule changes the order" {
    const Scheduler = SingleTileLPTScheduler(.{
        .block_m = 128,
        .row_bytes = 1024,
        .pack_gqa = true,
        .block_order = .ascending,
        .l2_swizzle = comptime configurations.backwardSm100(128).l2_swizzle,
    });
    const windows = [_][2]i32{ .{ -1, -1 }, .{ -1, 0 }, .{ -1, 32 }, .{ 32, -1 }, .{ 32, 0 } };
    var open: u32 = 0;
    var total: u32 = 0;
    for (windows) |window_sizes| for ([_]u32{ 1, 8, 32, 33 }) |num_block| for ([_]u32{ 1, 16 }) |num_head| for ([_]u32{ 1, 2 }) |num_qhead_splits| for ([_]u32{ 1, 8, 64 }) |num_batch| for ([_]u32{ 512, 2048, 4096 }) |seqlen_other| {
        const args: TileSchedulerArguments = .{
            .num_block = num_block,
            .num_head = num_head,
            .num_batch = num_batch,
            .num_qhead_splits = num_qhead_splits,
            .seqlen_other = seqlen_other,
            .qhead_per_kvhead = 8,
            .window_size_left = window_sizes[0],
            .window_size_right = window_sizes[1],
            .cu_seqlens = 0,
            .cu_seqlens_other = 0,
        };
        const scheduler: Scheduler = .create(args);
        total += 1;
        if (Scheduler.swizzles(args)) {
            open += 1;
        } else {
            try std.testing.expectEqual(@as(u32, 1), scheduler.l2_minor);
            try std.testing.expectEqual(.ascending, scheduler.block_order);
        }
    };
    try std.testing.expect(open > 0 and open < total);
}
