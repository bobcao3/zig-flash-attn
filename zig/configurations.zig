const std = @import("std");
const SchedulingMode = @import("tile_scheduler.zig").SchedulingMode;

pub const warps_per_warp_group = 4;
pub const registers_per_cta = 65536;
pub const head_dims = [_]u32{ 64, 72, 96, 128, 256 };

pub const Mask = enum {
    dense,
    causal,
    local_right,
    local_left,
    local_two_sided,

    pub const bounded: std.EnumSet(Mask) =
        .initMany(&.{ .causal, .local_right, .local_left, .local_two_sided });
};

pub const BlockOrder = enum { ascending, descending };

pub const Tier =
    enum(u3) { regular, limited_1, limited_2, local_two_sided, local_two_sided_limited_2 };

pub const Ex2EmulationPlacement = enum { interleaved, leading, inner_fragments };

pub const Mode = enum(u5) {
    regular = 0,
    varlen = 1,
    gqa = 2,
    varlen_gqa = 3,
    limited_1 = 4,
    varlen_limited_1 = 5,
    gqa_limited_1 = 6,
    varlen_gqa_limited_1 = 7,
    limited_2 = 8,
    varlen_limited_2 = 9,
    gqa_limited_2 = 10,
    varlen_gqa_limited_2 = 11,
    local_two_sided = 12,
    gqa_local_two_sided = 14,
    local_two_sided_limited_2 = 16,

    pub fn isVarlen(mode: Mode) bool {
        return @backingInt(mode) & 1 != 0;
    }

    pub fn tier(mode: Mode) Tier {
        return @fromBackingInt(@intCast(@backingInt(mode) >> 2));
    }

    pub fn isLimited1(mode: Mode) bool {
        return mode.tier() == .limited_1;
    }

    pub fn suffix(mode: Mode) []const u8 {
        return switch (mode) {
            .regular => "",
            inline else => |tag| "_" ++ @tagName(tag),
        };
    }
};

pub const SequenceLimits = struct {
    min_seqlen_q: u32 = 0,
    max_seqlen_q: u32 = 0,
    max_seqlen_k: u32 = 0,
    max_window_size: u32 = 0,
    two_sided_window: bool = false,
    masks: std.EnumSet(Mask) = .full,
    batched_only: bool = false,
};

pub const Specialization = struct {
    scheduler_metadata: bool = false,
    sort_batches: bool = false,
    pack_gqa: bool = false,
    head_split: bool = false,
    limits: SequenceLimits = .{},
};

pub fn Variant(comptime T: type) type {
    return struct {
        tiling: T,
        specialization: Specialization = .{},
    };
}

const varlen_sorted: Specialization = .{ .scheduler_metadata = true, .sort_batches = true };
const pack_gqa_only: Specialization = .{ .pack_gqa = true };

pub const Declaration = struct {
    varlen_head_dims: []const u32 = &.{},
    gqa_head_dims: []const u32 = &.{},
    gqa_limited_1_head_dims: []const u32 = &.{},
    varlen_gqa_head_dims: []const u32 = &.{},
    gqa_specialization: Specialization = .{},
    limited_1_limits: SequenceLimits = .{},
    limited_2_limits: ?SequenceLimits = null,
    local_head_dims: []const u32 = &.{},
};

fn declare(comptime T: type, tiling: T, mode: Mode, declaration: Declaration) ?Variant(T) {
    const head_dim = tiling.head_dim;
    const untiered: Mode = @fromBackingInt(@backingInt(mode) & 3);
    var specialization: Specialization = switch (untiered) {
        .regular => .{},
        .varlen => if (std.mem.indexOfScalar(
            u32,
            declaration.varlen_head_dims,
            head_dim,
        ) != null) varlen_sorted else return null,
        .gqa => if ((mode.tier() == .regular or mode.tier() == .local_two_sided) and
            std.mem.indexOfScalar(u32, declaration.gqa_head_dims, head_dim) != null)
            declaration.gqa_specialization
        else if (mode.tier() == .limited_1 and
            std.mem.indexOfScalar(u32, declaration.gqa_limited_1_head_dims, head_dim) != null)
            declaration.gqa_specialization
        else
            return null,
        .varlen_gqa => if (mode.tier() == .regular and std.mem.indexOfScalar(
            u32,
            declaration.varlen_gqa_head_dims,
            head_dim,
        ) != null) varlen_sorted else return null,
        else => return null,
    };
    switch (mode.tier()) {
        .regular => {},
        .limited_1 => specialization.limits = declaration.limited_1_limits,
        .limited_2 => specialization.limits = declaration.limited_2_limits orelse return null,
        .local_two_sided, .local_two_sided_limited_2 => {
            if (std.mem.indexOfScalar(
                u32,
                declaration.local_head_dims,
                head_dim,
            ) == null) return null;
            if (mode.tier() == .local_two_sided_limited_2) specialization.limits =
                declaration.limited_2_limits orelse return null;
            specialization.limits.two_sided_window = true;
        },
    }
    return .{ .tiling = tiling, .specialization = specialization };
}

pub const Source = enum {
    flash_fwd_sm80,
    flash_bwd_sm80,
    flash_fwd_sm120,
    flash_fwd_sm90,
    flash_bwd_sm90,
    flash_fwd_sm100,
    flash_bwd_sm100,
    flash_bwd_sm120,

    pub fn arch(source: Source) u32 {
        return switch (source) {
            .flash_fwd_sm80, .flash_bwd_sm80 => 80,
            .flash_fwd_sm120, .flash_bwd_sm120 => 120,
            .flash_fwd_sm90, .flash_bwd_sm90 => 90,
            .flash_fwd_sm100, .flash_bwd_sm100 => 100,
        };
    }
};

pub fn launch(comptime source: Source, head_dim: u32, mode: Mode) ?Launch {
    const variant = switch (source) {
        .flash_fwd_sm80 => forwardSm80Variant(head_dim, mode),
        .flash_bwd_sm80 => backwardSm80Variant(head_dim, mode),
        .flash_fwd_sm120 => forwardSm120Variant(head_dim, mode),
        .flash_fwd_sm90 => forwardSm90Variant(head_dim, mode),
        .flash_bwd_sm90 => backwardSm90Variant(head_dim, mode),
        .flash_fwd_sm100 => forwardSm100Variant(head_dim, mode),
        .flash_bwd_sm100 => backwardSm100Variant(head_dim, mode),
        .flash_bwd_sm120 => backwardSm120Variant(head_dim, mode),
    } orelse return null;
    const tile = variant.tiling;
    const specialization = variant.specialization;
    const limits = specialization.limits;
    const cta_tiler = source == .flash_fwd_sm100;
    var result: Launch = .{
        .threads = tile.threads(),
        .shared_bytes = tile.sharedBytes(),
        .block_m = if (cta_tiler) tile.ctaTilerM() else tile.block_m,
        .block_n = if (cta_tiler) tile.block_n / tile.clusterCtas() else tile.block_n,
        .scheduling_mode = tile.scheduling_mode,
        .scheduler_metadata = @intFromBool(specialization.scheduler_metadata),
        .pack_gqa = @intFromBool(specialization.pack_gqa),
        .head_split = @intFromBool(specialization.head_split),
        .min_seqlen_q = limits.min_seqlen_q,
        .max_seqlen_q = limits.max_seqlen_q,
        .max_seqlen_k = limits.max_seqlen_k,
        .max_window_size = limits.max_window_size,
        .two_sided_window = @intFromBool(limits.two_sided_window),
        .batched_only = @intFromBool(limits.batched_only),
        .arch = source.arch(),
    };
    var masks = limits.masks.iterator();
    while (masks.next()) |mask| result.masks |= @as(u32, 1) << @backingInt(mask);
    switch (source) {
        .flash_fwd_sm80 => {},
        .flash_bwd_sm80 => {
            const schedule = @import("Sm80BwdSchedule.zig").init(head_dim, mode);
            result.shared_bytes = schedule.sharedBytes(tile);
        },
        .flash_fwd_sm120, .flash_bwd_sm120 => {
            const pitch = tile.rowPitchChunks();
            const swizzle = swizzleBytes(pitch);
            result.box_columns = boxChunks(pitch, swizzle) * Sm80FwdTiling.chunk_elements;
            result.swizzle_bytes = swizzleBytes(tile.rowChunks());
        },
        else => {
            result.box_columns = tile.boxColumns();
            result.swizzle_bytes = tile.swizzleBytes();
        },
    }
    switch (source) {
        .flash_bwd_sm120 => {
            result.dq_accum_box_rows = if (tile.bulk_reduce) Sm80BwdTiling.dq_accum_box_rows else 0;
            result.bulk_reduce = @intFromBool(tile.bulk_reduce);
            result.dq_accum_box_columns = if (tile.bulk_reduce) tile.dqAccumBoxColumns() else 0;
            result.pdl = @intFromBool(tile.pdl);
        },
        .flash_fwd_sm90 => result.pdl = result.scheduler_metadata,
        .flash_bwd_sm90 => {
            result.bulk_reduce = 1;
            result.dq_accum_box_columns = Sm90BwdTiling.dq_accum_box_columns;
            result.pdl = 1;
            result.min_qhead_per_split = 2;
            result.dq_accum_interleave = 1;
            result.max_qhead_splits = 8;
            result.qhead_split_bounded_only = 1;
            result.qhead_split_by_waves = 1;
        },
        .flash_fwd_sm100 => {
            result.grid_per_tile_when_windowed = 1;
            result.q_box_rows = tile.m_block_size;
            result.cluster_ctas = if (tile.use_2cta_instrs) tile.clusterCtas() else 0;
            result.pdl = result.scheduler_metadata;
        },
        .flash_bwd_sm100 => {
            const separate_dq = tile.dq_block_m != 0;
            result.dq_accum_box_rows = if (tile.use_2cta_instrs) tile.block_m / 2 else 0;
            result.bulk_reduce = @intFromBool(!separate_dq);
            result.cluster_ctas = if (!separate_dq and !tile.use_2cta_instrs) 0 else 2;
            result.dq_block_m = tile.dq_block_m;
            result.dq_accum_box_columns = if (separate_dq) 0 else Sm100BwdTiling.dq_reduce_ncol;
            result.pdl = @intFromBool(tile.pdl);
            result.min_qhead_per_split = if (separate_dq) 0 else tile.min_qhead_per_split;
            result.max_qhead_splits = if (!separate_dq and !tile.use_2cta_instrs) 2 else 0;
            result.dkv_box_rows = if (tile.use_tma_store) Sm100BwdTiling.dkv_box_rows else 0;
        },
        else => {},
    }
    return result;
}

pub const L2SwizzleRule = struct {
    masks: std.EnumSet(Mask),
    size_l2: u32,
    max_l2_major: u32 = unlimited,
    min_seqlen_other: u32 = 0,
    max_seqlen_other: u32 = unlimited,
    max_num_block: u32 = unlimited,
    max_ctas: u32 = unlimited,
    varlen: bool = false,
    block_order: ?BlockOrder = null,

    pub const unlimited: u32 = std.math.maxInt(u32);
};

pub const L2SwizzlePolicy = []const L2SwizzleRule;

pub fn maxRegistersPerThread(thread_count: u32) u32 {
    return registers_per_cta / thread_count / 8 * 8;
}

/// Returns the XOR swizzle of a shared memory row layout that `cp.async` fills. Rows of 4 chunks
/// stay unswizzled and get padded instead.
pub fn rowSwizzleBytes(row_chunks: u32) u32 {
    return if (row_chunks % 8 == 0) 128 else if (row_chunks % 4 == 0 and row_chunks > 4) 64 else 0;
}

pub fn pitchChunks(row_chunks: u32) u32 {
    return if (rowSwizzleBytes(row_chunks) != 0 or row_chunks % 2 == 1) row_chunks else row_chunks +
        1;
}

pub fn swizzleBytes(row_chunks: u32) u32 {
    return if (row_chunks % 8 == 0) 128 else if (row_chunks % 4 == 0) 64 else 0;
}

pub fn wgmmaSwizzleBytes(columns: u32) u32 {
    return if (columns % 64 == 0) 128 else if (columns % 32 == 0) 64 else 32;
}

pub fn boxChunks(row_chunks: u32, swizzle_bytes: u32) u32 {
    return if (swizzle_bytes == 0) row_chunks else swizzle_bytes / Sm80FwdTiling.chunk_bytes;
}

fn alignedHeadDim(tiling: anytype) u32 {
    return std.mem.alignForward(u32, tiling.head_dim, tiling.hdim_multiple_of);
}

fn paddedRowChunks(tiling: anytype) u32 {
    return tiling.headDimPadded() / Sm80FwdTiling.chunk_elements;
}

fn paddedSwizzleBytes(tiling: anytype) u32 {
    return wgmmaSwizzleBytes(tiling.headDimPadded());
}

fn paddedBoxColumns(tiling: anytype) u32 {
    return tiling.swizzleBytes() / Sm80FwdTiling.element_bytes;
}

fn paddedTileBytes(tiling: anytype, rows: u32) u32 {
    return rows * tiling.headDimPadded() * Sm80FwdTiling.element_bytes;
}

pub const Sm80FwdTiling = struct {
    head_dim: u32,
    block_m: u32,
    block_n: u32,
    stages: u32,
    prefetch_stages: u32,
    q_in_regs: bool,
    /// Holds only this many 16-column steps of the query in registers. The kernel loads them
    /// from global memory, and the query tile in shared memory keeps the remaining columns.
    q_register_steps: ?u32 = null,
    tma: bool = false,
    max_registers: ?u32 = null,
    mma_m_per_warp: u32 = 1,
    separate_v_empty: bool = false,
    batch_major: bool = false,
    l2_swizzle: L2SwizzlePolicy = &.{},
    scheduling_mode: SchedulingMode = .single_tile,

    pub const chunk_bytes = 16;
    pub const chunk_elements = 8;
    pub const element_bytes = chunk_bytes / chunk_elements;

    pub fn rowChunks(tiling: Sm80FwdTiling) u32 {
        return tiling.head_dim / chunk_elements;
    }

    pub fn rowPitchChunks(tiling: Sm80FwdTiling) u32 {
        return if (tiling.tma) tiling.rowChunks() else pitchChunks(tiling.rowChunks());
    }

    pub fn warps(tiling: Sm80FwdTiling) u32 {
        return tiling.block_m / (16 * tiling.mma_m_per_warp);
    }

    pub fn threads(tiling: Sm80FwdTiling) u32 {
        return tiling.warps() * 32;
    }

    pub fn shareQvSmem(tiling: Sm80FwdTiling) bool {
        return tiling.q_in_regs and tiling.stages > 1 and tiling.block_m <= 2 * tiling.block_n;
    }

    pub fn tileBytes(tiling: Sm80FwdTiling, rows: u32) u32 {
        return rows * tiling.rowPitchChunks() * chunk_bytes;
    }

    /// Returns the number of chunks per row of the query tile in shared memory.
    pub fn queryChunks(tiling: Sm80FwdTiling) u32 {
        const steps = tiling.q_register_steps orelse return tiling.rowChunks();
        return tiling.rowChunks() - 2 * steps;
    }

    pub fn queryBytes(tiling: Sm80FwdTiling) u32 {
        if (tiling.q_register_steps == null) return tiling.tileBytes(tiling.block_m);
        return tiling.block_m * pitchChunks(tiling.queryChunks()) * chunk_bytes;
    }

    pub fn sharedBytes(tiling: Sm80FwdTiling) u32 {
        const stages = tiling.stages * 2 * tiling.tileBytes(tiling.block_n);
        return if (tiling.shareQvSmem()) @max(stages, tiling.queryBytes()) else stages +
            tiling.queryBytes();
    }
};

pub fn forwardSm80(head_dim: u32) Sm80FwdTiling {
    return switch (head_dim) {
        64 => .{
            .head_dim = 64,
            .block_m = 128,
            .block_n = 64,
            .stages = 3,
            .prefetch_stages = 1,
            .q_in_regs = true,
            .max_registers = 128,
        },
        72 => .{
            .head_dim = 72,
            .block_m = 128,
            .block_n = 64,
            .stages = 2,
            .prefetch_stages = 1,
            .q_in_regs = true,
            .max_registers = 128,
        },
        96 => .{
            .head_dim = 96,
            .block_m = 128,
            .block_n = 64,
            .stages = 3,
            .prefetch_stages = 1,
            .q_in_regs = true,
        },
        128 => .{
            .head_dim = 128,
            .block_m = 128,
            .block_n = 64,
            .stages = 3,
            .prefetch_stages = 1,
            .q_in_regs = true,
        },
        256 => .{
            .head_dim = 256,
            .block_m = 128,
            .block_n = 16,
            .stages = 2,
            .prefetch_stages = 1,
            .q_in_regs = true,
        },
        else => @panic("unsupported head dimension"),
    };
}

pub const Launch = struct {
    threads: u32,
    shared_bytes: u32,
    block_m: u32,
    block_n: u32,
    box_columns: u32 = 0,
    swizzle_bytes: u32 = 0,
    scheduling_mode: SchedulingMode = .single_tile,
    grid_per_tile_when_windowed: u32 = 0,
    q_box_rows: u32 = 0,
    dq_accum_box_rows: u32 = 0,
    bulk_reduce: u32 = 0,
    cluster_ctas: u32 = 0,
    dq_block_m: u32 = 0,
    dq_accum_box_columns: u32 = 0,
    pdl: u32 = 0,
    min_qhead_per_split: u32 = 0,
    dq_accum_interleave: u32 = 0,
    max_qhead_splits: u32 = 0,
    qhead_split_bounded_only: u32 = 0,
    qhead_split_by_waves: u32 = 0,
    scheduler_metadata: u32 = 0,
    pack_gqa: u32 = 0,
    head_split: u32 = 0,
    min_seqlen_q: u32 = 0,
    max_seqlen_q: u32 = 0,
    max_seqlen_k: u32 = 0,
    max_window_size: u32 = 0,
    dkv_box_rows: u32 = 0,
    two_sided_window: u32 = 0,
    masks: u32 = 0,
    batched_only: u32 = 0,
    arch: u32,
};

pub fn forwardSm80Limited1(head_dim: u32) ?Sm80FwdTiling {
    return switch (head_dim) {
        64, 96, 128 => .{
            .head_dim = head_dim,
            .block_m = 64,
            .block_n = 32,
            .stages = 2,
            .prefetch_stages = 1,
            .q_in_regs = true,
        },
        else => null,
    };
}

fn forwardSm80Regular(head_dim: u32) Sm80FwdTiling {
    var tiling = forwardSm80(head_dim);
    if (head_dim == 96) {
        tiling.block_n = 64;
        tiling.stages = 1;
        tiling.q_in_regs = false;
        tiling.mma_m_per_warp = 2;
        tiling.separate_v_empty = true;
    }
    if (head_dim == 256) {
        tiling.block_m = 64;
        tiling.block_n = 64;
        tiling.stages = 1;
        tiling.q_in_regs = false;
        tiling.separate_v_empty = true;
    }
    return tiling;
}

/// Returns the tiling for dense and causal attention over short sequences. Rows of 64 queries waste
/// less of the causal triangle, and two CTAs of 4 warps fit on an SM.
fn forwardSm80Short(head_dim: u32) Sm80FwdTiling {
    var tiling = forwardSm80(head_dim);
    tiling.block_m = 64;
    tiling.block_n = 64;
    tiling.stages = 1;
    tiling.q_in_regs = true;
    tiling.separate_v_empty = true;
    return tiling;
}

pub fn forwardSm80Variant(head_dim: u32, mode: Mode) ?Variant(Sm80FwdTiling) {
    const tiling = switch (mode.tier()) {
        .limited_1 => forwardSm80Limited1(head_dim) orelse return null,
        .limited_2 => if (head_dim == 128) forwardSm80Short(head_dim) else switch (mode) {
            .varlen_limited_2 => switch (head_dim) {
                72 => batch_major: {
                    var tiling = forwardSm80(head_dim);
                    tiling.batch_major = true;
                    break :batch_major tiling;
                },
                96 => forwardSm80Regular(head_dim),
                else => return null,
            },
            else => return null,
        },
        else => forwardSm80Regular(head_dim),
    };
    const limited_1_limits: SequenceLimits = .{ .max_seqlen_q = 256 };
    var variant =
        declare(Sm80FwdTiling, tiling, if (mode == .varlen_limited_2) .limited_2 else mode, .{
            .varlen_head_dims = &.{ 64, 128 },
            .gqa_head_dims = &.{ 64, 128 },
            .gqa_specialization = pack_gqa_only,
            .limited_1_limits = limited_1_limits,
            .limited_2_limits = .{ .max_seqlen_q = 1024 },
        }) orelse return null;
    if (mode.tier() == .limited_2 and head_dim == 128) {
        variant.specialization.limits.masks = .initMany(&.{ .causal, .dense });
        variant.specialization.limits.max_seqlen_k = 2048;
        if (mode == .varlen_limited_2) variant.specialization.limits.max_seqlen_q = 2048;
    }
    if (mode == .varlen_limited_2 and forwardSm80Limited1(head_dim) != null) {
        variant.specialization.limits.min_seqlen_q = limited_1_limits.max_seqlen_q + 1;
    }
    return variant;
}

pub const Sm80BwdTiling = struct {
    head_dim: u32,
    block_m: u32,
    block_n: u32,
    warps: u32,
    stages: u32,
    /// Holds the value operand of the dP GEMM in registers for the whole CTA.
    v_in_regs: bool = false,
    /// Sets the number of 16-row tiles of the S and dP GEMMs that one warp computes.
    score_row_tiles: u32 = 1,
    /// Sets the number of 16-row tiles of the dK, dV and dQ GEMMs that one warp computes.
    gradient_row_tiles: u32 = 1,
    tma: bool = false,
    bulk_reduce: bool = false,
    p_first: bool = false,
    alternate_gradient_order: bool = false,
    pdl: bool = false,
    batch_major: bool = false,
    l2_swizzle: L2SwizzlePolicy = &.{},
    scheduling_mode: SchedulingMode = .single_tile,

    pub const dq_accum_box_rows = 16;

    pub fn headDimPadded(tiling: Sm80BwdTiling) u32 {
        return std.mem.alignForward(u32, tiling.head_dim, 16);
    }

    pub const rowChunks = paddedRowChunks;

    pub fn rowPitchChunks(tiling: Sm80BwdTiling) u32 {
        const chunks = tiling.rowChunks();
        return if (tiling.tma and swizzleBytes(chunks) != 0) chunks else pitchChunks(chunks);
    }

    pub fn threads(tiling: Sm80BwdTiling) u32 {
        return tiling.warps * 32;
    }

    pub fn tileBytes(tiling: Sm80BwdTiling, rows: u32) u32 {
        return rows * tiling.rowPitchChunks() * Sm80FwdTiling.chunk_bytes;
    }

    /// Names the shared memory that the value tile occupies. When the warps hold the values in
    /// registers, they copy the values before anything else writes that memory.
    pub const ValueTile = enum { own, last_query_stage, shared_with_pds };

    pub fn valueTile(tiling: Sm80BwdTiling) ValueTile {
        if (!tiling.v_in_regs) return .own;
        if (tiling.stages > 1 and tiling.block_n <= tiling.block_m) return .last_query_stage;
        return .shared_with_pds;
    }

    pub fn pdsBytes(tiling: Sm80BwdTiling) u32 {
        return 2 * tiling.block_m *
            pitchChunks(tiling.block_n / Sm80FwdTiling.chunk_elements) * Sm80FwdTiling.chunk_bytes;
    }

    pub fn dqAccumBoxColumns(tiling: Sm80BwdTiling) u32 {
        const warp_columns = tiling.headDimPadded() * (tiling.block_m / 16) / tiling.warps;
        return if (warp_columns % 32 == 0) 32 else 16;
    }

    pub fn sharedBytes(tiling: Sm80BwdTiling) u32 {
        const pds_bytes = tiling.pdsBytes();
        const dq_accum_staging = if (tiling.bulk_reduce)
            tiling.warps * dq_accum_box_rows * tiling.dqAccumBoxColumns() * @sizeOf(f32)
        else
            0;
        const value_bytes = tiling.tileBytes(tiling.block_n);
        const values_and_pds = switch (tiling.valueTile()) {
            .own => value_bytes + pds_bytes,
            .last_query_stage => pds_bytes,
            .shared_with_pds => @max(value_bytes, pds_bytes),
        };
        return tiling.tileBytes(tiling.block_n) +
            tiling.stages * 2 * tiling.tileBytes(tiling.block_m) +
            values_and_pds + dq_accum_staging;
    }
};

/// Returns the tile shape that the SM80 and SM120 backward tilings start from.
fn backwardTiling(head_dim: u32) Sm80BwdTiling {
    return switch (head_dim) {
        64 => .{ .head_dim = 64, .block_m = 64, .block_n = 128, .warps = 8, .stages = 2 },
        72 => .{ .head_dim = 72, .block_m = 64, .block_n = 64, .warps = 8, .stages = 2 },
        96 => .{ .head_dim = 96, .block_m = 64, .block_n = 64, .warps = 8, .stages = 1 },
        128 => .{ .head_dim = 128, .block_m = 64, .block_n = 64, .warps = 8, .stages = 1 },
        256 => .{ .head_dim = 256, .block_m = 32, .block_n = 32, .warps = 4, .stages = 1 },
        else => @panic("unsupported head dimension"),
    };
}

pub fn backwardSm80(head_dim: u32) Sm80BwdTiling {
    var tiling = backwardTiling(head_dim);
    switch (head_dim) {
        64 => {
            tiling.p_first = true;
            tiling.alternate_gradient_order = true;
            tiling.gradient_row_tiles = 2;
        },
        72 => {
            tiling.block_n = 128;
            tiling.v_in_regs = true;
            tiling.score_row_tiles = 2;
            tiling.alternate_gradient_order = true;
        },
        96 => {
            tiling.block_n = 128;
            tiling.v_in_regs = true;
            tiling.score_row_tiles = 2;
            tiling.p_first = true;
            tiling.gradient_row_tiles = 2;
        },
        128 => {
            tiling.stages = 2;
            tiling.v_in_regs = true;
            tiling.alternate_gradient_order = true;
            tiling.p_first = true;
        },
        256 => {
            tiling.block_m = 64;
            tiling.warps = 8;
            tiling.v_in_regs = true;
            tiling.gradient_row_tiles = 2;
        },
        else => {},
    }
    return tiling;
}

pub fn backwardSm80Limited1(head_dim: u32) ?Sm80BwdTiling {
    return switch (head_dim) {
        64 => .{ .head_dim = 64, .block_m = 64, .block_n = 64, .warps = 8, .stages = 2 },
        72, 96, 128 => .{
            .head_dim = head_dim,
            .block_m = 32,
            .block_n = 64,
            .warps = 4,
            .stages = 2,
        },
        else => null,
    };
}

pub fn backwardSm80Variant(head_dim: u32, mode: Mode) ?Variant(Sm80BwdTiling) {
    const tiling = if (mode.isLimited1()) backwardSm80Limited1(head_dim) orelse
        return null else backwardSm80(head_dim);
    return declare(Sm80BwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 128 },
        .gqa_head_dims = &head_dims,
        .gqa_specialization = .{ .head_split = true },
        .limited_1_limits = .{ .max_seqlen_k = 128 },
    });
}

fn withTensorMaps(sm80_tiling: Sm80BwdTiling) Sm80BwdTiling {
    var tiling = sm80_tiling;
    tiling.tma = true;
    tiling.bulk_reduce = tiling.head_dim == 96 or tiling.head_dim == 128 or tiling.head_dim == 256;
    tiling.p_first = tiling.head_dim == 64;
    tiling.pdl = true;
    return tiling;
}

pub fn backwardSm120(head_dim: u32) Sm80BwdTiling {
    var tiling = withTensorMaps(backwardTiling(head_dim));
    if (head_dim == 96) tiling.stages = 2;
    if (head_dim == 72) tiling.batch_major = true;
    return tiling;
}

pub fn backwardSm120Variant(head_dim: u32, mode: Mode) ?Variant(Sm80BwdTiling) {
    var tiling = switch (mode.tier()) {
        .regular => backwardSm120(head_dim),
        .limited_1 => withTensorMaps(backwardSm80Limited1(head_dim) orelse return null),
        .limited_2 => if (mode == .limited_2 and (head_dim == 64 or head_dim == 96))
            backwardSm120(head_dim)
        else
            return null,
        .local_two_sided, .local_two_sided_limited_2 => return null,
    };
    if (mode == .regular or mode == .varlen) {
        if (head_dim == 64) {
            tiling.stages = 1;
            tiling.bulk_reduce = true;
        }
        if (head_dim == 96) tiling.alternate_gradient_order = true;
    }
    return declare(Sm80BwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 128 },
        .gqa_head_dims = &head_dims,
        .gqa_specialization = .{ .head_split = true },
        .limited_1_limits = .{ .max_seqlen_k = 128 },
        .limited_2_limits = .{ .max_seqlen_k = 1024 },
    });
}

pub fn forwardSm120Variant(head_dim: u32, mode: Mode) ?Variant(Sm80FwdTiling) {
    var tiling = switch (mode.tier()) {
        .regular => forwardSm80(head_dim),
        .limited_1 => if (head_dim == 96)
            forwardSm80(head_dim)
        else
            forwardSm80Limited1(head_dim) orelse return null,
        .limited_2 => if (mode == .limited_2 and head_dim == 96)
            forwardSm80(head_dim)
        else
            return null,
        .local_two_sided, .local_two_sided_limited_2 => return null,
    };
    tiling.tma = true;
    if (mode.tier() == .regular and head_dim == 96) {
        tiling.stages = 1;
        tiling.q_in_regs = false;
        tiling.mma_m_per_warp = 2;
        tiling.separate_v_empty = true;
    }
    if (mode.tier() == .regular and head_dim == 72) tiling.batch_major = true;
    if (mode.tier() == .limited_2) {
        tiling.stages = 2;
        tiling.max_registers = 128;
    }
    var variant = declare(Sm80FwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 128 },
        .gqa_head_dims = &.{ 64, 96, 128 },
        .gqa_specialization = pack_gqa_only,
        .limited_1_limits = .{ .max_seqlen_q = if (head_dim == 96) 1024 else 256 },
        .limited_2_limits = .{ .max_seqlen_q = 1024, .max_window_size = 2048 },
    }) orelse return null;
    if (mode.isLimited1() and head_dim == 96) variant.specialization.limits.max_window_size = 1024;
    if (mode == .gqa and head_dim == 96) variant.specialization.limits.max_seqlen_q = 512;
    return variant;
}

const causal_l2_swizzle: L2SwizzlePolicy = &.{
    .{ .masks = .initOne(.causal), .size_l2 = 16 << 20, .min_seqlen_other = 1024 },
};
const capped_causal_l2_swizzle: L2SwizzleRule = capped: {
    var rule = causal_l2_swizzle[0];
    rule.max_seqlen_other = 4095;
    break :capped rule;
};

pub fn registerRequirements(consumer_groups: u32) struct { u32, u32 } {
    return switch (consumer_groups) {
        1 => .{ 56, 256 },
        2 => .{ 24, 240 },
        3 => .{ 32, 160 },
        else => @panic("unsupported consumer group count"),
    };
}

pub const Sm90FwdTiling = struct {
    head_dim: u32,
    block_n: u32,
    stages: u32,
    block_m: u32 = 128,
    rescale_o_before_gemm: bool = false,
    use_scheduler_barrier: bool = true,
    intra_wg_overlap: bool = false,
    separate_masking_iterations: bool = true,
    overlap_so_sv: bool = false,
    suspend_time_hint_ns: u32 = 0,
    hdim_multiple_of: u32 = 16,
    bottom_aligned: bool = false,
    batch_major: bool = false,
    l2_swizzle: L2SwizzlePolicy = causal_l2_swizzle,
    scheduling_mode: SchedulingMode = .dynamic,

    pub const producer_warps = warps_per_warp_group;

    pub fn mmaHeadDim(tiling: Sm90FwdTiling) u32 {
        return std.mem.alignForward(u32, tiling.head_dim, 16);
    }

    pub const headDimPadded = alignedHeadDim;
    pub const rowChunks = paddedRowChunks;
    pub const swizzleBytes = paddedSwizzleBytes;
    pub const boxColumns = paddedBoxColumns;
    pub const tileBytes = paddedTileBytes;

    pub fn consumerGroups(tiling: Sm90FwdTiling) u32 {
        return tiling.block_m / 64;
    }

    pub fn consumerWarps(tiling: Sm90FwdTiling) u32 {
        return tiling.consumerGroups() * warps_per_warp_group;
    }

    pub fn threads(tiling: Sm90FwdTiling) u32 {
        return (producer_warps + tiling.consumerWarps()) * 32;
    }

    pub fn stagingColumns(tiling: Sm90FwdTiling) u32 {
        return @min(tiling.mmaHeadDim(), 128);
    }

    pub fn sharedBytes(tiling: Sm90FwdTiling) u32 {
        return tiling.tileBytes(tiling.block_m) +
            tiling.stages * 2 * tiling.tileBytes(tiling.block_n) +
            if (tiling.overlap_so_sv) 0 else tiling.block_m *
                tiling.stagingColumns() * Sm80FwdTiling.element_bytes;
    }
};

pub fn forwardSm90(head_dim: u32) Sm90FwdTiling {
    return switch (head_dim) {
        64 => .{
            .head_dim = 64,
            .block_m = 192,
            .block_n = 128,
            .stages = 2,
            .bottom_aligned = true,
            .suspend_time_hint_ns = 10_000_000,
        },
        72 => .{
            .head_dim = 72,
            .block_m = 192,
            .block_n = 96,
            .stages = 3,
            .use_scheduler_barrier = false,
            .hdim_multiple_of = 64,
            .bottom_aligned = true,
            .batch_major = true,
        },
        96 => .{
            .head_dim = 96,
            .block_m = 192,
            .block_n = 96,
            .stages = 3,
            .use_scheduler_barrier = false,
            .bottom_aligned = true,
        },
        128 => .{ .head_dim = 128, .block_n = 128, .stages = 2, .intra_wg_overlap = true },
        256 => .{
            .head_dim = 256,
            .block_n = 80,
            .stages = 2,
            .rescale_o_before_gemm = true,
            .use_scheduler_barrier = false,
            .overlap_so_sv = true,
            .l2_swizzle = &.{
                capped_causal_l2_swizzle,
                .{ .masks = .initOne(.causal), .size_l2 = 32 << 20, .min_seqlen_other = 4096 },
            },
        },
        else => @panic("unsupported head dimension"),
    };
}

pub fn forwardSm90Limited1(head_dim: u32) ?Sm90FwdTiling {
    var tiling = forwardSm90(head_dim);
    switch (head_dim) {
        64, 96 => {
            tiling.block_m = 128;
            tiling.separate_masking_iterations = false;
        },
        128 => tiling.intra_wg_overlap = false,
        256 => {
            tiling.block_n = 64;
            tiling.overlap_so_sv = false;
            tiling.l2_swizzle = causal_l2_swizzle;
        },
        else => return null,
    }
    return tiling;
}

pub fn forwardSm90Variant(head_dim: u32, mode: Mode) ?Variant(Sm90FwdTiling) {
    const regular = forwardSm90(head_dim);
    var tiling = switch (mode.tier()) {
        .regular => regular,
        .limited_1 => forwardSm90Limited1(head_dim) orelse return null,
        .limited_2 => if (mode != .limited_2) return null else switch (head_dim) {
            64 => regular,
            128 => forwardSm90Limited1(head_dim).?,
            else => return null,
        },
        .local_two_sided, .local_two_sided_limited_2 => return null,
    };
    if (mode.tier() == .limited_2) tiling.separate_masking_iterations = false;
    var variant = declare(Sm90FwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 128 },
        .gqa_head_dims = &.{ 64, 96 },
        .gqa_limited_1_head_dims = &.{256},
        .gqa_specialization = pack_gqa_only,
        .limited_1_limits = .{
            .max_seqlen_q = if (regular.intra_wg_overlap or regular.overlap_so_sv) 1024 else 256,
        },
        .limited_2_limits = .{ .max_seqlen_q = if (head_dim == 64) 512 else 256 },
    }) orelse return null;
    if (mode == .limited_1 and head_dim == 128) variant.specialization.limits.min_seqlen_q = 257;
    if (mode == .gqa_limited_1 and head_dim == 256) variant.specialization.limits.min_seqlen_q =
        513;
    if (mode == .varlen_limited_1 and
        tiling.block_m != regular.block_m) variant.specialization.limits.max_seqlen_q =
        tiling.block_m;
    if (mode == .varlen_limited_1 and head_dim == 128) variant.tiling.separate_masking_iterations =
        false;
    if (mode.isLimited1() and regular.overlap_so_sv) variant.specialization.limits.max_window_size =
        1280;
    return variant;
}

pub const Sm90BwdTiling = struct {
    head_dim: u32,
    stages: u32 = 2,
    block_m: u32 = 64,
    block_n: u32 = 128,
    hdim_multiple_of: u32 = 16,
    dq_split: DqSplit = .by_columns,
    staging_buffers: u32 = 2,
    pds_stage: u32 = 2,
    early_exp_tiles: u32 = 0,
    separate_do_pipeline: bool = false,
    batch_major: bool = false,
    separate_masking_iterations: bool = false,
    has_limited_2: bool = false,
    l2_swizzle: L2SwizzlePolicy = &.{},
    scheduling_mode: SchedulingMode = .single_tile,

    pub const DqSplit = enum { by_columns, by_rows, none };
    pub const dq_accum_box_columns = 32;
    pub const producer_warps = warps_per_warp_group;
    pub const consumer_warps = 2 * warps_per_warp_group;

    pub const headDimPadded = alignedHeadDim;
    pub const rowChunks = paddedRowChunks;
    pub const swizzleBytes = paddedSwizzleBytes;
    pub const boxColumns = paddedBoxColumns;
    pub const tileBytes = paddedTileBytes;

    pub fn threads(_: Sm90BwdTiling) u32 {
        return (producer_warps + consumer_warps) * 32;
    }

    pub fn stagingBoxes(tiling: Sm90BwdTiling) u32 {
        return (tiling.headDimPadded() + dq_accum_box_columns - 1) / dq_accum_box_columns;
    }

    pub fn residentStagingBoxes(tiling: Sm90BwdTiling) u32 {
        return if (tiling.pds_stage == 1) tiling.stagingBoxes() / 2 else tiling.stagingBoxes();
    }

    pub fn doStage(tiling: Sm90BwdTiling) u32 {
        return if (tiling.pds_stage == 1) 1 else tiling.stages;
    }

    pub fn sharedBytes(tiling: Sm90BwdTiling) u32 {
        const query_tiles = tiling.stages + tiling.doStage();
        const score_tiles: u32 = if (tiling.pds_stage == 1) 2 else tiling.stages;
        const scores = score_tiles * tiling.block_n * tiling.block_m * Sm80FwdTiling.element_bytes;
        const staging = tiling.staging_buffers *
            tiling.residentStagingBoxes() * tiling.block_m * dq_accum_box_columns * @sizeOf(f32);
        const vectors = query_tiles * tiling.block_m * @sizeOf(f32);
        return 2 * tiling.tileBytes(tiling.block_n) +
            query_tiles * tiling.tileBytes(tiling.block_m) + scores + staging + vectors;
    }
};

pub fn backwardSm90(head_dim: u32) Sm90BwdTiling {
    return switch (head_dim) {
        64 => .{
            .head_dim = 64,
            .block_m = 128,
            .dq_split = .by_rows,
            .early_exp_tiles = 4,
        },
        72 => .{
            .head_dim = 72,
            .dq_split = .none,
            .separate_do_pipeline = true,
            .batch_major = true,
        },
        96 => .{
            .head_dim = 96,
            .dq_split = .none,
            .separate_masking_iterations = true,
            .has_limited_2 = true,
        },
        128 => .{ .head_dim = 128, .early_exp_tiles = 4 },
        256 => .{
            .head_dim = 256,
            .block_n = 64,
            .staging_buffers = 1,
            .pds_stage = 1,
        },
        else => @panic("unsupported head dimension"),
    };
}

pub fn backwardSm90Variant(head_dim: u32, mode: Mode) ?Variant(Sm90BwdTiling) {
    var tiling: Sm90BwdTiling =
        if (!mode.isLimited1()) backwardSm90(head_dim) else switch (head_dim) {
            64 => .{ .head_dim = 64, .dq_split = .none },
            else => return null,
        };
    if (mode.tier() == .limited_2) {
        if (!tiling.has_limited_2) return null;
        tiling.separate_masking_iterations = false;
    }
    return declare(Sm90BwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 128 },
        .limited_1_limits = .{ .max_seqlen_q = 128 },
        .limited_2_limits = .{ .max_seqlen_q = 1024 },
    });
}

pub fn launchBackwardSm90(head_dim: u32, mode: Mode) ?Launch {
    return launch(.flash_bwd_sm90, head_dim, mode);
}

pub const Sm100FwdTiling = struct {
    head_dim: u32,
    kv_stage: u32,
    block_n: u32 = 128,
    m_block_size: u32 = 128,
    q_stage: u32 = 2,
    staging_buffers: u32 = 2,
    ex2_emulated_pairs: u32 = 0,
    ex2_emulation_placement: Ex2EmulationPlacement = .interleaved,
    load_backoff_ns: u32 = 0,
    l2_swizzle: L2SwizzlePolicy = &.{
        .{ .masks = Mask.bounded, .size_l2 = 64 << 20, .varlen = true },
    },
    late_row_sum: bool = false,
    p_split_pairs: u32 = 0,
    use_2cta_instrs: bool = false,
    scheduling_mode: SchedulingMode = .clc,
    prefetch_next_work_early: bool = true,
    fragment_stores: bool = false,
    divmod_decode: bool = false,
    prologue_loads: bool = true,
    batch_major: bool = false,

    pub const NumRegs = struct { softmax: u32, correction: u32, other: u32 };
    pub const num_regs_by_q_stage = [2]NumRegs{ .{
        .softmax = 256,
        .correction = 160,
        .other = 88,
    }, .{ .softmax = 192, .correction = 80, .other = 48 } };

    pub fn headDimPadded(tiling: Sm100FwdTiling) u32 {
        return if (tiling.head_dim == 72) 96 else tiling.head_dim;
    }

    pub const rowChunks = paddedRowChunks;
    pub const swizzleBytes = paddedSwizzleBytes;
    pub const boxColumns = paddedBoxColumns;
    pub const tileBytes = paddedTileBytes;

    pub fn clusterCtas(tiling: Sm100FwdTiling) u32 {
        return if (tiling.use_2cta_instrs) 2 else 1;
    }

    pub fn ctaTilerM(tiling: Sm100FwdTiling) u32 {
        return tiling.q_stage * tiling.m_block_size;
    }

    pub fn clusterTilerM(tiling: Sm100FwdTiling) u32 {
        return tiling.clusterCtas() * tiling.ctaTilerM();
    }

    pub fn threads(tiling: Sm100FwdTiling) u32 {
        return (tiling.q_stage + 2) * 128;
    }

    pub fn sharedBytes(tiling: Sm100FwdTiling) u32 {
        return tiling.tileBytes(tiling.ctaTilerM()) +
            tiling.kv_stage * tiling.tileBytes(tiling.block_n) / tiling.clusterCtas() +
            tiling.staging_buffers *
                tiling.m_block_size * tiling.boxColumns() * Sm80FwdTiling.element_bytes;
    }
};

pub fn forwardSm100(head_dim: u32) Sm100FwdTiling {
    return switch (head_dim) {
        64 => .{
            .head_dim = 64,
            .kv_stage = 8,
            .ex2_emulated_pairs = 2,
            .l2_swizzle = &.{
                .{ .masks = .initOne(.dense), .size_l2 = 16 << 20, .varlen = true },
                .{ .masks = Mask.bounded, .size_l2 = L2SwizzleRule.unlimited, .varlen = true },
            },
            .p_split_pairs = 40,
            .ex2_emulation_placement = .inner_fragments,
            .fragment_stores = true,
            .divmod_decode = true,
        },
        72 => .{
            .head_dim = 72,
            .kv_stage = 6,
            .ex2_emulated_pairs = 2,
            .p_split_pairs = 48,
            .prefetch_next_work_early = false,
        },
        96 => .{
            .head_dim = 96,
            .kv_stage = 6,
            .ex2_emulated_pairs = 2,
            .ex2_emulation_placement = .inner_fragments,
            .late_row_sum = true,
            .p_split_pairs = 48,
            .fragment_stores = true,
            .divmod_decode = true,
        },
        128 => .{
            .head_dim = 128,
            .kv_stage = 4,
            .ex2_emulated_pairs = 2,
            .ex2_emulation_placement = .leading,
            .load_backoff_ns = 100,
            .late_row_sum = true,
            .p_split_pairs = 48,
            .fragment_stores = true,
        },
        256 => .{
            .head_dim = 256,
            .kv_stage = 4,
            .q_stage = 1,
            .use_2cta_instrs = true,
            .prologue_loads = false,
        },
        else => @panic("unsupported head dimension"),
    };
}

pub fn forwardSm100Limited2(head_dim: u32) ?Sm100FwdTiling {
    var tiling = forwardSm100(head_dim);
    switch (head_dim) {
        64 => {},
        96 => tiling.ex2_emulated_pairs = 1,
        else => {
            tiling.divmod_decode = true;
            return tiling;
        },
    }
    tiling.ex2_emulation_placement = .interleaved;
    tiling.fragment_stores = false;
    return tiling;
}

pub fn forwardSm100Variant(head_dim: u32, mode: Mode) ?Variant(Sm100FwdTiling) {
    const tiling = switch (mode.tier()) {
        .regular, .local_two_sided => forwardSm100(head_dim),
        .limited_1 => limited_1: {
            if (head_dim != 64 and head_dim != 96 and head_dim != 128) return null;
            const regular = forwardSm100(head_dim);
            var tiling = regular;
            tiling.q_stage = 1;
            tiling.block_n = 64;
            tiling.kv_stage = regular.kv_stage * regular.block_n / tiling.block_n;
            tiling.ex2_emulated_pairs = 0;
            tiling.ex2_emulation_placement = .interleaved;
            tiling.p_split_pairs = 0;
            tiling.load_backoff_ns = 0;
            tiling.fragment_stores = false;
            break :limited_1 tiling;
        },
        .limited_2, .local_two_sided_limited_2 => switch (mode) {
            .limited_2, .local_two_sided_limited_2 => forwardSm100Limited2(head_dim) orelse
                return null,
            .varlen_limited_2 => if (head_dim == 96) forwardSm100Limited2(
                head_dim,
            ).? else return null,
            else => return null,
        },
    };
    const batch_major = mode == .varlen and head_dim == 72;
    var variant = declare(Sm100FwdTiling, tiling, mode, .{
        .varlen_head_dims = &.{ 64, 96, 128 },
        .gqa_head_dims = &.{ 64, 72, 96, 128 },
        .gqa_specialization = pack_gqa_only,
        .limited_1_limits = .{ .max_seqlen_q = 128 },
        .limited_2_limits = switch (head_dim) {
            64 => .{ .max_seqlen_q = 512 },
            96 => .{ .max_seqlen_q = 256 },
            else => .{ .max_seqlen_q = 1024, .masks = .initOne(.causal), .batched_only = true },
        },
        .local_head_dims = &.{64},
    }) orelse if (batch_major) Variant(Sm100FwdTiling){
        .tiling = tiling,
        .specialization = .{ .limits = .{ .masks = .initOne(.dense) } },
    } else return null;
    variant.tiling.batch_major = batch_major;
    if (variant.specialization.limits.two_sided_window) {
        variant.tiling.divmod_decode = false;
        variant.tiling.prologue_loads = false;
    }
    if (mode.isLimited1() and head_dim == 128) variant.specialization.limits.min_seqlen_q = 65;
    if (mode.isVarlen()) {
        variant.tiling.divmod_decode = false;
        variant.tiling.prologue_loads = false;
    }
    if (mode == .varlen_limited_2) variant.specialization.limits.max_seqlen_q = 1024;
    if (mode.isVarlen() and !mode.isLimited1()) switch (head_dim) {
        64, 96 => {},
        else => variant.tiling.fragment_stores = false,
    };
    return variant;
}

pub const Sm100BwdTiling = struct {
    head_dim: u32,
    block_m: u32 = 128,
    block_n: u32 = 128,
    q_stage: u32 = 2,
    do_stage: u32 = 1,
    staging_buffers: u32 = 2,
    head_dim_parts: u32 = 1,
    dq_block_m: u32 = 0,
    use_2cta_instrs: bool = false,
    staged_epilogue: bool = false,
    use_tma_store: bool = false,
    pdl: bool = true,
    min_qhead_per_split: u32 = 4,
    batch_major: bool = false,
    scheduling_mode: SchedulingMode = .single_tile,
    l2_swizzle: L2SwizzlePolicy = &lpt_l2_swizzle,

    pub const dq_reduce_ncol = 32;
    pub const compute_groups = 2;
    pub const other_warps = warps_per_warp_group;
    pub const num_regs = .{ .reduce = 152, .compute = 144, .other = 72 };
    pub const num_regs_separate_dq = .{ .compute = 200, .empty = 24, .other = 88 };
    pub const dkv_box_rows = 32;
    pub const lpt_l2_swizzle = [_]L2SwizzleRule{
        lptL2Swizzle(.initMany(&.{ .causal, .local_right }), .ascending),
        lptL2Swizzle(.initOne(.local_left), .descending),
    };

    fn lptL2Swizzle(masks: std.EnumSet(Mask), block_order: BlockOrder) L2SwizzleRule {
        return .{
            .masks = masks,
            .size_l2 = L2SwizzleRule.unlimited,
            .max_l2_major = 128,
            .max_num_block = 32,
            .max_ctas = 1024,
            .block_order = block_order,
        };
    }

    pub const rowChunks = paddedRowChunks;
    pub const swizzleBytes = paddedSwizzleBytes;
    pub const boxColumns = paddedBoxColumns;
    pub const tileBytes = paddedTileBytes;

    pub fn headDimPadded(tiling: Sm100BwdTiling) u32 {
        return forwardSm100(tiling.head_dim).headDimPadded();
    }

    pub fn threads(_: Sm100BwdTiling) u32 {
        return 4 * warps_per_warp_group * 32;
    }

    pub fn sharedBytes(tiling: Sm100BwdTiling) u32 {
        if (tiling.use_2cta_instrs) {
            const key_tiles = 3 * tiling.tileBytes(tiling.block_n);
            const ds = 2 * tiling.block_n * tiling.block_m * Sm80FwdTiling.element_bytes;
            const staging = 2 * tiling.block_m * dq_reduce_ncol * @sizeOf(f32);
            const exchange = tiling.block_n * tiling.block_m * Sm80FwdTiling.element_bytes;
            const query_tiles = 4 * tiling.tileBytes(tiling.block_m);
            const vectors = 2 * 2 * 2 * tiling.block_m * @sizeOf(f32);
            return key_tiles + ds + staging + exchange + query_tiles + vectors;
        }
        if (tiling.dq_block_m != 0) {
            const half_tiles =
                2 * tiling.tileBytes(tiling.block_n) + 2 * tiling.tileBytes(tiling.block_m);
            const column_tiles = 2 * tiling.tileBytes(2 * tiling.block_m) / 2;
            const operands = 2 * tiling.block_n * 2 * tiling.block_m * Sm80FwdTiling.element_bytes;
            const vectors = 2 * 2 * tiling.block_m * @sizeOf(f32);
            return half_tiles + column_tiles + operands + vectors;
        }
        const key_tiles = 2 * tiling.tileBytes(tiling.block_n);
        const query_tiles = (tiling.q_stage + tiling.do_stage) * tiling.tileBytes(tiling.block_m);
        const ds = tiling.block_n * tiling.block_m * Sm80FwdTiling.element_bytes;
        const staging = tiling.staging_buffers * tiling.block_m * dq_reduce_ncol * @sizeOf(f32);
        const vectors = 2 * tiling.q_stage * tiling.block_m * @sizeOf(f32);
        return key_tiles + query_tiles + ds + staging + vectors;
    }
};

pub fn backwardSm100(head_dim: u32) Sm100BwdTiling {
    return switch (head_dim) {
        64 => .{ .head_dim = 64, .staged_epilogue = true },
        72 => .{ .head_dim = 72, .batch_major = true },
        96 => .{ .head_dim = 96, .use_tma_store = true },
        128 => .{
            .head_dim = 128,
            .min_qhead_per_split = 2,
            .scheduling_mode = .clc,
            .use_tma_store = true,
        },
        256 => .{
            .head_dim = 256,
            .block_m = 64,
            .block_n = 64,
            .dq_block_m = 256,
            .l2_swizzle = &.{},
        },
        else => @panic("unsupported head dimension"),
    };
}

pub fn backwardSm100Variant(head_dim: u32, mode: Mode) ?Variant(Sm100BwdTiling) {
    if (head_dim != 128) {
        if (mode.isLimited1()) return null;
        return declare(
            Sm100BwdTiling,
            backwardSm100(head_dim),
            mode,
            .{ .varlen_head_dims = &.{64} },
        );
    }
    const two_cta_tiling: Sm100BwdTiling = .{
        .head_dim = 128,
        .block_m = 64,
        .q_stage = 1,
        .use_2cta_instrs = true,
        .min_qhead_per_split = 0,
        .l2_swizzle = &.{},
        .use_tma_store = true,
    };
    return switch (mode) {
        .regular => .{ .tiling = two_cta_tiling, .specialization = .{ .head_split = true } },
        .limited_1, .varlen_limited_1 => declare(Sm100BwdTiling, backwardSm100(head_dim), mode, .{
            .varlen_head_dims = &.{head_dim},
            .limited_1_limits = .{ .max_seqlen_q = if (mode.isVarlen()) 3072 else 1024 },
        }),
        .gqa, .varlen_gqa => declare(Sm100BwdTiling, backwardSm100(head_dim), mode, .{
            .gqa_head_dims = &.{head_dim},
            .varlen_gqa_head_dims = &.{head_dim},
        }),
        else => null,
    };
}
