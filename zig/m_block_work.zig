const ForwardParams = @import("ForwardParams.zig");
const SeqlenInfoQK = @import("SeqlenInfoQK.zig");
const BlockInfo = @import("BlockInfo.zig");
const ColLimits = BlockInfo.ColLimits;
const tile_scheduler = @import("tile_scheduler.zig");

pub const Options = struct {
    bottom_aligned: bool = false,
    pack_gqa: bool = false,
    divmod_pack: bool = false,
};

pub const PackGqa = PackGqaWith(u32);
const PackGqaDivmod = PackGqaWith(tile_scheduler.FastDivmodDivisor);

fn PackGqaWith(comptime Divisor: type) type {
    return struct {
        qhead_per_kvhead: Divisor,
        end_row: u32,

        pub fn mIdx(pack_gqa: @This(), row: u32) u32 {
            if (Divisor == u32) return row / pack_gqa.qhead_per_kvhead;
            return pack_gqa.qhead_per_kvhead.divide(row);
        }

        pub fn rowOffset(pack_gqa: @This(), row: u32, row_stride: u64, head_stride: u64) u64 {
            const m_idx = pack_gqa.mIdx(row);
            const qhead_per_kvhead = pack_gqa.qhead_per_kvhead;
            const divisor = if (Divisor == u32) qhead_per_kvhead else qhead_per_kvhead.divisor;
            return m_idx * row_stride + (row - m_idx * divisor) * head_stride;
        }
    };
}

pub fn MBlockWork(comptime block_m: u32, comptime block_n: u32, comptime options: Options) type {
    const PackGqaRows =
        if (!options.pack_gqa) void else if (options.divmod_pack) PackGqaDivmod else PackGqa;

    return struct {
        batch: u32,
        head: u32,
        kv_head: u32,
        sequence: SeqlenInfoQK,
        block_info: BlockInfo,
        n_block_min: u32,
        n_block_max: u32,
        col_limit_bounds: BlockInfo.ColLimitBounds,
        first_row: u32,
        pack_gqa: PackGqaRows,

        const Work = @This();

        pub fn init(params: ForwardParams, tile: tile_scheduler.WorkTileInfo) Work {
            const sequence: SeqlenInfoQK = .resolve(
                params.cu_seqlens_q,
                params.cu_seqlens_k,
                params.seqlen_q,
                params.seqlen_k,
                tile.batch,
            );
            const sequence_rows = if (options.pack_gqa) sequence.seqlen_q *
                params.qhead_per_kvhead else sequence.seqlen_q;
            const num_m_blocks = (sequence_rows + block_m - 1) / block_m;
            const end_row = if (options.bottom_aligned)
                sequence_rows - (num_m_blocks - 1 - tile.block) * block_m
            else
                @min((tile.block + 1) * block_m, sequence_rows);
            const first_row =
                if (options.bottom_aligned) end_row -| block_m else tile.block * block_m;
            const pack_gqa: PackGqaRows = if (!options.pack_gqa) {} else if (options.divmod_pack) .{
                .qhead_per_kvhead = params.qhead_per_kvhead_divmod,
                .end_row = end_row,
            } else .{ .qhead_per_kvhead = params.qhead_per_kvhead, .end_row = end_row };
            const first_token = if (options.pack_gqa) pack_gqa.mIdx(first_row) else first_row;
            const info: BlockInfo = .{
                .first_row = first_token,
                .row_count = if (options.pack_gqa) pack_gqa.mIdx(end_row - 1) +
                    1 - first_token else end_row - first_row,
                .seqlen_q = sequence.seqlen_q,
                .seqlen_k = sequence.seqlen_k,
                .window_size_left = params.window_size_left,
                .window_size_right = params.window_size_right,
            };
            const head = if (options.pack_gqa) tile.head * params.qhead_per_kvhead else tile.head;
            const kv_head =
                if (options.pack_gqa) tile.head else tile.head / params.qhead_per_kvhead;
            const n_block_min, const n_block_max = info.nBlockMinMax(block_n);
            return .{
                .batch = tile.batch,
                .head = head,
                .kv_head = kv_head,
                .sequence = sequence,
                .block_info = info,
                .n_block_min = n_block_min,
                .n_block_max = n_block_max,
                .col_limit_bounds = info.colLimitBounds(),
                .first_row = first_row,
                .pack_gqa = pack_gqa,
            };
        }

        pub fn needsMask(work: *const Work, n_block: u32) bool {
            return work.block_info.needsMask(work.col_limit_bounds, n_block, block_n);
        }

        pub fn unmaskedBlocks(work: *const Work) struct { u32, u32 } {
            return work.block_info.unmaskedBlocks(work.col_limit_bounds, block_n);
        }

        pub fn colLimits(work: *const Work, row: u32) ColLimits {
            const upper: ColLimits = .init(work.block_info, work.pack_gqa.mIdx(row));
            const lower: ColLimits = .init(work.block_info, work.pack_gqa.mIdx(row + 8));
            return .{ .col_limit_left = .{
                upper.col_limit_left[0], lower.col_limit_left[0],
            }, .col_limit_right = .{ upper.col_limit_right[0], lower.col_limit_right[0] } };
        }
    };
}
