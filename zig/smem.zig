const std = @import("std");
const configurations = @import("configurations.zig");
const cp_async = std.nvptx.cp_async;

pub const Chunk = @import("fragments.zig").Chunk;
pub const alignment = 1024;

pub fn swizzledColumn(comptime swizzle_bytes: u32, column: u32, row: u32) u32 {
    return switch (swizzle_bytes) {
        128 => column ^ row % 8,
        64 => column ^ (row >> 1 & 3),
        32 => column ^ (row >> 2 & 1),
        else => column,
    };
}

pub fn copyStagedRows(
    comptime rows: u32,
    comptime chunks_per_row: u32,
    comptime participants: u32,
    participant: anytype,
    copier: anytype,
) void {
    const chunks = rows * chunks_per_row;
    inline for (0..(chunks + participants - 1) / participants) |pass| {
        const index: u32 = @as(u32, pass) * participants +
            if (@typeInfo(@TypeOf(participant)) == .@"fn") participant() else participant;
        if ((pass + 1) * participants <=
            chunks or index < chunks) copier.copy(index / chunks_per_row, index % chunks_per_row);
    }
}

/// Copies rows of 16-byte chunks from global to shared memory with `cp.async`. Each thread copies
/// the same column of every `rows_per_pass`-th row, so that each pass only advances one pointer.
/// The copy fills rows at or past `limit` and the columns past `real_chunks` with zeros.
pub fn RowCopy(
    comptime TileLayout: type,
    comptime threads: u32,
    comptime row_chunks: u32,
    comptime real_chunks: u32,
) type {
    const thread_rows = threads / row_chunks;
    const rows_per_pass = if (TileLayout.swizzle_columns > 1 and thread_rows >= 8)
        thread_rows / 8 * 8
    else
        thread_rows;
    const fixed_swizzle = TileLayout.swizzle_columns == 1 or rows_per_pass % 8 == 0;
    return struct {
        pub fn copy(
            comptime count: u32,
            destination: *addrspace(.shared) TileLayout.Tile(count),
            origin: [*]addrspace(.global) const Chunk,
            row_stride: u64,
            first_row: u32,
            limit: u32,
        ) void {
            const tid = @workItemId(0);
            if (rows_per_pass * row_chunks != threads and tid >= rows_per_pass * row_chunks) return;
            const column = tid % row_chunks;
            const thread_row = tid / row_chunks;
            const thread_slot = TileLayout.slot(thread_row, column);
            var chunk = origin + (first_row + thread_row) * row_stride + column;
            inline for (0..(count + rows_per_pass - 1) / rows_per_pass) |pass| {
                const pass_row: u32 = pass * rows_per_pass;
                const row = thread_row + pass_row;
                if ((pass + 1) * rows_per_pass <= count or row < count) {
                    const slot = if (fixed_swizzle)
                        thread_slot + comptime TileLayout.slot(pass_row, 0)
                    else
                        TileLayout.slot(row, column);
                    const valid = first_row + row < limit and column < real_chunks;
                    cp_async.copyZeroFill(
                        .cache_global_level,
                        &destination[slot],
                        if (valid) &chunk[0] else &origin[0],
                        if (valid) @sizeOf(Chunk) else 0,
                    );
                }
                chunk += rows_per_pass * row_stride;
            }
        }
    };
}

pub fn TileAddresses(comptime TileLayout: type, comptime rows: comptime_int) type {
    const swizzle_columns = TileLayout.swizzle_columns;
    return struct {
        offsets: [swizzle_columns]u32,

        pub fn init(row: u32, column: u32) @This() {
            var addresses: @This() = undefined;
            inline for (&addresses.offsets, 0..) |*offset, key| {
                offset.* =
                    TileLayout.chunkIndex(rows, row, column + @as(u32, key)) * @sizeOf(Chunk);
            }
            return addresses;
        }

        pub fn chunk(
            addresses: @This(),
            tile: *addrspace(.shared) TileLayout.Tile(rows),
            comptime row: u32,
            comptime column: u32,
        ) *addrspace(.shared) Chunk {
            comptime std.debug.assert(row % 8 == 0);
            const offset =
                comptime TileLayout.chunkIndex(rows, row, column - column % swizzle_columns) *
                @sizeOf(Chunk);
            const bytes: [*]addrspace(.shared) u8 = @ptrCast(tile);
            return @ptrCast(@alignCast(
                bytes + addresses.offsets[column % swizzle_columns] + offset,
            ));
        }
    };
}

pub fn Layout(comptime row_chunks: comptime_int) type {
    const row_pitch = configurations.pitchChunks(row_chunks);
    const swizzle_bytes = configurations.rowSwizzleBytes(row_chunks);
    return struct {
        pub const swizzle_columns = @max(1, swizzle_bytes / 16);

        pub fn Tile(comptime rows: comptime_int) type {
            return [rows * row_pitch]Chunk;
        }

        pub fn Addresses(comptime rows: comptime_int) type {
            return TileAddresses(@This(), rows);
        }

        /// Locates the chunks at the even columns of a tile for a thread whose own column is 0 or
        /// 1. The swizzle permutes the chunks of a row within groups of `swizzle_columns` chunks by
        /// an XOR of their low bits, so the offset of an even column within a group follows from
        /// the offset of the thread's own column by an XOR with a constant. A thread keeps one
        /// register for the operand instead of one register per column.
        pub fn PairAddresses(comptime rows: comptime_int) type {
            return struct {
                base: u32,

                pub fn init(row: u32, column: u32) @This() {
                    return .{ .base = slot(row, column) * @sizeOf(Chunk) };
                }

                pub fn chunk(
                    addresses: @This(),
                    tile: *addrspace(.shared) Tile(rows),
                    comptime row: u32,
                    comptime column: u32,
                ) *addrspace(.shared) Chunk {
                    comptime std.debug.assert(row % 8 == 0 and column % 2 == 0);
                    const key = column % swizzle_columns;
                    const offset = comptime slot(row, column - key) * @sizeOf(Chunk);
                    const local = if (swizzle_columns == 1)
                        addresses.base
                    else
                        addresses.base ^ (key * @sizeOf(Chunk));
                    const bytes: [*]addrspace(.shared) u8 = @ptrCast(tile);
                    return @ptrCast(@alignCast(bytes + local + offset));
                }
            };
        }

        pub fn slot(row: u32, column: u32) u32 {
            return row * row_pitch + swizzledColumn(swizzle_bytes, column, row);
        }

        pub fn chunkIndex(comptime _: comptime_int, row: u32, column: u32) u32 {
            return slot(row, column);
        }

        pub fn chunk(
            tile: anytype,
            row: u32,
            column: u32,
            swizzle_key: u32,
        ) *addrspace(.shared) Chunk {
            return &tile[row * row_pitch + swizzledColumn(swizzle_bytes, column, swizzle_key)];
        }

        pub fn word(tile: anytype, row: u32, column: u32, word_index: u32) *addrspace(.shared) u32 {
            const words: [*]addrspace(.shared) u32 = @ptrCast(tile);
            return &words[slot(row, column) * (@sizeOf(Chunk) / @sizeOf(u32)) + word_index];
        }
    };
}
