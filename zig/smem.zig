const std = @import("std");
const configurations = @import("configurations.zig");

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
    const swizzle_bytes = if (configurations.swizzleBytes(row_chunks) == 128) 128 else 0;
    return struct {
        pub const swizzle_columns = if (swizzle_bytes == 0) 1 else 8;

        pub fn Tile(comptime rows: comptime_int) type {
            return [rows * row_pitch]Chunk;
        }

        pub fn Addresses(comptime rows: comptime_int) type {
            return TileAddresses(@This(), rows);
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
