const std = @import("std");
const nvptx = std.nvptx;
const config = @import("config");
const thread = @import("thread.zig");

pub const enabled = config.profile;

pub const chunk_records = 32;
pub const tick_bits = 24;
pub const header_offset = 64;
pub const layout_version = 1;

pub const Control = extern struct {
    next_chunk: u32,
    capacity: u32,
    dropped: u32,
    format: u32,
};

pub const Origin = packed struct(u32) {
    sm: u8,
    warp: u5,
    count: u6,
    shift: u4,
    spare: u9 = 0,
};

pub const Header = extern struct {
    clock: u64,
    globaltimer: u64,
    cta: u32,
    origin: Origin,
};

pub const Record = packed struct(u32) {
    ticks: u24,
    end: bool,
    zone: u7,
};

pub const Sink = enum { lanes, global };

pub const Options = struct {
    sink: Sink = .lanes,
    shift: u4 = 0,
};

pub fn recordsOffset(capacity: u32) u64 {
    const headers_end = header_offset + @sizeOf(Header) * (@as(u64, capacity) + 1);
    return std.mem.alignForward(u64, headers_end, chunk_records * @sizeOf(u32));
}

pub fn zoneNames(comptime Zone: type) []const u8 {
    comptime var text: []const u8 = "";
    inline for (@typeInfo(Zone).@"enum".field_names) |name| text = text ++ name ++ "\n";
    return text;
}

fn ZoneTable(comptime Zone: type) type {
    return struct {
        const names = zoneNames(Zone)[0..].*;
    };
}

/// Exports the names of `Zone` as `profile_zones`, one name per line, so that the host can label
/// the records of the kernels in this module.
pub fn exportZones(comptime Zone: type) void {
    if (enabled) @export(&ZoneTable(Zone).names, .{ .name = "profile_zones" });
}

/// Records the begin and end of the zones of one warp into the profile buffer at `address`. All
/// 32 lanes of the warp call `begin`, `end` and `finish` together. Without the `profile` build
/// option, every method compiles to nothing.
pub fn Recorder(comptime Zone: type, comptime options: Options) type {
    comptime {
        std.debug.assert(@typeInfo(Zone).@"enum".tag_type == u7);
        std.debug.assert(@as(u32, options.shift) + tick_bits <= 32);
    }
    if (!enabled) return struct {
        const Self = @This();

        pub fn init(address: u64) Self {
            _ = address;
            return .{};
        }

        pub inline fn begin(self: *Self, comptime zone: Zone) void {
            _ = self;
            _ = zone;
        }

        pub inline fn end(self: *Self, comptime zone: Zone) void {
            _ = self;
            _ = zone;
        }

        pub inline fn finish(self: *Self) void {
            _ = self;
        }
    };
    return struct {
        const Self = @This();
        const tick_mask = (1 << tick_bits) - 1;

        control: *addrspace(.global) Control,
        cursor: [*]addrspace(.global) u32,
        capacity: u32,
        chunk: u32,
        next: u32,
        count: u32,
        lane: u32,
        slot: u32,

        pub fn init(address: u64) Self {
            const control: *addrspace(.global) Control = @ptrFromInt(address);
            const lane = thread.lane();
            var first: u32 = 0;
            if (lane == 0) first = nvptx.atom(.gpu, .monotonic, .add, &control.next_chunk, 2);
            first = nvptx.shflSync(.idx, u32, first, 0, .fromWidth(.idx, 32), .all);
            var self: Self = .{
                .control = control,
                .cursor = undefined,
                .capacity = control.capacity,
                .chunk = undefined,
                .next = first + 1,
                .count = 0,
                .lane = lane,
                .slot = 0,
            };
            self.use(first);
            return self;
        }

        pub inline fn begin(self: *Self, comptime zone: Zone) void {
            self.mark(zone, false);
        }

        pub inline fn end(self: *Self, comptime zone: Zone) void {
            self.mark(zone, true);
        }

        pub inline fn finish(self: *Self) void {
            if (self.count != 0) self.close();
        }

        inline fn mark(self: *Self, comptime zone: Zone, comptime ends: bool) void {
            @setEvalBranchQuota(100_000);
            if (std.debug.runtime_safety) std.debug.assert(nvptx.activemask() == 0xffff_ffff);
            const event: u32 = comptime @bitCast(Record{
                .ticks = 0,
                .end = ends,
                .zone = @backingInt(zone),
            });
            const ticks: u32 = @truncate(nvptx.clock64() >> options.shift);
            const word = event | (ticks & tick_mask);
            switch (options.sink) {
                .lanes => if (self.lane == self.count) {
                    self.slot = word;
                },
                .global => if (self.lane == 0) {
                    self.cursor[self.count] = word;
                },
            }
            self.count += 1;
            if (self.count == chunk_records) {
                self.close();
                self.claim();
            }
        }

        inline fn close(self: *Self) void {
            if (options.sink == .lanes and self.lane < self.count) self.cursor[self.lane] = self.slot;
            if (self.lane != 0) return;
            const headers: [*]addrspace(.global) Header =
                @ptrFromInt(@intFromPtr(self.control) + header_offset);
            headers[self.chunk] = .{
                .clock = nvptx.clock64(),
                .globaltimer = nvptx.globaltimer(),
                .cta = @workGroupId(0) + nvptx.nctaid(0) *
                    (@workGroupId(1) + nvptx.nctaid(1) * @workGroupId(2)),
                .origin = .{
                    .sm = @truncate(nvptx.smid()),
                    .warp = @intCast(thread.warp()),
                    .count = @intCast(self.count),
                    .shift = options.shift,
                },
            };
            if (self.chunk == self.capacity) {
                _ = nvptx.atom(.gpu, .monotonic, .add, &self.control.dropped, self.count);
            }
        }

        inline fn claim(self: *Self) void {
            const claimed = nvptx.shflSync(.idx, u32, self.next, 0, .fromWidth(.idx, 32), .all);
            if (self.lane == 0) {
                self.next = nvptx.atom(.gpu, .monotonic, .add, &self.control.next_chunk, 1);
            }
            self.use(claimed);
            self.count = 0;
        }

        inline fn use(self: *Self, claimed: u32) void {
            self.chunk = @min(claimed, self.capacity);
            const records = @intFromPtr(self.control) + recordsOffset(self.capacity);
            self.cursor = @ptrFromInt(records + @as(u64, self.chunk) * chunk_records * @sizeOf(u32));
        }
    };
}

test Record {
    const word: u32 = @bitCast(Record{ .ticks = 0x123456, .end = true, .zone = 3 });
    try std.testing.expectEqual(0x0712_3456, word);
}

test Origin {
    const word: u32 = @bitCast(Origin{ .sm = 2, .warp = 4, .count = 32, .shift = 3 });
    try std.testing.expectEqual(2 | 4 << 8 | 32 << 13 | 3 << 19, word);
}

test recordsOffset {
    try std.testing.expectEqual(128, recordsOffset(0));
    try std.testing.expectEqual(256, recordsOffset(5));
    try std.testing.expectEqual(24, @sizeOf(Header));
}

test zoneNames {
    try std.testing.expectEqualStrings("tile\nload\n", zoneNames(enum(u7) { tile, load }));
}
