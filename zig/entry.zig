const std = @import("std");
const tma = @import("tma.zig");

pub const backward_sequence = .{
    "cu_seqlens_q",      "cu_seqlens_k",       "seqlen_q",      "seqlen_k",
    "qhead_per_kvhead",  "softmax_scale_log2", "softmax_scale", "window_size_left",
    "window_size_right",
};

pub fn strides(comptime tensor: []const u8) [3][]const u8 {
    return .{ tensor ++ "_batch_stride", tensor ++ "_row_stride", tensor ++ "_head_stride" };
}

pub fn Entry(
    comptime Params: type,
    comptime Args: type,
    comptime names: anytype,
    comptime options: std.lang.CallingConvention.NvptxKernelOptions,
    comptime launch: fn (Params, Args) void,
) type {
    return struct {
        const map: [43]bool = mapped: {
            var result: [43]bool = @splat(false);
            for (0..names.len) |index| result[index] = isMap(index);
            break :mapped result;
        };

        fn isMap(comptime index: usize) bool {
            return index < names.len and std.mem.endsWith(u8, names[index], "_map");
        }

        fn Argument(comptime index: usize) type {
            if (index >= names.len) return void;
            if (isMap(index)) return tma.TensorMap;
            if (@hasField(Params, names[index])) return @FieldType(Params, names[index]);
            return @FieldType(Args, names[index]);
        }

        pub fn entry(
            a0: Argument(0),
            a1: Argument(1),
            a2: Argument(2),
            a3: Argument(3),
            a4: Argument(4),
            a5: Argument(5),
            a6: Argument(6),
            a7: Argument(7),
            a8: Argument(8),
            a9: Argument(9),
            a10: Argument(10),
            a11: Argument(11),
            a12: Argument(12),
            a13: Argument(13),
            a14: Argument(14),
            a15: Argument(15),
            a16: Argument(16),
            a17: Argument(17),
            a18: Argument(18),
            a19: Argument(19),
            a20: Argument(20),
            a21: Argument(21),
            a22: Argument(22),
            a23: Argument(23),
            a24: Argument(24),
            a25: Argument(25),
            a26: Argument(26),
            a27: Argument(27),
            a28: Argument(28),
            a29: Argument(29),
            a30: Argument(30),
            a31: Argument(31),
            a32: Argument(32),
            a33: Argument(33),
            a34: Argument(34),
            a35: Argument(35),
            a36: Argument(36),
            a37: Argument(37),
            a38: Argument(38),
            a39: Argument(39),
            a40: Argument(40),
            a41: Argument(41),
            a42: Argument(42),
        ) callconv(.{ .nvptx_kernel = options }) void {
            @setEvalBranchQuota(100_000);
            comptime std.debug.assert(names.len <= 43);
            const arguments = .{
                if (map[0]) &a0 else a0,    if (map[1]) &a1 else a1,    if (map[2]) &a2 else a2,
                if (map[3]) &a3 else a3,    if (map[4]) &a4 else a4,    if (map[5]) &a5 else a5,
                if (map[6]) &a6 else a6,    if (map[7]) &a7 else a7,    if (map[8]) &a8 else a8,
                if (map[9]) &a9 else a9,    if (map[10]) &a10 else a10, if (map[11]) &a11 else a11,
                if (map[12]) &a12 else a12, if (map[13]) &a13 else a13, if (map[14]) &a14 else a14,
                if (map[15]) &a15 else a15, if (map[16]) &a16 else a16, if (map[17]) &a17 else a17,
                if (map[18]) &a18 else a18, if (map[19]) &a19 else a19, if (map[20]) &a20 else a20,
                if (map[21]) &a21 else a21, if (map[22]) &a22 else a22, if (map[23]) &a23 else a23,
                if (map[24]) &a24 else a24, if (map[25]) &a25 else a25, if (map[26]) &a26 else a26,
                if (map[27]) &a27 else a27, if (map[28]) &a28 else a28, if (map[29]) &a29 else a29,
                if (map[30]) &a30 else a30, if (map[31]) &a31 else a31, if (map[32]) &a32 else a32,
                if (map[33]) &a33 else a33, if (map[34]) &a34 else a34, if (map[35]) &a35 else a35,
                if (map[36]) &a36 else a36, if (map[37]) &a37 else a37, if (map[38]) &a38 else a38,
                if (map[39]) &a39 else a39, if (map[40]) &a40 else a40, if (map[41]) &a41 else a41,
                a42,
            };
            var params: Params = .{};
            var args: Args = undefined;
            inline for (0..names.len) |index| {
                const name = names[index];
                if (comptime isMap(index)) {
                    @field(args, name[0 .. name.len - "_map".len]) = arguments[index];
                } else if (@hasField(Params, name)) {
                    @field(params, name) = arguments[index];
                } else {
                    @field(args, name) = arguments[index];
                }
            }
            launch(params, args);
        }
    };
}
