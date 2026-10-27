const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const inter_sequence = @import("inter_sequence.zig");
const inter_sequence_forward = @import("inter_sequence_forward.zig");
const sm90 = @import("inter_sequence_forward_sm90.zig");
const profile = @import("profile.zig");
const Zone = @import("flash_fwd_sm80.zig").Zone;

const variants = configurations.variants(.flash_fwd_persistent_sm90, config.head_dim);
const variant = variants[config.variant];
const Kernel = sm90.Kernel(config.head_dim, variant.specialization.pack_gqa);
const Entry = entry.Entry(inter_sequence_forward.Params, sm90.Args, sm90.names, .{
    .required_work_group_size = .{ .x = Kernel.threads },
    .max_registers_per_work_item = Kernel.max_registers,
}, Kernel.run);
const BlockTable = inter_sequence.TableKernel(&inter_sequence.BlockTable.block_rows, true);

comptime {
    std.debug.assert(Kernel.threads == variant.tiling.threads());
    std.debug.assert(Kernel.shared_bytes == variant.tiling.sharedBytes());
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        @export(&Entry.entry, .{ .name = "flash_attn_fwd" });
        @export(&BlockTable.build, .{ .name = "inter_sequence_block_table" });
        @export(&inter_sequence.transposeVisibility, .{ .name = "inter_sequence_transpose" });
        @export(&inter_sequence.countVisibleTokens, .{
            .name = "inter_sequence_count_visible_tokens",
        });
        profile.exportZones(Zone);
    }
}
