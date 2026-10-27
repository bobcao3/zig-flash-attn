const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const entry = @import("entry.zig");
const configurations = @import("configurations.zig");
const inter_sequence = @import("inter_sequence.zig");
const inter_sequence_forward = @import("inter_sequence_forward.zig");
const sm100 = @import("inter_sequence_forward_sm100.zig");
const sm120 = @import("inter_sequence_forward_sm120.zig");
const profile = @import("profile.zig");
const Zone = @import("flash_fwd_sm80.zig").Zone;

const variants = configurations.variants(.flash_fwd_persistent_sm100, config.head_dim);
const variant = variants[config.variant];
const Forward = sm100.InterSequenceFwdSm100(variant.tiling.forward(), variant.specialization.pack_gqa);

fn launch(params: inter_sequence_forward.Params, args: sm120.Args) void {
    nvptx.griddepcontrol(.wait);
    Forward.Kernel.run(params, args);
}

const Entry = entry.Entry(inter_sequence_forward.Params, sm120.Args, sm120.names, .{
    .required_work_group_size = .{ .x = Forward.thread_count },
    .max_registers_per_work_item = Forward.max_registers,
}, launch);
const BlockTable = inter_sequence.TableKernel(&inter_sequence.BlockTable.block_rows, true);

comptime {
    std.debug.assert(Forward.thread_count == variant.tiling.threads());
    std.debug.assert(Forward.shared_bytes == variant.tiling.sharedBytes());
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
