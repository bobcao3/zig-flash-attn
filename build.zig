const std = @import("std");
const configurations = @import("zig/configurations.zig");

const KernelSource = struct {
    source: []const u8,
    launch: *const fn (u32, configurations.Mode) ?configurations.Launch,
    ptx_target: []const u8,
    cubin_targets: []const []const u8,

    fn instance(kernel_source: KernelSource, mode: configurations.Mode, head_dim: u32) []const u8 {
        return std.fmt.comptimePrint("{s}{s}_hd{d}", .{ kernel_source.source, mode.suffix(), head_dim });
    }
};

const matrix = [_]KernelSource{
    .{
        .source = "flash_fwd_sm80",
        .launch = configurations.launchForwardSm80,
        .ptx_target = "sm_80",
        .cubin_targets = &.{ "sm_80", "sm_86", "sm_89", "sm_120f" },
    },
    .{
        .source = "flash_bwd_sm80",
        .launch = configurations.launchBackwardSm80,
        .ptx_target = "sm_80",
        .cubin_targets = &.{ "sm_80", "sm_86", "sm_89", "sm_120f" },
    },
    .{
        .source = "flash_fwd_sm120",
        .launch = configurations.launchForwardSm120,
        .ptx_target = "sm_120f",
        .cubin_targets = &.{"sm_120f"},
    },
    .{
        .source = "flash_fwd_sm90",
        .launch = configurations.launchForwardSm90,
        .ptx_target = "sm_90a",
        .cubin_targets = &.{"sm_90a"},
    },
    .{
        .source = "flash_bwd_sm90",
        .launch = configurations.launchBackwardSm90,
        .ptx_target = "sm_90a",
        .cubin_targets = &.{"sm_90a"},
    },
    .{
        .source = "flash_fwd_sm100",
        .launch = configurations.launchForwardSm100,
        .ptx_target = "sm_100f",
        .cubin_targets = &.{"sm_100f"},
    },
    .{
        .source = "flash_bwd_sm100",
        .launch = configurations.launchBackwardSm100,
        .ptx_target = "sm_100f",
        .cubin_targets = &.{"sm_100f"},
    },
    .{
        .source = "flash_bwd_sm120",
        .launch = configurations.launchBackwardSm120,
        .ptx_target = "sm_120f",
        .cubin_targets = &.{"sm_120f"},
    },
};

pub fn build(b: *std.Build) void {
    const optimize = b.option(std.lang.Optimize, "kernel-optimize", "Kernel optimization mode") orelse .fast;
    const ptxas_option = b.option([]const u8, "ptxas", "Path to ptxas");
    const sources = b.option([]const []const u8, "source", "Only build these kernel sources");
    const targets = b.option([]const []const u8, "sm", "Only assemble cubins for these targets");
    const head_dims = b.option([]const []const u8, "head-dim", "Only build these head dimensions");
    const modes = b.option([]const []const u8, "mode", "Only build these modes");
    const ptxas = ptxas_option orelse b.findProgram(.{ .names = &.{"ptxas"} }) orelse
        b.findProgram(.{ .names = &.{"/usr/local/cuda/bin/ptxas"} }) orelse "ptxas";
    const fatbinary = b.pathJoin(&.{ std.fs.path.dirname(ptxas) orelse "", "fatbinary" });

    var manifest: std.Io.Writer.Allocating = .init(b.allocator);
    var declared: std.Io.Writer.Allocating = .init(b.allocator);
    if (sources != null or targets != null or head_dims != null or modes != null) {
        declared.writer.writeAll("partial\n") catch @panic("OOM");
    }
    manifest.writer.writeAll("# name") catch @panic("OOM");
    inline for (@typeInfo(configurations.Launch).@"struct".field_names) |field_name| {
        manifest.writer.print(" {s}", .{field_name}) catch @panic("OOM");
    }
    manifest.writer.writeByte('\n') catch @panic("OOM");
    @setEvalBranchQuota(300_000);
    inline for (matrix) |kernel_source| {
        inline for (configurations.head_dims) |head_dim| {
            inline for (comptime std.enums.values(configurations.Mode)) |mode| {
                if (kernel_source.launch(head_dim, mode)) |launch| {
                    const name = comptime kernel_source.instance(mode, head_dim);
                    declared.writer.print("{s}\n", .{name}) catch @panic("OOM");
                    if (selected(sources, kernel_source.source) and selectedHeadDim(head_dims, head_dim) and
                        selected(modes, @tagName(mode)))
                    {
                        manifest.writer.writeAll(name) catch @panic("OOM");
                        inline for (@typeInfo(configurations.Launch).@"struct".field_names) |field_name| {
                            const value = @field(launch, field_name);
                            if (@typeInfo(@TypeOf(value)) == .@"enum") {
                                manifest.writer.print(" {s}", .{@tagName(value)}) catch @panic("OOM");
                            } else {
                                manifest.writer.print(" {d}", .{value}) catch @panic("OOM");
                            }
                        }
                        manifest.writer.writeByte('\n') catch @panic("OOM");
                        addInstance(b, kernel_source, mode, head_dim, optimize, ptxas, fatbinary, targets);
                    }
                }
            }
        }
    }
    const files = b.addWriteFiles();
    inline for (.{ .{ "kernels.manifest", &manifest }, .{ "kernels.declared", &declared } }) |output| {
        b.getInstallStep().dependOn(&b.addInstallFile(files.add(output[0], output[1].written()), output[0]).step);
    }

    const tests = b.step("test", "Run the tile scheduler and sequence table tests");
    inline for (.{ "zig/tile_scheduler.zig", "zig/prepare_scheduler.zig", "zig/mbarrier.zig" }) |root| {
        const unit_tests = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(root),
            .target = b.graph.host,
        }) });
        tests.dependOn(&b.addRunArtifact(unit_tests).step);
    }
}

fn selected(filter: ?[]const []const u8, name: []const u8) bool {
    const names = filter orelse return true;
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn selectedHeadDim(filter: ?[]const []const u8, head_dim: u32) bool {
    const values = filter orelse return true;
    for (values) |text| {
        if ((std.fmt.parseInt(u32, text, 10) catch 0) == head_dim) return true;
    }
    return false;
}

fn addInstance(
    b: *std.Build,
    comptime kernel_source: KernelSource,
    comptime mode: configurations.Mode,
    comptime head_dim: u32,
    optimize: std.lang.Optimize,
    ptxas: []const u8,
    fatbinary: []const u8,
    targets: ?[]const []const u8,
) void {
    const name = comptime kernel_source.instance(mode, head_dim);
    const config = b.addOptions();
    config.addOption(u32, "head_dim", head_dim);
    config.addOption([]const u8, "mode", @tagName(mode));
    const kernel = b.addExecutable(.{
        .name = name,
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{kernel_source.source})),
            .target = b.resolveTargetQuery(std.Target.Query.parse(.{
                .arch_os_abi = "nvptx64-cuda",
                .cpu_features = b.fmt("{s}+ptx90", .{kernel_source.ptx_target}),
            }) catch @panic("invalid target")),
            .optimize = optimize,
            .imports = &.{.{ .name = "config", .module = config.createModule() }},
        }),
    });
    const ptx = kernel.getEmittedBin();
    b.getInstallStep().dependOn(&b.addInstallFile(ptx, b.fmt("ptx/{s}.ptx", .{name})).step);
    const pack = b.addSystemCommand(&.{ fatbinary, "-64" });
    const cubin = pack.addPrefixedOutputFileArg("--create=", b.fmt("{s}.cubin", .{name}));
    var images: u32 = 0;
    inline for (kernel_source.cubin_targets) |target| {
        if (selected(targets, target)) {
            const assemble = b.addSystemCommand(&.{ ptxas, b.fmt("-arch={s}", .{target}) });
            assemble.addFileArg(ptx);
            assemble.addArg("-o");
            const image = assemble.addOutputFileArg(b.fmt("{s}.{s}.cubin", .{ name, target }));
            pack.addPrefixedFileArg(b.fmt("--image3=kind=elf,sm={s},file=", .{target["sm_".len..]}), image);
            images += 1;
        }
    }
    if (images > 0) b.getInstallStep().dependOn(&b.addInstallFile(cubin, b.fmt("cubin/{s}.cubin", .{name})).step);
}
