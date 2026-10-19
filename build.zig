const std = @import("std");
const configurations = @import("zig/configurations.zig");

const Source = configurations.Source;

fn instance(comptime source: Source, comptime variant: []const u8, head_dim: u32) []const u8 {
    return std.fmt.comptimePrint("{s}{s}_hd{d}", .{ @tagName(source), variant, head_dim });
}

fn variantOptionName(comptime variant: []const u8) []const u8 {
    return if (variant.len == 0) "unrestricted" else variant[1..];
}

fn ptxTarget(comptime source: Source) []const u8 {
    return switch (source.arch()) {
        80 => "sm_80",
        90 => "sm_90a",
        100 => "sm_100f",
        else => "sm_120f",
    };
}

fn cubinTargets(comptime source: Source) []const []const u8 {
    if (source.arch() == 80) return &.{ "sm_80", "sm_86", "sm_89", "sm_120f" };
    return &.{ptxTarget(source)};
}

pub fn build(b: *std.Build) void {
    const optimize =
        b.option(std.lang.Optimize, "kernel-optimize", "Kernel optimization mode") orelse .fast;
    const ptxas_option = b.option([]const u8, "ptxas", "Path to ptxas");
    const profile = b.option(bool, "profile", "Record kernel zones into a profile buffer") orelse false;
    const sources = b.option([]const []const u8, "source", "Only build these kernel sources");
    const targets = b.option([]const []const u8, "sm", "Only assemble cubins for these targets");
    const head_dims = b.option([]const []const u8, "head-dim", "Only build these head dimensions");
    const variant_names = b.option(
        []const []const u8,
        "variant",
        "Only build these variants, named without the leading underscore or as unrestricted",
    );
    const ptxas = ptxas_option orelse b.findProgram(.{ .names = &.{"ptxas"} }) orelse
        b.findProgram(.{ .names = &.{"/usr/local/cuda/bin/ptxas"} }) orelse "ptxas";
    const fatbinary = b.pathJoin(&.{ std.fs.path.dirname(ptxas) orelse "", "fatbinary" });

    var manifest: std.Io.Writer.Allocating = .init(b.allocator);
    var declared: std.Io.Writer.Allocating = .init(b.allocator);
    if (sources != null or targets != null or head_dims != null or variant_names != null) {
        declared.writer.writeAll("partial\n") catch @panic("OOM");
    }
    manifest.writer.writeAll("# name") catch @panic("OOM");
    inline for (@typeInfo(configurations.Launch).@"struct".field_names) |field_name| {
        manifest.writer.print(" {s}", .{field_name}) catch @panic("OOM");
    }
    manifest.writer.writeByte('\n') catch @panic("OOM");
    @setEvalBranchQuota(300_000);
    inline for (comptime std.enums.values(Source)) |source| {
        inline for (comptime source.headDims()) |head_dim| {
            const variants = comptime configurations.variants(source, head_dim);
            inline for (0..variants.len) |index| {
                const variant = comptime variants[index].specialization.name();
                const launch = comptime configurations.launch(source, head_dim, index);
                const name = comptime instance(source, variant, head_dim);
                declared.writer.print("{s}\n", .{name}) catch @panic("OOM");
                if (selected(sources, @tagName(source)) and
                    selectedHeadDim(head_dims, head_dim) and
                    selected(variant_names, variantOptionName(variant)))
                {
                    manifest.writer.writeAll(name) catch @panic("OOM");
                    inline for (@typeInfo(
                        configurations.Launch,
                    ).@"struct".field_names) |field_name| {
                        const value = @field(launch, field_name);
                        if (@typeInfo(@TypeOf(value)) == .@"enum") {
                            manifest.writer.print(" {s}", .{@tagName(value)}) catch
                                @panic("OOM");
                        } else {
                            manifest.writer.print(" {d}", .{value}) catch @panic("OOM");
                        }
                    }
                    manifest.writer.writeByte('\n') catch @panic("OOM");
                    addInstance(b, source, index, head_dim, optimize, profile, ptxas, fatbinary, targets);
                }
            }
        }
    }
    const files = b.addWriteFiles();
    inline for (.{ .{
        "kernels.manifest", &manifest,
    }, .{ "kernels.declared", &declared } }) |output| {
        b.getInstallStep()
            .dependOn(&b.addInstallFile(files.add(output[0], output[1].written()), output[0]).step);
    }

    const tests = b.step("test", "Run the host unit tests");
    inline for (.{
        "zig/tile_scheduler.zig", "zig/prepare_scheduler.zig", "zig/mbarrier.zig",
    }) |root| {
        const unit_tests = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(root),
            .target = b.graph.host,
        }) });
        tests.dependOn(&b.addRunArtifact(unit_tests).step);
    }
    const profile_config = b.addOptions();
    profile_config.addOption(bool, "profile", true);
    const profile_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/profile.zig"),
        .target = b.graph.host,
        .imports = &.{.{ .name = "config", .module = profile_config.createModule() }},
    }) });
    tests.dependOn(&b.addRunArtifact(profile_tests).step);
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
    comptime source: Source,
    comptime index: usize,
    comptime head_dim: u32,
    optimize: std.lang.Optimize,
    profile: bool,
    ptxas: []const u8,
    fatbinary: []const u8,
    targets: ?[]const []const u8,
) void {
    const variant = comptime configurations.variants(source, head_dim)[index].specialization.name();
    const name = comptime instance(source, variant, head_dim);
    const config = b.addOptions();
    config.addOption(u32, "head_dim", head_dim);
    config.addOption(usize, "variant", index);
    config.addOption(bool, "profile", profile);
    const kernel = b.addExecutable(.{
        .name = name,
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{@tagName(source)})),
            .target = b.resolveTargetQuery(std.Target.Query.parse(.{
                .arch_os_abi = "nvptx64-cuda",
                .cpu_features = b.fmt("{s}+ptx90", .{ptxTarget(source)}),
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
    inline for (comptime cubinTargets(source)) |target| {
        if (selected(targets, target)) {
            const assemble = b.addSystemCommand(&.{ ptxas, b.fmt("-arch={s}", .{target}) });
            assemble.addFileArg(ptx);
            assemble.addArg("-o");
            const image = assemble.addOutputFileArg(b.fmt("{s}.{s}.cubin", .{ name, target }));
            pack.addPrefixedFileArg(b.fmt("--image3=kind=elf,sm={s},file=", .{
                target["sm_".len..],
            }), image);
            images += 1;
        }
    }
    if (images > 0) b.getInstallStep()
        .dependOn(&b.addInstallFile(cubin, b.fmt("cubin/{s}.cubin", .{name})).step);
}
