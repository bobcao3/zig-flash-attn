pub fn wait() void {
    asm volatile ("griddepcontrol.wait;" ::: .{ .memory = true });
}

pub fn launchDependents() void {
    asm volatile ("griddepcontrol.launch_dependents;" ::: .{ .memory = true });
}
