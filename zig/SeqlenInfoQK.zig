const SeqlenInfoQK = @This();

offset_q: u32,
offset_k: u32,
seqlen_q: u32,
seqlen_k: u32,

pub fn resolve(
    cu_seqlens_q: u64,
    cu_seqlens_k: u64,
    seqlen_q: u32,
    seqlen_k: u32,
    batch: u32,
) SeqlenInfoQK {
    if (cu_seqlens_q == 0) {
        return .{ .offset_q = 0, .offset_k = 0, .seqlen_q = seqlen_q, .seqlen_k = seqlen_k };
    }
    const m_cu_seqlens_q: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens_q);
    const m_cu_seqlens_k: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens_k);
    const offset_q = m_cu_seqlens_q[batch];
    const offset_k = m_cu_seqlens_k[batch];
    return .{
        .offset_q = @intCast(offset_q),
        .offset_k = @intCast(offset_k),
        .seqlen_q = @intCast(m_cu_seqlens_q[batch + 1] - offset_q),
        .seqlen_k = @intCast(m_cu_seqlens_k[batch + 1] - offset_k),
    };
}

pub fn flatRows(cu_seqlens: u64, num_batch: u32, seqlen: u32) SeqlenInfoQK {
    if (cu_seqlens == 0) return .{ .offset_q = 0, .offset_k = 0, .seqlen_q = seqlen, .seqlen_k = seqlen };
    const m_cu_seqlens: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens);
    const first: u32 = @intCast(m_cu_seqlens[0]);
    const rows: u32 = @intCast(m_cu_seqlens[num_batch] - m_cu_seqlens[0]);
    return .{ .offset_q = first, .offset_k = first, .seqlen_q = rows, .seqlen_k = rows };
}

pub fn rowsBeforeEnd(cu_seqlens: u64, num_batch: u32, first_row: u32, comptime rows: u32) u32 {
    if (cu_seqlens == 0) return rows;
    const m_cu_seqlens: [*]addrspace(.global) const i32 = @ptrFromInt(cu_seqlens);
    const end: u32 = @intCast(m_cu_seqlens[num_batch]);
    return @min(rows, end -| first_row);
}
