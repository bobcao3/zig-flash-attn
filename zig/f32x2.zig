const std = @import("std");
const nvptx = std.nvptx;

pub const Pair = @Vector(2, f32);

pub fn splat(value: f32) Pair {
    return .{ value, value };
}

pub fn fma(a: Pair, b: Pair, c: Pair) Pair {
    return asm ("{\n.reg .b64 a, b, c, d;\nmov.b64 a, %[a];\nmov.b64 b, %[b];\nmov.b64 c, %[c];\n" ++
            "fma.rn.ftz.f32x2 d, a, b, c;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> Pair),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

pub fn add(a: Pair, b: Pair) Pair {
    return asm ("{\n.reg .b64 a, b, d;\nmov.b64 a, %[a];\nmov.b64 b, %[b];\n" ++
            "add.rn.ftz.f32x2 d, a, b;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> Pair),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

pub fn mul(a: Pair, b: Pair) Pair {
    return asm ("{\n.reg .b64 a, b, d;\nmov.b64 a, %[a];\nmov.b64 b, %[b];\n" ++
            "mul.rn.ftz.f32x2 d, a, b;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> Pair),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

pub fn max3(a: f32, b: f32, c: f32) f32 {
    return asm ("max.f32 %[d], %[a], %[b], %[c];"
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

pub fn exp2f(x: Pair) Pair {
    return .{ nvptx.math.ex2ApproxF32(true, x[0]), nvptx.math.ex2ApproxF32(true, x[1]) };
}

pub fn ex2Emulation(x: Pair) Pair {
    return asm ("{\n" ++
            ".reg .f32 low, high;\n" ++
            ".reg .b64 v, rounded, whole, fraction, magic, c1, c2, c3, one, p;\n" ++
            ".reg .b32 r0, r1, p0, p1;\n" ++
            "mov.b64 v, %[x];\n" ++
            "mov.b64 {low, high}, v;\n" ++
            "max.ftz.f32 low, low, 0fC2FE0000;\n" ++
            "max.ftz.f32 high, high, 0fC2FE0000;\n" ++
            "mov.b64 v, {low, high};\n" ++
            "mov.b64 magic, 0x4B4000004B400000;\n" ++
            "add.rm.ftz.f32x2 rounded, v, magic;\n" ++
            "sub.rn.ftz.f32x2 whole, rounded, magic;\n" ++
            "sub.rn.ftz.f32x2 fraction, v, whole;\n" ++
            "mov.b64 c3, 0x3D9DF09D3D9DF09D;\n" ++
            "mov.b64 c2, 0x3E6906A43E6906A4;\n" ++
            "mov.b64 c1, 0x3F31F5193F31F519;\n" ++
            "mov.b64 one, 0x3F8000003F800000;\n" ++
            "fma.rn.ftz.f32x2 p, fraction, c3, c2;\n" ++
            "fma.rn.ftz.f32x2 p, p, fraction, c1;\n" ++
            "fma.rn.ftz.f32x2 p, p, fraction, one;\n" ++
            "mov.b64 {r0, r1}, rounded;\n" ++
            "mov.b64 {p0, p1}, p;\n" ++
            "shl.b32 r0, r0, 23;\n" ++
            "shl.b32 r1, r1, 23;\n" ++
            "add.s32 p0, p0, r0;\n" ++
            "add.s32 p1, p1, r1;\n" ++
            "mov.b64 p, {p0, p1};\n" ++
            "mov.b64 %[d], p;\n" ++
            "}"
        : [d] "=f" (-> Pair),
        : [x] "f" (x),
    );
}
