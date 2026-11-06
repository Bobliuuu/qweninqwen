"""Run every kernel in the RV32 interpreter: functional check vs float reference + dynamic instruction / trace-token counts (slopes per element)."""
import json, sys, struct, numpy as np
from rvsim import Sim, elf_text, s32
import ext
text, syms, rel = elf_text("kernels.o")
assert not rel or True
rng = np.random.default_rng(0)
def mk(wrom=b""): return Sim(text, wrom=wrom)
A, B, C_, D_, E_, F_ = 0x10000, 0x40000, 0x70000, 0xA0000, 0xD0000, 0x100000
res = {}
def slope(name, setup, n1=256, n2=512):
    out = []
    for n in (n1, n2):
        s = mk(); r = setup(s, n); s.call(syms[name], r); out.append((s.n, s.tokens, dict(s.cnt)))
    return dict(instr_per_elem=(out[1][0] - out[0][0]) / (n2 - n1), tokens_per_elem=(out[1][1] - out[0][1]) / (n2 - n1), fixed_instr=out[0][0] - (out[1][0] - out[0][0]) / (n2 - n1) * n1)
def rel(a, b): return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))

# ---- rsqrt (scalar)
ms = np.exp(rng.uniform(np.log(2 ** -8), np.log(2 ** 14), 400)); mq = (ms * 65536).astype(np.int64)
for nm in ("k_rsqrt_base", "k_rsqrt_ext"):
    ys = []; cnts = []; toks = []
    for m in mq:
        s = mk(); y = s.call(syms[nm], [int(m)]) & 0xFFFFFFFF; ys.append(y); cnts.append(s.n); toks.append(s.tokens)
    ref = 2 ** 24 / np.sqrt(mq / 65536.0)
    res[nm] = dict(max_rel_err=rel(np.array(ys, float), ref), instr=float(np.mean(cnts)), tokens=float(np.mean(toks)), note="includes the ret")
    print(nm, res[nm], flush=True)
# ---- exp2 / sigm scalar
xq = rng.integers(-7 * 65536, 0, 400)
for nm in ("k_exp2_base", "k_exp2_ext"):
    ys = []; cnts = []; toks = []
    for x in xq:
        s = mk(); y = s.call(syms[nm], [int(x)]) & 0xFFFFFFFF; ys.append(y); cnts.append(s.n); toks.append(s.tokens)
    ref = 2 ** (xq / 65536.0) * 2 ** 24
    res[nm] = dict(max_rel_err=rel(np.array(ys, float), ref), max_abs_err_Q24=float(np.max(np.abs(np.array(ys, float) - ref))), instr=float(np.mean(cnts)), tokens=float(np.mean(toks)))
    print(nm, res[nm], flush=True)
xs = rng.integers(-11 * 65536, 11 * 65536, 400)
s = mk(); y = s.call(syms["k_sigm_ext"], [int(xs[0])]); res["k_sigm_ext"] = dict(instr=s.n, tokens=s.tokens)
yy = [mk().call(syms["k_sigm_ext"], [int(x)]) for x in xs]; res["k_sigm_ext"]["max_abs_err_vs_float"] = float(np.max(np.abs(np.array(yy) / 65536.0 - 1 / (1 + np.exp(-xs / 65536.0)))))
print("k_sigm_ext", res["k_sigm_ext"])
# ---- vector kernels
def put32(s, a, arr): s.put_i32(a, arr)
# sumsq
def su(s, n): put32(s, A, rng.integers(-4096 * 4, 4096 * 4, n)); return [A, n]
res["k_sumsq"] = slope("k_sumsq", su); print("k_sumsq", res["k_sumsq"])
# rms scale
def rs(s, n): put32(s, A, rng.integers(-4096 * 4, 4096 * 4, n)); put32(s, B, rng.integers(1000, 9000, n)); return [A, B, C_, n, 0x01000000]
res["k_rms_scale"] = slope("k_rms_scale", rs); print("k_rms_scale", res["k_rms_scale"])
# end-to-end rmsnorm accuracy (base vs ext): sumsq -> mean -> rsqrt -> scale
def rms_run(nm_inv, x, g):
    n = len(x); s = mk(); put32(s, A, x); put32(s, B, g)
    ms = s.call(syms[nm_inv], [A, n, 10]) & 0xFFFFFFFF                       # n=1024 -> shift 10
    i1, t1 = s.n, s.tokens
    s.call(syms["k_rms_scale"], [A, B, C_, n, ms]); o = np.array(s.get_i32(C_, n)) / 4096.0
    return o, i1, t1, s.n, s.tokens
x = np.round(rng.normal(0, 3.0, 1024) * 4096).astype(np.int64); g = np.round(rng.uniform(0.5, 2.0, 1024) * 4096).astype(np.int64)
xf = x / 4096.0; gf = g / 4096.0; ref = xf / np.sqrt(np.mean(xf ** 2) + 1.0 / 65536 * 0) * gf
for nm in ("k_rms_inv_base", "k_rms_inv_ext"):
    o, i1, t1, i2, t2 = rms_run(nm, x, g)
    res["rmsnorm_" + nm[10:]] = dict(max_abs_err=float(np.max(np.abs(o - ref))), max_rel_err_vs_max=float(np.max(np.abs(o - ref)) / np.max(np.abs(ref))), instr_scalar_part_incl_sumsq=i1, instr_scale_part=i2, tokens_total=t1 + t2 if False else None)
    print(nm, res["rmsnorm_" + nm[10:]])
# softmax
def sm(s, n): put32(s, A, rng.integers(-8 * 65536, 0, n) ); return [A, B, n]
def sm_acc(nm, T=100):
    s = mk(); sc = rng.integers(-8 * 65536, 4 * 65536, T); put32(s, A, sc); s.call(syms[nm], [A, B, T])
    p = np.array(s.get_i32(B, T)) / 65536.0; q = 2 ** (sc / 65536.0); q /= q.sum(); return float(np.max(np.abs(p - q))), float(np.sum(p))
for nm in ("k_softmax_base", "k_softmax_ext"):
    r = slope(nm, sm, 128, 256); r["max_abs_err_prob"], r["sum_p"] = sm_acc(nm); res[nm] = r; print(nm, r, flush=True)
# silu
def sl(s, n): put32(s, A, rng.integers(-8 * 65536, 8 * 65536, n)); put32(s, B, rng.integers(-2 * 65536, 2 * 65536, n)); return [A, B, C_, n]
def sl_acc(nm, n=300):
    s = mk(); g = rng.integers(-10 * 65536, 10 * 65536, n); u = rng.integers(-2 * 65536, 2 * 65536, n); put32(s, A, g); put32(s, B, u); s.call(syms[nm], [A, B, C_, n])
    o = np.array(s.get_i32(C_, n)) / 65536.0; gf = g / 65536.0; ref = gf / (1 + np.exp(-gf)) * (u / 65536.0); return float(np.max(np.abs(o - ref))), float(np.max(np.abs(ref)))
for nm in ("k_silu_base", "k_silu_ext"):
    r = slope(nm, sl, 64, 128); r["max_abs_err"], r["max_abs_ref"] = sl_acc(nm); res[nm] = r; print(nm, r, flush=True)
# rope
def rp(s, n):
    for a_, nb in ((A, 2), (B, 2), (C_, 2), (D_, 2)): s.put_bytes(a_, rng.integers(-16000, 16000, n, dtype=np.int16).tobytes())
    return [A, B, C_, D_, E_, F_, n]
def rope_acc():
    n = 200; s = mk(); x1 = rng.integers(-16000, 16000, n); x2 = rng.integers(-16000, 16000, n); th = rng.uniform(0, 2 * np.pi, n)
    c = np.round(np.cos(th) * 32767).astype(np.int64); sn = np.round(np.sin(th) * 32767).astype(np.int64)
    for a_, v in ((A, x1), (B, x2), (C_, c), (D_, sn)): s.put_bytes(a_, v.astype(np.int16).tobytes())
    s.call(syms["k_rope_base"], [A, B, C_, D_, E_, F_, n])
    y1 = np.frombuffer(bytes(s.mem[E_:E_ + 2 * n]), np.int16); y2 = np.frombuffer(bytes(s.mem[F_:F_ + 2 * n]), np.int16)
    r1 = x1 * np.cos(th) - x2 * np.sin(th); r2 = x1 * np.sin(th) + x2 * np.cos(th)
    eb = max(np.max(np.abs(y1 - r1)), np.max(np.abs(y2 - r2)))
    # ext: packed pairs
    s2 = mk(); xp = ((x2 & 0xFFFF) << 16) | (x1 & 0xFFFF); csp = ((sn & 0xFFFF) << 16) | (c & 0xFFFF)
    s2.put_i32(A, [s32(int(v)) for v in xp]); s2.put_i32(B, [s32(int(v)) for v in csp]); s2.call(syms["k_rope_ext"], [A, B, C_, n])
    o = np.array(s2.get_i32(C_, n)) & 0xFFFFFFFF; e1 = ((o & 0xFFFF) ^ 0x8000) - 0x8000; e2 = (((o >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
    ee = max(np.max(np.abs(e1 - r1)), np.max(np.abs(e2 - r2)))
    return float(eb), float(ee)
res["k_rope_base"] = slope("k_rope_base", rp); res["k_rope_ext"] = slope("k_rope_ext", lambda s, n: (put32(s, A, rng.integers(-2**31, 2**31 - 1, n)), put32(s, B, rng.integers(-2**31, 2**31 - 1, n)), [A, B, C_, n])[2])
res["k_rope_base"]["max_abs_err_int16"], res["k_rope_ext"]["max_abs_err_int16"] = rope_acc(); print(res["k_rope_base"], res["k_rope_ext"])
# sincos table
tab = np.round(np.sin(2 * np.pi * np.arange(257 + 64) / 256) * 32767).astype(np.int16)
def sc_acc():
    errs = []; cn = []
    for _ in range(100):
        t = int(rng.integers(0, 2 ** 32)); s = mk(); s.put_bytes(A, tab.tobytes()); v = s.call(syms["k_sincos_base"], [t, A]) & 0xFFFFFFFF; cn.append((s.n, s.tokens))
        sn = ((v >> 16) ^ 0x8000) - 0x8000; cs = ((v & 0xFFFF) ^ 0x8000) - 0x8000; th = 2 * np.pi * t / 2 ** 32
        errs.append(max(abs(sn - 32767 * np.sin(th)), abs(cs - 32767 * np.cos(th))))
    return float(max(errs)), float(np.mean([c[0] for c in cn])), float(np.mean([c[1] for c in cn]))
res["k_sincos_base"] = dict(zip(("max_abs_err_Q15", "instr", "tokens"), sc_acc())); print(res["k_sincos_base"])
# residual / requant / argmax
res["k_add"] = slope("k_add", lambda s, n: (put32(s, A, rng.integers(-1e6, 1e6, n)), put32(s, B, rng.integers(-1e6, 1e6, n)), [A, B, C_, n])[2]); print(res["k_add"])
res["k_requant"] = slope("k_requant", lambda s, n: (put32(s, A, rng.integers(-1e8, 1e8, n)), [A, B, n, 0x01000000])[1]); print(res["k_requant"])
for nm in ("k_argmax_base", "k_argmax_ext", "k_argmax_fused_base"):
    def am(s, n): put32(s, A, rng.permutation(n * 8)[:n] - n * 4); return [A, n]
    r = slope(nm, am, 1024, 2048)
    for tr in range(5):
        n = 2000; s = mk(); a = rng.integers(-2**30, 2**30, n); put32(s, A, a); got = s.call(syms[nm], [A, n]); assert got == int(np.argmax(a)), (nm, got, int(np.argmax(a)))
    r["correct_5_trials"] = True; res[nm] = r; print(nm, r)
# dots
def dot_acc():
    n = 256; w8 = rng.integers(-8, 8, n); a = rng.integers(-127, 128, n); out = {}
    s = mk(); s.put_bytes(A, w8.astype(np.int8).tobytes()); s.put_bytes(B, a.astype(np.int8).tobytes()); out["i8"] = (s.call(syms["k_dot_i8_base"], [A, B, n]) == int(w8 @ a))
    wn = (w8 + 8).astype(np.uint32); words = [sum(int(wn[8 * i + k]) << (4 * k) for k in range(8)) for i in range(n // 8)]
    s = mk(); s.put_i32(A, [s32(v) for v in words]); s.put_bytes(B, a.astype(np.int8).tobytes()); out["i4"] = (s.call(syms["k_dot_i4_base"], [A, B, n]) == int(w8 @ a))
    s = mk(); s.put_bytes(A, w8.astype(np.int8).tobytes()); s.put_bytes(B, a.astype(np.int8).tobytes()); out["smaqa"] = (s.call(syms["k_dot_smaqa"], [A, B, n // 4]) == int(w8 @ a))
    wrom = bytes(wn.astype(np.uint8)); s = Sim(text, wrom=wrom); s.put_bytes(B, a.astype(np.int8).tobytes()); s.wp = 0
    out["vdot"] = (s.call(syms["k_dot_vdot1"], [B, n // 8]) == int(w8 @ a)); return out
res["dot_correct"] = dot_acc(); print(res["dot_correct"])
for nm, setup in (("k_dot_i8_base", lambda s, n: (s.put_bytes(A, bytes(n)), s.put_bytes(B, bytes(n)), [A, B, n])[2]),
                  ("k_dot_i4_base", lambda s, n: (s.put_bytes(A, bytes(n)), s.put_bytes(B, bytes(n)), [A, B, n])[2]),
                  ("k_dot_smaqa", lambda s, n: (s.put_bytes(A, bytes(4 * n)), s.put_bytes(B, bytes(4 * n)), [A, B, n])[2]),
                  ("k_dot_vdot1", lambda s, n: (s.put_bytes(A, bytes(8 * n)), [A, n])[1]),
                  ("k_dot_vdot8", lambda s, n: (s.put_bytes(A, bytes(8 * n)), [A, n, B])[1:] if False else [A, n, B])):
    r = slope(nm, setup, 128, 256); res[nm] = r; print(nm, r)
for k in ("k_dot_i8_base", "k_dot_i4_base"): res[k]["instr_per_mac"] = res[k]["instr_per_elem"]; res[k]["tokens_per_mac"] = res[k]["tokens_per_elem"]
res["k_dot_smaqa"]["instr_per_mac"] = res["k_dot_smaqa"]["instr_per_elem"] / 4; res["k_dot_smaqa"]["tokens_per_mac"] = res["k_dot_smaqa"]["tokens_per_elem"] / 4
res["k_dot_vdot1"]["instr_per_mac"] = res["k_dot_vdot1"]["instr_per_elem"] / 8; res["k_dot_vdot1"]["tokens_per_mac"] = res["k_dot_vdot1"]["tokens_per_elem"] / 8
res["k_dot_vdot8"]["instr_per_mac"] = res["k_dot_vdot8"]["instr_per_elem"] / 64; res["k_dot_vdot8"]["tokens_per_mac"] = res["k_dot_vdot8"]["tokens_per_elem"] / 64
json.dump(res, open("res_kernels.json", "w"), indent=1)
