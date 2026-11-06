"""Per-op error of the integer nonlinear kernels (nl_fixed.h, compiled for RV32 and run in the interpreter) vs HF Qwen3-0.6B fp32 on REAL activations (acts.npz from capture_acts.py)."""
import json, sys, numpy as np
from rvsim import Sim, elf_text, s32
acts = np.load("acts.npz"); LAY = [0, 1, 7, 14, 21, 27]
objs = {"soft": "k2_soft.o", "ext": "k2_ext.o"}; S = {k: elf_text(v) for k, v in objs.items()}
A, B, C_, D_ = 0x10000, 0x40000, 0x70000, 0xA0000
def mk(v): return Sim(S[v][0])
def call(v, fn, args, setup):
    s = mk(v); setup(s); r = s.call(S[v][1][fn], args); return s, r
res = {v: {} for v in objs}
CURL = [0]
def rec(v, key, errmax, errrel, ins):
    key = f"{key}@L{CURL[0]}"
    d = res[v].setdefault(key, dict(max_err_over_max_ref=0.0, rel_l2=0.0, instr=0, n_vec=0)); d["max_err_over_max_ref"] = max(d["max_err_over_max_ref"], errmax); d["rel_l2"] = max(d["rel_l2"], errrel); d["instr"] += ins; d["n_vec"] += 1
def rms_test(v, L, which, x, g, ref, l2n, eps=1e-6):
    xq = np.round(x * 4096).astype(np.int64); gq = np.round(g * 4096).astype(np.int64); n = len(x)
    def st(s): s.put_i32(A, [int(t) for t in xq]); s.put_i32(B, [int(t) for t in gq])
    s = mk(v); st(s); y = s.call(S[v][1]["t_rms_inv"], [A, n, l2n]) & 0xFFFFFFFF; i1 = s.n
    s.call(S[v][1]["t_rms_scale"], [A, B, C_, n, y]); o = np.array(s.get_i32(C_, n)) / 4096.0
    rec(v, f"{which}", float(np.max(np.abs(o - ref)) / np.max(np.abs(ref))), float(np.linalg.norm(o - ref) / np.linalg.norm(ref)), s.n)
    return y
pos_n = [0, 1, 2, 17, 40, 63]
for v in objs:
    for L in LAY:
        CURL[0] = L
        for nm, key in (("in_norm", "in_g"), ("post_norm", "post_g")):
            xin, xo = acts[f"L{L}.{nm}.0"], acts[f"L{L}.{nm}.1"]; g = acts[f"L{L}.{key}"]
            for p in pos_n: rms_test(v, L, "rmsnorm_d1024", xin[p], g, xo[p], 10)
        for nm, key in (("q_norm", "q_g"), ("k_norm", "k_g")):
            xin, xo = acts[f"L{L}.{nm}.0"], acts[f"L{L}.{nm}.1"]; g = acts[f"L{L}.{key}"]
            for p in (0, 17, 63):
                for h in range(xin.shape[1]): rms_test(v, L, "qk_norm_d128", xin[p, h], g, xo[p, h], 7)
    print(v, "norms done", flush=True)
# rope + softmax + swiglu
log2e = 1.4426950408889634
for v in objs:
    for L in LAY:
        CURL[0] = L
        q0, q1 = acts[f"L{L}.q_rope.0"], acts[f"L{L}.q_rope.1"]      # [h,T,hd]
        cos, sin = acts[f"L{L}.cossin.0"], acts[f"L{L}.cossin.1"]       # [T,hd]
        sc = 2 ** np.floor(14 - np.log2(max(np.abs(q0).max(), np.abs(q1).max())))
        for p in (0, 5, 40, 63):
            for h in (0, 7, 15):
                x1 = np.round(q0[h, p, :64] * sc).astype(np.int64); x2 = np.round(q0[h, p, 64:] * sc).astype(np.int64)
                c = np.round(cos[p, :64] * 32767).astype(np.int64); sn = np.round(sin[p, :64] * 32767).astype(np.int64)
                xp = [s32(int(((b & 0xFFFF) << 16) | (a & 0xFFFF))) for a, b in zip(x1, x2)]; cs = [s32(int(((b & 0xFFFF) << 16) | (a & 0xFFFF))) for a, b in zip(c, sn)]
                def st(s): s.put_i32(A, xp); s.put_i32(B, cs)
                s, _ = call(v, "t_rope", [A, B, C_, 64], st); o = np.array(s.get_i32(C_, 64)) & 0xFFFFFFFF
                y1 = (((o & 0xFFFF) ^ 0x8000) - 0x8000) / sc; y2 = ((((o >> 16) & 0xFFFF) ^ 0x8000) - 0x8000) / sc
                ref = np.concatenate([q1[h, p, :64], q1[h, p, 64:]]); got = np.concatenate([y1, y2])
                rec(v, "rope_pairs64", float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))), float(np.linalg.norm(got - ref) / np.linalg.norm(ref)), s.n)
        k0 = acts[f"L{L}.k_rope.1"]; probs = acts[f"L{L}.probs.0"] if False else acts[f"L{L}.probs"]
        for p in (9, 31, 63):
            for h in range(16):
                sco = q1[h, p] @ k0[h // 2, :p + 1].T / np.sqrt(128.0)             # GQA: 16 q heads, 8 kv heads
                sq = np.round(sco * log2e * 65536).astype(np.int64)
                def st(s): s.put_i32(A, [int(t) for t in sq])
                s, _ = call(v, "t_softmax", [A, B, p + 1], st); o = np.array(s.get_i32(B, p + 1)) / 65536.0
                ref = probs[h, p, :p + 1]
                rec(v, "softmax_rows", float(np.max(np.abs(o - ref))), float(np.linalg.norm(o - ref) / np.linalg.norm(ref)), s.n)
        g_, u_, ref_ = acts[f"L{L}.gate"], acts[f"L{L}.up"], acts[f"L{L}.swiglu_out"]
        for p in (0, 3, 40, 63):
            gq = np.round(np.clip(g_[p] * 65536, -2**31 + 1, 2**31 - 1)).astype(np.int64); uq = np.round(np.clip(u_[p] * 65536, -2**31 + 1, 2**31 - 1)).astype(np.int64)
            n = 3072
            def st(s): s.put_i32(A, [int(t) for t in gq]); s.put_i32(B, [int(t) for t in uq])
            s, _ = call(v, "t_swiglu", [A, B, C_, n], st); o = np.array(s.get_i32(C_, n)) / 65536.0
            rec(v, "swiglu_d3072", float(np.max(np.abs(o - ref_[p])) / np.max(np.abs(ref_[p]))), float(np.linalg.norm(o - ref_[p]) / np.linalg.norm(ref_[p])), s.n)
    print(v, "rope/softmax/swiglu done", flush=True)
json.dump(res, open("res_real.json", "w"), indent=1)
for v in res:
    print("==", v)
    for k, d in res[v].items(): print(f"  {k:16s} max|err|/max|ref| = {d['max_err_over_max_ref']:.3e}   rel L2 = {d['rel_l2']:.3e}   instr/vec = {d['instr']/d['n_vec']:.0f}   ({d['n_vec']} vectors)")
