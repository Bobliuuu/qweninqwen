"""Per-element instruction / trace-token slopes of the nl_fixed.h kernels (soft and ext builds) -> res_header.json (keys mirror res_kernels.json so counts.py can use them)."""
import json, numpy as np
from rvsim import Sim, elf_text
rng = np.random.default_rng(0); out = {}
A, B, C_ = 0x10000, 0x40000, 0x70000
def slope(text, syms, fn, setup, n1, n2):
    r = []
    for n in (n1, n2):
        s = Sim(text); args = setup(s, n); s.call(syms[fn], args); r.append((s.n, s.tokens))
    ip = (r[1][0] - r[0][0]) / (n2 - n1); tp = (r[1][1] - r[0][1]) / (n2 - n1)
    return dict(instr_per_elem=ip, tokens_per_elem=tp, fixed_instr=r[0][0] - ip * n1)
for v in ("soft", "ext"):
    text, syms, _ = elf_text(f"k2_{v}.o"); tag = "base" if v == "soft" else "ext"
    def put(s, a, arr): s.put_i32(a, [int(t) for t in arr])
    out[f"k_sumsq"] = slope(text, syms, "t_meansq", lambda s, n: (put(s, A, rng.integers(-2**24, 2**24, n)), [A, n, 10])[1], 256, 512) if v == "soft" else out["k_sumsq"]
    out[f"k_rms_scale"] = slope(text, syms, "t_rms_scale", lambda s, n: (put(s, A, rng.integers(-2**24, 2**24, n)), put(s, B, rng.integers(0, 2**15, n)), [A, B, C_, n, 5000000])[2], 256, 512) if v == "soft" else out["k_rms_scale"]
    out[f"k_softmax_{tag}"] = slope(text, syms, "t_softmax", lambda s, n: (put(s, A, rng.integers(-8 * 65536, 0, n)), [A, B, n])[1], 128, 256)
    out[f"k_silu_{tag}"] = slope(text, syms, "t_swiglu", lambda s, n: (put(s, A, rng.integers(-8 * 65536, 8 * 65536, n)), put(s, B, rng.integers(-2 * 65536, 2 * 65536, n)), [A, B, C_, n])[2], 64, 128)
    out[f"k_rope_{tag}"] = slope(text, syms, "t_rope", lambda s, n: (put(s, A, rng.integers(-2**31, 2**31, n)), put(s, B, rng.integers(-2**31, 2**31, n)), [A, B, C_, n])[2], 128, 256)
    out[f"k_argmax_{tag}"] = slope(text, syms, "t_argmax", lambda s, n: (put(s, A, rng.integers(-2**30, 2**30, n)), [A, n])[1], 1024, 2048)
    cs = []; ts = []
    for m in (1000, 54321, 7654321, 123456789):
        s = Sim(text); s.call(syms["t_rsqrt"], [m]); cs.append(s.n); ts.append(s.tokens)
    out[f"k_rsqrt_{tag}"] = dict(instr=float(np.mean(cs)), tokens=float(np.mean(ts)))
    cs = []; ts = []
    for x in (-3 * 65536, -65536, -5):
        s = Sim(text); s.call(syms["t_exp2"], [x]); cs.append(s.n); ts.append(s.tokens)
    out[f"k_exp2_{tag}"] = dict(instr=float(np.mean(cs)), tokens=float(np.mean(ts)))
json.dump(out, open("res_header.json", "w"), indent=1)
for k, d in out.items(): print(k, {a: round(b, 2) for a, b in d.items()})
