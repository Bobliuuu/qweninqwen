import sys, time, json, numpy as np, torch
from netlist import *; import fp
dev = "cuda" if torch.cuda.is_available() else "cpu"
def build(f, na):
    n = Net(); ins = [n.inp(32) for _ in range(na)]; out = f(n, *ins); return n, ins, out
def run(n, ins, out, arrs, N):
    s = Sim(n, out, dev); feed = {}
    for bus, x in zip(ins, arrs):
        u = x.astype(np.float32).view(np.uint32)
        bits = ((u[None, :] >> np.arange(32, dtype=np.uint32)[:, None]) & 1).astype(bool); P = pack(bits)
        for i, w in enumerate(bus): feed[w] = torch.from_numpy(P[i].copy()).to(dev)
    o, _ = s.run(feed)
    b = np.stack([unpack(o[w], N) if w in o else np.zeros(N, bool) for w in out]); u = np.zeros(N, np.uint64)
    for i in range(32): u |= b[i].astype(np.uint64) << np.uint64(i)
    return u.astype(np.uint32).view(np.float32)
def ulp_err(y, ref):
    ref = ref.astype(np.float64); y = y.astype(np.float64); nz = ref != 0
    e = np.zeros_like(ref); e[nz] = np.abs(y[nz] - ref[nz]) / np.abs(ref[nz]); return e
if __name__ == "__main__":
    N = 200000; rng = np.random.default_rng(0); res = {}
    def rnd(lo, hi): return (rng.choice([-1, 1], N) * 2.0 ** rng.uniform(lo, hi, N) * rng.uniform(1, 2, N)).astype(np.float32)
    a, b = rnd(-20, 20), rnd(-20, 20)
    for name, f, ref in [("fmul", fp.fmul, a * b), ("fadd", fp.fadd, a + b), ("fsub", fp.fsub, a - b)]:
        n, ins, out = build(f, 2); st = n.stats(out); t = time.time(); y = run(n, ins, out, [a, b], N); dt = time.time() - t
        e = ulp_err(y, ref); res[name] = dict(st, max_rel_err=float(e.max()), mean_rel_err=float(e.mean()), frac_bit_exact=float((y == ref).mean()), n_samples=N, sim_sec=dt)
        print(name, res[name], flush=True)
    # near-cancellation adds
    c = a * (1 + rng.uniform(-1e-3, 1e-3, N).astype(np.float32)); n, ins, out = build(fp.fadd, 2); y = run(n, ins, out, [a, -c], N)
    ref = a - c; e = ulp_err(y, ref); amax = np.abs(a).astype(np.float64)
    res["fadd_cancel_abs_err_over_max_operand"] = float((np.abs(y.astype(np.float64) - ref) / amax).max()); print(res["fadd_cancel_abs_err_over_max_operand"])
    json.dump(res, open("res_fp.json", "w"), indent=1)
