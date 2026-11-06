"""Step time of the CORRECTED ROM+SELECT+MAC+accumulation circuit (MEMORY_DESIGN s.10) on the fused-executor-style sparse treatment, from the exact sparsity
captured by capture_layers.py (layers.pt), incl. the flag-gated skip (kernels predicated on a device flag; flag=0 = non-ROM_DOT token).
TIMING ONLY: relu-gated sparse arithmetic, inputs = the captured first test position (so the number of active units/scatter work is the real one); values are checked by memory/*.py.
usage: python bench_corrected.py layers.pt --chip_ms 0.2002 --out res.json"""
import argparse, json, time, torch, triton, triton.language as tl

@triton.jit
def _k(r, yn, flag, D, U, rpa, ixa, wa, rpb, ixb, wb, rpd, ixd, wd, ndmax, BLOCK: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, KD: tl.constexpr):
    on = tl.load(flag) != 0
    u = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = (u < U) & on
    ka = tl.arange(0, KA); a0 = tl.load(rpa + u, mask=m, other=0); na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    ma = m[:, None] & (ka[None, :] < na[:, None])
    a = tl.sum(tl.load(wa + a0[:, None] + ka[None, :], mask=ma, other=0.0) * tl.load(r + tl.load(ixa + a0[:, None] + ka[None, :], mask=ma, other=0), mask=ma, other=0.0), axis=1)
    kb = tl.arange(0, KB); b0 = tl.load(rpb + u, mask=m, other=0); nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    mb = m[:, None] & (kb[None, :] < nb[:, None])
    b = tl.sum(tl.load(wb + b0[:, None] + kb[None, :], mask=mb, other=0.0) * tl.load(r + tl.load(ixb + b0[:, None] + kb[None, :], mask=mb, other=0), mask=mb, other=0.0), axis=1)
    val = a * tl.maximum(b, 0.0); act = m & (val != 0.0)
    if tl.max(act.to(tl.int32), axis=0) > 0:
        d0 = tl.load(rpd + u, mask=m, other=0); nd = tl.load(rpd + u + 1, mask=m, other=0) - d0
        for k0 in range(0, ndmax, KD):
            kd = k0 + tl.arange(0, KD); md = act[:, None] & (kd[None, :] < nd[:, None])
            di = tl.load(ixd + d0[:, None] + kd[None, :], mask=md, other=0); dw = tl.load(wd + d0[:, None] + kd[None, :], mask=md, other=0.0)
            tl.atomic_add(yn + di, dw * val[:, None], mask=md)

def pw(x): return max(2, 1 << (int(x) - 1).bit_length())
def prep(rec, dev):
    out = {}
    for k in ("gate", "up", "down"):
        rp, ix, w = rec[k]; out[k] = [torch.cat([rp.int()]).to(dev), torch.cat([ix.int(), torch.zeros(1, dtype=torch.int32)]).to(dev), torch.cat([w.float(), torch.zeros(1)]).to(dev)]
    n = lambda k: torch.diff(rec[k][0])
    out["KA"], out["KB"], out["ndmax"] = pw(n("up").max()), pw(n("gate").max()), int(n("down").max())
    out["KD"] = 8 if out["ndmax"] <= 8 else 256; out["BLOCK"] = 8 if out["ndmax"] > 64 else 32
    out["U"], out["H"], out["x0"] = rec["U"], rec["H"], rec["x0"].to(dev); out["name"] = rec["name"]; return out

def launch(L, y, flag, nw):
    _k[(triton.cdiv(L["U"], L["BLOCK"]),)](L["x0"], y, flag, L["H"], L["U"], *L["up"], *L["gate"], *L["down"], L["ndmax"], BLOCK=L["BLOCK"], KA=L["KA"], KB=L["KB"], KD=L["KD"], num_warps=nw)

def tgraph(fn, reps=100):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): g.replay()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1e3

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("layers"); ap.add_argument("--chip_ms", type=float, default=0.2002); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = "cuda"; recs = torch.load(a.layers, weights_only=False); Ls = {r["name"]: prep(r, dev) for r in recs}
    y = torch.zeros(2560, device=dev); on = torch.ones(1, dtype=torch.int32, device=dev); off = torch.zeros(1, dtype=torch.int32, device=dev)
    t = {}; res = dict(gpu=torch.cuda.get_device_name(), chip_step_ms=a.chip_ms, layers={})
    for nm, L in Ls.items():
        best = min((tgraph(lambda: launch(L, y, on, nw)), nw) for nw in (1, 2, 4, 8)); L["nw"] = best[1]
        e = tgraph(lambda: launch(L, y, off, L["nw"])); t[nm] = (best[0], e)
        res["layers"][nm] = dict(ms=best[0], ms_flag_off=e, num_warps=best[1], KA=L["KA"], KB=L["KB"], ndmax=L["ndmax"]); print(nm, res["layers"][nm], flush=True)
    acc = [n for n in Ls if n.startswith("ACC")]
    dn = 0.011   # DeltaNet activation-read layer: measured head kernel ~11 us (engine README)
    for k, banks in ((14, 29), (29, 29), (4, 29), (1, 29)):
        # layers: banks ROM_full + k x (SELECT, DN, PRODUCT, DIGITIZE, LOSUM) + accumulation (11 layers; 14-pipeline variant, reused for k=29 as a lower bound)
        on_ms = banks * t["ROM_full"][0] + k * (t["SELECT"][0] + dn + t["PRODUCT"][0] + t["DIGITIZE"][0] + t["LOSUM"][0]) + sum(t[n][0] for n in acc)
        off_ms = banks * t["ROM_full"][1] + k * (t["SELECT"][1] + dn_off(dn) + t["PRODUCT"][1] + t["DIGITIZE"][1] + t["LOSUM"][1]) + sum(t[n][1] for n in acc)
        nl = banks + 5 * k + len(acc)
        res[f"k{k}"] = dict(pipelines=k, banks=banks, appended_layers=nl, appended_ms_flag_on=on_ms, appended_ms_flag_off=off_ms, step_ms_ROM_DOT=a.chip_ms + on_ms, step_ms_other=a.chip_ms + off_ms,
                            tok_per_s_ROM_DOT=1e3 / (a.chip_ms + on_ms), tok_per_s_other=1e3 / (a.chip_ms + off_ms), step_ms_other_with_no_skip=a.chip_ms + on_ms)
        print(json.dumps(res[f"k{k}"]), flush=True)
    # real ROM bank memory: 29 banks x CSR
    nb = sum(sum(x.numel() * x.element_size() for k in ("gate", "up", "down") for x in Ls["ROM_full"][k]) for _ in [0])
    res["rom_bank_csr_mb_each"] = nb / 2**20; res["rom_banks_csr_gb_29"] = 29 * nb / 2**30
    if a.out: json.dump(res, open(a.out, "w"), indent=1)

def dn_off(dn): return 0.003    # an early-exit head kernel is bounded by the ~3 us kernel floor
main()
