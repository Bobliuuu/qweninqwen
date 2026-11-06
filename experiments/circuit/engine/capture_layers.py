"""Capture the exact sparse structure of the corrected ROM/SELECT/MAC/accumulation stages (memory/rom_select.py, mac_layer.py, acc_stage.py) by wrapping
mac_layer.fwd, and test the flag-gated skip: add a rail `notflag` with gate coefficient -P to every unit; with notflag=1 the sublayer must leave the residual
bit-identical (silu(z-P) rounds to exactly 0 in bf16), with notflag=0 it must be unchanged. Run from <qwenputer>/experimental/riscv with memory/ on the path.
usage: python capture_layers.py --out layers.pt"""
import sys, json, math, argparse, torch
sys.path.insert(0, "memory")
import mac_layer as ML, rom_select as RS, acc_stage as AC
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="layers.pt"); ap.add_argument("--json", default="skip_exactness.json"); a = ap.parse_args()
rec = []; orig = ML.fwd
def csr(m):   # m: [rows, H] dense -> CSR over rows
    nz = m.float().nonzero(as_tuple=False); rp = torch.zeros(m.shape[0] + 1, dtype=torch.int64, device=m.device)
    rp[1:] = torch.cumsum(torch.bincount(nz[:, 0], minlength=m.shape[0]), 0)
    return rp.int().cpu(), nz[:, 1].int().cpu(), m.float()[nz[:, 0], nz[:, 1]].cpu()
def fwd(X, gate, up, down):
    out, nu = orig(X, gate, up, down)
    # skip test: notflag rail in a spare residual dim (last dim), gate coefficient -P*SC
    H = X.shape[1]; xf = X.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(ML.DT)
    zmax = float((h @ gate.t()).float().abs().max()); P = 2.0 ** math.ceil(math.log2(2 * zmax + 128))
    used = (X.abs().amax(0) != 0) | (gate.abs().amax(0) != 0) | (up.abs().amax(0) != 0) | (down.abs().amax(1) != 0)
    free = (~used).nonzero().flatten()
    if len(free) == 0:
        rec.append(dict(gate=csr(gate), up=csr(up), down=csr(down.t().contiguous()), U=gate.shape[0], H=H, zmax=zmax, P=P, skip_exact=None, flag_off_unchanged=None)); return out, nu
    spare = int(free[-1])
    g2 = gate.clone(); g2[:, spare] = (-P * ML.SC) if False else (-P)       # coefficient on the (pre-scaled) rail value below
    X1 = X.clone(); X1[:, spare] = ML.SC                                       # notflag=1 rail carries SC (so x'=1)
    o1, _ = orig(X1, g2, up, down); skip_ok = bool(torch.equal(o1, X1))
    X0 = X.clone(); o0, _ = orig(X0, g2, up, down); noskip_ok = bool(torch.equal(o0, out))
    rec.append(dict(x0=X[0].float().cpu(), gate=csr(gate), up=csr(up), down=csr(down.t().contiguous()), U=gate.shape[0], H=H, zmax=zmax, P=P, skip_exact=skip_ok, flag_off_unchanged=noskip_ok))
    return out, nu
ML.fwd = RS.fwd = AC.fwd = fwd
res = {}
res["rom_select"] = RS.run(); n_rs = len(rec)
res["mac_layer"] = ML.run(230); n_ml = len(rec)
res["acc_stage"] = AC.run(Pg=14); n_ac = len(rec)
names = ["ROM_full", "SELECT"] + ["ROM_230", "PRODUCT", "DIGITIZE", "LOSUM"] + [f"ACC{i}" for i in range(n_ac - n_ml)]
for r, nme in zip(rec, names): r["name"] = nme
print(json.dumps([dict(name=r["name"], U=r["U"], nnz_gate=int(r["gate"][0][-1]), nnz_up=int(r["up"][0][-1]), nnz_down=int(r["down"][0][-1]), zmax=r["zmax"], P=r["P"], skip_exact=r["skip_exact"], flag_off_unchanged=r["flag_off_unchanged"]) for r in rec]), flush=True)
json.dump([{k: v for k, v in r.items() if k in ("name", "U", "zmax", "P", "skip_exact", "flag_off_unchanged")} for r in rec], open(a.json, "w"), indent=1)
torch.save(rec, a.out)
