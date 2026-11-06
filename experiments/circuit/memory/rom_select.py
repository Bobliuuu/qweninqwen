"""Full-width ROM word + slot select.  A ROM layer (I words) can only be read through I address units, so to use its
whole H-wide width one read must return many pipelines' worth of weights; the residual then cannot also hold activation
planes.  This test: ROM stage returns a (Wfull=2300)-entry word of 4-bit weights; a SELECT layer picks one of 10 slots of
230 weights using 4 slot bits (dual rail) and clears the rest, in place.  Exact check vs table.   python rom_select.py"""
import math, json, torch
from mac_layer import SC, DEV, DT, Res, fwd, mk

def run(H=2560, I=9216, K=230, S=10, P=64):
    n = math.ceil(math.log2(I)); ns = math.ceil(math.log2(S)); Wf = K * S
    R = Res(H); ab = R.new('addr', 2 * n); sb = R.new('slot', 2 * ns); car = R.new('car'); R.new('cst', H // 256)
    W = R.new('W', Wf)
    addr = torch.randint(0, I, (P,)); slot = torch.randint(0, S, (P,)); table = torch.randint(0, 16, (I, Wf))
    X = torch.zeros(P, H)
    for b in range(n): X[:, ab + b] = (addr >> b) & 1; X[:, ab + n + b] = 1 - ((addr >> b) & 1)
    for b in range(ns): X[:, sb + b] = (slot >> b) & 1; X[:, sb + ns + b] = 1 - ((slot >> b) & 1)
    X[:, car] = SC; X[:, R.d['cst']:R.d['cst'] + H // 256] = 2.0 ** 18
    X = X.to(DEV, DT); Gt = 32.0
    # ROM (vectorised build)
    j = torch.arange(I); bj = ((j[:, None] >> torch.arange(n)[None]) & 1).float()
    gate = torch.zeros(I, H); up = torch.zeros(I, H); down = torch.zeros(H, I)
    gate[:, ab:ab + n] = Gt * bj * SC; gate[:, ab + n:ab + 2 * n] = Gt * (1 - bj) * SC; gate[:, car] = -Gt * n + Gt / 2
    up[:, car] = 1.0; down[W:W + Wf, :] = table.float().t() / (Gt / 2)
    X, _ = fwd(X, gate.to(DEV, DT), up.to(DEV, DT), down.to(DEV, DT))
    # SELECT: unit (s,k): gate = AND of slot match, up = entry; plus a clear unit per dim
    rows = []
    for s in range(S):
        g = {}
        for b in range(ns):
            g[sb + b if (s >> b) & 1 else sb + ns + b] = Gt * SC
        g[car] = -Gt * ns + Gt / 2
        for k in range(K):
            rows.append((dict(g), {W + s * K + k: SC}, {W + k: 1.0 / (Gt / 2)}))
    for d in range(Wf): rows.append(({car: 16.0}, {W + d: SC}, {W + d: -1.0 / 16}))
    X, _ = fwd(X, *mk(I, H, rows))
    Xf = X.float().cpu()
    want = torch.stack([table[addr[i], slot[i] * K:(slot[i] + 1) * K] for i in range(P)]).float()
    ok = bool((Xf[:, W:W + K].round() == want).all()); cleared = bool((Xf[:, W + K:W + Wf].abs() < 0.01).all())
    return dict(H=H, I=I, K=K, slots=S, word_entries=Wf, select_units=len(rows), select_exact=ok, rest_cleared=cleared,
                max_err=(Xf[:, W:W + K] - want).abs().max().item(), dims_used=R.n)
if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
