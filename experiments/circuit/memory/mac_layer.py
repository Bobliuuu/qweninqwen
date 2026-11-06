"""ROM read + exact int4 x int8 multiply-accumulate on a ROM word, built from stock-shaped SwiGLU sublayers and run in
bf16 exactly as HF runs them (RMSNorm fp32 with (1+w)=1, silu in bf16, act*up rounded to bf16, linears bf16 out).
Stages: ROM (address rails -> K int4 weights w'), PRODUCT ((8+w')*a_bit per activation bit-plane, group sums <=115),
DIGITIZE (split each <=115 group sum into hi/lo nibble with 7 thresholds, sum hi per chunk), LOSUM (sum lo digits).
Result is a set of small digits (<=127) whose weighted sum is  sum_k (8+w'_k) a'_k  (w' in 0..15, a' in 0..255);
verified equal to the integer reference on every test position.  Every residual scalar must stay <=255 (bf16 mantissa),
which is why the reduction needs these stages.   python mac_layer.py [K]"""
import math, sys, json, torch
torch.manual_seed(1)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.bfloat16
SC = 2.0 ** 14                    # rms of the residual is pinned to SC by ten constant dims of 2^18 => x' = x / SC

class Res:                         # residual dim allocator
    def __init__(s, H): s.H, s.n, s.d = H, 0, {}
    def new(s, name, cnt=1):
        i = s.n; s.n += cnt; assert s.n <= s.H, (name, s.n, s.H); s.d[name] = i; return i

def fwd(X, gate, up, down):
    xf = X.float(); y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    h = y.to(DT)                                                  # (1+w) = 1
    z = h @ gate.t(); u = h @ up.t()
    a = torch.nn.functional.silu(z) * u
    return X + a @ down.t(), gate.shape[0]

def mk(I, H, rows):               # rows: list of (gate{dim:coef}, up{dim:coef}, down{dim:coef}) ; coefs are on x' (caller scales)
    g = torch.zeros(I, H); u = torch.zeros(I, H); d = torch.zeros(H, I)
    for j, (gg, uu, dd) in enumerate(rows):
        for k, v in gg.items(): g[j, k] = v
        for k, v in uu.items(): u[j, k] = v
        for k, v in dd.items(): d[k, j] = v
    return g.to(DEV, DT), u.to(DEV, DT), d.to(DEV, DT)

def run(K, H=2560, I=9216, P=64, m=5):
    R = Res(H); n = math.ceil(math.log2(I))
    ab = R.new('addr', 2 * n); car = R.new('car'); R.new('cst', H // 256)
    W = R.new('W', K); A = R.new('A', 8 * K)
    G = math.ceil(K / m); C = R.new('C', G * 8)
    chH = 18; nH = math.ceil(G / chH); HS = R.new('HS', 8 * nH)
    chL = 8; nL = math.ceil(G / chL); LS = R.new('LS', 8 * nL)
    # --- inputs
    addr = torch.randint(0, I, (P,)); table = torch.randint(0, 16, (I, K))
    a_int = torch.randint(0, 256, (P, K))
    X = torch.zeros(P, H)
    for b in range(n):
        X[:, ab + b] = (addr >> b) & 1; X[:, ab + n + b] = 1 - ((addr >> b) & 1)
    X[:, car] = SC; X[:, R.d['cst']:R.d['cst'] + H // 256] = 2.0 ** 18
    for p in range(8): X[:, A + p * K:A + (p + 1) * K] = ((a_int >> p) & 1).float()
    X = X.to(DEV, DT); stats = []
    # --- stage 0: ROM  (gate z = G*(match) ; match => +16, else <= -16)
    Gt = 32.0; rows = []
    for j in range(I):
        gg = {}
        for b in range(n): gg[ab + b if (j >> b) & 1 else ab + n + b] = Gt * SC
        gg[car] = -Gt * n + Gt / 2
        rows.append((gg, {car: 1.0}, {W + k: float(table[j, k]) / (Gt / 2) for k in range(K)}))
    X, nu = fwd(X, *mk(I, H, rows)); stats.append(('ROM', nu))
    # --- stage 1: products  (8 + w') * a_bit
    S = 8.0; rows = []
    for k in range(K):
        for p in range(8):
            rows.append(({W + k: SC, car: S}, {A + p * K + k: SC}, {C + (k // m) * 8 + p: 1.0}))
    X, nu = fwd(X, *mk(I, H, rows)); stats.append(('PRODUCT', len(rows)))
    # --- stage 2: digitize c (<=115) into hi=floor(c/16) (thresholds 16..112) and lo = c-16hi in place; hi summed per chunk
    Gd = 16.0; rows = []
    for g in range(G):
        for p in range(8):
            ci = C + g * 8 + p; hs = HS + p * nH + g // chH
            for t in range(16, 128, 16):
                for sgn, off in ((1, 1.0), (-1, 0.0)):
                    rows.append(({ci: Gd * SC, car: Gd * (off - t)}, {car: 1.0},
                                 {hs: sgn / Gd, ci: -sgn * 1.0}))
    assert len(rows) <= I, (len(rows), I)
    X, nu = fwd(X, *mk(I, H, rows)); stats.append(('DIGITIZE', len(rows)))
    # --- stage 3: sum lo digits per chunk (identity units)
    rows = []
    for g in range(G):
        for p in range(8):
            rows.append(({car: 16.0}, {C + g * 8 + p: SC}, {LS + p * nL + g // chL: 1.0 / 16}))
    X, nu = fwd(X, *mk(I, H, rows)); stats.append(('LOSUM', len(rows)))
    # --- check against the integer reference
    Xf = X.float().cpu()
    wp = table[addr]                                              # (P,K)
    ref = ((8 + wp) * a_int).sum(1)
    hs = Xf[:, HS:HS + 8 * nH].round().long().view(P, 8, nH).sum(2)
    ls = Xf[:, LS:LS + 8 * nL].round().long().view(P, 8, nL).sum(2)
    got = sum((1 << p) * (16 * hs[:, p] + ls[:, p]) for p in range(8))
    # exactness of the raw digit dims (not only the recomposed sum)
    hraw = Xf[:, HS:HS + 8 * nH]; lraw = Xf[:, LS:LS + 8 * nL]
    maxfrac = max((hraw - hraw.round()).abs().max().item(), (lraw - lraw.round()).abs().max().item())
    w_ok = bool((Xf[:, W:W + K].round() == wp.float()).all())
    return dict(K=K, H=H, I=I, positions=P, dims_used=R.n, stage_units=stats, layers=len(stats),
                rom_exact=w_ok, dot_exact=bool((got == ref).all()), max_digit_frac_err=maxfrac,
                max_digit=max(hraw.max().item(), lraw.max().item()), dot_range=int(ref.max()))

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    print(json.dumps(run(K)), flush=True)
