"""Risk 2: drive the ROM address from a DeltaNet-state register instead of injecting it into the residual.
One stock-shaped Gated DeltaNet layer (pre-RMSNorm, in_proj_qkv/z/b/a, depthwise causal conv k=4 + silu, l2norm q/k,
q scale dk^-0.5, gated delta rule in fp32, gated RMSNorm with silu(z), out_proj; bf16 as HF runs it) holds the address
as one register row (dual-rail bits, 14 ones among 2n rails => constant norm).  Position t0 writes (one-hot key, beta=1,
no decay); position t1 reads (one-hot query) and the layer's output lands in the residual as the 0/1 rails that the ROM
sublayer of rom_mlp.py consumes.  Chain checked: write -> read -> ROM word exact.  python dn_reg.py"""
import math, json, torch
from mac_layer import SC, DEV, DT, Res, fwd, mk
torch.manual_seed(2)

def dn_layer(X, Wqkv, Wz, Wb, Wa, Wo, normw, dk=128, dv=128, a_bias=-40.0):
    """X (P,T,H) residual in bf16.  Returns X + out.  Conv taps are [0,0,0,1] (current token only)."""
    P, T, H = X.shape
    xf = X.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(DT)
    qkv = h @ Wqkv.t(); z = h @ Wz.t(); b = h @ Wb.t(); a = h @ Wa.t()
    qkv = torch.nn.functional.silu(qkv)                          # conv (identity taps) then silu, bf16
    q, k, v = qkv[..., :dk].float(), qkv[..., dk:2 * dk].float(), qkv[..., 2 * dk:].float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * dk ** -0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b.float()); g = -torch.nn.functional.softplus(a.float() + a_bias)   # decay exp(g) ~ 1
    S = torch.zeros(P, dk, dv, device=X.device); outs = []
    for t in range(T):
        S = S * g[:, t, 0].exp()[:, None, None]
        kv = (S * k[:, t, :, None]).sum(1)
        d = (v[:, t] - kv) * beta[:, t]
        S = S + k[:, t, :, None] * d[:, None, :]
        outs.append((S * q[:, t, :, None]).sum(1))
    o = torch.stack(outs, 1).to(DT)                              # (P,T,dv)
    of = o.float(); hs = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-6)).to(DT)
    hs = normw.to(DT) * hs
    hs = (hs.float() * torch.nn.functional.silu(z.float())).to(DT)
    return X + hs @ Wo.t()

def run(H=2560, I=9216, P=64, K=32, nrows=64):
    n = math.ceil(math.log2(I)); dk = dv = 128
    R = Res(H); ab = R.new('addr', 2 * n); car = R.new('car'); R.new('cst', 10)
    fw = R.new('flag_w'); fr = R.new('flag_r'); rowc = R.new('rowc', nrows); W = R.new('W', K)
    V = 16.0
    # v layout: 2n rails (value V) + filler so that sum v^2 = dv * V^2 * 4 ... pick sumsq = 128*V^2
    rails = 2 * n; fill = []; rem = int(dv * V * V - n * V * V)
    while rem > 0 and len(fill) < dv - rails:
        f = min(int(math.isqrt(rem)), 255); fill.append(f); rem -= f * f
    assert rem == 0, rem
    # projections (act on x' = x/SC for variable dims, carrier/flags have x' = 1)
    Wqkv = torch.zeros(3 * dk, H); Wz = torch.zeros(dv, H); Wb = torch.zeros(1, H); Wa = torch.zeros(1, H)
    L = 16.0
    for r in range(nrows):
        Wqkv[r, rowc + r] = L                                    # q = L * e_r when row-select dim r is set (read)
        Wqkv[dk + r, rowc + r] = L                               # k = L * e_r
    for i in range(rails): Wqkv[2 * dk + i, ab + i] = V * SC     # v rails from the residual's address rails (0/1)
    for i, f in enumerate(fill): Wqkv[2 * dk + rails + i, car] = float(f)
    Wz[:, car] = 16.0
    Wb[0, fw] = 64.0; Wb[0, car] = -32.0                         # beta ~1 on write positions, ~0 otherwise
    Wa[0, car] = 0.0
    normw = torch.full((dv,), 1.0 / 16)
    Wo = torch.zeros(H, dv)
    for i in range(rails): Wo[ab + i, i] = 1.0
    # NOTE: q and k share the row-select dims; read vs write is set by flag_w/flag_r gating the row dims
    rows = torch.randint(0, nrows, (P,)); addr = torch.randint(0, I, (P,)); table = torch.randint(0, 16, (I, K))
    X = torch.zeros(P, 2, H)
    X[:, :, car] = SC; X[:, :, R.d['cst']:R.d['cst'] + 10] = 2.0 ** 18
    for b in range(n):
        X[:, 0, ab + b] = (addr >> b) & 1; X[:, 0, ab + n + b] = 1 - ((addr >> b) & 1)   # only at t0 (the write)
    X[:, 0, fw] = SC; X[:, 1, fr] = SC
    X[torch.arange(P), :, rowc + rows[:, None].expand(P, 2).t()[0]] = SC if False else 0
    for t in range(2): X[torch.arange(P), t, rowc + rows] = SC
    X = X.to(DEV, DT)
    Wqkv, Wz, Wb, Wa, Wo, normw = [t.to(DEV, DT) for t in (Wqkv, Wz, Wb, Wa, Wo, normw)]
    # the write-vs-read split: at t1 beta~0 (flag_w=0 -> b=-32), k nonzero but beta tiny; at t0 q read sees the fresh write.
    Y = dn_layer(X, Wqkv, Wz, Wb, Wa, Wo, normw)
    rec = Y[:, 1, ab:ab + 2 * n].float()
    want = torch.zeros(P, 2 * n)
    for b in range(n): want[:, b] = ((addr >> b) & 1); want[:, n + b] = 1 - ((addr >> b) & 1)
    reg_ok = bool((rec.cpu() == want).all()); maxerr = (rec.cpu() - want).abs().max().item()
    # ROM sublayer on the register-read address (position 1 only), table of K weights, same construction as rom_mlp
    G = 32.0; j = torch.arange(I); bj = ((j[:, None] >> torch.arange(n)[None]) & 1).float()
    gate = torch.zeros(I, H); up = torch.zeros(I, H); down = torch.zeros(H, I)
    gate[:, ab:ab + n] = G * bj * SC; gate[:, ab + n:ab + 2 * n] = G * (1 - bj) * SC; gate[:, car] = -G * n + G / 2
    up[:, car] = 1.0; down[W:W + K, :] = table.float().t() / (G / 2)
    Z = fwd(Y[:, 1], gate.to(DEV, DT), up.to(DEV, DT), down.to(DEV, DT))[0].float().cpu()
    rom_ok = bool((Z[:, W:W + K].round() == table[addr].float()).all())
    return dict(H=H, I=I, positions=P, register_roundtrip_exact=reg_ok, max_rail_err=maxerr,
                rom_word_exact_from_register=rom_ok, extra_layers_before_rom=1)

if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
