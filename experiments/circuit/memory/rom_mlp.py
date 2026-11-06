"""De-risking experiment for option (d): a stock-shaped Qwen3.5 decoder MLP sublayer used as a ROM, in bf16 exactly as
HF runs it (RMSNorm in fp32 with the (1+w) gain, SwiGLU, down_proj).  The residual carries the address in dual rail
(n bits and their complements, so the norm statistic is constant) plus constant dims chosen so that mean(x^2)=1 exactly
(rms norm becomes the identity up to eps).  gate row j fires silu(+G/2) iff the address equals j and <= -G/2 otherwise;
down_proj column j holds table[j,:]/(G/2).  One position reads a whole H-wide word.  Reports how many bits per table
entry read back exactly after rounding at 0.8B (H=1024, I=3584) and 4B (H=2560, I=9216) geometry.
python rom_mlp.py   (use a GPU: the CPU bf16 matmul is slow)"""
import math, json, torch
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.bfloat16

def rmsnorm(x, w, eps=1e-6):                      # Qwen3_5RMSNorm: fp32 stats, out * (1 + w), cast back
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(x.dtype)

def const_dims(total, c0):
    """integers c_i with c_0 = c0 and sum c_i^2 = total (greedy squares)."""
    cs = [c0]; rem = total - c0 * c0; assert rem >= 0
    while rem > 0:
        c = int(math.isqrt(rem)); cs.append(c); rem -= c * c
    return cs

def run(H, I, bits, G=32.0, c0=16):
    n = math.ceil(math.log2(I)); cs = const_dims(H - n, c0)
    assert 2 * n + len(cs) <= H, (2 * n, len(cs))
    j = torch.arange(I, device=DEV)
    bj = ((j[:, None] >> torch.arange(n, device=DEV)[None]) & 1)               # (I, n)
    gate = torch.zeros(I, H, device=DEV); up = torch.zeros(I, H, device=DEV)
    ci = 2 * n                                                                  # const carrier index (value c0)
    gate[:, :n] = G * bj.float(); gate[:, n:2 * n] = G * (1 - bj).float()      # rail of bit b where address bit = 1 / 0
    gate[:, ci] = (-G * n + G / 2) / c0
    up[:, ci] = 1.0 / c0
    table = torch.randint(0, 2 ** bits, (I, H), device=DEV)
    down = (table.float().t() / (G / 2))
    nw = torch.zeros(H, device=DEV)                                             # (1+w) = 1 exactly
    x = torch.zeros(I, H, device=DEV)
    x[:, :n] = bj.float(); x[:, n:2 * n] = 1 - bj.float()
    for k, c in enumerate(cs): x[:, 2 * n + k] = c
    x, gate, up, down, nw = [t.to(DT) for t in (x, gate, up, down, nw)]
    h = rmsnorm(x, nw)
    z = h @ gate.t(); u = h @ up.t()
    a = torch.nn.functional.silu(z) * u
    out = a @ down.t()
    ok = (out.float().round() == table.float()); err = (out.float() - table.float()).abs()
    return dict(H=H, I=I, bits=bits, entries=I * H, exact_frac=ok.float().mean().item(),
                max_abs_err=err.max().item(), words_all_exact=ok.all(-1).float().mean().item())

if __name__ == "__main__":
    res = []
    for (H, I, name) in [(1024, 3584, "0.8B"), (2560, 9216, "4B")]:
        for bits in (1, 4, 6, 7, 8):
            r = run(H, I, bits); r["geom"] = name; res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open("rom_mlp_results.json", "w"), indent=1)
