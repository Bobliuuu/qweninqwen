"""Candidate (b): SwiGLU MLP of a STOCK Qwen3_5 outer layer as a streamed-weight multiplier.
Frame = [BALLAST | X (C) | W (R*C) | Y (R)].  Units pair (r,j,+/-):
   gate = +-eps*W[r,j]   up = +-X[j]   ->  silu(eps w)x + silu(-eps w)(-x) = eps*w*x   (odd part of silu: EXACT)
down_proj sums the 2C units of row r into Y[r] (scaled 1/eps).  One outer step = R*C MACs, host streams only W.
Variant 'inv': one unit per MAC: gate = silu^-1(w+s) (host reformat of weights only), correction -s*sum_j x_j from C extra units."""
import os
import sys, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'qweninqwen'))
from machine import Layout, make_outer
from safetensors import safe_open
import glob

def load_w(name, model='Qwen2.5-0.5B'):
    f = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--Qwen--{model}/snapshots/*/model.safetensors'))[0]
    with safe_open(f, 'pt') as s: return s.get_tensor(name).float()

class MulMLP:
    def __init__(self, C, R=1, B=1e6, eps=1.0, dtype=torch.float32, device='cpu', ltype='linear_attention'):
        self.C, self.R, self.B, self.eps = C, R, B, eps
        L = Layout(); L.alloc('X', C); L.alloc('W', R*C); L.alloc('Y', R)
        self.L = L; self.D = L.n; self.I = 2*R*C
        torch.set_grad_enabled(False)
        m = make_outer(L, [ltype], self.I, vocab=8, attn=dict(heads=1,kv=1,hd=64))   # layer0 attn zero => identity
        lay = m.model.layers[0]
        m.model.norm.weight.fill_(B/self.D**.5 - 1)
        lay.input_layernorm.weight.fill_(B/self.D**.5 - 1)
        lay.post_attention_layernorm.weight.fill_(B/self.D**.5 - 1)
        g, u, d = lay.mlp.gate_proj.weight, lay.mlp.up_proj.weight, lay.mlp.down_proj.weight
        X0, W0, Y0 = L['X'].start, L['W'].start, L['Y'].start
        with torch.no_grad():
            idx = torch.arange(R*C); r = idx // C; j = idx % C
            kp = 2*idx; kn = 2*idx + 1
            g[kp, W0 + idx] = eps;  g[kn, W0 + idx] = -eps
            u[kp, X0 + j] = 1.;     u[kn, X0 + j] = -1.
            d[Y0 + r, kp] = 1.0/eps; d[Y0 + r, kn] = 1.0/eps
        self.m = m.to(dtype).to(device).eval(); self.dtype, self.device = dtype, device
    def frames(self, W, x):
        """W: [T, R, C] tile weights per step; x: [C] or [T,C] (x validation-fed; in integration comes from attention copy)."""
        T = W.shape[0]; F = torch.zeros(T, self.D, dtype=self.dtype, device=self.device)
        F[:, 0] = self.B; F[:, self.L['X']] = x.expand(T, -1) if x.dim() == 1 else x
        F[:, self.L['W']] = W.reshape(T, -1); return F
    @torch.no_grad()
    def run(self, F):
        """stock forward over T positions (prefill, no cache needed: MLP is position-local). returns Y [T,R]"""
        o = self.m.model(inputs_embeds=F[None], use_cache=False).last_hidden_state[0]
        return o[:, self.L['Y']]
