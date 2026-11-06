"""Full matvec y = W x on STOCK Qwen3_5 layers with only weight tiles streamed from the host (K tiled into nt tiles of Ct columns):
 L0 full-attention layer: round A stores nt x-tile entries (key = code(a), valid +); round B each weight step (row-group t, tile a) selects x-tile a by hard
      attention (code match, weight-step keys valid -) -> copies x_a into its residual, then the SwiGLU multiplier of the same layer computes the R partial dots
      (odd-silu trick, exact).
 L1 full-attention layer: stores each step's R partials as v with key = code(row group t) (same code for all nt tiles); round C: one query step, T heads,
      head t attends EQUALLY to its nt entries -> mean of partials; o_proj gain nt => exact K-tile reduction + gather into X2[t*R:(t+1)*R].
Host sends only BALLAST, flags, address codes (+-1) and raw weight rows."""
import sys, math, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'qweninqwen'))
from machine import Layout, make_outer
from transformers import DynamicCache

def r4(n): return (n + 3)//4*4
def nbits(n): return max(1, math.ceil(math.log2(max(2, n))))

class Matvec:
    def __init__(self, Ct, O, R, nt=1, B=1e6, device='cpu', dtype=torch.float32):
        assert O % R == 0
        self.C, self.O, self.R, self.nt, self.B, self.dev, self.dtype = Ct, O, R, nt, B, device, dtype
        C = Ct; self.K = Ct*nt
        T = self.T = O//R; nb = self.nb = nbits(T); na = self.na = nbits(nt)
        L = Layout(); self.L = L
        L.alloc('X', C); L.alloc('W', R*C); L.alloc('Y', R); L.alloc('XS', 1); L.alloc('WS', 1)
        L.alloc('XADDR', na); L.alloc('QADDR', na); L.alloc('ADDR', nb); L.alloc('X2', O)
        self.D = L.n; hd = self.hd = r4(max(C + 1, R, 4 + max(na, nb) + 3) + 4)
        torch.set_grad_enabled(False)
        m = make_outer(L, ['full_attention', 'full_attention'], 2*R*C, vocab=8, attn=dict(heads=T, kv=1, hd=hd))
        D = self.D; nrm = B/D**.5 - 1
        m.model.norm.weight.fill_(nrm)
        for lay in m.model.layers:
            lay.input_layernorm.weight.fill_(nrm); lay.post_attention_layernorm.weight.fill_(nrm)
        ix = lambda n: L.slots[n].start
        g = 30.0
        c0 = hd//4 + 1
        def set_gain(a, nbit):
            nact = nbit + 1; sc = (hd/nact)**.5
            need = (40*hd**.5/(2*sc*sc))**.5          # one-bit-mismatch score gap >= 40
            for nm in (a.q_norm, a.k_norm): nm.weight[c0:c0+nbit+1] = need - 1
        # ---- layer 0
        a0 = m.model.layers[0].self_attn; vd = c0 + na
        for h in range(T):
            base = h*2*hd
            for b in range(na): a0.q_proj.weight[base + c0 + b, ix('QADDR') + b] = 1.0
            a0.q_proj.weight[base + vd, 0] = 1.0/B
            a0.q_proj.weight[base + hd + torch.arange(hd), 0] = g/B
        for b in range(na): a0.k_proj.weight[c0 + b, ix('XADDR') + b] = 1.0
        a0.k_proj.weight[vd, ix('XS')] = 2.0; a0.k_proj.weight[vd, 0] = -1.0/B
        a0.v_proj.weight[torch.arange(C), ix('X') + torch.arange(C)] = 1.0
        set_gain(a0, na)
        a0.o_proj.weight[ix('X') + torch.arange(C), torch.arange(C)] = 1.0
        mlp = m.model.layers[0].mlp; Gp, Up, Dn = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        idx = torch.arange(R*C); r = idx//C; j = idx % C
        Gp[2*idx, ix('W') + idx] = 1.; Gp[2*idx+1, ix('W') + idx] = -1.
        Up[2*idx, ix('X') + j] = 1.;   Up[2*idx+1, ix('X') + j] = -1.
        Dn[ix('Y') + r, 2*idx] = 1.;   Dn[ix('Y') + r, 2*idx+1] = 1.
        # ---- layer 1: gather + K-tile reduce
        a1 = m.model.layers[1].self_attn; vd = c0 + nb
        for t in range(T):
            base = t*2*hd
            for b in range(nb): a1.q_proj.weight[base + c0 + b, 0] = (1.0 if (t >> b) & 1 else -1.0)/B
            a1.q_proj.weight[base + vd, 0] = 1.0/B
            a1.q_proj.weight[base + hd + torch.arange(hd), 0] = g/B
        for b in range(nb): a1.k_proj.weight[c0 + b, ix('ADDR') + b] = 1.0
        a1.k_proj.weight[vd, ix('WS')] = 2.0; a1.k_proj.weight[vd, 0] = -1.0/B
        a1.v_proj.weight[torch.arange(R), ix('Y') + torch.arange(R)] = 1.0
        set_gain(a1, nb)
        tt = torch.arange(T)[:, None]; rr = torch.arange(R)[None, :]
        a1.o_proj.weight[ix('X2') + (tt*R + rr).reshape(-1), (tt*hd + rr).reshape(-1)] = float(nt)
        self.m = m.to(device).to(dtype).eval()

    def code(self, v, nbit):
        return torch.stack([((v >> b) & 1) for b in range(nbit)], -1).to(self.dtype)*2 - 1

    def frames_B(self, Wm):
        """Wm [O,K] raw weights.  step s = a*T + t : rows t*R.., tile a"""
        S = self.L.slots; T, R, C, nt = self.T, self.R, self.C, self.nt
        n = T*nt
        FB = torch.zeros(n, self.D, device=self.dev, dtype=self.dtype)
        FB[:, 0] = self.B; FB[:, S['WS']] = 1
        Wt = Wm.reshape(T, R, nt, C).permute(2, 0, 1, 3).reshape(n, R*C)
        FB[:, S['W']] = Wt.to(self.dev)
        s = torch.arange(n, device=self.dev); t = s % T; a = s // T
        FB[:, S['ADDR']] = self.code(t, self.nb); FB[:, S['QADDR']] = self.code(a, self.na)
        return FB

    def run(self, Wm, x, FB=None):
        S = self.L.slots; C, nt = self.C, self.nt
        cache = DynamicCache()
        def fwd(F):
            pos = cache.get_seq_length() + torch.arange(F.shape[0], device=F.device)
            return self.m(inputs_embeds=F[None], past_key_values=cache, use_cache=True, position_ids=pos[None], output_hidden_states=True)
        FA = torch.zeros(nt, self.D, device=self.dev, dtype=self.dtype); FA[:, 0] = self.B
        FA[:, S['X']] = x.reshape(nt, C).to(self.dev); FA[:, S['XS'].start] = 1
        FA[:, S['XADDR']] = self.code(torch.arange(nt, device=self.dev), self.na)
        fwd(FA)
        if FB is None: FB = self.frames_B(Wm)
        outB = fwd(FB)
        FC = torch.zeros(1, self.D, device=self.dev, dtype=self.dtype); FC[0, 0] = self.B
        outC = fwd(FC)
        return outB.hidden_states[-1][0][:, S['Y']], outC.hidden_states[-1][0, 0][S['X2']]
