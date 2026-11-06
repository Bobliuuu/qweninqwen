"""recursiveqwen / merge2: N inner Qwen3 models hosted side by side in ONE stock Qwen3.5-0.8B checkpoint.

Changes vs the single-model builder (../hosted-nesting/qweninqwen/merge.py):
  * all N inner models share the SAME appended full-attention layers (layer i of every inner model is
    hosted in appended layer 24+i, in disjoint hidden subspaces / heads / MLP neurons / head-dim slots)
  * RoPE: rope_type='longrope' gives an arbitrary per-pair frequency table (short_factor==long_factor,
    rope_theta=1, attention_factor=1) so every model gets its own exact frequencies in its own head-dim
    slots. No shared integer grid needed -> head_dim = 2*(32+sum hd_m/2)+192 instead of 640 in the single-model builder,
    and no grid-growth limit on which inner models can be hosted.
  * one flag dim per inner model plus one for the outer model: gate saturation (own flag) and
    lm_head masking of every foreign vocab range (-LAM*flag), so greedy argmax is exact even when the
    hosted model's top logit is negative.
  * q_norm/k_norm weights are per-slot, so models with different qk-norm weights coexist.
Vocab layout: [0,248320) outer, then one range of V_m rows per inner model.
"""
import json, math, os, sys, torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.initialization import no_init_weights

HO, NO, OD, OP = 1024, 248320, 256, 64      # outer hidden, vocab, head dim, rotated dims
NQ_O, NKV_O_ORIG = 8, 2
NQ_O_LAYERS = 24
LAM = 2.0 ** 32
FLAGV = 2.0 ** -14


def _snap(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])


OUTER_DIR = _snap('Qwen/Qwen3.5-0.8B')


def outer_text_config():
    return json.load(open(OUTER_DIR + '/config.json'))['text_config']


def load_outer_state():
    sd = {}
    with safe_open(OUTER_DIR + '/model.safetensors-00001-of-00001.safetensors', 'pt') as f:
        for k in f.keys():
            if k.startswith('model.language_model.'):
                sd[k[len('model.language_model.'):]] = f.get_tensor(k)
    return sd


def reference_outer(dtype):
    tc = dict(outer_text_config()); tc['mtp_num_hidden_layers'] = 0
    m = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc)).to(dtype)
    m.model.load_state_dict({k: v.to(dtype) for k, v in load_outer_state().items()}, strict=True)
    m.lm_head.weight = m.model.embed_tokens.weight
    return m.eval()


def load_inner(repo, dtype, nlayers=None):
    from transformers import AutoModelForCausalLM, AutoConfig
    p = _snap(repo)
    cfg = AutoConfig.from_pretrained(p)
    if nlayers: cfg.num_hidden_layers = nlayers
    return AutoModelForCausalLM.from_pretrained(p, config=cfg, dtype=dtype).eval()


class Layout:
    """All index bookkeeping for the merged model."""
    def __init__(self, inners, n_app=None):
        self.inners = inners
        self.N = len(inners)
        self.cfgs = [m.config for m in inners]
        self.hi = [c.hidden_size for c in self.cfgs]
        self.hd = [getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads for c in self.cfgs]
        self.nq = [c.num_attention_heads for c in self.cfgs]
        self.nkv = [c.num_key_value_heads for c in self.cfgs]
        self.inter = [c.intermediate_size for c in self.cfgs]
        self.V = [c.vocab_size for c in self.cfgs]
        self.L = [len(m.model.layers) for m in inners]
        self.theta = [float((getattr(c, 'rope_parameters', None) or {}).get('rope_theta') or getattr(c, 'rope_theta', None)) for c in self.cfgs]
        assert all(not getattr(c, 'attention_bias', False) for c in self.cfgs), 'biases not supported'
        self.n_app = n_app or max(self.L)
        # hidden: outer | inner slices | flags (outer, inner...) | pad
        self.off = []; o = HO
        for h in self.hi: self.off.append(o); o += h
        self.F_O = o; self.F = [o + 1 + i for i in range(self.N)]
        self.H = (o + 1 + self.N + 31) // 32 * 32
        # vocab
        self.voff = []; v = NO
        for x in self.V: self.voff.append(v); v += x
        self.vocab = v
        # heads (global q:kv ratio 2)
        self.NQ = NQ_O + sum(self.nq); self.NKV = self.NQ // 2
        self.qo = []; q = NQ_O
        for n in self.nq: self.qo.append(q); q += n
        self.kvo = [q_ // 2 for q_ in self.qo]
        # head-dim slots
        self.P = 32 + sum(h // 2 for h in self.hd)
        self.R = 2 * self.P
        self.HD = self.R + (OD - OP)
        self.pbase = []; b = 32
        for h in self.hd: self.pbase.append(b); b += h // 2
        # mlp neurons
        self.noff = []; n = 0
        for x in self.inter: self.noff.append(n); n += x
        self.I = max(outer_text_config()['intermediate_size'], n)
        self.slot_o = torch.tensor([self.so(d) for d in range(OD)])
        self.slot_m = [torch.tensor([self.sm(m, d) for d in range(self.hd[m])]) for m in range(self.N)]

    def so(self, d):
        return d if d < 32 else self.P + d - 32 if d < 64 else self.R + d - 64

    def sm(self, m, d):
        h2 = self.hd[m] // 2
        return self.pbase[m] + d if d < h2 else self.P + self.pbase[m] + d - h2

    def inv_freq(self):
        f = [0.0] * self.P
        for j in range(32):
            f[j] = 1e7 ** (-(2 * j) / 64)
        for m in range(self.N):
            h2 = self.hd[m] // 2
            for i in range(h2):
                f[self.pbase[m] + i] = self.theta[m] ** (-(2 * i) / self.hd[m])
        return f

    def config(self, dtype):
        tc = dict(outer_text_config()); n_out = tc['num_hidden_layers']
        tc.update(hidden_size=self.H, num_hidden_layers=n_out + self.n_app, num_attention_heads=self.NQ,
                  num_key_value_heads=self.NKV, head_dim=self.HD, intermediate_size=self.I,
                  vocab_size=self.vocab, tie_word_embeddings=False,
                  layer_types=tc['layer_types'] + ['full_attention'] * self.n_app, dtype=dtype,
                  mtp_num_hidden_layers=0, bos_token_id=None, pad_token_id=None)
        fac = [1.0 / x for x in self.inv_freq()]
        tc['rope_parameters'] = dict(rope_type='longrope', rope_theta=1.0, partial_rotary_factor=self.R / self.HD,
                                     short_factor=fac, long_factor=fac, factor=1.0, attention_factor=1.0,
                                     original_max_position_embeddings=tc['max_position_embeddings'],
                                     mrope_section=[self.P // 3 + 1] * 3, mrope_interleaved=True)
        return Qwen3_5TextConfig(**tc)

    def nparams_useful(self):
        return sum(sum(p.numel() for p in m.parameters()) for m in self.inners)


def vec_norm(L, parts, flag_idx=(), dt=torch.float32):
    """stored (zero-centred) norm weight; parts = [(offset, size, eff_weight)]; slices scaled sqrt(size/H)."""
    v = torch.zeros(L.H, dtype=torch.float32)
    for off, size, w in parts:
        v[off:off + size] = w.float().cpu() * math.sqrt(size / L.H)
    for f in flag_idx: v[f] = 1.0
    return (v - 1).to(dt)


def build(inners, dtype=torch.bfloat16, device='cpu', n_app=None, verbose=True):
    inners = [i.to(device) for i in inners]
    L = Layout(inners, n_app)
    cfg = L.config(str(dtype).replace('torch.', ''))
    torch.set_default_dtype(dtype)
    with no_init_weights(), torch.device(device):
        m = Qwen3_5ForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        # non-persistent buffers (rope) were built from config; GDN etc have no others
    dv = torch.device(device); dt = dtype
    osd = {k: v.to(dv) for k, v in load_outer_state().items()}
    types = cfg.layer_types; HD, H, NQ, NKV = L.HD, L.H, L.NQ, L.NKV
    SO = L.slot_o.to(dv)
    with torch.no_grad():
        E = m.model.embed_tokens.weight; Lh = m.lm_head.weight
        emb = osd['embed_tokens.weight'].to(dt)
        E[:NO, :HO] = emb; E[:NO, L.F_O] = FLAGV
        Lh[:NO, :HO] = emb
        for f in L.F: Lh[:NO, f] = -LAM
        for k, im in enumerate(inners):
            s, e = L.voff[k], L.voff[k] + L.V[k]; o, h = L.off[k], L.hi[k]
            E[s:e, o:o + h] = im.model.embed_tokens.weight.to(dt).to(dv); E[s:e, L.F[k]] = FLAGV
            Lh[s:e, o:o + h] = im.lm_head.weight.to(dt).to(dv)
            Lh[s:e, L.F_O] = -LAM
            for k2 in range(L.N):
                if k2 != k: Lh[s:e, L.F[k2]] = -LAM
        parts = [(0, HO, 1 + osd['norm.weight'].float())] + [(L.off[k], L.hi[k], inners[k].model.norm.weight.float()) for k in range(L.N)]
        m.model.norm.weight.copy_(vec_norm(L, parts, [L.F_O] + L.F, dt))
        # ---------------- outer layers
        for l in range(NQ_O_LAYERS):
            ly = m.model.layers[l]; p = f'layers.{l}.'
            ly.input_layernorm.weight.copy_(vec_norm(L, [(0, HO, 1 + osd[p + 'input_layernorm.weight'].float())], (), dt))
            ly.post_attention_layernorm.weight.copy_(vec_norm(L, [(0, HO, 1 + osd[p + 'post_attention_layernorm.weight'].float())], (), dt))
            ly.mlp.gate_proj.weight[:3584, :HO] = osd[p + 'mlp.gate_proj.weight'].to(dt)
            ly.mlp.up_proj.weight[:3584, :HO] = osd[p + 'mlp.up_proj.weight'].to(dt)
            ly.mlp.down_proj.weight[:HO, :3584] = osd[p + 'mlp.down_proj.weight'].to(dt)
            if types[l] == 'linear_attention':
                la = ly.linear_attn
                for n in ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b'):
                    getattr(la, n).weight[:, :HO] = osd[p + f'linear_attn.{n}.weight'].to(dt)
                la.out_proj.weight[:HO, :] = osd[p + 'linear_attn.out_proj.weight'].to(dt)
                for n in ('conv1d.weight', 'dt_bias', 'A_log', 'norm.weight'):
                    a = la
                    for part in n.split('.')[:-1]: a = getattr(a, part)
                    getattr(a, n.split('.')[-1]).copy_(osd[p + 'linear_attn.' + n].to(dt))
            else:
                sa = ly.self_attn
                q = osd[p + 'self_attn.q_proj.weight'].to(dt).view(NQ_O, 2, OD, HO)
                Wq = sa.q_proj.weight.view(NQ, 2, HD, H)
                tmp = torch.zeros(NQ_O, HD, HO, dtype=dt, device=dv); tmp[:, SO] = q[:, 0]
                Wq[:NQ_O, 0, :, :HO] = tmp
                Wq[:NQ_O, 1, :OD, :HO] = q[:, 1]
                k = osd[p + 'self_attn.k_proj.weight'].to(dt).view(NKV_O_ORIG, OD, HO)
                v = osd[p + 'self_attn.v_proj.weight'].to(dt).view(NKV_O_ORIG, OD, HO)
                Wk = sa.k_proj.weight.view(NKV, HD, H); Wv = sa.v_proj.weight.view(NKV, HD, H)
                for mi in range(NQ_O // 2):
                    tk = torch.zeros(HD, HO, dtype=dt, device=dv); tk[SO] = k[mi // 2]; Wk[mi, :, :HO] = tk
                    Wv[mi, :OD, :HO] = v[mi // 2]
                sa.o_proj.weight.view(H, NQ, HD)[:HO, :NQ_O, :OD] = osd[p + 'self_attn.o_proj.weight'].to(dt).view(HO, NQ_O, OD)
                qn = torch.zeros(HD); kn = torch.zeros(HD)
                qw = 1 + osd[p + 'self_attn.q_norm.weight'].float().cpu(); kw = 1 + osd[p + 'self_attn.k_norm.weight'].float().cpu()
                qn[L.slot_o] = qw; kn[L.slot_o] = kw * math.sqrt(OD / HD)
                sa.q_norm.weight.copy_((qn - 1).to(dt)); sa.k_norm.weight.copy_((kn - 1).to(dt))
        # ---------------- inner layers: layer i of every model -> appended layer 24+i
        for i in range(L.n_app):
            ly = m.model.layers[NQ_O_LAYERS + i]
            ly.input_layernorm.weight.fill_(-1); ly.post_attention_layernorm.weight.fill_(-1)
        for k, im in enumerate(inners):
            o, h, hd, nq, nkv = L.off[k], L.hi[k], L.hd[k], L.nq[k], L.nkv[k]
            SM = L.slot_m[k].to(dv); rep = (nq // 2) // nkv; nkvn = nq // 2
            qo, kvo, no, it = L.qo[k], L.kvo[k], L.noff[k], L.inter[k]
            for i, il in enumerate(im.model.layers):
                ly = m.model.layers[NQ_O_LAYERS + i]
                # accumulate norms across models (disjoint slices): read-modify-write on stored weight
                for nm, src in (('input_layernorm', il.input_layernorm), ('post_attention_layernorm', il.post_attention_layernorm)):
                    w = getattr(ly, nm).weight
                    eff = w.float() + 1
                    eff[o:o + h] = src.weight.float().to(eff.device) * math.sqrt(h / H)
                    if nm == 'input_layernorm': eff[L.F[k]] = 1.0
                    w.copy_((eff - 1).to(dt))
                ly.mlp.gate_proj.weight[no:no + it, o:o + h] = il.mlp.gate_proj.weight.to(dt)
                ly.mlp.up_proj.weight[no:no + it, o:o + h] = il.mlp.up_proj.weight.to(dt)
                ly.mlp.down_proj.weight[o:o + h, no:no + it] = il.mlp.down_proj.weight.to(dt)
                a = il.self_attn; sa = ly.self_attn
                q = a.q_proj.weight.to(dt).view(nq, hd, h)
                Wq = sa.q_proj.weight.view(NQ, 2, HD, H)
                tmp = torch.zeros(nq, HD, h, dtype=dt, device=dv); tmp[:, SM] = q
                Wq[qo:qo + nq, 0, :, o:o + h] = tmp
                Wq[qo:qo + nq, 1, :, L.F[k]] = LAM
                kk = a.k_proj.weight.to(dt).view(nkv, hd, h); vv = a.v_proj.weight.to(dt).view(nkv, hd, h)
                Wk = sa.k_proj.weight.view(NKV, HD, H); Wv = sa.v_proj.weight.view(NKV, HD, H)
                tk = torch.zeros(nkv, HD, h, dtype=dt, device=dv); tk[:, SM] = kk
                for j in range(nkvn):
                    Wk[kvo + j, :, o:o + h] = tk[j // rep]
                    Wv[kvo + j, :hd, o:o + h] = vv[j // rep]
                sa.o_proj.weight.view(H, NQ, HD)[o:o + h, qo:qo + nq, :hd] = a.o_proj.weight.to(dt).view(h, nq, hd)
                for nm, src, sc in (('q_norm', a.q_norm, 1.0), ('k_norm', a.k_norm, math.sqrt(hd / HD))):
                    w = getattr(sa, nm).weight
                    eff = w.float() + 1
                    eff[L.slot_m[k].to(eff.device)] = src.weight.float().to(eff.device) * sc
                    w.copy_((eff - 1).to(dt))
    return m.eval(), cfg, L


def count_params(m):
    return sum(p.numel() for p in m.parameters())
