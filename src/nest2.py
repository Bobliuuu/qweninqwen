"""Qwen3-0.6B (inner) merged into the real pretrained Qwen3.5-0.8B (outer).

One Qwen3.5 checkpoint, hidden = 1024 (outer) + 1024 (inner) + 1 (flag) padded to 2080.
Outer tokens live in dims [0,1024), inner tokens (vocab rows >= 248320) in [1024,2048).
Both halves are block-diagonal, so each half only ever reads/writes its own subspace.
Fixes done purely by weights (no code change to Qwen3.5):
  RMSNorm over wider hidden      : norm weight x sqrt(1024/2080)
  per-head q/k norm, wider head  : zero padding keeps sumsq; weight x sqrt(old_hd/new_hd) on k
  attention scale 1/sqrt(hd)     : folded (q norm weight unchanged <=> q scaled sqrt(new/old))
  RoPE: outer 64 rot dims @1e7, inner 128 rot dims @1e6 -> one shared grid 10^(-k/32),
        k=0..223 (theta=1e7, 448 rot dims). outer pair j -> k=7j, inner pair i -> k=3i.
  attn output gate               : inner heads: gate = 2^26 * flag_normed  (flag only on inner tokens) -> sigmoid == 1
  outer logits during inner mode : lm_head outer rows read the flag with weight -2^26
"""
import json, glob, math, os, torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

HO, HI = 1024, 1024
FLAG = HO + HI
H = 2080
NO, NI = 248320, 151936
NQ, NKV, HD, ROT = 24, 12, 640, 448
HALF = ROT // 2
OD, ID = 256, 128
NQ_O, NQ_I, NKV_O, NKV_I = 8, 16, 4, 8
LAM = 2.0 ** 26
FLAGV = 2.0 ** -10
S = math.sqrt(HO / H)
OUTER_DIR = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*'))[0]


def pos_out(d):
    return 7 * d if d < 32 else 7 * (d - 32) + HALF if d < 64 else ROT + (d - 64)


def pos_in(d):
    return 3 * d if d < 64 else 3 * (d - 64) + HALF


IDX_O = torch.tensor([pos_out(d) for d in range(OD)])

IDX_I = torch.tensor([pos_in(d) for d in range(ID)])


def outer_text_config():
    return json.load(open(OUTER_DIR + '/config.json'))['text_config']


def merged_config(dtype='bfloat16'):
    tc = dict(outer_text_config())
    n_out = tc['num_hidden_layers']
    tc.update(hidden_size=H, num_hidden_layers=n_out + 28, num_attention_heads=NQ, num_key_value_heads=NKV,
              head_dim=HD, vocab_size=NO + NI, tie_word_embeddings=False, eos_token_id=NO + 151645,
              layer_types=tc['layer_types'] + ['full_attention'] * 28, dtype=dtype, mtp_num_hidden_layers=0,
              bos_token_id=None, pad_token_id=None)
    tc['rope_parameters'] = dict(rope_type='default', rope_theta=10000000, partial_rotary_factor=ROT / HD,
                                 mrope_section=[75, 75, 74], mrope_interleaved=True)
    return Qwen3_5TextConfig(**tc)


def load_outer_state():
    sd = {}
    with safe_open(OUTER_DIR + '/model.safetensors-00001-of-00001.safetensors', 'pt') as f:
        for k in f.keys():
            if k.startswith('model.language_model.'):
                sd[k[len('model.language_model.'):]] = f.get_tensor(k)
    return sd


def reference_outer(dtype):
    """The untouched Qwen3.5-0.8B text model (for outer-mode comparison)."""
    tc = outer_text_config(); tc = dict(tc); tc['mtp_num_hidden_layers'] = 0
    cfg = Qwen3_5TextConfig(**tc)
    m = Qwen3_5ForCausalLM(cfg).to(dtype)
    sd = load_outer_state()
    m.model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    m.lm_head.weight = m.model.embed_tokens.weight
    return m.eval()


def zero_(m):
    with torch.no_grad():
        for p in m.parameters(): p.zero_()


def vec_norm(outer_w=None, inner_w=None, flag=False, dt=torch.float32):
    """stored (zero-centred) norm weight over the merged hidden."""
    v = torch.zeros(H, dtype=torch.float32)   # effective multiplier (1+w)
    if outer_w is not None: v[:HO] = (1 + outer_w.float().cpu()) * S
    if inner_w is not None: v[HO:HO + HI] = inner_w.float().cpu() * S
    if flag: v[FLAG] = 1.0
    return (v - 1).to(dt)


def build(inner, dtype=torch.bfloat16, device='cpu'):
    cfg = merged_config(str(dtype).replace('torch.', ''))
    torch.set_default_dtype(dtype)
    with torch.device(device):
        m = Qwen3_5ForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    zero_(m)
    osd = {k: v.to(device) for k, v in load_outer_state().items()}
    nl = len(osd and [1 for k in osd if k.endswith('.input_layernorm.weight')])
    types = cfg.layer_types
    dt = dtype; dv = torch.device(device)
    IO, II = IDX_O.to(dv), IDX_I.to(dv)
    with torch.no_grad():
        # ---------------- embeddings / head / final norm
        E = m.model.embed_tokens.weight
        E[:NO, :HO] = osd['embed_tokens.weight'].to(dt).to(dv)
        E[NO:, HO:HO + HI] = inner.model.embed_tokens.weight.to(dt)
        E[NO:, FLAG] = FLAGV
        Lh = m.lm_head.weight
        Lh[:NO, :HO] = osd['embed_tokens.weight'].to(dt).to(dv)
        Lh[:NO, FLAG] = -LAM
        Lh[NO:, HO:HO + HI] = inner.lm_head.weight.to(dt)
        m.model.norm.weight.copy_(vec_norm(osd['norm.weight'], inner.model.norm.weight, True, dt))
        # ---------------- outer layers 0..nl-1
        for L in range(nl):
            ly = m.model.layers[L]; p = f'layers.{L}.'
            ly.input_layernorm.weight.copy_(vec_norm(osd[p + 'input_layernorm.weight'], None, False, dt))
            ly.post_attention_layernorm.weight.copy_(vec_norm(osd[p + 'post_attention_layernorm.weight'], None, False, dt))
            ly.mlp.gate_proj.weight[:, :HO] = osd[p + 'mlp.gate_proj.weight'].to(dt)
            ly.mlp.up_proj.weight[:, :HO] = osd[p + 'mlp.up_proj.weight'].to(dt)
            ly.mlp.down_proj.weight[:HO, :] = osd[p + 'mlp.down_proj.weight'].to(dt)
            if types[L] == 'linear_attention':
                la = ly.linear_attn
                for n in ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b'):
                    getattr(la, n).weight[:, :HO] = osd[p + f'linear_attn.{n}.weight'].to(dt)
                la.out_proj.weight[:HO, :] = osd[p + 'linear_attn.out_proj.weight'].to(dt)
                la.conv1d.weight.copy_(osd[p + 'linear_attn.conv1d.weight'].to(dt))
                la.dt_bias.copy_(osd[p + 'linear_attn.dt_bias'].to(dt))
                la.A_log.copy_(osd[p + 'linear_attn.A_log'].to(dt))
                la.norm.weight.copy_(osd[p + 'linear_attn.norm.weight'].to(dt))
            else:
                sa = ly.self_attn
                q = osd[p + 'self_attn.q_proj.weight'].to(dt).view(NQ_O, 2, OD, HO)
                Wq = sa.q_proj.weight.view(NQ, 2, HD, H)
                tmp = torch.zeros(NQ_O, HD, HO, dtype=dt, device=dv); tmp[:, IO] = q[:, 0]
                Wq[:NQ_O, 0, :, :HO] = tmp
                Wq[:NQ_O, 1, :OD, :HO] = q[:, 1]
                k = osd[p + 'self_attn.k_proj.weight'].to(dt).view(2, OD, HO)
                v = osd[p + 'self_attn.v_proj.weight'].to(dt).view(2, OD, HO)
                Wk = sa.k_proj.weight.view(NKV, HD, H); Wv = sa.v_proj.weight.view(NKV, HD, H)
                for mi in range(NKV_O):
                    tk = torch.zeros(HD, HO, dtype=dt, device=dv); tk[IO] = k[mi // 2]; Wk[mi, :, :HO] = tk
                    Wv[mi, :OD, :HO] = v[mi // 2]
                sa.o_proj.weight.view(H, NQ, HD)[:HO, :NQ_O, :OD] = osd[p + 'self_attn.o_proj.weight'].to(dt).view(HO, NQ_O, OD)
                qn = torch.full((HD,), 0.0); kn = torch.full((HD,), 0.0)   # effective (1+w)
                qn[IDX_O] = 1 + osd[p + 'self_attn.q_norm.weight'].float().cpu()
                kn[IDX_O] = (1 + osd[p + 'self_attn.k_norm.weight'].float().cpu()) * math.sqrt(OD / HD)
                # NoPE dims of the outer head
                qn[ROT:ROT + OD - 64] = 1 + osd[p + 'self_attn.q_norm.weight'].float().cpu()[64:]
                kn[ROT:ROT + OD - 64] = (1 + osd[p + 'self_attn.k_norm.weight'].float().cpu()[64:]) * math.sqrt(OD / HD)
                sa.q_norm.weight.copy_((qn - 1).to(dt)); sa.k_norm.weight.copy_((kn - 1).to(dt))
        # ---------------- inner layers appended
        for i, il in enumerate(inner.model.layers):
            ly = m.model.layers[nl + i]
            ly.input_layernorm.weight.copy_(vec_norm(None, il.input_layernorm.weight, True, dt))
            ly.post_attention_layernorm.weight.copy_(vec_norm(None, il.post_attention_layernorm.weight, False, dt))
            ly.mlp.gate_proj.weight[:3072, HO:HO + HI] = il.mlp.gate_proj.weight.to(dt)
            ly.mlp.up_proj.weight[:3072, HO:HO + HI] = il.mlp.up_proj.weight.to(dt)
            ly.mlp.down_proj.weight[HO:HO + HI, :3072] = il.mlp.down_proj.weight.to(dt)
            a = il.self_attn; sa = ly.self_attn
            q = a.q_proj.weight.to(dt).view(NQ_I, ID, HI)
            Wq = sa.q_proj.weight.view(NQ, 2, HD, H)
            tmp = torch.zeros(NQ_I, HD, HI, dtype=dt, device=dv); tmp[:, II] = q
            Wq[NQ_O:, 0, :, HO:HO + HI] = tmp
            Wq[NQ_O:, 1, :, FLAG] = LAM
            k = a.k_proj.weight.to(dt).view(NKV_I, ID, HI); v = a.v_proj.weight.to(dt).view(NKV_I, ID, HI)
            Wk = sa.k_proj.weight.view(NKV, HD, H); Wv = sa.v_proj.weight.view(NKV, HD, H)
            tk = torch.zeros(NKV_I, HD, HI, dtype=dt, device=dv); tk[:, II] = k
            Wk[NKV_O:, :, HO:HO + HI] = tk
            Wv[NKV_O:, :ID, HO:HO + HI] = v
            sa.o_proj.weight.view(H, NQ, HD)[HO:HO + HI, NQ_O:, :ID] = a.o_proj.weight.to(dt).view(HI, NQ_I, ID)
            qn = torch.zeros(HD); kn = torch.zeros(HD)
            qn[IDX_I] = a.q_norm.weight.float().cpu()
            kn[IDX_I] = a.k_norm.weight.float().cpu() * math.sqrt(ID / HD)
            sa.q_norm.weight.copy_((qn - 1).to(dt)); sa.k_norm.weight.copy_((kn - 1).to(dt))
    return m.eval(), cfg
