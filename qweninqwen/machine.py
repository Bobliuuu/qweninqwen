"""Shared outer-machine definitions for qweninqwen.  See INTERFACE.md."""
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

BALLAST = float(2**27)          # value of slot 0 in every frame (constant); makes outer RMSNorm ~ constant scale
GATE_ON = 30.0           # silu(30)=30-9e-12 ; sigmoid(30)=1-9e-14 : 'gate fully open' constant


class Layout:
    """Allocates named slots in the residual stream (frame). slot 'ONE' (index 0) = BALLAST."""
    def __init__(self):
        self.slots = {}; self.n = 0
        self.alloc('BALLAST', 1)
    def alloc(self, name, size):
        assert name not in self.slots, name
        self.slots[name] = slice(self.n, self.n + size); self.n += size
        return self.slots[name]
    def __getitem__(self, k): return self.slots[k]


def make_outer(layout_or_D, layer_types, intermediate, gdn=dict(hk=16, hv=16, dk=128, dv=128),
               attn=dict(heads=0, kv=0, hd=64), vocab=151936, eps=1e-12, dtype=torch.float32,
               tie=False):
    """Stock Qwen3_5ForCausalLM, ALL parameters zeroed (every layer = identity on the stream).
    layer_types: list of 'linear_attention' | 'full_attention'.  Compilers then write weights."""
    D = layout_or_D.n if isinstance(layout_or_D, Layout) else layout_or_D
    kw = dict(hidden_size=D, intermediate_size=intermediate, num_hidden_layers=len(layer_types),
              layer_types=layer_types, vocab_size=vocab, rms_norm_eps=eps, tie_word_embeddings=tie,
              linear_num_key_heads=gdn['hk'], linear_num_value_heads=gdn['hv'],
              linear_key_head_dim=gdn['dk'], linear_value_head_dim=gdn['dv'], linear_conv_kernel_dim=4,
              num_attention_heads=max(attn['heads'], 1), num_key_value_heads=max(attn['kv'], 1),
              head_dim=attn['hd'], max_position_embeddings=4096, hidden_act='silu',
              attn_output_gate=True) if False else dict(
              hidden_size=D, intermediate_size=intermediate, num_hidden_layers=len(layer_types),
              layer_types=layer_types, vocab_size=vocab, rms_norm_eps=eps, tie_word_embeddings=tie,
              linear_num_key_heads=gdn['hk'], linear_num_value_heads=gdn['hv'],
              linear_key_head_dim=gdn['dk'], linear_value_head_dim=gdn['dv'], linear_conv_kernel_dim=4,
              num_attention_heads=max(attn['heads'], 1), num_key_value_heads=max(attn['kv'], 1),
              head_dim=attn['hd'], max_position_embeddings=4096)
    cfg = Qwen3_5TextConfig(**kw)
    m = Qwen3_5ForCausalLM(cfg).to(dtype)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
    return m


def gdn_const_init(layer, layout, hv=None):
    """Make a GDN layer an 'overwrite memory': beta ~ 1 (1-1e-13), decay g ~ 0 (-9e-14), conv = current tap only.
    Inputs q/k/v still pass through the conv's SiLU (see INTERFACE.md: feed z>=20 or small-signal)."""
    g = layer.linear_attn
    with torch.no_grad():
        g.conv1d.weight.zero_(); g.conv1d.weight[:, 0, -1] = 1.0   # last tap = current token
        g.A_log.zero_(); g.dt_bias.zero_()
        g.in_proj_b.weight[:, 0] = GATE_ON / BALLAST               # beta = sigmoid(30)
        g.in_proj_a.weight[:, 0] = -GATE_ON / BALLAST              # softplus(-30) ~ 0 -> exp(g)=1
        g.in_proj_z.weight[:, 0] = GATE_ON / BALLAST               # silu(z)=30 constant gate
        g.norm.weight.fill_(1.0 / GATE_ON)                         # gated-norm output = o/rms(o)
