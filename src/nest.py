"""Compile Qwen2.5-0.5B (inner) into a Qwen3.5-architecture model (outer).

Every inner tensor is written into the matching outer tensor; the outer's own
forward code (Qwen3_5Attention / RMSNorm / SwiGLU) then computes the inner model.
Outer-only features are neutralised *by weights*:
  - zero-centred RMSNorm       : stored weight = w - 1
  - attn output gate           : gate bias = +40 -> sigmoid == 1
  - per-head q_norm / k_norm   : ballast dims (bias = C) dominate the rms; norm
                                 weight restores the scale, ballast weight = 0
  - head_dim padding           : v / o_proj columns zero
  - attention scale 1/sqrt(hd) : folded into q (x sqrt(hd)/8)
"""
import math, torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

BALLAST = 2.0 ** 16
GATE_ON = 40.0


def outer_config(ic, head_dim=128):
    rot = ic.hidden_size // ic.num_attention_heads  # 64
    assert head_dim > rot and head_dim % 2 == 0
    return Qwen3_5TextConfig(
        vocab_size=ic.vocab_size, hidden_size=ic.hidden_size,
        intermediate_size=ic.intermediate_size, num_hidden_layers=ic.num_hidden_layers,
        num_attention_heads=ic.num_attention_heads, num_key_value_heads=ic.num_key_value_heads,
        head_dim=head_dim, hidden_act="silu", rms_norm_eps=ic.rms_norm_eps,
        attention_bias=True, attn_output_gate=True,
        layer_types=["full_attention"] * ic.num_hidden_layers,
        max_position_embeddings=ic.max_position_embeddings,
        tie_word_embeddings=True, pad_token_id=None, eos_token_id=ic.eos_token_id,
        rope_parameters=dict(rope_type="default", rope_theta=ic.rope_parameters["rope_theta"],
                             partial_rotary_factor=rot / head_dim,
                             mrope_section=[rot // 6 + 1, rot // 6 + 1, rot // 2 - 2 * (rot // 6 + 1)],
                             mrope_interleaved=True),
    )


def put_attn(sa, a, ic, hd):
    """sa: outer Qwen3_5Attention, a: inner Qwen2Attention."""
    nq, nkv, d = ic.num_attention_heads, ic.num_key_value_heads, ic.hidden_size // ic.num_attention_heads
    H = ic.hidden_size
    pad = hd - d
    nlen = ((hd - d) * BALLAST ** 2 + 0.0)  # sum of squares of ballast part
    inv = math.sqrt(hd / (pad * 1.0))        # rms ~= C*sqrt(pad/hd)  -> restore factor C*sqrt(pad/hd)
    restore = BALLAST * math.sqrt(pad / hd)
    with torch.no_grad():
        # q_proj rows per head: [q(hd) | gate(hd)]
        W = torch.zeros(nq, 2 * hd, H); b = torch.zeros(nq, 2 * hd)
        W[:, :d] = a.q_proj.weight.view(nq, d, H); b[:, :d] = a.q_proj.bias.view(nq, d)
        b[:, d:hd] = BALLAST; b[:, hd:] = GATE_ON
        sa.q_proj.weight.copy_(W.view(-1, H)); sa.q_proj.bias.copy_(b.view(-1))
        Wk = torch.zeros(nkv, hd, H); bk = torch.zeros(nkv, hd)
        Wk[:, :d] = a.k_proj.weight.view(nkv, d, H); bk[:, :d] = a.k_proj.bias.view(nkv, d); bk[:, d:] = BALLAST
        sa.k_proj.weight.copy_(Wk.view(-1, H)); sa.k_proj.bias.copy_(bk.view(-1))
        Wv = torch.zeros(nkv, hd, H); bv = torch.zeros(nkv, hd)
        Wv[:, :d] = a.v_proj.weight.view(nkv, d, H); bv[:, :d] = a.v_proj.bias.view(nkv, d)
        sa.v_proj.weight.copy_(Wv.view(-1, H)); sa.v_proj.bias.copy_(bv.view(-1))
        Wo = torch.zeros(H, nq, hd); Wo[:, :, :d] = a.o_proj.weight.view(H, nq, d)
        sa.o_proj.weight.copy_(Wo.view(H, -1)); sa.o_proj.bias.zero_()
        # zero-centred norms: (1+w): first d dims -> restore (q also x sqrt(hd/d_inner_scale)), ballast -> 0
        scale_q = math.sqrt(hd) / math.sqrt(d)  # outer 1/sqrt(hd) -> inner 1/sqrt(d)
        wq = torch.full((hd,), -1.0); wq[:d] = restore * scale_q - 1
        wk = torch.full((hd,), -1.0); wk[:d] = restore - 1
        sa.q_norm.weight.copy_(wq); sa.k_norm.weight.copy_(wk)


def compile_nested(inner, head_dim=128, dtype=torch.float32):
    ic = inner.config
    cfg = outer_config(ic, head_dim)
    outer = Qwen3_5ForCausalLM(cfg).to(torch.float32)
    with torch.no_grad():
        outer.model.embed_tokens.weight.copy_(inner.model.embed_tokens.weight)
        for ol, il in zip(outer.model.layers, inner.model.layers):
            ol.input_layernorm.weight.copy_(il.input_layernorm.weight - 1)
            ol.post_attention_layernorm.weight.copy_(il.post_attention_layernorm.weight - 1)
            put_attn(ol.self_attn, il.self_attn, ic, head_dim)
            ol.mlp.gate_proj.weight.copy_(il.mlp.gate_proj.weight)
            ol.mlp.up_proj.weight.copy_(il.mlp.up_proj.weight)
            ol.mlp.down_proj.weight.copy_(il.mlp.down_proj.weight)
        outer.model.norm.weight.copy_(inner.model.norm.weight - 1)
    outer.tie_weights()
    return outer.to(dtype), cfg
