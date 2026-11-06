"""Select the GDN kernel used by the stock Qwen3_5GatedDeltaNet: 'torch' (reference fp32, exact) or 'fla' (fused triton)."""
import transformers.models.qwen3_5.modeling_qwen3_5 as M
_orig = {n: getattr(M, n) for n in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')}
def set_impl(chunk='torch', recurrent='torch'):
    """modes: 'torch' (exact fp32 reference), 'fla' (fla kernel of that name), 'fla_rec' (use fla fused_recurrent for the chunk slot)"""
    for n, mode in (('torch_chunk_gated_delta_rule', chunk), ('torch_recurrent_gated_delta_rule', recurrent)):
        f = _orig[n]
        if mode == 'torch': g = getattr(f, '__wrapped__', f)
        elif mode == 'fla_rec': g = _orig['torch_recurrent_gated_delta_rule']
        else: g = f
        setattr(M, n, g)


def use_best():
    """exact + fastest available: fla fused_recurrent on GPU if flash-linear-attention is importable, else the torch fp32 reference.
    (fla *chunk* kernel is NOT exact: ~6e-4 rel error, TF32 inside; do not use it.)"""
    try:
        import fla  # noqa
        set_impl('fla_rec', 'fla_rec'); return 'fla_rec'
    except Exception:
        set_impl('torch', 'torch'); return 'torch'
