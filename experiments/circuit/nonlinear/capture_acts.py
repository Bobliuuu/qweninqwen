"""Run HF Qwen3-0.6B (fp32) on a prompt and save the inputs/outputs of every nonlinear op for a few layers (real activations).  Runs on the GPU box."""
import torch, numpy as np, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "Qwen/Qwen3-0.6B"; tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager").eval()
text = ("The capital of France is Paris, a city on the river Seine. In 1889 the Eiffel Tower was completed for the World's Fair. "
        "Water boils at one hundred degrees Celsius at sea level, while the speed of light is roughly 299,792 kilometres per second.")
ids = tok(text, return_tensors="pt")["input_ids"]; T = ids.shape[1]; print("T", T)
LAY = [0, 1, 7, 14, 21, 27]; out = {}
def pre(name_, idx=0):
    def f(mod, args, kwargs=None): out[name_] = (args[idx] if args else kwargs["hidden_states"]).detach()[0].numpy().copy()
    return f
def post(name_):
    def f(mod, args, kwargs, res=None): pass
    return f
hs = []
for L in LAY:
    ly = m.model.layers[L]; at = ly.self_attn
    ly.input_layernorm.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.in_norm", (a[0][0].numpy().copy(), r[0].numpy().copy())))
    ly.post_attention_layernorm.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.post_norm", (a[0][0].numpy().copy(), r[0].numpy().copy())))
    at.q_norm.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.q_norm", (a[0][0].numpy().copy(), r[0].numpy().copy())))
    at.k_norm.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.k_norm", (a[0][0].numpy().copy(), r[0].numpy().copy())))
    def ah(mod, args, kwargs, res, L=L):
        out[f"L{L}.cossin"] = tuple(t[0].numpy().copy() for t in kwargs["position_embeddings"]); out[f"L{L}.probs"] = res[1][0].numpy().copy()
    at.register_forward_hook(ah, with_kwargs=True)
    ly.mlp.down_proj.register_forward_pre_hook(lambda mod, a, L=L: out.__setitem__(f"L{L}.swiglu_out", a[0][0].numpy().copy()))
    ly.mlp.gate_proj.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.gate", r[0].numpy().copy()))
    ly.mlp.up_proj.register_forward_hook(lambda mod, a, r, L=L: out.__setitem__(f"L{L}.up", r[0].numpy().copy()))
for L in LAY:
    ly = m.model.layers[L]; out[f'L{L}.in_g'] = ly.input_layernorm.weight.detach().numpy().copy(); out[f'L{L}.post_g'] = ly.post_attention_layernorm.weight.detach().numpy().copy()
    out[f'L{L}.q_g'] = ly.self_attn.q_norm.weight.detach().numpy().copy(); out[f'L{L}.k_g'] = ly.self_attn.k_norm.weight.detach().numpy().copy()
with torch.no_grad(): res = m(ids, output_attentions=True)
# q,k after rope for scores: recompute from q_norm/k_norm outputs
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
for L in LAY:
    qn = torch.tensor(out[f"L{L}.q_norm"][1]); kn = torch.tensor(out[f"L{L}.k_norm"][1]); cos, sin = (torch.tensor(t) for t in out[f"L{L}.cossin"])
    q = qn.transpose(0, 1)[None]; k = kn.transpose(0, 1)[None]          # [1,h,T,hd]
    q2, k2 = apply_rotary_pos_emb(q, k, cos[None], sin[None])
    out[f"L{L}.q_rope"] = (q[0].numpy(), q2[0].numpy()); out[f"L{L}.k_rope"] = (k[0].numpy(), k2[0].numpy())
flat = {}
for k, v in out.items():
    if isinstance(v, tuple):
        for i, a in enumerate(v): flat[f"{k}.{i}"] = a
    else: flat[k] = v
flat["tokens"] = ids[0].numpy(); np.savez_compressed("acts.npz", **flat); print("saved", len(flat))
for L in LAY: print(L, "resid absmax", np.abs(out[f"L{L}.in_norm"][0]).max(), "rms", np.sqrt((out[f"L{L}.in_norm"][0]**2).mean()))
