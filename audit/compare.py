"""Reference-comparison kit (T1 op / T2 layer / T3 model).  Usable by every agent.

  ref = hf_trace(hf_inner, input_ids)                 # dict name -> tensor (fp32 cpu), canonical names below
  cand = {...}  same names (any subset)               # produced by the in-model pipeline (read from outer slots/logits)
  rows = compare_traces(ref, cand, tol=Tol(...))      # list of dict rows
  print(table(rows)); assert all_pass(rows)

Canonical names (Qwen2/Qwen3 HF inner):  embed | L{i}.attn_norm | L{i}.q | L{i}.k | L{i}.v | L{i}.attn_out(o_proj out) |
  L{i}.resid_mid (hidden after attn residual add; derived) | L{i}.mlp_norm | L{i}.gate | L{i}.up | L{i}.down | L{i}.out |
  final_norm | logits.   q/k/v are the raw projections (pre-RoPE, pre-qk-norm); shapes [B,T,dim].
Metrics: max_abs, rel_l2=||d||/||ref||, scale_err=max_abs/max|ref|, cosine; PASS if scale_err<=rtol or max_abs<=atol.
For 'logits' also: argmax agreement per position and top-5 overlap.   Different tolerance per op by regex.
"""
import re, torch


class Tol:
    def __init__(self, atol=1e-4, rtol=1e-3, per=None):
        """per: {regex: (atol, rtol)} first match wins (e.g. {'.*mlp_norm': (1e-3,1e-3), 'logits': (1e-2,1e-3)})"""
        self.atol, self.rtol, self.per = atol, rtol, per or {}
    def get(self, name):
        for pat, v in self.per.items():
            if re.fullmatch(pat, name): return v
        return self.atol, self.rtol


def hf_trace(model, input_ids=None, inputs_embeds=None, keep_logits=True):
    """Canonical per-op trace of a Qwen2/Qwen3-style HF causal LM (fp32 recommended).  Runs the REFERENCE (HF) model:
    call it inside hostguard.reference() when a HostGuard is active."""
    tr, hs = {}, []
    base = model.model
    def put(n):
        def f(m, a, o):
            o = o[0] if isinstance(o, tuple) else o
            tr[n] = o.detach().float().cpu()
        return f
    hs.append(base.embed_tokens.register_forward_hook(put("embed")))
    for i, l in enumerate(base.layers):
        hs += [l.input_layernorm.register_forward_hook(put(f"L{i}.attn_norm")),
               l.self_attn.q_proj.register_forward_hook(put(f"L{i}.q")),
               l.self_attn.k_proj.register_forward_hook(put(f"L{i}.k")),
               l.self_attn.v_proj.register_forward_hook(put(f"L{i}.v")),
               l.self_attn.o_proj.register_forward_hook(put(f"L{i}.attn_out")),
               l.post_attention_layernorm.register_forward_hook(put(f"L{i}.mlp_norm")),
               l.mlp.gate_proj.register_forward_hook(put(f"L{i}.gate")),
               l.mlp.up_proj.register_forward_hook(put(f"L{i}.up")),
               l.mlp.down_proj.register_forward_hook(put(f"L{i}.down")),
               l.register_forward_hook(put(f"L{i}.out"))]
    hs.append(base.norm.register_forward_hook(put("final_norm")))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, inputs_embeds=inputs_embeds) if inputs_embeds is not None else model(input_ids=input_ids)
    finally:
        for h in hs: h.remove()
    if keep_logits: tr["logits"] = out.logits.detach().float().cpu()
    for i in range(len(base.layers)):
        prev = tr["embed"] if i == 0 else tr[f"L{i-1}.out"]
        tr[f"L{i}.resid_mid"] = prev + tr[f"L{i}.attn_out"]
    return tr


def _order(k):
    if k == "embed": return (-1, 0, k)
    m = re.match(r"L(\d+)\.(\w+)", k)
    ops = ["attn_norm", "q", "k", "v", "attn_out", "resid_mid", "mlp_norm", "gate", "up", "down", "out"]
    if m: return (int(m.group(1)), ops.index(m.group(2)) if m.group(2) in ops else 99, k)
    return (10**6, 0, k)


def compare_tensors(a, b):
    a = a.double(); b = b.double().reshape(a.shape) if b.numel() == a.numel() else None
    if b is None: return None
    d = (a - b).abs()
    scale = a.abs().max().item() + 1e-30
    na = a.norm().item() + 1e-30
    cos = (a.flatten() @ b.flatten() / (na * (b.norm().item() + 1e-30))).item()
    return dict(max_abs=d.max().item(), rel_l2=(d.norm().item() / na), scale_err=d.max().item() / scale, cos=cos,
                ref_scale=scale, nan=bool(torch.isnan(b).any()))


def compare_traces(ref, cand, tol=None, names=None):
    tol = tol or Tol(); rows = []
    for k in sorted(names or ref.keys(), key=_order):
        if k not in cand: rows.append(dict(name=k, status="MISSING")); continue
        c = compare_tensors(ref[k], cand[k] if isinstance(cand[k], torch.Tensor) else torch.as_tensor(cand[k]))
        if c is None: rows.append(dict(name=k, status="SHAPE", shape=(tuple(ref[k].shape), tuple(cand[k].shape)))); continue
        at, rt = tol.get(k)
        c.update(name=k, shape=tuple(ref[k].shape), atol=at, rtol=rt,
                 status="PASS" if (not c["nan"] and (c["max_abs"] <= at or c["scale_err"] <= rt)) else "FAIL")
        if k == "logits":
            ra, ca = ref[k].argmax(-1), cand[k].float().reshape(ref[k].shape).argmax(-1)
            c["argmax_agree"] = float((ra == ca).float().mean())
            r5 = ref[k].topk(5, -1).indices; c5 = cand[k].float().reshape(ref[k].shape).topk(5, -1).indices
            c["top5_overlap"] = float(torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 5 for x, y in
                                                   zip(r5.reshape(-1, 5), c5.reshape(-1, 5))]).mean())
        rows.append(c)
    return rows


def table(rows):
    L = [f"{'op':16s} {'shape':18s} {'max_abs':>10s} {'rel_l2':>10s} {'scale_err':>10s} {'cos':>10s}  tol(a,r)   status"]
    first = None
    for r in rows:
        if r["status"] in ("MISSING", "SHAPE"):
            L.append(f"{r['name']:16s} {r['status']}  {r.get('shape','')}"); first = first or r["name"]; continue
        extra = f"  argmax_agree={r['argmax_agree']:.3f} top5={r['top5_overlap']:.3f}" if "argmax_agree" in r else ""
        L.append(f"{r['name']:16s} {str(r['shape']):18s} {r['max_abs']:10.3e} {r['rel_l2']:10.3e} {r['scale_err']:10.3e} "
                 f"{r['cos']:10.6f}  ({r['atol']:.0e},{r['rtol']:.0e}) {r['status']}{extra}")
        if r["status"] == "FAIL" and first is None: first = r["name"]
    L.append(f"FIRST DIVERGENCE: {first}" if first else "ALL OPS WITHIN TOLERANCE")
    return "\n".join(L)


def all_pass(rows): return all(r["status"] == "PASS" for r in rows)


def selftest():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                                     num_key_value_heads=2, vocab_size=100)).float().eval()
    ids = torch.randint(0, 100, (1, 7))
    ref = hf_trace(m, ids)
    cand = {k: v + 1e-6 * torch.randn_like(v) for k, v in ref.items()}
    r1 = compare_traces(ref, cand); print(table(r1)); assert all_pass(r1)
    cand["L1.gate"] = cand["L1.gate"] * 1.05           # corrupt one op
    cand.pop("L2.up")
    r2 = compare_traces(ref, cand); print(table(r2)); assert not all_pass(r2)
    assert next(r["name"] for r in r2 if r["status"] != "PASS") == "L1.gate"
    print("compare selftest OK")

if __name__ == "__main__": selftest()
