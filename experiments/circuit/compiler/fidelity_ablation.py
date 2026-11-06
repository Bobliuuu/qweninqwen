"""Float-HF ablation of weight quantisation schemes on real Qwen3-0.6B (isolates weight error from activation/fixed-point error).
Metrics over a 48-token prompt: top-1 agreement vs fp32 HF at every position, position-0 logit error, mean KL.  Output res_fidelity_ablation.json"""
import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'Qwen/Qwen3-0.6B'; tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
text = "The capital of France is Paris. The capital of Germany is Berlin. Once upon a time, a small robot learned to read. " * 6
ids = tok(text, return_tensors='pt').input_ids[:, :48]
orig = {n: p.detach().clone() for n, p in m.named_parameters()}
with torch.no_grad(): ref = m(ids).logits[0]
def fq(W, bits, group=None, sym=True):
    qmax = 2 ** (bits - 1) - 1
    if group:
        R, C = W.shape; Wg = W.reshape(R, C // group, group); s = Wg.abs().amax(-1, keepdim=True) / qmax
        return (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s).reshape(R, C)
    s = W.abs().amax(1, keepdim=True) / qmax; return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
def run(label, bits, group=None, skip=()):
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(orig[n])
            if p.ndim == 2 and ('proj' in n or 'embed' in n) and not any(s in n for s in skip): p.copy_(fq(orig[n], bits, group))
        lg = m(ids).logits[0]
    kl = torch.nn.functional.kl_div(lg.log_softmax(-1), ref.softmax(-1), reduction='none').sum(-1)
    r = dict(label=label, top1_agree=float((lg.argmax(-1) == ref.argmax(-1)).float().mean()), pos0_ok=bool(lg[0].argmax() == ref[0].argmax()),
             pos0_rel_err=float((lg[0] - ref[0]).abs().max() / ref[0].abs().max()), mean_kl=float(kl.mean()), kl_pos0=float(kl[0]))
    print(json.dumps(r)); return r
def run_layers(label, bits, keep_layers, mats):
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(orig[n])
            if p.ndim == 2 and ('proj' in n or 'embed' in n):
                lay = int(n.split('layers.')[1].split('.')[0]) if 'layers.' in n else -1
                if not (lay in keep_layers and any(s in n for s in mats)): p.copy_(fq(orig[n], bits))
        lg = m(ids).logits[0]
    kl = torch.nn.functional.kl_div(lg.log_softmax(-1), ref.softmax(-1), reduction='none').sum(-1)
    r = dict(label=label, top1_agree=float((lg.argmax(-1) == ref.argmax(-1)).float().mean()), pos0_ok=bool(lg[0].argmax() == ref[0].argmax()), pos0_rel_err=float((lg[0] - ref[0]).abs().max() / ref[0].abs().max()), mean_kl=float(kl.mean()), kl_pos0=float(kl[0]))
    print(json.dumps(r)); return r
import sys
if len(sys.argv) > 1:
    out = [run_layers(f'int8 per-row, layer {l} gate/up/down fp', 8, {l}, ('gate_proj', 'up_proj', 'down_proj')) for l in range(28)]
    json.dump(out, open('res_fidelity_layers.json', 'w'), indent=1); sys.exit()
out = [run('int8 per-row all', 8), run('int8 per-row, embed/lm_head kept fp', 8, skip=('embed',)), run('int8 per-row, down_proj kept fp', 8, skip=('down_proj',)),
       run('int8 per-row, attn kept fp', 8, skip=('self_attn',)), run('int8 per-row, gate/up kept fp', 8, skip=('gate_proj', 'up_proj')),
       run('int8 g=64', 8, 64), run('int4 per-row all', 4), run('int4 g=128 all', 4, 128), run('int4 g=64 all', 4, 64), run('int4 g=64 embed fp', 4, 64, skip=('embed',))]
json.dump(out, open('res_fidelity_ablation.json', 'w'), indent=1)
