"""Real Qwen3-0.6B -> integer firmware -> emulator; per-position instruction counts, per-symbol profile, fidelity vs HF.
usage: run_full_fx.py WBITS(4|8) EXT(0|1) NTOK OUTDIR [dump(0|1)]    (CPU only; ~1e9 instr/token -> tens of s per token in the emulator)"""
import sys, os, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen_fx, test_tiny_fx as T
wbits, ext, ntok, out = int(sys.argv[1]), bool(int(sys.argv[2])), int(sys.argv[3]), sys.argv[4]; dump = len(sys.argv) > 5 and sys.argv[5] == '1'
# optional 6th arg: boost spec 'L:mat:bits,...' e.g. 27:mlp:12,2:mlp:12
boost = {(int(a), b): int(c) for a, b, c in (x.split(':') for x in sys.argv[6].split(','))} if len(sys.argv) > 6 else None
tag = '' if not boost else '_boost' + '_'.join(f'{a}{b}{c}' for (a, b), c in boost.items())
name = 'Qwen/Qwen3-0.6B'
tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
text = ("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. " * 40)
prompt = tok(text, return_tensors='pt').input_ids[0][:ntok].tolist()
with torch.no_grad(): ref = m(torch.tensor([prompt])).logits[0].numpy()
c = m.config; cd = dict(hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
    head_dim=c.head_dim, intermediate_size=c.intermediate_size, vocab_size=c.vocab_size, rms_norm_eps=c.rms_norm_eps, rope_theta=1000000.0)
sd = {k: v.numpy() for k, v in m.state_dict().items()}
# weight-only fake quant (same per-row RTN as the firmware) in float HF: isolates the weight-quantisation error
def fq(W, wb):
    qmax = 7.0 if wb == 4 else 119.0; lo, hi = (-8, 7) if wb == 4 else (-120, 119)
    s = W.abs().max(1, keepdim=True).values / qmax; return torch.clamp(torch.round(W / s), lo, hi) * s
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.ndim == 2 and ('proj' in n or 'embed' in n): p.copy_(fq(p, wbits))
    refq = m(torch.tensor([prompt])).logits[0].numpy()
del m
st, log = T.build_run(sd, cd, prompt, out, wbits, ext, dump=dump, boost=boost)
am, lg, marks, syms, first = T.parse(log, out, len(prompt), cd['vocab_size'], dump)
ip = [b - a for a, b in zip(marks, marks[1:])]
res = dict(model=name, wbits=wbits, ext=ext, ntok=ntok, instr_per_pos=ip, classes=first, by_symbol=syms, blob=st,
           argmax_fw=am.tolist(), argmax_hf=ref.argmax(1).tolist(), argmax_hf_weightquant_only=refq.argmax(1).tolist(),
           top1_agree_fw_vs_hf=float((am == ref.argmax(1)).mean()), top1_agree_weightquant_vs_hf=float((refq.argmax(1) == ref.argmax(1)).mean()),
           top1_agree_fw_vs_weightquant=float((am == refq.argmax(1)).mean()))
srt = np.sort(ref, 1); res['hf_top1_top2_gap_pos0'] = float(srt[0, -1] - srt[0, -2])
if dump:
    res['rel_err_fw_vs_hf'] = float(np.abs(lg - ref).max() / np.abs(ref).max()); res['rel_err_weightquant_vs_hf'] = float(np.abs(refq - ref).max() / np.abs(ref).max())
    res['rel_err_fw_vs_weightquant'] = float(np.abs(lg - refq).max() / np.abs(ref).max())
    res['rms_err_fw_vs_hf'] = float(np.sqrt(((lg - ref) ** 2).mean())); res['rms_err_weightquant_vs_hf'] = float(np.sqrt(((refq - ref) ** 2).mean()))
json.dump(res, open(f'{HERE}/res_full_fx_w{wbits}_e{int(ext)}_n{ntok}{tag}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'by_symbol'}))
