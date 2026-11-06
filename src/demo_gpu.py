import sys, time, torch, glob, os, json
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from nest2 import *
dev = 'cuda'; dtype = torch.bfloat16
I3 = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
tin = AutoTokenizer.from_pretrained(I3)
inner = AutoModelForCausalLM.from_pretrained(I3, dtype=dtype).to(dev).eval()
t = time.time(); merged, cfg = build(inner, dtype, dev); print('built', round(time.time() - t), 's', flush=True)
res = {}
prompts = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n\n", "Explain in one sentence why the sky is blue:",
           "1, 2, 3, 4, 5,", "Q: What is 17 * 3?\nA:"]
ok = 0; tot = 0; maxd = 0; rows = []
with torch.no_grad():
    for p in prompts:
        x = tin(p, return_tensors='pt').input_ids.to(dev)
        g1 = inner.generate(x, max_new_tokens=48, do_sample=False)
        g2 = merged.generate(x + NO, max_new_tokens=48, do_sample=False) - NO
        n = min(g1.shape[1], g2.shape[1]); same = torch.equal(g1, g2)
        first_div = next((i for i in range(n) if g1[0, i] != g2[0, i]), None)
        rows.append(dict(prompt=p, identical=same, first_divergence=first_div, inner=tin.decode(g1[0]), nested=tin.decode(g2[0])))
        print(rows[-1], flush=True)
    # speed
    x = tin("Once upon a time", return_tensors='pt').input_ids.to(dev)
    for name, mdl, off in (('inner Qwen3-0.6B', inner, 0), ('nested (Qwen3-0.6B inside Qwen3.5-0.8B)', merged, NO)):
        mdl.generate(x + off, max_new_tokens=8, do_sample=False); torch.cuda.synchronize(); t = time.time()
        mdl.generate(x + off, max_new_tokens=64, min_new_tokens=64, do_sample=False); torch.cuda.synchronize()
        print(name, round(64 / (time.time() - t), 1), 'tok/s', flush=True); res[name] = 64 / (time.time() - t)
    # mutation: zero one inner MLP -> must change output
    x = tin("The capital of France is", return_tensors='pt').input_ids.to(dev)
    before = merged.generate(x + NO, max_new_tokens=16, do_sample=False)
    with torch.no_grad():
        for L in range(24 + 8, 24 + 20): merged.model.layers[L].mlp.down_proj.weight.zero_()
    after = merged.generate(x + NO, max_new_tokens=16, do_sample=False)
    print('mutation (zero inner MLPs 8-19) changes output:', not torch.equal(before, after), '|', tin.decode(after[0] - NO))
json.dump(dict(rows=rows, speed=res, params_B=sum(p.numel() for p in merged.parameters()) / 1e9), open('demo_gpu_results.json', 'w'), indent=1)
