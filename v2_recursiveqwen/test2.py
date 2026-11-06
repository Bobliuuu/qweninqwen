"""Verify N hosted inner models (+ outer) vs standalone HF models.
usage: python test2.py DTYPE DEVICE NLAYERS REPO1[,REPO2,...]   (NLAYERS=0 -> all layers)"""
import sys, os, time, json, torch
sys.path.insert(0, '.')
import merge2 as M
from transformers import AutoTokenizer

dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[sys.argv[1]]; dev = sys.argv[2]
nl = int(sys.argv[3]) or None; repos = sys.argv[4].split(',')
tok = AutoTokenizer.from_pretrained(M._snap('Qwen/Qwen3-0.6B'))
prompts = ['The capital of France is', 'def fibonacci(n):', 'Once upon a time, in a small village by the sea,',
           'Explain why the sky is blue in one sentence:', '2 + 2 * 3 =']
prompts = prompts[:int(os.environ.get('NP', 5))]; NT = int(os.environ.get('NT', 32))
inners = [M.load_inner(r, dtype, nl) for r in repos]
VT = int(os.environ.get('VT', 0))
if VT: prompts = [p for p in prompts if max(tok(p).input_ids) < VT]
if VT:   # test-only: trim inner vocab so bigger inner models fit in memory (ref trimmed identically)
    for im in inners:
        w = im.model.embed_tokens.weight.data[:VT].clone()
        im.model.embed_tokens = torch.nn.Embedding.from_pretrained(w, freeze=True)
        im.lm_head = torch.nn.Linear(w.shape[1], VT, bias=False, dtype=w.dtype); im.lm_head.weight.data = w.clone()
        im.config.vocab_size = VT
t0 = time.time()
m, cfg, L = M.build(inners, dtype, dev, n_app=nl)
print(f'built in {time.time()-t0:.0f}s: params {M.count_params(m)/1e9:.3f}B hidden {L.H} layers {cfg.num_hidden_layers} NQ {L.NQ} NKV {L.NKV} HD {L.HD} I {L.I} vocab {L.vocab}', flush=True)
res = {}
@torch.no_grad()
def greedy(model, ids, n, eos=None):
    out = model.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=None, pad_token_id=0)
    return out[0, ids.shape[1]:]
for k, (repo, ref) in enumerate(zip(repos, inners)):
    ref = ref.to(dev); off = L.voff[k]; V = L.V[k]
    md, na, mt, ident, bit = 0.0, 0, 0, 0, 0
    for pr in prompts:
        ids = tok(pr, return_tensors='pt').input_ids.to(dev)
        with torch.no_grad():
            lr = ref(ids).logits.float(); lm = m(input_ids=ids + off).logits[..., off:off + V].float()
        md = max(md, (lr - lm).abs().max().item()); na += (lr.argmax(-1) == lm.argmax(-1)).sum().item(); mt += ids.shape[1]
        gr = greedy(ref, ids, NT); gm = greedy(m, ids + off, NT) - off
        ident += int(torch.equal(gr, gm))
    res[repo] = dict(max_dlogit=md, argmax_match=f'{na}/{mt}', greedy_identical=f'{ident}/{len(prompts)}')
    print(repo, res[repo], flush=True)
    ref.cpu()
# outer mode
ro = M.reference_outer(dtype).to(dev)
md, na, mt, ident = 0.0, 0, 0, 0
for pr in prompts:
    ids = tok(pr, return_tensors='pt').input_ids.to(dev)
    with torch.no_grad():
        lr = ro(ids).logits.float(); lm = m(input_ids=ids).logits[..., :M.NO].float()
    md = max(md, (lr - lm).abs().max().item()); na += (lr.argmax(-1) == lm.argmax(-1)).sum().item(); mt += ids.shape[1]
    ident += int(torch.equal(greedy(ro, ids, NT), greedy(m, ids, NT)))
res['outer'] = dict(max_dlogit=md, argmax_match=f'{na}/{mt}', greedy_identical=f'{ident}/{len(prompts)}')
print('outer', res['outer'], flush=True)
# speed
if dev == 'cuda':
    ids = tok(prompts[0], return_tensors='pt').input_ids.to(dev) + L.voff[0]
    torch.cuda.synchronize(); t = time.time(); greedy(m, ids, 64); torch.cuda.synchronize()
    res['tok_per_s_merged'] = 64 / (time.time() - t); print('tok/s', res['tok_per_s_merged'])
json.dump(res, open(f'res_{sys.argv[1]}_{dev}_{nl}_{len(repos)}.json', 'w'), indent=1)
