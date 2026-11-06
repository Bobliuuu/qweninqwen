"""Stream a FULL inner token's worth of real weights (all matrices, real values) through B's paged-GDN bank on GPU.
usage: run_token.py [model q3|q25] [nL bank layers] [layers to run (default all)] [lm_head 0/1] [seg_cap_frames]
Activations X are harness-fed random (timing/accuracy of the paging machine; NOT yet a chained token).  Each matrix = one round
(one forward call); LM head split in calls of <=seg_cap frames.  Error checked vs fp64 reference on EVERY output of EVERY matrix."""
import sys, os, time, json, torch
if os.environ.get('NOFLA'): sys.modules['fla'] = None
 
import os, sys  # PATHS_SET
_h=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[os.path.join(_h,'..','..','qweninqwen'), os.path.join(_h,'..','state')]
from machine import Layout, make_outer
from engine import StateMV
from stream import *
from transformers import DynamicCache
from safetensors import safe_open
mod = sys.argv[1] if len(sys.argv) > 1 else 'q3'; nL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
nlay = int(sys.argv[3]) if len(sys.argv) > 3 else 999; do_head = int(sys.argv[4]) if len(sys.argv) > 4 else 1
segcap = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
dev = 'cuda'; torch.backends.cuda.matmul.allow_tf32 = False
cfg, st = inner_cfg(mod); prog, vec = inner_program(cfg)
prog = [p for p in prog if p[0] != 'lm_head.weight' and int(p[0].split('.')[2]) < nlay] + ([p for p in prog if p[0] == 'lm_head.weight'] if do_head else [])
lay = Layout(); HH = int(os.environ.get('HH', 16)); spec = StateMV(lay, nL=nL, H=HH, K=nL * HH * 63, B=1e6)
model = make_outer(lay, ['linear_attention'] * nL + ['full_attention'], 8, gdn=dict(hk=HH, hv=HH, dk=128, dv=128), vocab=8)
spec.compile(model); model = model.to(dev).eval(); fa = FrameAssembler(spec, dev)
print('D', lay.n, 'nL', nL, 'matrices', len(prog), flush=True)
f = safe_open(st, 'pt'); P2 = 2 * spec.P; rows = torch.arange(P2, device=dev)
# ---- build resident stream (token independent) ----
t0 = time.time(); store = {}; nbytes = 0
for nm, o, k in prog:
    key = nm if nm in f.keys() else 'model.embed_tokens.weight'
    W = f.get_tensor(key).float(); na = -(-(-(-k // spec.P)) // spec.H)
    store[nm] = (build_matrix_payload(spec, W, device=dev, nL_act=na), na, W.to(dev))
    nbytes += store[nm][0].numel() * 4
torch.cuda.synchronize(); t_build = time.time() - t0
print(f'stream built {nbytes/1e9:.2f} GB resident in {t_build:.1f}s; gpu alloc {torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
cap = {}; model.model.layers[nL - 1].register_forward_hook(lambda m, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
cache = DynamicCache(config=model.config); pos = [0]
def run(fr):
    n = fr.shape[0]; cp = torch.arange(pos[0], pos[0] + n, device=dev); pos[0] += n
    model.model(inputs_embeds=fr[None], past_key_values=cache, use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
    return cap['o'][0]
with torch.no_grad(): run(spec.init_frames().float().to(dev))
torch.manual_seed(0); res = []; tot_t = 0.0; tot_f = 0; worst = 0.0
with torch.no_grad():
  for nm, o, k in prog:
    pay, na, W = store[nm]; nOB = pay.shape[0]; X = torch.randn(1, k, device=dev) * 3
    Xp = torch.zeros(1, spec.K, device=dev); Xp[:, :k] = X; ys = []
    torch.cuda.synchronize(); t0 = time.time(); nfr = 0
    segb = max(1, segcap // (P2 + 1))
    for b0 in range(0, nOB, segb):
        b1 = min(nOB, b0 + segb); parts = []
        for b in range(b0, b1): parts += [fa.write(pay[b], rows, na), fa.read(Xp)]
        out = run(torch.cat(parts)); nfr += (b1 - b0) * (P2 + 1)
        ys += [out[i * (P2 + 1) + P2][spec.Y].double() for i in range(b1 - b0)]
    torch.cuda.synchronize(); dt = time.time() - t0
    y = torch.cat(ys)[:o]; ref = (X.double() @ W.double().T)[0]
    e = ((y - ref).abs().max() / ref.abs().max()).item(); worst = max(worst, e)
    am = bool(y.argmax() == ref.argmax())
    res.append(dict(name=nm, O=o, K=k, na=na, frames=nfr, sec=dt, maxrel=e, argmax_ok=am)); tot_t += dt; tot_f += nfr
    if len(res) <= 8 or nm == 'lm_head.weight': print(res[-1], flush=True)
summ = dict(model=mod, nL=nL, D=lay.n, matrices=len(prog), layers=min(nlay, cfg['num_hidden_layers']), head=bool(do_head), frames=tot_f, sec=tot_t, frames_per_s=tot_f / tot_t,
            worst_maxrel=worst, argmax_all_ok=all(r['argmax_ok'] for r in res), weights_streamed=sum(o * k for _, o, k in prog), stream_gb=nbytes / 1e9, build_s=t_build,
            peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, nofla=bool(os.environ.get('NOFLA')))
print('SUMMARY', json.dumps(summ), flush=True)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'token_results.jsonl'), 'a').write(json.dumps(summ) + '\n')
