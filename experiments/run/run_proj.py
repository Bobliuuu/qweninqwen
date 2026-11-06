"""Run a FULL real inner matrix through B's paged-GDN matvec engine on GPU. usage: run_proj.py <tensor name> [seg_blocks] [model q3|q25] [nL]
X (the activation) is given to the READ frames by the harness (test harness, as in B's tiled.py); outputs read from slot Y."""
import sys, time, json, glob, torch, os

import os, sys  # PATHS_SET
_h=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[os.path.join(_h,'..','..','qweninqwen'), os.path.join(_h,'..','state'), os.path.join(_h,'..','ops')]
from machine import Layout, make_outer
from engine import StateMV
from stream import *
from transformers import DynamicCache
from safetensors import safe_open
nm = sys.argv[1]; segb = int(sys.argv[2]) if len(sys.argv) > 2 else 16; mod = sys.argv[3] if len(sys.argv) > 3 else 'q3'
dev = 'cuda'; torch.backends.cuda.matmul.allow_tf32 = False
cfg, st = inner_cfg(mod)
with safe_open(st, 'pt') as f:
    key = nm if nm in f.keys() else 'model.embed_tokens.weight'
    W = f.get_tensor(key).float()
Oall, Kall = W.shape
nL = int(sys.argv[4]) if len(sys.argv) > 4 else -(-Kall // 1008)
lay = Layout(); spec = StateMV(lay, nL=nL, H=16, K=Kall, B=1e6)
model = make_outer(lay, ['linear_attention'] * nL + ['full_attention'], 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8)
spec.compile(model); model = model.to(dev).eval()
torch.manual_seed(0); X = (torch.randn(1, Kall) * 3).to(dev)
t0 = time.time(); pay = build_matrix_payload(spec, W, device=dev); torch.cuda.synchronize(); t_build = time.time() - t0
fa = FrameAssembler(spec, dev); nOB = pay.shape[0]; P2 = 2 * spec.P
Xp = torch.zeros(1, spec.K, device=dev); Xp[:, :Kall] = X
cap = {}; model.model.layers[nL - 1].register_forward_hook(lambda m, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
cache = DynamicCache(config=model.config); pos = 0; ys = []
rows = torch.arange(P2, device=dev); torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    fr = spec.init_frames().float().to(dev)
    def run(fr):
        global pos
        n = fr.shape[0]; cp = torch.arange(pos, pos + n, device=dev); pos += n
        model.model(inputs_embeds=fr[None], past_key_values=cache, use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
        return cap['o'][0]
    run(fr)
    for b0 in range(0, nOB, segb):
        b1 = min(nOB, b0 + segb); parts = []
        for b in range(b0, b1):
            parts += [fa.write(pay[b], rows), fa.read(Xp)]
        out = run(torch.cat(parts))
        for i in range(b1 - b0): ys.append(out[i * (P2 + 1) + P2][spec.Y].double())
torch.cuda.synchronize(); t_run = time.time() - t0
y = torch.cat(ys)[:Oall]; ref = (X.double().cpu() @ W.double().T)[0].to(dev)
nfr = nOB * (P2 + 1) + nL
res = dict(tensor=key, shape=[Oall, Kall], nL=nL, D=lay.n, blocks=nOB, frames=nfr, seg_blocks=segb, t_build_payload_s=t_build, t_run_s=t_run, frames_per_s=nfr / t_run,
           maxrel_vs_maxref=((y - ref).abs().max() / ref.abs().max()).item(), relL2=((y - ref).norm() / ref.norm()).item(), weights=W.numel(),
           peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, payload_gb=pay.numel() * 4 / 1e9)
print(json.dumps(res)); open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results.jsonl'), 'a').write(json.dumps(res) + '\n')
