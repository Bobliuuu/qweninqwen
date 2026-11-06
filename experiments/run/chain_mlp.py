"""First chained result at REAL dims (T2-lite): y = W_down (W_gate x) with real Qwen3-0.6B layer-0 weights, the intermediate (3072) handed
bank1 -> REG -> bank2 IN-MODEL (B's RegFile); host supplies only the layer input x to bank1 and weight frames. usage: chain_mlp.py [layer]"""
import os
import sys, os, time, json, glob, torch
sys.path.insert(0, os.path.expanduser('~/work/qq/trueqiq')); sys.path.insert(0, os.path.expanduser('~/work/qq/trueqiq_state'))
sys.modules['fla'] = None
from engine import *
from transformers import DynamicCache
from safetensors import safe_open
dev = 'cuda'; layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
p = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors')))[0]
with safe_open(p, 'pt') as f:
    Wg = f.get_tensor(f'model.layers.{layer}.mlp.gate_proj.weight').float(); Wd = f.get_tensor(f'model.layers.{layer}.mlp.down_proj.weight').float()
torch.manual_seed(0)
K1, K2 = Wg.shape[1], Wd.shape[1]
lay = Layout()
bank = StateMV(lay, nL=-(-K1 // 1008), H=16, K=K1, name='B1'); bank2 = StateMV(lay, nL=-(-K2 // 1008), H=16, K=K2, name='B2'); reg = RegFile(lay, bank, bank2, nmax=4)
types = ['linear_attention'] * (bank.nL + reg.nL + bank2.nL)
m = make_outer(lay, types, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8)
bank.compile(m, 0); reg.compile(m, bank.nL); bank2.compile(m, bank.nL + reg.nL); m.to(dev).eval(); bank.dev = dev; bank2.dev = dev
D = lay.n; print('D', D, 'layers', len(types), 'nL', bank.nL, reg.nL, bank2.nL, flush=True)
st = dict(cache=DynamicCache(config=m.config), pos=0); cap = {}
m.model.layers[-1].register_forward_hook(lambda mod, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
def push(fr, seg=2048):
    outs = []
    with torch.no_grad():
        for a in range(0, fr.shape[0], seg):
            fb = fr[a:a + seg].float().to(dev); cp = torch.arange(st['pos'], st['pos'] + fb.shape[0], device=dev); st['pos'] += fb.shape[0]
            m.model(inputs_embeds=fb[None], past_key_values=st['cache'], use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
            outs.append(cap['o'][0])
    return torch.cat(outs)
def rnd(bk, W, mk_read, wr=None, rd=None):
    ys = []; t0 = time.time(); nf = 0
    for bi, o0 in enumerate(range(0, W.shape[0], bk.O)):
        rows, pl = bk.tile_weights(W, o0); fr = bk.write_frames(rows, pl).to(dev)
        rf = mk_read().to(dev)
        if wr is not None: reg.add_write(rf, bi, wr)
        if rd is not None: reg.add_read(rf, rd)
        out = push(torch.cat([fr, rf])); nf += fr.shape[0] + 1
        ys.append(out[-1, bk.Y].double())
    torch.cuda.synchronize(); return torch.cat(ys)[:W.shape[0]], time.time() - t0, nf
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
push(torch.cat([bank.init_frames(), bank2.init_frames()]).to(dev))
x0 = torch.randn(K1) * 3
y1, t1, n1 = rnd(bank, Wg, lambda: bank.read_frames(x0[None]), wr=0)
ref1 = Wg.double() @ x0.double(); print('round1 gate_proj x: rel', rel(y1.cpu(), ref1), 'time', t1, 'frames', n1, flush=True)
y2, t2, n2 = rnd(bank2, Wd, lambda: bank2.read_frames(torch.zeros(1, K2)), rd=[0])
ref2 = Wd.double() @ ref1; print('round2 down_proj(gate_proj x) [in-model hand-off, X=0 from host]: rel', rel(y2.cpu(), ref2), 'time', t2, 'frames', n2, flush=True)
print(json.dumps(dict(layer=layer, D=D, rel1=rel(y1.cpu(), ref1), rel2=rel(y2.cpu(), ref2), t=t1 + t2, frames=n1 + n2, argmax_ok=bool(y2.argmax().cpu() == ref2.argmax()))))
