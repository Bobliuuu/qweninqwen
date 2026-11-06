"""In-model chaining: x1 = W1 x0 (round 1, REG-write), y = W2 x1 (round 2, X slot filled in-model by REG-read; host supplies no activation)
and multi-pass K accumulation (two K-halves summed in-model).  usage: python test_chain.py [cpu|cuda]"""
import sys, torch, time
from engine import *
from tiled import DynamicCache
import gdn_impl; gdn_impl.set_impl('torch', 'torch')
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
torch.manual_seed(0)
from run_block import load_w
Kd = 896
W1 = load_w('model.layers.0.self_attn.q_proj.weight'); W2 = load_w('model.layers.0.self_attn.o_proj.weight')
lay = Layout(); bank = StateMV(lay, nL=1, H=16, K=Kd, name='B1'); bank2 = StateMV(lay, nL=1, H=16, K=Kd, name='B2'); reg = RegFile(lay, bank, bank2, nmax=4)
types = ['linear_attention'] * (bank.nL + reg.nL + bank2.nL)
m = make_outer(lay, types, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8)
bank.compile(m, 0); reg.compile(m, 1); bank2.compile(m, 1 + reg.nL); m.to(dev); bank.dev = dev; bank2.dev = dev
D = lay.n; print('D', D, 'layers', len(types))
state = dict(cache=DynamicCache(config=m.config), pos=0)
def push(fr):
    cap = {}; hk = m.model.layers[-1].register_forward_hook(lambda mod, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
    outs = []
    with torch.no_grad():
        for a in range(0, fr.shape[0], 1024):
            fb = fr[a:a + 1024].float().to(dev); cp = torch.arange(state['pos'], state['pos'] + fb.shape[0], device=dev); state['pos'] += fb.shape[0]
            m.model(inputs_embeds=fb[None], past_key_values=state['cache'], use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
            outs.append(cap['o'][0].cpu())
    hk.remove(); return torch.cat(outs)
def matvec_round(bk, W, x_frames_fn, reg_write_row=None, reg_read_rows=None):
    """one projection on bank bk: for each 126-block: write rows, then a read frame. returns y (host-visible only for VALIDATION)."""
    ys = []
    for bi, o0 in enumerate(range(0, W.shape[0], bk.O)):
        rows, pl = bk.tile_weights(W, o0); fr = bk.write_frames(rows, pl).cpu()
        rf = x_frames_fn(bi)
        if reg_write_row is not None: reg.add_write(rf, bi, reg_write_row)
        if reg_read_rows is not None: reg.add_read(rf, reg_read_rows)
        out = push(torch.cat([fr, rf]))
        ys.append(out[-1, bk.Y].double())
    return torch.cat(ys)[:W.shape[0]]
def rel(a, b): return ((a - b).norm() / b.norm()).item()
push(torch.cat([bank.init_frames().cpu(), bank2.init_frames().cpu()]))
x0 = torch.randn(Kd) * 3
# round 1: host supplies x0 (program input embedding); Y blocks go to register row 0
y1 = matvec_round(bank, W1, lambda bi: bank.read_frames(x0[None]).cpu(), reg_write_row=0)
ref1 = W1.double() @ x0.double(); print('round1 W1 x0 rel', rel(y1, ref1))
# round 2: bank2's X slot is filled IN-MODEL from register row 0 (host frame has no activation)
y2 = matvec_round(bank2, W2, lambda bi: bank2.read_frames(torch.zeros(1, Kd)).cpu(), reg_read_rows=[0])
ref2 = W2.double() @ ref1; print('round2 W2 (W1 x0) rel [in-model hand-off]', rel(y2, ref2))
# multi-pass K accumulation: two producer passes write register rows 1,2; one read of rows [1,2] feeds bank2 with their SUM
xa = torch.randn(Kd) * 3
pa = matvec_round(bank, W1[:126], lambda bi: bank.read_frames(xa[None]).cpu(), reg_write_row=1)
pb = matvec_round(bank, (W1[:126] * 2.0), lambda bi: bank.read_frames(xa[None]).cpu(), reg_write_row=2)
ysum = matvec_round(bank2, W2[:126], lambda bi: bank2.read_frames(torch.zeros(1, Kd)).cpu(), reg_read_rows=[1, 2])
vec = torch.zeros(Kd, dtype=torch.float64); vec[:126] = (W1[:126].double() @ xa.double()) * 3.0
print('multi-row sum (n=2) rel', rel(ysum, W2[:126].double() @ vec))
