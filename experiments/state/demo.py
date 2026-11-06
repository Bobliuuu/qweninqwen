"""demo.py: real Qwen2.5-0.5B weights paged through outer GDN recurrent state.  python demo.py [cpu|cuda]
1) full q_proj (896x896) and gate_proj (4864x896) vs torch   2) chained, in-model hand-off  W2(W1 x)   3) (cuda) full LM head 151936x896, argmax check.
Host sends only flags, one-hot addresses, silu-pre-inverted weight payload; activations x0 supplied as the program input (embedding)."""
import sys, torch, time
from run_block import load_w
from tiled import *
import gdn_impl; print('GDN impl:', gdn_impl.use_best() if (len(sys.argv) > 1 and sys.argv[1] == 'cuda') else (gdn_impl.set_impl('torch', 'torch') or 'torch'))
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
torch.manual_seed(0)
def bank(nL, K, G=1):
    lay = Layout(); spec = StateMV(lay, nL=nL, H=16, K=K, G=G)
    m = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8); spec.compile(m); return spec, m.to(dev)
def rel(y, r): return ((y - r).norm() / r.norm()).item()
for nm in ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.gate_proj.weight'):
    W = load_w(nm); spec, m = bank(1, 896); x = torch.randn(1, 896) * 3
    y, st = project(spec, m, W, x, return_stats=True)
    print(nm, tuple(W.shape), 'rel L2 vs fp64 torch:', f"{rel(y, x.double() @ W.double().T):.2e}", f"frames={st['frames']} run={st['run_s']:.3f}s")
if dev == 'cuda':
    W = load_w('model.embed_tokens.weight'); spec, m = bank(1, 896)
    x = W[12345:12346].clone() * 3 + torch.randn(1, 896) * 0.1       # hidden vector close to a vocab row so the argmax is well separated
    y, st = project(spec, m, W, x, return_stats=True); ref = (x.double() @ W.double().T)
    print('LM head 151936x896: rel L2', f"{rel(y, ref):.2e}", 'argmax state/torch', y.argmax().item(), ref.argmax().item(), f"run={st['run_s']:.2f}s frames={st['frames']}")
print('chained in-model hand-off: run python test_chain.py', dev)
