import sys, time, torch, glob, json
sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '.'))
from engine import *
from safetensors import safe_open

def load_w(name, model='Qwen2.5-0.5B'):
    import os
    c = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--Qwen--{model}/snapshots/*/model.safetensors')) + [os.path.expanduser('~/work/state/w/q25.safetensors')]
    p = [x for x in c if os.path.exists(x)][0]
    with safe_open(p, 'pt') as f: return f.get_tensor(name).float()

def run_frames(model, frames, cache=None):
    cap = {}
    h = model.model.layers[-1].register_forward_hook(lambda m, i, o: cap.__setitem__('o', o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        model.model(inputs_embeds=frames.float()[None], use_cache=False)
    h.remove(); return cap['o'][0]

def matvec_pass(spec, model, W, o0, X):
    """one output block: INIT, 2P write frames, then reads of X [m,K]. returns y [m, O] (read from the Y slot)."""
    rows, pl = spec.tile_weights(W, o0)
    fr = torch.cat([spec.init_frames(), spec.write_frames(rows, pl), spec.read_frames(X)])
    out = run_frames(model, fr)
    return out[-X.shape[0]:, spec.Y].double(), fr.shape[0]

if __name__ == '__main__':
    B = float(sys.argv[1]) if len(sys.argv) > 1 else 1e4
    nH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    torch.manual_seed(0)
    W = load_w('model.layers.0.self_attn.q_proj.weight')   # 896x896
    lay = Layout(); spec = StateMV(lay, nL=1, H=nH, K=896 if nH == 16 else nH * 63, B=B)
    model = make_outer(lay, ['linear_attention']*spec.nL, 8, gdn=dict(hk=spec.H,hv=spec.H,dk=spec.dk,dv=spec.dv), vocab=8)
    spec.compile(model)
    K = spec.K
    X = torch.randn(3, K) * 3
    X[1] *= 30  # large activations
    y, nf = matvec_pass(spec, model, W[:, :K], 0, X)
    ref = X.double() @ W[:spec.O, :K].double().T
    err = (y - ref).abs().max(1).values / ref.abs().max(1).values
    print('B', B, 'D', lay.n, 'frames', nf, 'maxrel(vs max|ref|)', err.tolist())
    print('elem rel median', ((y-ref).abs()/ref.abs().clamp(min=1e-3)).median().item())
