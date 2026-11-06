"""pytest-style quick tests (CPU ok):  python -m pytest -q test_engine.py   (or run directly)"""
import torch, math
from engine import *
from tiled import project
import gdn_impl; gdn_impl.set_impl('torch', 'torch')

def mk(nL, H, K, B=1e6, G=1, dev='cpu'):
    lay = Layout(); spec = StateMV(lay, nL=nL, H=H, K=K, B=B, G=G)
    m = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=H, hv=H, dk=128, dv=128), vocab=8); spec.compile(m); return spec, m.to(dev)

def rel(y, r): return ((y - r).norm() / r.norm()).item()

def test_silu_inverse():
    t = torch.cat([torch.linspace(-0.2784, 0, 500), torch.logspace(-8, 3, 2000)]).double()
    assert (silu64(silu_inv(t)) - t).abs().max() < 1e-13

def test_single_tile_exact():
    torch.manual_seed(0); W = torch.randn(126, 63) * 0.05; X = torch.randn(3, 63) * 5
    spec, m = mk(1, 1, 63); y = project(spec, m, W, X)
    assert rel(y, X.double() @ W.double().T) < 3e-6

def test_overwrite_and_resident_reads():
    """second tile overwrites first completely (no stale state); several reads of one resident tile are consistent."""
    torch.manual_seed(1); spec, m = mk(1, 2, 126)
    W = torch.randn(252, 126) * 0.05; X = torch.randn(4, 126)         # two output blocks -> second write overwrites first
    y = project(spec, m, W, X); assert rel(y, X.double() @ W.double().T) < 3e-6

def test_outlier_weights_and_big_activations():
    torch.manual_seed(2); W = torch.randn(126, 126) * 0.05; W[3, 7] = -4.0; W[50, 100] = 3.0; X = torch.randn(2, 126) * 100
    spec, m = mk(1, 2, 126); y = project(spec, m, W, X); assert rel(y, X.double() @ W.double().T) < 3e-6

def test_kloop_accumulation_across_layers_and_groups():
    torch.manual_seed(3); W = torch.randn(300, 1900) * 0.03; X = torch.randn(2, 1900)
    spec, m = mk(4, 16, 1900, G=2)                                      # 2 output groups x 2 K-layers
    y = project(spec, m, W, X); assert rel(y, X.double() @ W.double().T) < 3e-6

def test_low_ballast_fails_as_documented():
    torch.manual_seed(4); W = torch.randn(126, 126) * 0.05; X = torch.randn(2, 126) * 300
    spec, m = mk(1, 2, 126, B=1e4); y = project(spec, m, W, X)
    assert rel(y, X.double() @ W.double().T) > 1e-4          # |x|~3000: transparent-norm scale error visible; B>=1e6 required

if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'): v(); print('ok', k)
