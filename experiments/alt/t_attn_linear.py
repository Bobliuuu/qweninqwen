"""(a) KV cache as weight memory in the LINEAR (near-uniform softmax) regime: keys +-e_i*s, values +-W[i,:]; q = eps*x.
 o = sum_i (p_i^+ - p_i^-) W_i = sum_i 2 sinh(eps x_i)/Z W_i  ~ (2 eps/Z) W^T x.  Normaliser Z=sum 2cosh(eps x_i) is x dependent => error eps^2|x|^2/(2K).
 Measure with real weights, fp32 (stock F.scaled_dot_product_attention)."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0,'.')
from mlpmul import load_w
W = load_w('model.layers.3.self_attn.q_proj.weight')[:128,:896]   # [O=128 value dims, K=896]
K = W.shape[1]; x = torch.randn(K)*2+.5
ref = (W.double()@x.double())
for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4]:
    keys = torch.cat([torch.eye(K), -torch.eye(K)])           # 2K keys
    vals = torch.cat([W.T, -W.T])                              # [2K, O]
    q = (eps*x)[None,None,None,:]                              # head dim K
    # sdpa applies scale=1/sqrt(hd); set scale=1 to make score=q.k
    o = F.scaled_dot_product_attention(q, keys[None,None], vals[None,None], scale=1.0)[0,0,0]
    Z = (torch.exp(eps*x)+torch.exp(-eps*x)).sum()/ (2*K)       # not available to the machine; only for the oracle
    est = o.double()*K/eps                                     # known constant gain only
    err = ((est-ref).abs().max()/ref.abs().max()).item()
    print(f'eps={eps:g}  err (constant-gain readout) = {err:.2e}   max|eps x|={float((eps*x).abs().max()):.3g}')
