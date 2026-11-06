import os
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qweninqwen"))
from machine import Layout, make_outer
from hostguard import transport
L = Layout(); L.alloc("x", 31)
m = make_outer(L, ["linear_attention", "full_attention"], 64, gdn=dict(hk=2, hv=2, dk=16, dv=16), attn=dict(heads=2, kv=1, hd=16), vocab=50)
for r in range(3):
    e = torch.zeros(1, 4, L.n); e[..., 0] = 1e4
    lg = m(inputs_embeds=e).logits
    t = lg[:, -1].argmax(-1)
x = e @ e.transpose(1,2)
