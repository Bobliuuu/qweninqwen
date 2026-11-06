import torch
from engine import silu_inv, silu64
t = torch.cat([torch.linspace(-0.2784, 0, 1000), torch.logspace(-8, 3, 4000)]).double()
u = silu_inv(t); print('silu_inv max abs err', (silu64(u) - t).abs().max().item(), 'rel', ((silu64(u)-t).abs()/t.abs().clamp(min=1e-9)).max().item())
