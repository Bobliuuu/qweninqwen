"""Outer-step throughput: frames/s for (a) one cached single-frame call (decode-style step), (b) long chunked/recurrent streams.
usage: python bench_steps.py [cpu|cuda] [impl]"""
import sys, time, json, torch
from engine import *
from tiled import DynamicCache
import gdn_impl
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
impl = sys.argv[2] if len(sys.argv) > 2 else ('torch' if dev == 'cpu' else 'fla_rec')
gdn_impl.set_impl(impl, impl)
res = {}
for nL, G in [(1, 1), (2, 2), (5, 1)]:
    lay = Layout(); spec = StateMV(lay, nL=nL, H=16, K=min(nL // G * 16 * 63, 896 if nL == 1 else 4864) if nL != 5 else 4864, B=1e6, G=G)
    m = make_outer(lay, ['linear_attention'] * nL, 8, gdn=dict(hk=16, hv=16, dk=128, dv=128), vocab=8); spec.compile(m); m.to(dev)
    spec.dev = dev; D = lay.n
    def run(n, seg):
        cache = DynamicCache(config=m.config); fr = torch.randn(n, D, device=dev) * 0.01; fr[:, 0] = 1e6
        for rep in range(2):
            if dev == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); pos = 0
            with torch.no_grad():
                for a in range(0, n, seg):
                    fb = fr[a:a + seg]; cp = torch.arange(pos, pos + fb.shape[0], device=dev); pos += fb.shape[0]
                    m.model(inputs_embeds=fb[None], past_key_values=cache, use_cache=True, attention_mask={'full_attention': None, 'linear_attention': None}, cache_position=cp, position_ids=cp[None])
            if dev == 'cuda': torch.cuda.synchronize()
            dt = time.time() - t0
        return n / dt
    n1 = 40 if dev == 'cpu' else 200
    r = dict(D=D, nL=nL, G=G, single_frame_steps_per_s=run(n1, 1), stream_frames_per_s=run(512 if dev == 'cpu' else 4096, 512 if dev == 'cpu' else 2048))
    r['unique_weights_per_frame'] = 1008 * nL // (1 if nL != 5 else 1)
    r['state_MB'] = nL * 16 * 128 * 128 * 4 / 1e6
    print(r); res[f'nL{nL}_G{G}'] = r
json.dump(res, open(f'results_steps_{dev}.json', 'w'), indent=1)
