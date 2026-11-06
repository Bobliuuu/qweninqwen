"""Projected seconds/token for Qwen3-0.6B from the measured tok/MAC (res_kernels.json): 440M layer MACs + 156M LM-head MACs.
tok/s default 5,562 (Qwenputer fused executor, L4, stories260K; chip depth 11 layers).  Extra chip layers scale step time ~linearly.
Nonlinear glue (RMSNorm/RoPE/softmax/SiLU/attention, see experiments/circuit/nonlinear) is NOT included."""
import json, sys
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 5562.0
LAYER_MACS, HEAD_MACS = 440e6, 156e6
rows = json.load(open('res_kernels.json'))
g = lambda sub, dist='real': next(r for r in rows if sub in r['name'] and r['dist'] == dist)['tok_per_mac']
epi = g('K6') - g('RB=20 C=2')                    # fused requant/clamp/pack epilogue, tokens per MAC (K=1024 rows)
plan = [   # name, layer tok/MAC, head tok/MAC, chip layers (11 now)
 ('T0  plain C int8 (K0)',                         g('K0'), g('K0'), 11),
 ('T0+ plain unrolled int8 (K1)',                  g('K1 '), g('K1 '), 11),
 ('T0++ plain, 2 rows/mul lane trick (K1b)',       g('K1b'), g('K1b'), 11),
 ('SMAQA int8 (K2)',                               g('K2'), g('K2'), 11),
 ('VDOT existing compiler structure (K5)',         g('K5'), g('K5'), 11),
 ('VDOT blocked RB=22 C=2 + requant (K6) / argmax (K7)', g('K6'), g('K7'), 11),
 ('PROPOSED VDOT16 W4A4 (K9) +requant, 11-12 layers', g('K9') + epi, g('K9') + 0.002, 12),
 ('PROPOSED VDOTA W4A8 acts-from-RAM (K8) +requant, 14 layers', g('K8') + epi, g('K8') + 0.002, 14),
 ('PROPOSED VDOTA16 W4A8 16-wide from RAM (K10) +requant, 14 layers', g('K10') + epi, g('K10') + 0.002, 14),
]
out = []
print(f'rate {RATE:.0f} tok/s at 11 chip layers; epilogue {epi:.4f} tok/MAC')
print(f"{'build':70s} {'tok/MAC':>8s} {'tokens/token':>13s} {'s/token':>10s}  {'h/token':>8s}")
for n, a, b, L in plan:
    toks = LAYER_MACS * a + HEAD_MACS * b
    sec = toks / (RATE * 11 / L)
    out.append(dict(build=n, layer_tok_per_mac=a, head_tok_per_mac=b, chip_layers=L, tokens_per_token=toks, seconds_per_token=sec))
    print(f'{n:70s} {toks/596e6:8.4f} {toks:13.4g} {sec:10.4g}  {sec/3600:8.3f}')
json.dump(out, open('res_projection.json', 'w'), indent=1)
