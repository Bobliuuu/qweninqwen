"""Counts table from res_full_{f32,q4v}.json + res_proj_instr_per_mac.json -> RESULTS.md (T0 seconds/token at Qwenputer trace rates)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); L = lambda f: json.load(open(f'{H}/{f}'))
TPI = (1.19, 1.40); RATES = (3200, 5500)   # trace tokens per instruction (Qwenputer docs); trace tokens/s (HF generation / fused L4)
rows = []
for mode in ('f32', 'q4v'):
    p = f'res_full_{mode}.json'
    if not os.path.exists(f'{H}/{p}'): continue
    r = L(p); ip = r['instr_per_pos']
    # per-pos instr: pos0 includes cold-start; slope from the last two positions = per-context-token attention cost
    base = ip[0]; slope = (ip[-1] - ip[-2]) if len(ip) > 2 else 0
    for ctx in (1, 128, 1024):
        ins = base + slope * (ctx - 1)
        rows.append((mode, ctx, ins, ins * TPI[1] / RATES[1], ins * TPI[1] / RATES[0], ins * TPI[0] / RATES[1]))
out = ['# Qwen3-0.6B on the Qwenputer RV32IM CPU: instructions and seconds per token\n',
       'Measured: instructions by the emulator (`fw/rvemu.c`) on real HF weights (`run_full.py`), positions 0..3. Larger contexts extrapolate the measured per-context-token slope.',
       'Seconds = instr x trace-tokens/instr / trace-tokens/s. Best = 1.19 tok/instr @5500/s; worst = 1.40 @3200/s; mid = 1.40 @5500/s.\n',
       '| build | ctx | instr/token | s/token (1.40 tok/instr, 5500/s) | worst (1.40, 3200/s) | best (1.19, 5500/s) | days (mid) |', '|---|---|---|---|---|---|---|']
for m, c, i, a, b, d in rows: out.append(f'| {m} | {c} | {i:.3e} | {a:.3e} | {b:.3e} | {d:.3e} | {a/86400:.2f} |')
open(f'{H}/RESULTS.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
