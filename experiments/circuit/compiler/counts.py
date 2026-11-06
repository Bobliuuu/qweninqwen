"""Builds RESULTS.md from the measured res_full_*.json (seconds/token at Qwenputer trace rates)."""
import json, os, glob
H = os.path.dirname(os.path.abspath(__file__)); L = lambda f: json.load(open(f'{H}/{f}'))
def secs(i): return i * 1.19 / 5500, i * 1.40 / 5500, i * 1.40 / 3200
rows = []
def add(label, ins, ctx, note=''): rows.append((label, ctx, ins, *secs(ins), note))
tiny = L('res_tiny.json')[0]; mac_tiny = 81920   # 2 layers d64 + lm head 128x64
f32_per_mac = tiny['instr_per_pos'][0] / mac_tiny; full_macs = 595984384
if os.path.exists(f'{H}/res_full_f32.json'):
    r = L('res_full_f32.json'); add('T0a fp32 soft-float llama2.c (MEASURED)', r['instr_per_pos'][0], 1)
else: add('T0a fp32 soft-float llama2.c (EXTRAPOLATED from tiny: %.0f instr/MAC)' % f32_per_mac, f32_per_mac * full_macs, 1, 'measurement running; replaced when res_full_f32.json lands')
q = L('res_full_q4v.json'); add('v1 int4 VDOT + soft-float glue', q['instr_per_pos'][0], 1)
for wb, ex in ((8, 0), (8, 1), (4, 0), (4, 1)):
    f = f'res_full_fx_w{wb}_e{ex}_n4.json'
    if os.path.exists(f'{H}/{f}'):
        r = L(f); add(f'v2 fixed-point int{wb} weights, {"custom-1 EXT" if ex else "plain RV32IM"} nonlinears', r['instr_per_pos'][0], 1)
r = L('res_full_fx_w4_e1_n130.json'); ip = r['instr_per_pos']
add('v2 int4 EXT (MEASURED at position 129)', ip[-1], 130, 'per-context-token slope %.2e instr' % ((ip[-1] - ip[0]) / 129))
out = ['# Qwen3-0.6B as RV32IM firmware on the Qwenputer CPU: instructions and seconds per token', '',
       'Instruction counts: `fw/rvemu.c` on real Qwen3-0.6B weights (`run_full.py`, `run_full_fx.py`). Seconds = instr x trace-tokens/instr / trace-tokens/s.',
       'Columns: best = 1.19 tok/instr @5500/s; mid = 1.40 @5500/s (Qwenputer fused executor on an L4); worst = 1.40 @3200/s. VDOT results carry 8 products per instruction.', '',
       '| build | ctx | instr/token | s/token best | mid | worst | note |', '|---|---|---|---|---|---|---|']
for l, c, i, a, b, w, n in rows: out.append(f'| {l} | {c} | {i:.3e} | {a:.3e} | {b:.3e} | {w:.3e} | {n} |')
open(f'{H}/RESULTS.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
