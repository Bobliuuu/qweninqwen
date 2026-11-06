"""Collect /out/endcheck/<tag>/endcheck_*.json (after `modal volume get qfr-out endcheck/<tag> logs/endcheck/`) into the per-boundary diff table (markdown + json).
usage: python collect_endcheck.py logs/endcheck/<tag> [out.md]"""
import json, glob, sys, os
d = sys.argv[1]; rows = []
for f in sorted(glob.glob(f'{d}/endcheck_*.json')): rows += json.load(open(f))['boundaries']
rows.sort(key=lambda r: r['chunk']); L = ['| boundary | steps | regs rd/rs1/rs2 bad (of 31) | pc anchor / n | wb15 | bn bad (of 1024) | latch words bad (of 8) | RAM bad (of 8192) | free-token mismatches | OK |', '|---|---|---|---|---|---|---|---|---|---|']
nb = nok = 0
for r in rows:
    if r.get('no_next_snapshot'): L.append(f"| {r['boundary']} (last chunk, no snapshot) | | | {r['anchor']}/{r['n']} | | | | | | n/a |"); continue
    nb += 1; nok += bool(r['ok_full'])
    L.append(f"| {r['boundary']} | {r['step']:,} | {r['regs_rd_bad_n']}/{r['regs_rs1_bad']}/{r['regs_rs2_bad']} | {'=' if r['pc_ok'] else 'DIFF'} {r['anchor'][0]}/{r['n'][0]} | {'=' if r['wb15_ok'] else 'DIFF'} | {r.get('bn_bad')} | {r.get('latch_bad_words')} | {r['ram_full_bad']} | {r['free_mismatches']}/{r['free_tokens']:,} | {'yes' if r['ok_full'] else 'NO'} |")
L.append(f'\n{nok} of {nb} boundaries have 0 diffs in every compared field.')
txt = '\n'.join(L); print(txt)
if len(sys.argv) > 2: open(sys.argv[2], 'w').write(txt + '\n')
json.dump(rows, open(f'{d}/all_boundaries.json', 'w'), indent=1, default=str)
