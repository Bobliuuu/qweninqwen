"""Download WF chunk results from Modal volume qfr-out (/wf/chunkNNNN/result_*.json + .stdout) into logs/wf/ and print a table."""
import subprocess, json, os, glob, sys
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'wf', 'chunks'); os.makedirs(dst, exist_ok=True)
ls = subprocess.run(['modal', 'volume', 'ls', 'qfr-out', '/wf'], capture_output=True, text=True).stdout.split()
names = sorted({x.split('/')[-1] for x in ls if x.startswith('wf/chunk') and '_neg' not in x})
tot = dict(chunks=0, free=0, mism=0, ram=0); rows = []
for n in names:
    if n.endswith('.stdout'):
        if not os.path.exists(f'{dst}/{n}'): subprocess.run(['modal', 'volume', 'get', 'qfr-out', f'/wf/{n}', f'{dst}/{n}'], capture_output=True)
        continue
    d = f'{dst}/{n}'
    if not glob.glob(d + '/result_*.json'):
        os.makedirs(d, exist_ok=True); subprocess.run(['modal', 'volume', 'get', 'qfr-out', f'/wf/{n}/', dst + '/'], capture_output=True)
per = {}
for p in sorted(glob.glob(dst + '/chunk*/result_*.json')):
    r = json.load(open(p))
    if '_bt' in p: continue   # skip benchmark runs
    for k, ch in enumerate(r['chunks']):
        if r['steps_per_row'][k] < 2_000_000 and ch != 83: continue   # partial (--steps) run
        per[ch] = (r['steps_per_row'][k], r['free_tokens'][k], r['free_mismatches'][k], r['first_free_mismatch'][k], r['ram_injections'][k], r['gpu'], round(r['agg_tok_per_s']), r['rows'])
for ch in sorted(per): print(ch, *per[ch])
print('chunks done', len(per), 'free tokens', sum(v[1] for v in per.values()), 'free mismatches', sum(v[2] for v in per.values()), 'chunks with mismatch', sum(1 for v in per.values() if v[2]))
json.dump({str(k): v for k, v in per.items()}, open(dst + '/../summary.json', 'w'), indent=1)
