"""Collect CF chunk results from Modal volume qfr-out (/cf/chunkNNNN/result_*.json) into ../logs/cf/ and print the per-chunk table with the reference-side blob-injection counts (/tmp/cfx/ref785cf/stats_*.json copies in ../logs/cf/ref_stats)."""
import subprocess, json, os, glob
H = os.path.dirname(os.path.abspath(__file__)); dst = H + '/../logs/cf/chunks'; os.makedirs(dst, exist_ok=True)
ls = subprocess.run(['modal', 'volume', 'ls', 'qfr-out', '/cf'], capture_output=True, text=True).stdout.split()
names = sorted({x.split('/')[-1] for x in ls if x.startswith('cf/chunk')})
for n in names:
    if n.endswith('.stdout'):
        if not os.path.exists(f'{dst}/{n}'): subprocess.run(['modal', 'volume', 'get', 'qfr-out', f'/cf/{n}', f'{dst}/{n}'], capture_output=True)
    elif not glob.glob(f'{dst}/{n}/result_*.json'):
        os.makedirs(f'{dst}/{n}', exist_ok=True); subprocess.run(['modal', 'volume', 'get', 'qfr-out', f'/cf/{n}/', dst + '/'], capture_output=True)
per = {}
for p in sorted(glob.glob(dst + '/chunk*/result_*.json')):
    if '_neg' in p: continue
    r = json.load(open(p))
    for k, ch in enumerate(r['chunks']):
        per[ch] = dict(steps=r['steps_per_row'][k], free=r['free_tokens'][k], mism=r['free_mismatches'][k], first=r['first_free_mismatch'][k], ram_inj=r['ram_injections'][k], gpu=r['gpu'], total_inj=r['injections'][k])
for ch in sorted(per):
    sp = f'{H}/../logs/cf/ref_stats/stats_{ch:04d}.json'; st = json.load(open(sp)) if os.path.exists(sp) else {}
    per[ch]['ref_blob_loads_injected'] = st.get('blob_loads_injected'); per[ch]['ref_ram_loads_injected'] = st.get('ram_loads_injected')
    print(ch, *per[ch].values())
print('chunks', len(per), 'free tokens', sum(v['free'] for v in per.values()), 'free mismatches', sum(v['mism'] for v in per.values()), 'chunks with mismatch', sum(1 for v in per.values() if v['mism']),
      'blob inj (ref)', sum(v['ref_blob_loads_injected'] or 0 for v in per.values()), 'circuit type-1 inj', sum(v['ram_inj'] for v in per.values()))
json.dump({str(k): v for k, v in per.items()}, open(H + '/../logs/cf/summary.json', 'w'), indent=1)
