"""Sync 6-bit WF reference chunks lambda -> modal qfr-data and spawn batched L40S groups (8 chunks per GPU) on deployed app qwen-in-qwen-wf6."""
import subprocess, time, json, os, modal
SP = os.environ['SP']; os.makedirs(SP, exist_ok=True)
f = modal.Function.from_name('qwen-in-qwen-wf6', 'l40s'); ids = {}
def ssh(c): return subprocess.run(['ssh', 'lambda-final', c], capture_output=True, text=True).stdout
groups = [list(range(i, min(i + 8, 84))) for i in range(0, 84, 8)]
for g in groups:
    while True:
        if ssh(f'test -f ~/fr2/ref785wf6/state_{g[-1]+1:04d}.pkl -o -f ~/fr2/ref785wf6/summary.json && echo Y').strip() == 'Y': break
        time.sleep(15)
    for c in g:
        for n in (c, c + 1):
            subprocess.run(f'scp -q "lambda-final:~/fr2/ref785wf6/*_{n:04d}.*" {SP}/ 2>/dev/null', shell=True)
        for fn in os.listdir(SP):
            if any(f'_{n:04d}.' in fn for n in (c, c + 1)): subprocess.run(['modal', 'volume', 'put', '-f', 'qfr-data', f'{SP}/{fn}', f'/ref785wf6/{fn}'], capture_output=True)
    h = f.spawn(','.join(map(str, g))); ids[g[0]] = (h.object_id, g); print(g, h.object_id, flush=True)
    json.dump(ids, open(f'{SP}/groups.json', 'w'))
print('ALLSPAWNED', flush=True)
