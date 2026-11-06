"""Spawn BATCHED WF groups (several chunks as batch rows on one GPU) on deployed Modal app qwen-in-qwen-wf once their reference chunks are synced (log lines 'synced NNNN')."""
import modal, time, json, sys
log = '/tmp/syncwf.log'; gpu = sys.argv[1]; groups = [[int(x) for x in g.split(',')] for g in sys.argv[2:]]
f = modal.Function.from_name('qwen-in-qwen-wf', gpu); ids = {}
for g in groups:
    while f'synced {g[-1]:04d}' not in open(log).read(): time.sleep(15)
    c = f.spawn(','.join(map(str, g))); ids[g[0]] = (c.object_id, gpu, g); print(g, c.object_id, flush=True)
    json.dump(ids, open(f'/tmp/wf_groups_{gpu}.json', 'w'))
