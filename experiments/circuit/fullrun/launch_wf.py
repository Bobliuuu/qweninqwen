"""Spawn the WF chunks on the deployed Modal app qwen-in-qwen-wf as their reference chunks land on the volume (log /tmp/syncwf.log lines 'synced NNNN'). Calls persist without the client."""
import modal, time, json, sys
log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/syncwf.log'; lo, hi = int(sys.argv[2]), int(sys.argv[3]); L40 = set(int(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []))
fa = modal.Function.from_name('qwen-in-qwen-wf', 'a10g'); fl = modal.Function.from_name('qwen-in-qwen-wf', 'l40s'); ids = {}
for ch in range(lo, hi + 1):
    while f'synced {ch:04d}' not in open(log).read(): time.sleep(15)
    c = (fl if ch in L40 else fa).spawn(ch); ids[ch] = (c.object_id, 'L40S' if ch in L40 else 'A10G'); print(ch, ids[ch], flush=True)
    json.dump(ids, open('/tmp/wf_calls.json', 'w'))
