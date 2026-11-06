"""python launch_cf.py GPU(a10g|l40s|a100) 0,1,2 [steps]  -- spawn one batched group (chunks as rows on one GPU) on the deployed app qwen-in-qwen-wfcfram; prints the call id."""
import modal, sys, json
f = modal.Function.from_name('qwen-in-qwen-wfcfram', sys.argv[1]); chs = sys.argv[2]; extra = ('--steps', sys.argv[3]) if len(sys.argv) > 3 else ()
c = f.spawn(chs, list(extra)); print(json.dumps(dict(call=c.object_id, gpu=sys.argv[1], chunks=chs)), flush=True)
