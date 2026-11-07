"""Modal CPU check (no GPU): the table decoded from the merged checkpoint's embed_tokens.weight (ckpt_table.load_rowtab) is bit-identical to /data/cfram_q/cf/rowtab_cf.npy.
Needed because the Vast boxes read the npy (--rowtab) instead of the 10 GB checkpoint shard. run: modal run --detach cf/vast/verify_ckpt_vs_npy.py ; result printed + qfr-out /cfram_q_vast/ckpt_vs_npy.json"""
import modal, json, hashlib, os
app = modal.App('qwen-in-qwen-vast-verify')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'safetensors')
data = modal.Volume.from_name('qfr-data'); wfhf = modal.Volume.from_name('qfr-wfhf'); out = modal.Volume.from_name('qfr-out')

@app.function(image=img, cpu=4, memory=32768, timeout=3600, volumes={'/data': data, '/wf': wfhf, '/out': out})
def check():
    import sys, numpy as np; sys.path.insert(0, '/data/circuit_wfram/fullrun')
    from ckpt_table import load_rowtab
    a = load_rowtab('/wf/ckpt_cf_q'); b = np.load('/data/cfram_q/cf/rowtab_cf.npy')
    r = dict(shape_ckpt=list(a.shape), shape_npy=list(b.shape), dtype=[str(a.dtype), str(b.dtype)], identical=bool(a.shape == b.shape and (a == b).all()), md5_ckpt=hashlib.md5(a.tobytes()).hexdigest(), md5_npy=hashlib.md5(b.tobytes()).hexdigest())
    os.makedirs('/out/cfram_q_vast', exist_ok=True); json.dump(r, open('/out/cfram_q_vast/ckpt_vs_npy.json', 'w')); out.commit(); return r

@app.local_entrypoint()
def main(): print(json.dumps(check.remote()), flush=True)
