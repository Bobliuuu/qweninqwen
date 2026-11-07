"""Verify that the row table decoded FROM THE MERGED CHECKPOINT (qfr-wfhf:/ckpt_cf embed_tokens.weight, ckpt_table.load_rowtab) is BIT-IDENTICAL to rowtab_cf.npy for all 156,396 rows x 3,840 cells. CPU only.
  modal run cf/modal_ckpt_table.py     -> prints + writes /out/ckpt_table/verify.json (qfr-out)"""
import modal, os, sys, json, time, hashlib
app = modal.App('qwen-in-qwen-ckpt-table')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'safetensors')
data = modal.Volume.from_name('qfr-data'); wf = modal.Volume.from_name('qfr-wfhf'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)

@app.function(image=img, cpu=4, memory=24576, timeout=3600, volumes={'/data': data, '/wf': wf, '/out': out})
def verify(ck='/wf/ckpt_cf', npy='/data/rowtab_cf.npy'):
    import numpy as np
    data.reload(); wf.reload(); sys.path.insert(0, '/data/circuit_wfram/fullrun'); import ckpt_table
    t0 = time.time(); a = ckpt_table.load_rowtab(ck); t_load = time.time() - t0
    b = np.load(npy, mmap_mode='r'); same_shape = a.shape == b.shape; neq = int((a != b).sum()) if same_shape else -1
    r = dict(ckpt=ck, npy=npy, shape_ckpt=list(a.shape), shape_npy=list(b.shape), cells_different=neq, bit_identical=bool(same_shape and neq == 0), load_seconds=round(t_load, 1), md5_ckpt=hashlib.md5(a.tobytes()).hexdigest(), md5_npy=hashlib.md5(np.ascontiguousarray(b).tobytes()).hexdigest())
    os.makedirs('/out/ckpt_table', exist_ok=True); json.dump(r, open('/out/ckpt_table/verify.json', 'w'), indent=1); out.commit(); return r

@app.local_entrypoint()
def main(): print(json.dumps(verify.remote(), indent=1), flush=True)
