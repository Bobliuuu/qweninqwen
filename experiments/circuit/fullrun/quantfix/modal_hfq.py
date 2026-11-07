"""Merged Qwen3.5-4B checkpoint for the quantfix CF table (168,684 rows) + hfcheck, mirrors hfwf/modal_hfwf.py::mergecf/checkcf with new paths (nothing existing is overwritten).
  modal run --detach quantfix/modal_hfq.py --what merge                    -> qfr-wfhf:/ckpt_cf_q
  modal run --detach quantfix/modal_hfq.py --what check [--neg] [--steps N] -> qfr-wfhf:/logs/check_cfq{,_neg}.json   (chunk 0 of /data/cfram_q/refQ, program-only start, 0 injections asserted)"""
import modal, os, subprocess, json
HFW = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../hfwf')
app = modal.App('qwen-in-qwen-hfq')
img = (modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'numpy', 'triton', 'safetensors', 'transformers==5.19.0', 'tokenizers', 'huggingface_hub', 'accelerate').add_local_dir(HFW, '/hfwf', ignore=['__pycache__']))
data = modal.Volume.from_name('qfr-data'); wf = modal.Volume.from_name('qfr-wfhf', create_if_missing=True)
KW = dict(image=img, timeout=24 * 3600, volumes={'/data': data, '/wf': wf}, gpu=['H100', 'A100-80GB', 'L40S', 'A100-40GB'], memory=131072, ephemeral_disk=600 * 1024)
NROWS = os.environ.get('CFQ_NROWS', '168684')
ENV = dict(os.environ, PYTHONPATH='/hfwf:/data/qpriscv_wf', PYTHONUNBUFFERED='1', WF_NROWS=NROWS, REALMAP='1')
def sh(cmd, log):
    with open(log, 'a') as f: r = subprocess.run(cmd, cwd='/hfwf', env=ENV, stdout=f, stderr=subprocess.STDOUT, text=True)
    wf.commit(); return r.returncode
@app.function(**KW)
def merge():
    from huggingface_hub import snapshot_download
    data.reload(); os.makedirs('/wf/logs', exist_ok=True); src = snapshot_download('Qwen/Qwen3.5-4B'); log = '/wf/logs/merge_cfq.log'
    rc = sh(['python', '-u', 'merge_wf.py', src, '/data/cfram_q/cf/rowtab_cf.npy', '/wf/ckpt_cf_q'], log); return rc, open(log).read()[-3000:]
@app.function(**KW)
def check(steps: int = 130000, neg: bool = False):
    data.reload(); wf.reload(); os.makedirs('/wf/logs', exist_ok=True); tag = '_neg' if neg else ''; log = f'/wf/logs/check_cfq{tag}.log'
    cmd = ['python', '-u', 'check_wf.py', '/wf/ckpt_cf_q', '/data/cfram_q/ramcf_q.elf', '/data/cfram_q/refQ', f'/wf/logs/check_cfq{tag}.json', '--steps', str(steps)] + (['--neg'] if neg else [])
    rc = sh(cmd, log); return rc, open(log).read()[-3000:]
@app.local_entrypoint()
def main(what: str = 'merge', steps: int = 130000, neg: bool = False):
    print(*(merge.remote() if what == 'merge' else check.remote(steps, neg)))
