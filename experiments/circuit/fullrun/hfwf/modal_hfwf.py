"""Modal runner: merge (A100-80GB) -> checkpoint in volume qfr-wfhf -> HF check.
  modal run --detach modal_hfwf.py::merge
  modal run --detach modal_hfwf.py::check --steps 130000 [--neg]
Inputs from volume qfr-data: /qpriscv_wf (patched Qwenputer riscv tree), /w785/wrom.bin, /w785wf/ref.elf, /ref785wf (WF reference trace chunk 0)."""
import modal, os, subprocess, json, shutil
app = modal.App('qwen-in-qwen-hfwf')
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'numpy', 'triton', 'safetensors', 'transformers==5.19.0', 'tokenizers', 'huggingface_hub', 'accelerate')
       .add_local_dir(HERE, '/hfwf', ignore=['__pycache__']))
data = modal.Volume.from_name('qfr-data'); wf = modal.Volume.from_name('qfr-wfhf', create_if_missing=True)
KW = dict(image=img, timeout=24 * 3600, volumes={'/data': data, '/wf': wf}, gpu='A100-80GB', memory=131072, ephemeral_disk=600 * 1024)
ENV = dict(os.environ, PYTHONPATH='/hfwf:/data/qpriscv_wf', PYTHONUNBUFFERED='1')
def sh(cmd, log):
    with open(log, 'a') as f: r = subprocess.run(cmd, cwd='/hfwf', env=ENV, stdout=f, stderr=subprocess.STDOUT, text=True)
    wf.commit(); return r.returncode
@app.function(**KW)
def merge():
    from huggingface_hub import snapshot_download
    data.reload(); os.makedirs('/wf/logs', exist_ok=True)
    src = snapshot_download('Qwen/Qwen3.5-4B'); log = '/wf/logs/merge.log'
    rc = sh(['python', '-u', 'merge_wf.py', src, '/data/w785/wrom.bin', '/wf/ckpt'], log); return rc, open(log).read()[-3000:]
@app.function(**KW)
def check(steps: int = 130000, neg: bool = False, tag: str = ''):
    data.reload(); wf.reload(); os.makedirs('/wf/logs', exist_ok=True); log = f'/wf/logs/check{tag}.log'
    cmd = ['python', '-u', 'check_wf.py', '/wf/ckpt', '/data/w785wf/ref.elf', '/data/ref785wf', f'/wf/logs/check{tag}.json', '--steps', str(steps)] + (['--neg'] if neg else [])
    rc = sh(cmd, log); return rc, open(log).read()[-3000:]
@app.local_entrypoint()
def main(what: str = 'merge', steps: int = 130000, neg: bool = False):
    print(*(merge.remote() if what == 'merge' else check.remote(steps, neg, '_neg' if neg else '')))
