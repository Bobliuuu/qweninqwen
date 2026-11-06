"""Modal launcher for experiments/circuit/algo.  modal run --detach experiments/circuit/algo/modal_algo.py::main --job quant|tps|probe [--gpu H100]
Results print as 'ALGOJSON {...}' and are written to Volume qweninqwen-algo:/out."""
import modal, os, json, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
app = modal.App('qweninqwen-algo')
vol = modal.Volume.from_name('qweninqwen-algo', create_if_missing=True); hf = modal.Volume.from_name('qweninqwen-hf', create_if_missing=True)
ign = ['**/__pycache__/**', '**/.git/**', '**/*.json', '**/*.log']
base = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate', 'transformers>=5.2')
img = (base.add_local_dir(HERE, '/repo/algo', ignore=ign).add_local_dir(REPO + '/reference/nested-qwen-lab/qwentpu', '/repo/lab/qwentpu', ignore=ign))
KW = dict(timeout=4 * 3600, volumes={'/out': vol, '/hf': hf}, image=img)

def _go(job, args):
    os.environ['HF_HOME'] = '/hf'; os.environ['QWENTPU_LAB'] = '/repo/lab'; sys.path.insert(0, '/repo/algo')
    if job == 'probe': import depth_probe as m; r = m.main(**args)
    elif job == 'quant': import quant_study as m; r = m.run(**args)
    elif job == 'tps': import tps_bench as m; r = m.run(**args)
    elif job == 'hybrid': import tps_bench as m; r = m.run(**args)
    r['job'] = job; print('ALGOJSON ' + json.dumps(r), flush=True)
    os.makedirs('/out', exist_ok=True); json.dump(r, open(f'/out/{job}_{time.strftime("%H%M%S")}.json', 'w'), indent=1); vol.commit(); return r

@app.function(gpu='H100', **KW)
def job_h100(job, args): return _go(job, args)

@app.function(gpu='A100-80GB', **KW)
def job_a100(job, args): return _go(job, args)

@app.function(**KW)
def job_cpu(job, args): return _go(job, args)

@app.local_entrypoint()
def main(job: str = 'probe', args: str = '{}', gpu: str = 'H100'):
    f = {'H100': job_h100, 'A100': job_a100, 'cpu': job_cpu}[gpu]
    print('DONE', json.dumps(f.remote(job, json.loads(args)))[:3000])
