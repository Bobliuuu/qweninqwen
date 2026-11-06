"""Modal launcher for the vs2 performance work.  Usage (always detached):
  modal run --detach experiments/vslice/perf/modal_perf.py::main --job micro
  modal run --detach experiments/vslice/perf/modal_perf.py::main --job run --args '{"variant":"base","positions":2,...}'
Results: printed as JSON lines (grep 'PERFJSON') and written to Modal Volume 'qweninqwen-perf' under /out/."""
import modal, os, json, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
app = modal.App('qweninqwen-perf')
vol = modal.Volume.from_name('qweninqwen-perf', create_if_missing=True)
hf = modal.Volume.from_name('qweninqwen-hf', create_if_missing=True)
ign = ['**/__pycache__/**', '**/.git/**', '**/results/**', '**/logs/**', '**/*.json', '**/older/**', '**/reference/**']
img = (modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch', 'numpy', 'safetensors', 'huggingface_hub', 'accelerate', 'transformers>=5.2')
       .add_local_dir(REPO + '/experiments', '/repo/experiments', ignore=ign)
       .add_local_dir(REPO + '/qweninqwen', '/repo/qweninqwen', ignore=ign)
       .add_local_dir(REPO + '/audit', '/repo/audit', ignore=ign))
img_fla = img.pip_install('flash-linear-attention')
KW = dict(timeout=6 * 3600, volumes={'/out': vol, '/hf': hf}, gpu='A100-80GB')

def _go(job, args):
    os.environ['HF_HOME'] = '/hf'
    sys.path.insert(0, '/repo/experiments/vslice/perf'); sys.path.insert(0, '/repo/experiments/vslice')
    import perf_jobs
    res = getattr(perf_jobs, job)(**args)
    res['job'] = job; res['args'] = args
    print('PERFJSON ' + json.dumps(res), flush=True)
    os.makedirs('/out', exist_ok=True)
    name = args.get('variant', job) + '_' + time.strftime('%H%M%S')
    json.dump(res, open(f'/out/{name}.json', 'w'), indent=1); vol.commit()
    return res

@app.function(image=img, **KW)
def job_a100(job, args): return _go(job, args)

@app.function(image=img_fla, **KW)
def job_a100_fla(job, args): return _go(job, args)

@app.local_entrypoint()
def main(job: str = 'micro', args: str = '{}', fla: bool = False):
    a = json.loads(args)
    r = (job_a100_fla if fla else job_a100).remote(job, a)
    print('DONE', json.dumps(r)[:2000])
