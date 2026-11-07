"""Modal runner for hfcheck_sampled.py (new file; does not touch modal_hfwf.py). Needs the CF merged checkpoint at qfr-wfhf:/ckpt_cf (modal_hfwf.py::mergecf).
  modal run hfwf/modal_hfsampled.py --plan-only                         (CPU container: print window plan + coverage)
  modal run --detach hfwf/modal_hfsampled.py --worker 0 --nworkers 2    (GPU worker 0; run a second process with --worker 1; max 2 GPUs concurrently)
Output: qfr-wfhf:/logs/sampled_w<i>of<n>.json (+ .log). Stop idle apps when done (skill keep-runs-alive-and-in-repo rule 7)."""
import modal, os, subprocess
app = modal.App('qwen-in-qwen-hfsampled')
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'numpy', 'triton', 'safetensors', 'transformers==5.19.0', 'tokenizers', 'huggingface_hub', 'accelerate')
       .add_local_dir(HERE, '/hfwf', ignore=['__pycache__']))
data = modal.Volume.from_name('qfr-data'); wf = modal.Volume.from_name('qfr-wfhf', create_if_missing=True)
ENV = dict(os.environ, PYTHONPATH='/hfwf:/data/qpriscv_wf', PYTHONUNBUFFERED='1', WF_NROWS='156396', REALMAP='1')
KW = dict(image=img, timeout=24 * 3600, volumes={'/data': data, '/wf': wf}, memory=131072, ephemeral_disk=600 * 1024)

def _run(worker, nworkers, steps, final_steps, nsample, plan_only, chunks):
    data.reload(); wf.reload(); os.makedirs('/wf/logs', exist_ok=True); tag = f'sampled_w{worker}of{nworkers}'; log = f'/wf/logs/{tag}{"_plan" if plan_only else ""}.log'
    cmd = ['python', '-u', 'hfcheck_sampled.py', '/wf/ckpt_cf', '/data/cfram/ramcf.elf', '/data/cfram/refRC', f'/wf/logs/{tag}.json', '--steps', str(steps), '--final-steps', str(final_steps),
           '--nsample', str(nsample), '--worker', str(worker), '--nworkers', str(nworkers)] + (['--plan-only'] if plan_only else []) + (['--chunks', chunks] if chunks else [])
    with open(log, 'a') as f:
        p = subprocess.Popen(cmd, cwd='/hfwf', env=ENV, stdout=f, stderr=subprocess.STDOUT, text=True)
        while p.poll() is None:
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            wf.commit()   # progress readable from outside every 5 min
    wf.commit(); return p.returncode, open(log).read()[-3500:]

@app.function(cpu=4, memory=16384, **{k: v for k, v in KW.items() if k not in ('memory', 'ephemeral_disk')})
def plan(nworkers: int = 2, steps: int = 100000, final_steps: int = 0, nsample: int = 13): return _run(0, nworkers, steps, final_steps, nsample, True, '')
@app.function(gpu=['H100', 'A100-80GB', 'L40S', 'A100-40GB'], **KW)
def gpu(worker: int = 0, nworkers: int = 2, steps: int = 100000, final_steps: int = 0, nsample: int = 13, chunks: str = ''): return _run(worker, nworkers, steps, final_steps, nsample, False, chunks)

@app.local_entrypoint()
def main(worker: int = 0, nworkers: int = 2, steps: int = 100000, final_steps: int = 0, nsample: int = 13, chunks: str = '', plan_only: bool = False):
    print(*(plan.remote(nworkers, steps, final_steps, nsample) if plan_only else gpu.remote(worker, nworkers, steps, final_steps, nsample, chunks)))
