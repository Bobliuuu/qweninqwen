"""PARALLEL CF+RAM run on Modal: one A100 job per chunk (75 chunks), FINAL-PATH flags (real in_ram/in_drom decode, whole row table, circuit's own HALT stop), each job saves the circuit's OWN end-of-chunk state.
Job k is seeded with the reference START state of chunk k (state_k.pkl + ramimg_k.npy; chunk 0 from the PROGRAM ALONE with --reset0). The seam collector (cf/collect_seams.py) then checks that job k's circuit end state
EXACTLY equals the seed job k+1 started from; if all 74 seams match, the 75 parallel runs equal one unbroken run by induction (deterministic circuit: same state in => same run out).
Deployed app (functions spawned from a deployed app survive the laptop dying; concurrency capped by max_containers = env CFRAM_MAX_CONTAINERS at DEPLOY time, default 25):
  CFRAM_MAX_CONTAINERS=25 modal deploy cf/modal_cfram_par.py ; python cf/spawn_cfram_parallel.py 0-74 ; python cf/collect_seams.py
  modal app stop -y <id>   (or `modal app stop qwen-in-qwen-cfram-par`) when done: skill keep-runs-alive-and-in-repo rule 7."""
import modal, subprocess, os, json, time
app = modal.App('qwen-in-qwen-cfram-par')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True); wfhf = modal.Volume.from_name('qfr-wfhf')
MAXC = int(os.environ.get('CFRAM_MAX_CONTAINERS', '25'))

@app.function(image=img, gpu='A100-40GB', timeout=3 * 3600, volumes={'/data': data, '/out': out, '/wf': wfhf}, retries=0, max_containers=MAXC)
def chunk_job(k, steps=0, root='cfram_par'):
    data.reload(); wfhf.reload(); o = f'/out/{root}/chunk_{k:04d}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', REALDECODE='1', WF_ROW_CELLS='3840', ROWEMB_CACHE='/data/cfram/rowemb_cf.npy')
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', str(k), '--ramimg', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy', '--rowtab-ckpt', '/wf/ckpt_cf',
           '--chain', '--save-state', '--fulltable', '--stop-on-halt', '--no-host-writes'] + (['--reset0'] if k == 0 else []) + (['--steps', str(steps)] if steps else [])
    t0 = time.time()
    with open(f'{o}/job.stdout', 'w') as f:
        p = subprocess.Popen(cmd, cwd='/data/qpriscv_wf', env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            out.commit()
    res = dict(chunk=k, rc=p.returncode, seconds=time.time() - t0, cmd=' '.join(cmd)); json.dump(res, open(f'{o}/job_done.json', 'w')); out.commit()
    return dict(res, tail=open(f'{o}/job.stdout').read()[-1500:])
