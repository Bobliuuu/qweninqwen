"""Reference trace for the quantfix CF+RAM firmware (cf/ref_chain_cf.py, Python ref.Machine, CPU only) on a Modal CPU container writing straight to volume qfr-data /cfram_q/refQ (lambda-final's disk is full).
Resumable (ckpt.pkl) and the volume is committed every 5 min, so a restart continues. Inputs from modal_cfq.py::build: /data/cfram_q/{ramcf_q.elf, cfg/{blob.bin,wrom.bin}, cf/rowtab_cf.npy}.
  modal run --detach quantfix/modal_refq.py            (detached: survives the laptop; progress: modal volume get qfr-data cfram_q/refQ/summary.json when done, or app logs)"""
import modal, subprocess, os, json, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); CIRC = os.path.abspath(HERE + '/../..')
app = modal.App('qwen-in-qwen-refq')
img = (modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors').add_local_dir(CIRC, '/root/circuit', ignore=['__pycache__', '*.elf', 'logs', '*.npy', '*.pkl', '*.bin']))
data = modal.Volume.from_name('qfr-data')

@app.function(image=img, cpu=2, memory=32768, timeout=24 * 3600 - 60, volumes={'/data': data}, retries=0)
def ref(root='/data/cfram_q', outname='refQ'):
    data.reload(); out = f'{root}/{outname}'; os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CF_ROWTAB=f'{root}/cf/rowtab_cf.npy', WF='1', WF_ROW_CELLS='3840', PYTHONPATH='/data/qpriscv_wf')
    cmd = ['python', '-u', '/root/circuit/fullrun/cf/ref_chain_cf.py', f'{root}/ramcf_q.elf', f'{root}/cfg/blob.bin', f'{root}/cfg/wrom.bin', out, '/data/qpriscv_wf']
    log = open(out + '/ref.log', 'a'); p = subprocess.Popen(cmd, cwd='/data/qpriscv_wf', env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while p.poll() is None:
        try: p.wait(timeout=300)
        except subprocess.TimeoutExpired: pass
        data.commit(); print(f'{time.time() - t0:.0f}s', open(out + '/ref.log').read()[-300:].replace('\n', ' | '), flush=True)
    data.commit()
    s = open(out + '/summary.json').read() if os.path.exists(out + '/summary.json') else ''
    return dict(rc=p.returncode, summary=s[:3000], tail=open(out + '/ref.log').read()[-2000:])

@app.local_entrypoint()
def main(root: str = '/data/cfram_q', outname: str = 'refQ'):
    print(json.dumps(ref.remote(root, outname))[:4000], flush=True)
