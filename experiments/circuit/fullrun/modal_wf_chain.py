"""WF + WB27 chained circuit run (one continuous circuit run over consecutive chunks, WF firmware, circuit-held 27-bit WB pointer, latch window and row-id register read back at every boundary).
  modal volume put --force qfr-data fullrun/circuit_run.py /circuit_wf/fullrun/circuit_run.py ; modal volume put --force qfr-data ramown/wb27/wb27.py /circuit_wf/ramown/wb27/wb27.py
  modal run --detach modal_wf_chain.py --chunks 0,1,2,3,4,5 --gpu A100"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-wf-chain')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=4 * 3600, volumes={'/data': data, '/out': out}, retries=0)

def _run(chunks, extra):
    data.reload(); tag = 'chain_' + chunks.replace(',', '_') + ''.join(extra).replace('-', '')
    o = f'/out/wfchain/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', WB27='1', WF_ROW_CELLS='3840')
    cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_run.py', '/data/w785wf/ref.elf', '/data/ref785wf', o, '--chunks', chunks, '--block', '100000', '--rowtab', '/data/rowtab.npy', '--chain'] + list(extra)
    with open(o + '.stdout', 'w') as f:
        p = subprocess.Popen(cmd, cwd='/data/qpriscv_wf', env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            out.commit()
    out.commit()
    return dict(tag=tag, rc=p.returncode, tail=open(o + '.stdout').read()[-3000:])

@app.function(gpu='A100-40GB', **KW)
def a100(chunks, extra=()): return _run(chunks, list(extra))

@app.local_entrypoint()
def main(chunks: str = '0,1,2,3,4,5', reset0: bool = True):
    print(json.dumps(a100.remote(chunks, ('--reset0',) if reset0 else ())), flush=True)
