"""PURE free-run (circuit_run_pure.py: zero host writes, zero token overrides) on Modal.  Output: qfr-out /pure/<name>/.
  modal run --detach modal_pure.py --name wf_c1 --mode wf --chunks 1 --gpu A100 [--extra "--steps 300000"]
  modal run --detach modal_pure.py --name rm_c012 --mode realmap --chunks 0,1,2 --gpu A100 --extra "--ramimg --reset0"
mode wf: WF firmware (/w785wf), row-table weights, WF=1.   mode realmap: remapped RAM firmware (/ramrc), REALMAP=1."""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-pure')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=14400, volumes={'/data': data, '/out': out}, memory=65536, cpu=8, retries=0)

def _run(name, mode, chunks, extra):
    data.reload(); o = f'/out/pure/{name}'; os.makedirs(o, exist_ok=True)
    if mode == 'wf':
        env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', WF_ROW_CELLS='3840')
        cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_run_pure.py', '/data/w785wf/ref.elf', '/data/ref785wf', o, '--chunks', chunks, '--block', '100000', '--rowtab', '/data/rowtab.npy']; cwd = '/data/qpriscv_wf'
    else:
        env = dict(os.environ, PYTHONPATH='.:/data/circuit/engine', REALMAP='1')
        cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_run_pure.py', '/data/ramrc/ramfw.elf', '/data/ramrc/refRC', o, '--chunks', chunks, '--block', '100000']; cwd = '/data/qpriscv'
    cmd += extra
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    open(o + '/stdout.log', 'w').write('CMD ' + ' '.join(cmd) + '\n' + r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(name=name, rc=r.returncode, tail=(r.stdout[-2500:] + r.stderr[-1500:]))

@app.function(gpu='A100-40GB', **KW)
def a100(name, mode, chunks, extra): return _run(name, mode, chunks, extra)
@app.function(gpu='A10G', **KW)
def a10g(name, mode, chunks, extra): return _run(name, mode, chunks, extra)

@app.local_entrypoint()
def main(name: str, mode: str, chunks: str, gpu: str = 'A100', extra: str = ''):
    print(json.dumps((a100 if gpu == 'A100' else a10g).remote(name, mode, chunks, extra.split())), flush=True)
