"""Chained circuit runs on Modal for the REMAPPED firmware (REALMAP=1, real in_ram, circuit-owned RAM), fed by the reference dir produced by ramown/ref_chain.py.
Volume qfr-data holds: /ramrc/refRC (reference dir), /ramrc/ramfw.elf, /w785/wrom.bin, /circuit2/fullrun/circuit_run.py (this repo's version), /circuit/engine, /qpriscv.
Output: volume qfr-out /chain/<name>/ (log streamed, chain_partial.json at every chunk boundary, final json).
  modal run --detach modal_chain.py --name seg0 --chunks 0-7 --gpu A100 [--extra "--reset0 --ramimg"]
  modal run --detach modal_chain.py --name full --chunks 0-66 --gpu H100 ...
One call = ONE continuous circuit chain (no snapshot restore between chunks)."""
import modal, subprocess, os, json, threading, time
app = modal.App('qwen-in-qwen-chain')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=86400, volumes={'/data': data, '/out': out}, memory=65536, cpu=8)

def _run(name, chunks, extra):
    o = f'/out/chain/{name}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit/engine', REALMAP='1')
    cmd = ['python', '-u', '/data/circuit2/fullrun/circuit_run.py', '/data/ramrc/ramfw.elf', '/data/ramrc/refRC', o, '--chunks', chunks, '--chain', '--block', '100000', '--wrom', '/data/w785/wrom.bin'] + extra
    lf = open(o + '/stdout.log', 'a'); lf.write('CMD ' + ' '.join(cmd) + '\n'); lf.flush()
    stop = threading.Event()
    def committer():
        while not stop.wait(180): lf.flush(); out.commit()
    th = threading.Thread(target=committer, daemon=True); th.start()
    r = subprocess.run(cmd, cwd='/data/qpriscv', env=env, stdout=lf, stderr=subprocess.STDOUT)
    stop.set(); lf.flush(); out.commit()
    return dict(name=name, rc=r.returncode)

@app.function(gpu='A100-40GB', **KW)
def a100(name, chunks, extra): return _run(name, chunks, extra)
@app.function(gpu='H100', **KW)
def h100(name, chunks, extra): return _run(name, chunks, extra)
@app.function(gpu='A10G', **KW)
def a10g(name, chunks, extra): return _run(name, chunks, extra)
FN = dict(A100=a100, H100=h100, A10G=a10g)

@app.local_entrypoint()
def main(name: str, chunks: str, gpu: str = 'A100', extra: str = ''):
    a, b = chunks.split('-') if '-' in chunks else (chunks, chunks)
    cl = ','.join(str(i) for i in range(int(a), int(b) + 1))
    print(json.dumps(FN[gpu].remote(name, cl, extra.split())), flush=True)
