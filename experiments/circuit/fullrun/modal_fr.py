"""Fan the 78 chunks of the full Qwen3-0.6B reference token (w785 firmware, ref785t) out over Modal GPUs.
Each container runs circuit_run.py on ONE chunk from its reference snapshot (same bring-up mode as the lambda chain run), writes to volume qfr-out.
  modal volume create qfr-data; modal volume put qfr-data <modalfr>/ref785t /ref785t ... (see launch notes)
  modal run --detach modal_fr.py --chunks 6-77 --gpu A10G
"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-fr')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=7200, volumes={'/data': data, '/out': out}, retries=1)

def _run(ch, extra):
    o = f'/out/chunk{ch:04d}'
    env = dict(os.environ, PYTHONPATH='.:/data/circuit/engine')
    cmd = ['python', '/data/circuit/fullrun/circuit_run.py', '/data/w785/ref.elf', '/data/ref785t', o, '--chunks', str(ch), '--block', '100000', '--wrom', '/data/w785/wrom.bin'] + extra
    r = subprocess.run(cmd, cwd='/data/qpriscv', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-200000:] + '\n--STDERR--\n' + r.stderr[-50000:]) if os.path.isdir(os.path.dirname(o)) else None
    out.commit()
    return dict(chunk=ch, rc=r.returncode, tail=(r.stdout[-1500:] + r.stderr[-1500:]))

@app.function(gpu='A10G', **KW)
def a10g(ch, extra=()): return _run(ch, list(extra))
@app.function(gpu='L40S', **KW)
def l40s(ch, extra=()): return _run(ch, list(extra))
@app.function(gpu='A100-40GB', **KW)
def a100(ch, extra=()): return _run(ch, list(extra))
FN = dict(A10G=a10g, L40S=l40s, A100=a100)

@app.local_entrypoint()
def main(chunks: str = '6-77', gpu: str = 'A10G', steps: int = 0):
    a, b = chunks.split('-') if '-' in chunks else (chunks, chunks)
    cl = list(range(int(a), int(b) + 1)); extra = ('--steps', str(steps)) if steps else ()
    for r in FN[gpu].starmap([(c, extra) for c in cl]): print(json.dumps(r)[:600], flush=True)
