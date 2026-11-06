"""Weight-fetch full-token run on Modal GPUs: VDOT weights from the outer model's own embedding rows (ROW tokens), no weight injection.
Volumes: qfr-data (inputs: /qpriscv_wf = Qwenputer riscv tree + weight_fetch/wf_npu.patch, /circuit_wf = this repo's experiments/circuit, /w785wf/ref.elf = WF firmware,
/ref785wf = WF reference trace (ref_big.py WF=1), /w785/wrom.bin -> /rowtab.npy (built by `modal run modal_wf.py::mktab`)); qfr-out (outputs, /out/wf/chunkNNNN).
  modal run --detach modal_wf.py --chunks 0-0 --gpu A10G          (smoke)
  modal run --detach modal_wf.py --chunks 0-77 --gpu A10G
"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-wf')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=14400, volumes={'/data': data, '/out': out}, retries=0)

@app.function(image=img, timeout=3600, volumes={'/data': data}, memory=16384)
def mktab():
    import sys; sys.path.insert(0, '/data/circuit_wf/fullrun')
    import numpy as np; from wf_table import build_table
    t = build_table(open('/data/w785/wrom.bin', 'rb').read()); np.save('/data/rowtab.npy', t); data.commit()
    return dict(shape=t.shape, nbytes=int(t.nbytes), md5_first_rows=__import__('hashlib').md5(t[:64].tobytes()).hexdigest())

def _run(ch, extra, tag='', extra_env=()):
    os.makedirs('/out/wf', exist_ok=True); chs = str(ch); first = int(chs.split(',')[0]); o = f'/out/wf/chunk{first:04d}' + (f'_b{len(chs.split(","))}' if ',' in chs else '') + tag   # ch: one chunk or a comma list (batched rows on one GPU)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', WF_ROW_CELLS='3840'); env.update(dict(x.split('=') for x in extra_env))
    cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_run.py', '/data/w785wf/ref.elf', '/data/ref785wf', o, '--chunks', chs, '--block', '100000', '--rowtab', '/data/rowtab.npy'] + extra
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(chunk=chs, rc=r.returncode, tail=(r.stdout[-1800:] + r.stderr[-1800:]))

@app.function(gpu='A10G', **KW)
def a10g(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='L40S', **KW)
def l40s(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='A100-40GB', **KW)
def a100(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
FN = dict(A10G=a10g, L40S=l40s, A100=a100)

@app.local_entrypoint()
def main(chunks: str = '0-0', gpu: str = 'A10G', steps: int = 0, neg: bool = False):
    a, b = chunks.split('-') if '-' in chunks else (chunks, chunks)
    cl = list(range(int(a), int(b) + 1)); extra = ('--steps', str(steps)) if steps else ()
    tag, ee = ('_neg', ('WF_NEG=1',)) if neg else ('', ())
    for r in FN[gpu].starmap([(c, extra, tag, ee) for c in cl]): print(json.dumps(r)[:900], flush=True)
