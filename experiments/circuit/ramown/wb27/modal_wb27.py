"""WB27 validation (2) on Modal: --chain over chunks 0..K of the legacy reference (ref785t, w785) with WB27=1, plus a short
speed A/B (chunk 0, --steps N, with and without WB27). Same image/volumes/tree as fullrun/modal_fr.py.
  modal volume put qfr-data circuit_run.py /wb27/circuit_run.py ; modal volume put qfr-data wb27.py /wb27/wb27.py
  modal run --detach modal_wb27.py --chunks 0,1,2 --gpu A100
"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-wb27')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=4 * 3600, volumes={'/data': data, '/out': out})


def _run(tag, chunks, extra, wb27):
    o = f'/out/wb27/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit/engine:/data/wb27')
    if wb27: env['WB27'] = '1'
    cmd = ['python', '/data/wb27/circuit_run.py', '/data/w785/ref.elf', '/data/ref785t', o, '--chunks', chunks, '--block', '100000', '--wrom', '/data/w785/wrom.bin'] + list(extra)
    with open(o + '.stdout', 'w') as f:
        p = subprocess.Popen(cmd, cwd='/data/qpriscv', env=env, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired: pass
            out.commit()   # progress durable every 5 min
    out.commit()
    return dict(tag=tag, rc=p.returncode, tail=open(o + '.stdout').read()[-3000:])


@app.function(gpu='A100-40GB', **KW)
def a100(tag, chunks, extra=(), wb27=True): return _run(tag, chunks, extra, wb27)


@app.function(gpu='A10G', **KW)
def a10g(tag, chunks, extra=(), wb27=True): return _run(tag, chunks, extra, wb27)


@app.local_entrypoint()
def main(chunks: str = '0,1,2', gpu: str = 'A100', speed_steps: int = 300000):
    fn = dict(A100=a100, A10G=a10g)[gpu]
    jobs = [(f'chain_{chunks.replace(",", "_")}', chunks, ('--chain',), True),
            ('speed_wb27', '0', ('--steps', str(speed_steps)), True), ('speed_base', '0', ('--steps', str(speed_steps)), False)]
    for r in fn.starmap(jobs): print(json.dumps(r)[:4000], flush=True)
