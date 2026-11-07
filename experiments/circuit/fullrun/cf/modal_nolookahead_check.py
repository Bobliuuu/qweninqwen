"""FIX H check: CF+RAM chunk 0 from the PROGRAM ALONE with the reference look-ahead removed: --fulltable (whole row table, not the slice the reference trace touches) and --stop-on-halt (stop on the circuit's own HALT token). Same as modal_reset0_check.py otherwise. Uses circuit_run_h.py (uploaded next to circuit_run.py so running jobs are untouched).
  modal volume put -f qfr-data circuit_run.py /circuit_wfram/fullrun/circuit_run_h.py ; modal run --detach cf/modal_nolookahead_check.py --steps 300000"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-cfram-nolookahead')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)

@app.function(image=img, gpu='A100-40GB', timeout=3600, memory=65536, volumes={'/data': data, '/out': out}, retries=0)
def run(steps=300000, chunks='0', tag='nolookahead_chain', flags='--fulltable --stop-on-halt'):
    data.reload(); o = f'/out/cfram_nolookahead/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840')
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run_h.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', chunks, '--ramimg', '--reset0', '--chain', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy', '--steps', str(steps)] + flags.split()
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(rc=r.returncode, tail=(r.stdout[-2500:] + r.stderr[-2500:]))

@app.local_entrypoint()
def main(steps: int = 300000, chunks: str = '0', tag: str = 'nolookahead_chain', flags: str = '--fulltable --stop-on-halt'): print(json.dumps(run.remote(steps, chunks, tag, flags), indent=1))
