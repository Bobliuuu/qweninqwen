"""CF+RAM chunk 0 from the PROGRAM ALONE (--reset0, no reference state_0000/ramimg_0000), --chain, ~N steps, one A100. Output /out/cfram_reset0/<tag>.
  modal run --detach cf/modal_reset0_check.py --steps 300000"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-cfram-reset0')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)

@app.function(image=img, gpu='A100-40GB', timeout=3600, volumes={'/data': data, '/out': out}, retries=0)
def run(steps=300000, chunks='0', tag='reset0_chain'):
    data.reload(); o = f'/out/cfram_reset0/{tag}'; os.makedirs(o, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840')
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', chunks, '--ramimg', '--reset0', '--chain', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy', '--steps', str(steps)]
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(rc=r.returncode, tail=(r.stdout[-2500:] + r.stderr[-2500:]))

@app.local_entrypoint()
def main(steps: int = 300000, chunks: str = '0', tag: str = 'reset0_chain'): print(json.dumps(run.remote(steps, chunks, tag), indent=1))
