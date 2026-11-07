"""Modal runner for the real-decode check (1 A100, detached). Same as cf/modal_wf_cfram.py (per chunk, --reset0 --ramimg) but through run_realdecode.py.
  modal run --detach modal_realdecode.py --chunks 0,1,2                       (CF+RAM, real decode; expect 0 mismatches, 0 injections)
  modal run --detach modal_realdecode.py --chunks 0 --neg --steps 600000     (negative control: firmware WITH blob loads (WF+RAM ramwf.elf); real decode must NOT hide them)
Needs on qfr-data: /circuit_wf/ramown/realdecode/run_realdecode.py (modal volume put) and the existing cfram/wfram inputs. Outputs qfr-out /realdecode/."""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-realdecode')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)

@app.function(gpu='A100-40GB', image=img, timeout=7200, volumes={'/data': data, '/out': out}, retries=0)
def run(chs, neg=False, steps=0):
    data.reload(); os.makedirs('/out/realdecode', exist_ok=True)
    o = f'/out/realdecode/{"neg_" if neg else ""}chunk{int(chs.split(",")[0]):04d}_b{len(chs.split(","))}'
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840', CIRCUIT_RUN='/data/circuit_wfram/fullrun/circuit_run.py')
    elf, ref, tab = ('/data/wfram/ramwf.elf', '/data/wfram/refRW', '/data/rowtab.npy') if neg else ('/data/cfram/ramcf.elf', '/data/cfram/refRC', '/data/rowtab_cf.npy')
    cmd = ['python', '-u', '/data/circuit_wf/ramown/realdecode/run_realdecode.py', elf, ref, o, '--chunks', chs, '--ramimg', '--reset0', '--block', '100000', '--rowtab', tab] + (['--steps', str(steps)] if steps else [])
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(chunks=chs, neg=neg, rc=r.returncode, tail=r.stdout[-2500:] + r.stderr[-2500:])

@app.local_entrypoint()
def main(chunks: str = '0,1,2', neg: bool = False, steps: int = 0):
    print(json.dumps(run.remote(chunks, neg, steps)), flush=True)
