"""CF + REMAPPED RAM (REALMAP; circuit-owned 32 KB RAM, ramown firmware port cf/fw_qwen3_ram_cf.c) replay on Modal; derived from modal_wf_cf.py (CF = constants-fetch full-token replay): weights AND the blob constants come from the outer model's own table rows (ROW tokens). Same executor/circuit as fullrun/modal_wf.py.
Volumes: qfr-data (/circuit_wf = repo experiments/circuit (engine + circuit_run.py), /qpriscv_wf = patched Qwenputer tree, /circuit_cf = this directory, /w785 = blob.bin wrom.bin cfg.h,
/w785cf/ref.elf = CF firmware, /ref785cf = CF reference trace (cf/ref_big_cf.py), /rowtab_cf.npy built by `modal run modal_wf_cf.py::mktab`); qfr-out (/out/cfram/chunkNNNN).
  modal run modal_wf_cf.py::mktab                                  (build /rowtab_cf.npy with cf_gen.py on a CPU container)
  modal deploy modal_wf_cf.py ; python launch_cf.py GPU c0,c1,..   (batched rows on one GPU)"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-wfcfram')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=14400, volumes={'/data': data, '/out': out}, retries=0)

def _run(ch, extra, tag='', extra_env=()):
    data.reload()
    os.makedirs('/out/cfram', exist_ok=True); chs = str(ch); first = int(chs.split(',')[0]); o = f'/out/cfram/chunk{first:04d}' + (f'_b{len(chs.split(","))}' if ',' in chs else '') + tag
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', REALMAP='1', WF_ROW_CELLS='3840'); env.update(dict(x.split('=') for x in extra_env))
    cmd = ['python', '-u', '/data/circuit_wfram/fullrun/circuit_run.py', '/data/cfram/ramcf.elf', '/data/cfram/refRC', o, '--chunks', chs, '--ramimg', '--reset0', '--block', '100000', '--rowtab', '/data/rowtab_cf.npy'] + extra
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(chunk=chs, rc=r.returncode, tail=(r.stdout[-1800:] + r.stderr[-1800:]))

@app.function(gpu='A10G', **KW)
def a10g(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='L40S', **KW)
def l40s(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='A100-40GB', **KW)
def a100(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
