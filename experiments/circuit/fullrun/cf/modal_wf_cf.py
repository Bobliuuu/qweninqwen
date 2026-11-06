"""CF (constants-fetch) full-token replay on Modal: weights AND the blob constants come from the outer model's own table rows (ROW tokens). Same executor/circuit as fullrun/modal_wf.py.
Volumes: qfr-data (/circuit_wf = repo experiments/circuit (engine + circuit_run.py), /qpriscv_wf = patched Qwenputer tree, /circuit_cf = this directory, /w785 = blob.bin wrom.bin cfg.h,
/w785cf/ref.elf = CF firmware, /ref785cf = CF reference trace (cf/ref_big_cf.py), /rowtab_cf.npy built by `modal run modal_wf_cf.py::mktab`); qfr-out (/out/cf/chunkNNNN).
  modal run modal_wf_cf.py::mktab                                  (build /rowtab_cf.npy with cf_gen.py on a CPU container)
  modal deploy modal_wf_cf.py ; python launch_cf.py GPU c0,c1,..   (batched rows on one GPU)"""
import modal, subprocess, os, json
app = modal.App('qwen-in-qwen-wfcf')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=14400, volumes={'/data': data, '/out': out}, retries=0)

@app.function(image=img, timeout=3600, volumes={'/data': data}, memory=32768)
def mktab():
    data.reload(); import sys, shutil; sys.path.insert(0, '/data/circuit_cf')
    os.makedirs('/tmp/cfg', exist_ok=True)
    for f in ('cfg.h',): shutil.copy('/data/w785/' + f, '/tmp/cfg/' + f)
    for f in ('blob.bin', 'wrom.bin'): os.symlink('/data/w785/' + f, '/tmp/cfg/' + f)
    r = subprocess.run(['python', '/data/circuit_cf/cf_gen.py', '/tmp/cfg', '/tmp/cfout'], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='/data/circuit_wf/fullrun'))
    shutil.copy('/tmp/cfout/rowtab_cf.npy', '/data/rowtab_cf.npy'); shutil.copy('/tmp/cfout/cf_layout.json', '/data/cf_layout.json'); data.commit()
    return dict(rc=r.returncode, out=r.stdout[-600:], err=r.stderr[-600:])

@app.function(image=img, timeout=3600, volumes={'/data': data}, memory=32768)
def mkneg():
    """NEGATIVE-CONTROL table: every CONSTANT row (emb/rope, per-layer norms+scales, final norm, LM scales) +1 mod 16 in every cell; weight rows untouched."""
    data.reload(); import numpy as np; L = json.load(open('/data/cf_layout.json')); t = np.load('/data/rowtab_cf.npy'); KH, KL = L['KH'], L['KL']
    rows = list(range(KH)) + [r for l in range(28) for r in range(KH + l * (4096 + KL) + 4096, KH + (l + 1) * (4096 + KL))] + list(range(KH + 28 * (4096 + KL), KH + 28 * (4096 + KL) + L['KF'])) + list(range(L['LS_ROW'], L['rows']))
    t[rows] = (t[rows] + 1) & 15; np.save('/data/rowtab_cf_neg.npy', t); data.commit(); return dict(const_rows_flipped=len(rows))

def _run(ch, extra, tag='', extra_env=()):
    data.reload()
    os.makedirs('/out/cf', exist_ok=True); chs = str(ch); first = int(chs.split(',')[0]); o = f'/out/cf/chunk{first:04d}' + (f'_b{len(chs.split(","))}' if ',' in chs else '') + tag
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF='1', WF_ROW_CELLS='3840'); env.update(dict(x.split('=') for x in extra_env))
    cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_run.py', '/data/w785cf/ref.elf', '/data/ref785cf', o, '--chunks', chs, '--block', '100000', '--rowtab', '/data/rowtab_cf.npy'] + extra
    r = subprocess.run(cmd, cwd='/data/qpriscv_wf', env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-300000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(chunk=chs, rc=r.returncode, tail=(r.stdout[-1800:] + r.stderr[-1800:]))

@app.function(gpu='A10G', **KW)
def a10g(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='L40S', **KW)
def l40s(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
@app.function(gpu='A100-40GB', **KW)
def a100(ch, extra=(), tag='', extra_env=()): return _run(ch, list(extra), tag, list(extra_env))
