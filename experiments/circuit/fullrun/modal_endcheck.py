"""Parallel-chain proof runner on Modal (firmware-agnostic). Each chunk k runs from snapshot k and is compared at its END, in FULL, with snapshot k+1 (circuit_endcheck.py --endcheck);
0 diffs on every boundary => chain by induction. Also drives true --chain / --handoff runs of the first chunks. Same volumes/pattern as modal_wf.py (qfr-data inputs, qfr-out outputs).
  modal deploy modal_endcheck.py
  python modal_endcheck.py spawn --groups 8 [--nchunks 84] [--gpu l40s] [--elf /data/w785wf/ref.elf --refdir /data/ref785wf --tag wf --env WF=1 --env WF_ROW_CELLS=3840]
  python modal_endcheck.py spawn-chain --mode chain --chunks 0-5 / --mode handoff --chunks 0-3
  python modal_endcheck.py collect --tag wf     (downloads /out/endcheck/<tag>/* and prints the per-boundary table)
Re-run on another build (e.g. zero-injection / REALMAP): same command with --elf/--refdir/--env/--tag for that build (the tool reads whatever injections/latch/bn the refdir snapshots contain)."""
import modal, subprocess, os, json, sys, argparse
app = modal.App('qwen-in-qwen-endcheck')
img = modal.Image.debian_slim(python_version='3.11').pip_install('torch', 'numpy', 'triton', 'safetensors')
data = modal.Volume.from_name('qfr-data'); out = modal.Volume.from_name('qfr-out', create_if_missing=True)
KW = dict(image=img, timeout=14400, volumes={'/data': data, '/out': out}, retries=0)

def _run(chunks, mode, tag, elf, refdir, rowtab, env_extra, extra, tree):
    data.reload(); first = chunks.split(',')[0].split('-')[0]; od = f'/out/endcheck/{tag}'; os.makedirs(od, exist_ok=True)
    o = f'{od}/{mode}_{first}_n{len(chunks.split(","))}' if ',' in chunks else f'{od}/{mode}_{chunks}'
    env = dict(os.environ, PYTHONPATH='.:/data/circuit_wf/engine', WF_ROW_CELLS='3840'); env.update(dict(x.split('=', 1) for x in env_extra))
    flag = {'endcheck': ['--endcheck', '--reset0'], 'chain': ['--chain', '--reset0'], 'handoff': ['--handoff', '--reset0']}[mode]
    cl = chunks
    if '-' in chunks and ',' not in chunks: a, b = chunks.split('-'); cl = ','.join(str(i) for i in range(int(a), int(b) + 1))
    cmd = ['python', '-u', '/data/circuit_wf/fullrun/circuit_endcheck.py', elf, refdir, o, '--chunks', cl, '--block', '100000', '--rowtab', rowtab] + flag + list(extra)
    r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True)
    open(o + '.stdout', 'w').write(r.stdout[-400000:] + '\n--STDERR--\n' + r.stderr[-60000:]); out.commit()
    return dict(chunks=cl, rc=r.returncode, tail=r.stdout[-1500:] + r.stderr[-1500:])

@app.function(gpu='L40S', **KW)
def l40s(chunks, mode='endcheck', tag='wf', elf='/data/w785wf/ref.elf', refdir='/data/ref785wf', rowtab='/data/rowtab.npy', env_extra=('WF=1',), extra=(), tree='/data/qpriscv_wf'): return _run(chunks, mode, tag, elf, refdir, rowtab, env_extra, extra, tree)
@app.function(gpu='A100-40GB', **KW)
def a100(chunks, mode='endcheck', tag='wf', elf='/data/w785wf/ref.elf', refdir='/data/ref785wf', rowtab='/data/rowtab.npy', env_extra=('WF=1',), extra=(), tree='/data/qpriscv_wf'): return _run(chunks, mode, tag, elf, refdir, rowtab, env_extra, extra, tree)
@app.function(gpu='A10G', **KW)
def a10g(chunks, mode='endcheck', tag='wf', elf='/data/w785wf/ref.elf', refdir='/data/ref785wf', rowtab='/data/rowtab.npy', env_extra=('WF=1',), extra=(), tree='/data/qpriscv_wf'): return _run(chunks, mode, tag, elf, refdir, rowtab, env_extra, extra, tree)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd'); p.add_argument('--groups', type=int, default=8); p.add_argument('--nchunks', type=int, default=84); p.add_argument('--first', type=int, default=0)
    p.add_argument('--gpu', default='l40s'); p.add_argument('--elf', default='/data/w785wf/ref.elf'); p.add_argument('--refdir', default='/data/ref785wf'); p.add_argument('--rowtab', default='/data/rowtab.npy')
    p.add_argument('--tag', default='wf'); p.add_argument('--env', action='append', default=None); p.add_argument('--mode', default='chain'); p.add_argument('--chunks', default='0-5'); p.add_argument('--steps', type=int, default=0)
    p.add_argument('--extra', action='append', default=[]); p.add_argument('--tree', default='/data/qpriscv_wf'); a = p.parse_args(); envs = tuple(a.env or ['WF=1'])
    extra = tuple(a.extra) + (('--steps', str(a.steps)) if a.steps else ())
    if a.cmd == 'spawn':
        f = modal.Function.from_name('qwen-in-qwen-endcheck', a.gpu); ids = {}; cl = list(range(a.first, a.first + a.nchunks))
        for i in range(0, len(cl), a.groups):
            g = ','.join(map(str, cl[i:i + a.groups])); c = f.spawn(g, 'endcheck', a.tag, a.elf, a.refdir, a.rowtab, envs, extra, a.tree); ids[g] = c.object_id; print(g, c.object_id, flush=True)
        json.dump(ids, open(f'endcheck_calls_{a.tag}.json', 'w'), indent=1)
    elif a.cmd == 'spawn-chain':
        f = modal.Function.from_name('qwen-in-qwen-endcheck', a.gpu); c = f.spawn(a.chunks, a.mode, a.tag, a.elf, a.refdir, a.rowtab, envs, extra, a.tree); print(a.mode, a.chunks, c.object_id, flush=True)
    elif a.cmd == 'collect':
        subprocess.run(['modal', 'volume', 'get', 'qfr-out', f'endcheck/{a.tag}', f'logs/endcheck/', '--force'])
