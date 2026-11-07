"""Spawn the PARALLEL final-path run: one single-chunk job per chunk on the DEPLOYED app qwen-in-qwen-cfram-par (functions spawned from a deployed app survive this laptop dying; each job writes to qfr-out /<root>/chunk_NNNN).
EVERY path is a parameter, so the same command runs unchanged on any (firmware, reference, table) triple. Paths: --elf/--refdir/--rowtab/--rowemb are on volume qfr-data (as /data/...), --ckpt on volume qfr-wfhf (as /wf/...), outputs on qfr-out.

  1) CFRAM_MAX_CONTAINERS=25 modal deploy cf/modal_cfram_par.py            (re-deploy after editing it; max parallel A100s = the number)
  2) python cf/spawn_cfram_parallel.py --build-cache --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB     (ONCE per table: ~20 min, one A100; skip if EMB already matches the table - jobs fail fast otherwise)
  3) python cf/spawn_cfram_parallel.py --n N --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB               (spawns chunks 0..N-1, detached)
  4) python cf/collect_seams.py --n N --root R --refvol-dir <refdir relative to qfr-data> --expect-word W                 (ONE verdict; writes logs/cfram_par/seams_*.json/.md)
  5) modal app stop -y <app id>  (skill rule 7)
Current CF+RAM firmware (defaults):  --n 75 --root cfram_par --elf /data/cfram/ramcf.elf --refdir /data/cfram/refRC --ckpt /wf/ckpt_cf --rowtab /data/rowtab_cf.npy --rowemb /data/cfram/rowemb_cf.npy   (collect: --refvol-dir cfram/refRC --expect-word 13)
New firmware, placeholders:          --n <N> --root <R> --elf /data/<fw>/<name>.elf --refdir /data/<fw>/<ref> --ckpt /wf/<ckpt dir> --rowtab /data/<table>.npy --rowemb /data/<fw>/rowemb_<x>.npy   (collect: --refvol-dir <fw>/<ref> --expect-word <emulator token>)
Extra circuit_run.py flags: --extra "--flag value"; extra env: --env K=V (repeatable). Writes FunctionCall ids to ~/qweninqwen-work/cfram_par_out/calls_<root>.json. The full run needs the user's go."""
import sys, os, json, time, argparse, shlex
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=75, help='number of chunks (0..N-1)'); ap.add_argument('--chunks', default='', help='subset a-b (overrides --n)')
ap.add_argument('--root', default='cfram_par'); ap.add_argument('--steps', type=int, default=0)
ap.add_argument('--elf', default='/data/cfram/ramcf.elf'); ap.add_argument('--refdir', default='/data/cfram/refRC'); ap.add_argument('--ckpt', default='/wf/ckpt_cf', help="'' = read the table from --rowtab instead of a checkpoint")
ap.add_argument('--rowtab', default='/data/rowtab_cf.npy'); ap.add_argument('--rowemb', default='/data/cfram/rowemb_cf.npy')
ap.add_argument('--build-cache', action='store_true', help='build the row-embedding cache at --rowemb (one job: chunk 0, 2000 steps) and wait for it'); ap.add_argument('--extra', default=''); ap.add_argument('--env', action='append', default=[])
ap.add_argument('--app', default='qwen-in-qwen-cfram-par'); ap.add_argument('--dry', action='store_true'); a = ap.parse_args()
lo, hi = (int(x) for x in a.chunks.split('-')) if '-' in a.chunks else (0, a.n - 1)
od = os.path.expanduser('~/qweninqwen-work/cfram_par_out'); os.makedirs(od, exist_ok=True)
kw = dict(root=a.root, elf=a.elf, refdir=a.refdir, ckpt=a.ckpt, rowtab=a.rowtab, rowemb=a.rowemb, extra=tuple(shlex.split(a.extra)), env_extra=tuple(a.env))
print(f'{"build-cache" if a.build_cache else f"spawn chunks {lo}..{hi} ({hi - lo + 1} jobs)"} root={a.root} elf={a.elf} refdir={a.refdir} ckpt={a.ckpt or "(npy)"} rowtab={a.rowtab} rowemb={a.rowemb} steps={a.steps or "all"}')
if a.dry: sys.exit(0)
import modal
f = modal.Function.from_name(a.app, 'chunk_job')
if a.build_cache:
    h = f.spawn(0, 2000, a.root + '_cachebuild', write_cache=True, **{k: v for k, v in kw.items() if k != 'root'}); print('cache-build call', h.object_id, flush=True)
    r = h.get(); print(json.dumps({k: r[k] for k in ('rc', 'seconds')}), flush=True); sys.exit(0 if r['rc'] == 0 else 1)
calls = {}
for k in range(lo, hi + 1):
    h = f.spawn(k, a.steps, **kw); calls[k] = h.object_id; print(k, h.object_id, flush=True)
    json.dump(calls, open(f'{od}/calls_{a.root}.json', 'w'))
print('ALLSPAWNED', len(calls), flush=True)
