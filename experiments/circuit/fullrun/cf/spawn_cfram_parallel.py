"""Spawn the PARALLEL CF+RAM run: one single-chunk job per chunk on the DEPLOYED app qwen-in-qwen-cfram-par (functions spawned from a deployed app survive this laptop dying; results are written to volume qfr-out /cfram_par/chunk_NNNN).
Concurrency is set by max_containers at deploy time (CFRAM_MAX_CONTAINERS, default 25); extra spawns queue on Modal.
usage: CFRAM_MAX_CONTAINERS=25 modal deploy cf/modal_cfram_par.py
       python cf/spawn_cfram_parallel.py 0-74 [--steps N] [--root cfram_par] [--dry]
Writes the FunctionCall ids to ~/qweninqwen-work/cfram_par_out/calls_<root>.json (so a lost client can re-attach: modal.FunctionCall.from_id). NOT launched by anything else; the full 0-74 run needs the user's go."""
import sys, os, json, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument('chunks'); ap.add_argument('--steps', type=int, default=0); ap.add_argument('--root', default='cfram_par'); ap.add_argument('--dry', action='store_true'); a = ap.parse_args()
lo, hi = (int(x) for x in a.chunks.split('-')) if '-' in a.chunks else (int(a.chunks),) * 2
od = os.path.expanduser('~/qweninqwen-work/cfram_par_out'); os.makedirs(od, exist_ok=True)
print(f'spawn chunks {lo}..{hi} ({hi - lo + 1} jobs) root={a.root} steps={a.steps or "all"}')
if a.dry: sys.exit(0)
import modal
f = modal.Function.from_name('qwen-in-qwen-cfram-par', 'chunk_job'); calls = {}
for k in range(lo, hi + 1):
    h = f.spawn(k, a.steps, a.root); calls[k] = h.object_id; print(k, h.object_id, flush=True)
    json.dump(calls, open(f'{od}/calls_{a.root}.json', 'w'))
print('ALLSPAWNED', len(calls), flush=True)
