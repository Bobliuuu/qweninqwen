"""Integrated tier run: one command, one JSON result.

usage: python run_tier.py --build builds/<name>.json [--out results/<name>.json]

A build manifest names the components and the command that executes one token of the chosen model on the chosen
executor. The command must print a final line `RESULT {json}` with keys:
  model, trace_tokens, instructions, argmax, ref_argmax
The harness adds wall time, derived rates and a verdict, and refuses to call a run measured unless argmax == ref_argmax.
Manifest keys: name, tier, model, components {firmware, isa, executor, memory, lanes}, hardware, cmd (list), cwd.
Heavy builds must be launched on the shared A100 (flock ~/.gpu.lock) or Modal, not on the Mac; wrap that in `cmd`.
"""
import argparse, json, subprocess, time, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    b = json.load(open(a.build))
    t0 = time.time()
    p = subprocess.run(b['cmd'], cwd=b.get('cwd', '.'), capture_output=True, text=True)
    wall = time.time() - t0
    res = None
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith('RESULT '):
            res = json.loads(line[7:]); break
    out = dict(build=b['name'], tier=b.get('tier'), model=b.get('model'), components=b.get('components'),
               hardware=b.get('hardware'), wall_s=wall, returncode=p.returncode, tail=p.stdout[-2000:] + p.stderr[-1000:])
    if res:
        out.update(res)
        out['trace_tok_per_s'] = res['trace_tokens'] / wall
        out['tok_per_instr'] = res['trace_tokens'] / max(res['instructions'], 1)
        out['argmax_match'] = res['argmax'] == res['ref_argmax']
        out['measured'] = bool(out['argmax_match'] and p.returncode == 0)
    else:
        out['measured'] = False
    path = a.out or os.path.join('results', b['name'] + '.json')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps({k: out.get(k) for k in ('build', 'model', 'trace_tokens', 'wall_s', 'trace_tok_per_s', 'argmax_match', 'measured')}))
    sys.exit(0 if out['measured'] else 1)


if __name__ == '__main__':
    main()
