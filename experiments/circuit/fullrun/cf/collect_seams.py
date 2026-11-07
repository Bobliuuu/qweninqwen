"""Seam collector for the PARALLEL CF+RAM run (cf/modal_cfram_par.py). Pulls every chunk job's output from Modal volume qfr-out and the reference seeds from qfr-data, then checks, for each chunk k:
  (a) the job is clean: rc 0, 0 free-token mismatches, 0 forced tokens, 0 injections of any kind (the 'injections per row [N]' line), no divergence;
  (b) the SEAM: the circuit's OWN end-of-chunk state (circuit_state_{k+1}.pkl, read back from the circuit's head states by --save-state) EXACTLY equals the seed that job k+1 starts from (reference state_{k+1}.pkl + ramimg_{k+1}.npy).
Fields compared EXACTLY (these are the fields the next job is seeded with, machine() in circuit_run.py):
  regs x1..x31 (32-bit), pc anchor, n, weight base low 15 bits (wb & 0x7FFF; the CF firmware keeps the row id in a register, the weight pointer's high 12 bits are not used), the 8x8 WROW latch window, all 8192 RAM words (vs ramimg_{k+1}.npy).
Compared but NOT part of the verdict (informational): bn (4x2x128 4-bit head state; the next job is NOT seeded with it - if it is equal there is nothing hidden, if not it is reported), j/vcnt.
HONEST CLAIM: if every job is clean and every seam is exact, then by induction the parallel runs equal one unbroken run (the circuit is deterministic: identical state in => identical tokens out), PROVIDED bn (and any other
state not listed) does not influence later tokens - supported by bn equality when reported, and by 0 mismatches of every job that started without it. The reference is the comparator and the source of the seeds, not an input to any job's computation after its start.
The final token is read from the circuit's own raw tokens of the last chunk (readout in the chunk result JSON), compared with the reference only afterwards.
usage: python cf/collect_seams.py [--root cfram_par] [--chunks 0-74] [--refvol-dir cfram/refRC] [--out logs/cfram_par]   -> prints ONE verdict line, writes seams_<ts>.json + .md"""
import argparse, json, os, pickle, re, subprocess, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='cfram_par'); ap.add_argument('--chunks', default='0-74'); ap.add_argument('--refvol-dir', default='cfram/refRC')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'cfram_par')); ap.add_argument('--pull', default=os.path.expanduser('~/qweninqwen-work/cfram_par_pull')); ap.add_argument('--no-pull', action='store_true')
ap.add_argument('--expect-word', type=int, default=13, help='reference final word (argmax) for the last chunk read-out comparison')
a = ap.parse_args(); lo, hi = (int(x) for x in a.chunks.split('-')) if '-' in a.chunks else (int(a.chunks),) * 2
os.makedirs(a.pull, exist_ok=True); os.makedirs(a.out, exist_ok=True)
def get(vol, remote, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    r = subprocess.run(['modal', 'volume', 'get', '--force', vol, remote, local], capture_output=True, text=True); return r.returncode == 0 and os.path.exists(local)
rows, ok_all, t0 = [], True, time.time()
for k in range(lo, hi + 1):
    d = f'{a.pull}/{a.root}/chunk_{k:04d}'; r = dict(chunk=k, problems=[])
    if not a.no_pull:
        for f in (f'chain_{k}_to_{k}.json', 'job.stdout', 'job_done.json', f'circuit_state_{k + 1:04d}.pkl', 'circuit_tail.npy'): get('qfr-out', f'{a.root}/chunk_{k:04d}/{f}', f'{d}/{f}')
        for f in (f'state_{k + 1:04d}.pkl', f'ramimg_{k + 1:04d}.npy'): get('qfr-data', f'{a.refvol_dir}/{f}', f'{a.pull}/ref/{f}')
    try:
        res = json.load(open(f'{d}/chain_{k}_to_{k}.json')); done = json.load(open(f'{d}/job_done.json')); so = open(f'{d}/job.stdout').read()
    except Exception as e:
        r['problems'].append(f'no result/stdout: {e}'); rows.append(r); ok_all = False; continue
    m = re.search(r'injections per row \[(\d+)', so); r['injections'] = int(m.group(1)) if m else None
    r.update(rc=done['rc'], steps_run=res.get('steps_run'), seconds=round(res.get('seconds', 0)), free_tokens=res.get('free_tokens'), free_mismatches=res.get('free_mismatches'), forced_tokens=res.get('forced_tokens'), diverged=res.get('diverged'), halt_step=res.get('halt_step'))
    if done['rc'] != 0: r['problems'].append(f"rc {done['rc']}")
    if res.get('free_mismatches') != 0: r['problems'].append(f"free_mismatches {res.get('free_mismatches')}")
    if res.get('forced_tokens') != 0: r['problems'].append(f"forced_tokens {res.get('forced_tokens')}")
    if res.get('diverged'): r['problems'].append('diverged')
    if r['injections'] != 0: r['problems'].append(f"injections {r['injections']}")
    if k < 74 and res.get('halt_step') is not None: r['problems'].append('circuit HALT before the last chunk')
    sp = f'{d}/circuit_state_{k + 1:04d}.pkl'; rsp = f'{a.pull}/ref/state_{k + 1:04d}.pkl'; rip = f'{a.pull}/ref/ramimg_{k + 1:04d}.npy'
    if os.path.exists(sp) and os.path.exists(rsp) and os.path.exists(rip):
        cs = pickle.load(open(sp, 'rb')); stt = cs['stt']; rs = pickle.load(open(rsp, 'rb')); ri = np.load(rip); seam = {}
        seam['regs_bad'] = [(i, stt['x'][i] & 0xFFFFFFFF, rs['x'][i] & 0xFFFFFFFF) for i in range(1, 32) if (stt['x'][i] & 0xFFFFFFFF) != (rs['x'][i] & 0xFFFFFFFF)]
        seam['anchor'] = [stt['anchor'], rs['anchor']]; seam['n'] = [stt['n'], rs['n']]; seam['wb15'] = [stt['wb'] & 0x7FFF, rs['wb'] & 0x7FFF]
        cl = [list(map(int, w)) for w in stt.get('latch', [])]; rl = [list(map(int, w)) for w in rs.get('latch', [])]; seam['latch_equal'] = (cl == rl) or (not rl and not any(any(w) for w in cl))
        cr = np.array([int(v) & 0xFFFFFFFF for v in cs['ram']], np.uint64); rr = np.array([int(v) & 0xFFFFFFFF for v in ri], np.uint64); seam['ram_words'] = int(len(cr)); seam['ram_bad'] = int((cr != rr).sum())
        seam['exact'] = (not seam['regs_bad'] and stt['anchor'] == rs['anchor'] and stt['n'] == rs['n'] and seam['wb15'][0] == seam['wb15'][1] and seam['latch_equal'] and seam['ram_bad'] == 0 and len(cr) == len(rr) == 8192)
        if 'bn' in rs and cs.get('bn'): seam['bn_bad_informational'] = int(sum(int(cs['bn'][i][j]) != int(rs['bn'][i][j]) for i in range(len(rs['bn'])) for j in range(len(rs['bn'][i]))))
        r['seam'] = seam
        if not seam['exact']: r['problems'].append('seam NOT exact')
    elif k < 74:
        r['problems'].append('seam files missing (circuit_state or reference seed)')
    if k == 74 or res.get('halt_step') is not None:
        ro = res.get('readout') or {}; r['readout'] = ro
        if ro.get('circuit_word') != a.expect_word: r['problems'].append(f"final word {ro.get('circuit_word')} != reference {a.expect_word}")
    r['clean'] = not r['problems']; ok_all &= r['clean']; rows.append(r)
n = len(rows); nclean = sum(r['clean'] for r in rows); nseam = sum(1 for r in rows if r.get('seam', {}).get('exact')); nseam_chk = sum(1 for r in rows if 'seam' in r)
word = next((r['readout'].get('circuit_word') for r in rows if r.get('readout')), None)
verdict = f"{'PASS' if ok_all else 'FAIL'}: chunks {lo}-{hi}: {nclean}/{n} jobs clean (0 mismatches, 0 forced, 0 injections); seams exact {nseam}/{nseam_chk}; final word from circuit tokens: {word} (reference {a.expect_word}); {time.time() - t0:.0f}s to collect"
print(verdict)
for r in rows:
    if r['problems']: print(f"  chunk {r['chunk']}: {r['problems']}")
ts = time.strftime('%Y%m%d_%H%M%S'); json.dump(dict(verdict=verdict, ok=ok_all, chunks=[lo, hi], rows=rows), open(f'{a.out}/seams_{lo}-{hi}_{ts}.json', 'w'), indent=1, default=str)
with open(f'{a.out}/seams_{lo}-{hi}_{ts}.md', 'w') as f:
    f.write(f'# {verdict}\n\n| chunk | clean | steps | free mism | forced | inj | seam exact | ram_bad | bn_bad (info) | problems |\n|---|---|---|---|---|---|---|---|---|---|\n')
    for r in rows: s = r.get('seam', {}); f.write(f"| {r['chunk']} | {r['clean']} | {r.get('steps_run')} | {r.get('free_mismatches')} | {r.get('forced_tokens')} | {r.get('injections')} | {s.get('exact')} | {s.get('ram_bad')} | {s.get('bn_bad_informational')} | {'; '.join(r['problems'])} |\n")
sys.exit(0 if ok_all else 1)
