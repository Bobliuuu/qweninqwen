"""SAMPLED teacher-forced check of the CF+RAM firmware trace through the REAL merged checkpoint (stock Qwen3_5 HF modules: 12 chip layers + Qwen's 32; the CF table is rows of
embed_tokens.weight; next-token argmax over the FULL tied vocabulary must equal the reference trace's next token), over windows spread across the WHOLE trace.

WHY SAMPLED: the HF-module path runs ~149 steps/s (check_wf.py, 130,000 steps in 875 s on one GPU/Modal fallback list). The whole CF+RAM trace is ~1.5e8 tokens
(chunks of 2,097,152 steps): 1.5e8 / 149 = 1.0e6 s = ~280 GPU-hours = ~12 days on one GPU, ~35 h on 8 GPUs, and at the 59-91 tok/s of the HF dense path it is 20-29 days.
So full-trace HF checking is infeasible; the full trace is covered by the fused-executor chain run (same arithmetic, different kernels), and THIS script checks that the
stock HF modules agree with the reference on windows that start from the reference snapshot of chunk k (state_k, ramimg_k, latch), for chunks spread over the trace plus the
densest ROW / WB / taken-branch (J) chunks plus the last present chunk run to its end (final argmax / HALT region).

Each window: reference snapshot of chunk k -> circuit state (core.snapshot) -> HF chip layers inject -> feed [GO] + tok_k teacher-forced for N steps; at every step the FULL-vocab
argmax (lm_head(norm(h)), all ~666 K rows incl. the table rows) must equal the reference next token. Reference injections must be EMPTY (CF+RAM: REALMAP=1, WF_NROWS=156396).
Host-assisted (labelled, as check_wf.py): window start state comes from the reference snapshot (chunk 0: program-only start), bring-up core (in_drom widened), harness loop/compare.
Not covered: anything between windows (state carried across windows is NOT checked here), the fused-executor-vs-HF kernel equivalence outside the windows.

usage: REALMAP=1 WF_NROWS=156396 python hfcheck_sampled.py MODEL_DIR ELF REFDIR OUT.json [--steps 100000] [--final-steps 0] [--worker i --nworkers n] [--nsample 13] [--plan-only] [--chunks a,b,c]
--plan-only: print the window plan + coverage and exit (no GPU needed). --chunks overrides the auto plan. --final-steps 0 = run the last present chunk to its end."""
import argparse, json, time, sys, os, pickle
from common import *
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('elf'); ap.add_argument('refdir'); ap.add_argument('out')
ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--final-steps', type=int, default=0)
ap.add_argument('--nsample', type=int, default=13); ap.add_argument('--chunks', default='')
ap.add_argument('--worker', type=int, default=0); ap.add_argument('--nworkers', type=int, default=1)
ap.add_argument('--plan-only', action='store_true'); ap.add_argument('--every', type=int, default=10000); ap.add_argument('--maxbad', type=int, default=20)
a = ap.parse_args()
assert REALMAP and NROWS == 156396, 'CF+RAM check needs REALMAP=1 and WF_NROWS=156396'
fam = lambda t: S.family(int(t))[0]

# ---------------- plan (CPU only) ----------------
have = sorted(k for k in range(0, 200) if all(os.path.exists(f'{a.refdir}/{p}_{k:04d}.{e}') for p, e in (('tok', 'npy'), ('state', 'pkl'), ('ramimg', 'npy'), ('inj', 'npy'))))
gaps = [k for k in range(have[0], have[-1] + 1) if k not in have]
print(f'reference chunks present: {len(have)} ({have[0]}..{have[-1]}); MISSING in between: {gaps}', flush=True)
def stats(k):
    t = remap_trace(np.load(f'{a.refdir}/tok_{k:04d}.npy', mmap_mode='r')); f = np.empty(len(t), object)
    b = {n: S.FAMILIES[n] for n in ('ROW', 'WB', 'J')}; c = {n: int(((t >= lo) & (t < lo + sz)).sum()) for n, (lo, sz) in b.items()}
    return dict(chunk=k, n=len(t), **c)
if a.chunks: sel = [(int(k), 'manual') for k in a.chunks.split(',')]
else:
    why = {}
    for k in np.unique(np.round(np.linspace(0, len(have) - 1, a.nsample)).astype(int)): why.setdefault(have[k], []).append('spread')
    st = {k: stats(k) for k in have}
    for n in ('ROW', 'WB', 'J'):
        kk = max(have, key=lambda k: st[k][n] / st[k]['n']); why.setdefault(kk, []).append(f'densest-{n}({st[kk][n]}/{st[kk]["n"]})')
    why.setdefault(have[-1], []).append('last-present-chunk-to-end(final argmax/HALT)')
    sel = sorted((k, '+'.join(w)) for k, w in why.items())
tot = {k: len(np.load(f'{a.refdir}/tok_{k:04d}.npy', mmap_mode='r')) for k in have}
def nsteps(k): return (a.final_steps or tot[k] - 1) if (k == have[-1] and not a.chunks) else min(a.steps, tot[k] - 1)
plan = [dict(chunk=k, why=w, steps=nsteps(k)) for k, w in sel]
# greedy balance over workers by steps
loads = [0] * a.nworkers; asg = [[] for _ in range(a.nworkers)]
for w in sorted(plan, key=lambda w: -w['steps']): i = loads.index(min(loads)); asg[i].append(w); loads[i] += w['steps']
mine = sorted(asg[a.worker], key=lambda w: w['chunk'])
cov = sum(w['steps'] for w in plan); tt = sum(tot.values())
print(f'PLAN {len(plan)} windows, {cov:,} steps of {tt:,} present trace tokens = {100*cov/tt:.3f}% ; per-worker steps {loads}; est. {max(loads)/149/60:.0f} min on the busiest worker @149 step/s', flush=True)
for w in plan: print('  ', w, flush=True)
if a.plan_only: sys.exit(0)

# ---------------- model / circuit (GPU) ----------------
import torch
from chipformer.rv import elf, ref, hf, chat as CH
_b = core.build; core.build = lambda target=None, text=None, wrow=8: _b(target, text, wrow)
CH.core = core
chat = CH.RvChat(a.model); c = chat.c; v0 = chat.v0; dev = chat.dev; meta = chat.meta
assert meta['row_cells'] == S.ROW_CELLS and meta['chip_layers'] == c.n_layers() == 12 and meta['row_rows'] == NROWS, (meta['chip_layers'], meta['row_rows'])
assert chat.model.config.tie_word_embeddings and chat.model.lm_head.weight.data_ptr() == chat.lm.embed_tokens.weight.data_ptr(), 'head must be tied'
prog = elf.program(a.elf); GO_old = int(np.load(f'{a.refdir}/tok_0000.npy', mmap_mode='r')[0]); GO = int(remap_trace(np.array([GO_old]))[0]); assert GO == S.CONTROLS['GO'], (GO, S.CONTROLS['GO'])
ids = chat.prompt_ids([{"role": "user", "content": "Hi"}])
nm = lambda t: S.name(int(t)) if 0 <= t < S.VOCAB else f'<{t}>'
def machine(k):
    stt = pickle.load(open(f'{a.refdir}/state_{k:04d}.pkl', 'rb')); m = ref.Machine.from_program(prog)
    m.ram = [int(v) for v in np.load(f'{a.refdir}/ramimg_{k:04d}.npy')]
    if k == 0:   # program-only start, asserted equal to the reference (as check_wf.py --reset0)
        assert list(stt['x']) == list(m.x) and stt['anchor'] == m.anchor and m.n == 0 and stt['wb'] == 0 and not stt.get('latch'), 'reference state_0000 != program reset state'
        m.wrow_w = 8; m.latch = []
    else:
        m.wrow_w = 8; m.latch = [list(w) for w in stt.get('latch', [])]; m.x = list(stt['x']); m.anchor = stt['anchor']; m.n = stt['n']; m.wb = stt['wb'] & 0x7FFF; m.wb27 = stt.get('wb27', stt['wb'])
    return m
results = []; t00 = time.time()
def dump(final=False): json.dump(dict(final=final, model=a.model, worker=a.worker, nworkers=a.nworkers, present_chunks=have, missing_chunks=gaps, plan=plan, windows=results, seconds=time.time() - t00), open(a.out, 'w'), indent=1)
for w in mine:
    k, N = w['chunk'], w['steps']
    inj = np.load(f'{a.refdir}/inj_{k:04d}.npy'); assert len(inj) == 0, f'chunk {k}: CF+RAM reference must have 0 host injections, found {len(inj)}'
    t = remap_trace(np.load(f'{a.refdir}/tok_{k:04d}.npy')); stream = t if k == 0 else np.concatenate([[GO], t])   # chunk 0's trace already starts with GO; later chunks: the circuit is fed GO and predicts t[0]
    assert int(stream[0]) == GO; N = min(N, len(stream) - 1)
    m = machine(k); ccache = chat._chip_cache(); h = chat.cm.embed_tokens(torch.tensor([ids], device=dev))
    with torch.no_grad():
        for L in range(chat.n_chip): h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=ccache)
    core.snapshot(c, m); hf.inject(c, ccache)
    if chat.graph is None: chat._capture()
    g, bad, t0 = len(ids), [], time.time(); margin_min = 1e30; n_code_ne = 0; fams = {}
    with torch.no_grad():
        for s in range(N):
            hc = chat._chip_step(v0 + int(stream[s]), g, ccache); g += 1
            lg = chat.model.lm_head(chat.lm.norm(hc.view(1, 1, -1)))[0, 0].float()
            top = torch.topk(lg, 2); got = int(top.indices[0]); want = v0 + int(stream[s + 1]); mg = float(top.values[0] - top.values[1])
            if got == want: margin_min = min(margin_min, mg)
            if chat.circuit_token(hc) != got: n_code_ne += 1
            f = fam(stream[s + 1]); fams[f] = fams.get(f, 0) + 1
            if got != want:
                bad.append(dict(step=s, want=nm(stream[s + 1]), got=nm(got - v0) if got >= v0 else f'text {got}', margin=mg, ctx=[nm(x) for x in stream[max(0, s - 5):s + 1]]))
                if len(bad) >= a.maxbad: break
            if (s + 1) % a.every == 0:
                print(f'[chunk {k}] step {s+1}/{N} mismatches {len(bad)} {time.time()-t0:.0f}s ({(s+1)/(time.time()-t0):.0f} step/s)', flush=True)
    done = s + 1 if not bad else bad[-1]['step']
    r = dict(chunk=k, why=w['why'], steps_requested=N, steps_checked=done, n_mismatch=len(bad), first_mismatch=bad[0] if bad else None, mismatches=bad, family_counts=fams,
             n_row_tokens=fams.get('ROW', 0), n_wb_tokens=fams.get('WB', 0), n_j_tokens=fams.get('J', 0), full_vocab=int(lg.numel()), min_top1_top2_logit_margin=margin_min,
             steps_code_decode_ne_fullvocab_argmax=n_code_ne, last_token=nm(stream[done]), seconds=time.time() - t0)
    results.append(r); dump(); print('WINDOW ' + json.dumps({x: r[x] for x in r if x not in ('mismatches', 'family_counts')}), flush=True)
dump(True)
tot_ck = sum(r['steps_checked'] for r in results); nbad = sum(r['n_mismatch'] for r in results)
print(f'DONE worker {a.worker}: {len(results)} windows, {tot_ck:,} steps checked, {nbad} mismatches', flush=True); sys.exit(0 if not nbad else 1)
