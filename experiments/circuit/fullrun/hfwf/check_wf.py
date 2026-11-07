"""Teacher-forced check of the WF firmware trace through the REAL merged checkpoint (stock Qwen3_5 HF modules: 12 chip layers + Qwen's 32), the weight table being
rows of embed_tokens.weight. Per step: next-token argmax over the FULL tied vocabulary (lm_head(norm(h)), all ~666 K rows incl. the 155,205 table rows) must equal the
reference trace's next token. Pattern of e2e/hfcheck.py. Host-assisted (labelled): type-1 RAM/blob word injections into the RAM heads' recurrent state (same records as
circuit_run.py), chunk-0 start state from the program (as --reset0), bring-up core (in_ram := one; with REALMAP=1: CF+RAM firmware, real RAM, 0 injections, program-only start state).
usage: python check_wf.py MODEL_DIR ELF REFDIR OUT.json [--steps N] [--neg] [--nohead]"""
import argparse, json, time, sys, pickle
from common import *
import numpy as np, torch
from chipformer.rv import elf, ref, hf, chat as CH
from chipformer.rv.asm import Program
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('elf'); ap.add_argument('refdir'); ap.add_argument('out')
ap.add_argument('--steps', type=int, default=130000); ap.add_argument('--neg', action='store_true', help='negative control: every table entry +1 (4-bit wrap)')
ap.add_argument('--every', type=int, default=5000); ap.add_argument('--maxbad', type=int, default=20)
a = ap.parse_args()
_b = core.build; core.build = lambda target=None, text=None, wrow=8: _b(target, text, wrow)   # RvChat builds the circuit: make it the WF one
CH.core = core
chat = CH.RvChat(a.model); c = chat.c; v0 = chat.v0; dev = chat.dev
meta = chat.meta; assert meta['row_cells'] == S.ROW_CELLS and meta['chip_layers'] == c.n_layers() == 12, (meta['chip_layers'], c.n_layers())
assert chat.model.config.tie_word_embeddings and chat.model.lm_head.weight.data_ptr() == chat.lm.embed_tokens.weight.data_ptr(), 'head must be tied'
if a.neg:
    cols = torch.tensor(meta['cell_columns'], device=dev); r0 = v0 + S.FAMILIES['ROW'][0]; W = chat.lm.embed_tokens.weight
    blk = W[r0:r0 + S.FAMILIES['ROW'][1]][:, cols].float()
    with torch.no_grad(): W[r0:r0 + S.FAMILIES['ROW'][1], cols] = ((blk + 1) % 16).to(W.dtype)
    print('NEGATIVE CONTROL: all table entries +1 mod 16', flush=True)
# --- reference trace, state, injections (chunk 0) ---
tok_all = remap_trace(np.load(f'{a.refdir}/tok_0000.npy')); stt = pickle.load(open(f'{a.refdir}/state_0000.pkl', 'rb'))
inj = np.load(f'{a.refdir}/inj_0000.npy')
if REALMAP: assert len(inj) == 0, f'CF+RAM reference must have 0 host injections, found {len(inj)}'
else: assert (inj[:, 1] == 1).all(), 'only type-1 (RAM/blob word) injections expected'
N = min(a.steps, len(tok_all) - 1); inj = inj[inj[:, 0] - 1 < N]
if not REALMAP: entry, segs = elf.segments(a.elf); text = [s for s in segs if s[0] == S.ROM_BASE][0][1]
prog = elf.program(a.elf) if REALMAP else Program([int.from_bytes(text[i:i + 4], 'little') for i in range(0, len(text), 4)], b'', {'_start': entry}, data_base=S.RAM_BASE, rodata=b'', wrom=b'')
m = ref.Machine.from_program(prog); assert stt['n'] == 0 and stt['wb'] == 0 and not stt.get('latch')
if REALMAP:   # --reset0: chunk 0 starts from the PROGRAM ONLY (reset registers/pc/n/wb + initial .data RAM image); asserted equal to the reference state_0000 and ramimg_0000
    assert list(stt['x']) == list(m.x) and stt['anchor'] == m.anchor and m.n == 0, 'reference state_0000 != program reset state'
    import os
    if os.path.exists(f'{a.refdir}/ramimg_0000.npy'): assert all(int(v) == int(w) for v, w in zip(np.load(f'{a.refdir}/ramimg_0000.npy'), m.ram)), 'ramimg_0000 != program initial RAM'
    m.wrow_w = 8; m.latch = []
else:
    m.wrow_w = 8; m.latch = []; m.x = list(stt['x']); m.anchor = stt['anchor']; m.n = stt['n']; m.wb = stt['wb'] & 0x7FFF   # chunk-0 start state (registers, pc anchor) from the reference, as circuit_run.py
# --- chip layers: prompt prefill, snapshot, inject (hfcheck pattern) ---
ids = chat.prompt_ids([{"role": "user", "content": "Hi"}]); ccache = chat._chip_cache()
h = chat.cm.embed_tokens(torch.tensor([ids], device=dev))
with torch.no_grad():
    for L in range(chat.n_chip): h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=ccache)
core.snapshot(c, m); hf.inject(c, ccache)
if chat.graph is None: chat._capture()
# RAM-word injection targets: slot wi -> (layer, state head, row)
slots = hf.head_slots(c); r = c.target.value_heads // c.target.key_heads
ram_loc = {}
for wi in range(8192):
    hh = (wi & 15) | ((wi >> 12) << 4); ni = c.nodes.index(c.tables[f'ram{hh}']); L, j = slots[ni]; ram_loc[wi] = (L, j * r + (wi >> 4 & 1), wi >> 5 & 127)
def put_ram(wi, word):
    L, hv, row = ram_loc[wi]; Sx = ccache.layers[L].recurrent_states[0][0, hv, row]
    v = np.zeros(core.HD); v[list(core.REF_COLS)] = core.A_REF
    for i in range(32): v[2 + i] = core.A_BIT * (int(word) >> i & 1)
    Sx.copy_(torch.tensor(v, dtype=Sx.dtype, device=Sx.device))
inj_by = {}
for k in range(len(inj)): inj_by.setdefault(int(inj[k, 0]) - 1, []).append((int(inj[k, 2]), int(inj[k, 3]) & 0xFFFFFFFF))
fam = lambda t: S.family(int(t))[0]
g, bad, t0 = len(ids), [], time.time(); nrow = nvdot = 0; margin_min = 1e30; n_code_ne = 0
nm = lambda t: S.name(int(t)) if 0 <= t < S.VOCAB else f'<{t}>'
with torch.no_grad():
    for s in range(N):
        for wi, w in inj_by.get(s, ()): put_ram(wi, w)
        hc = chat._chip_step(v0 + int(tok_all[s]), g, ccache); g += 1
        lg = chat.model.lm_head(chat.lm.norm(hc.view(1, 1, -1)))[0, 0].float()           # FULL-vocabulary logits
        top = torch.topk(lg, 2); got = int(top.indices[0]); want = v0 + int(tok_all[s + 1])
        mg = float(top.values[0] - top.values[1]); margin_min = min(margin_min, mg) if got == want else margin_min
        code = chat.circuit_token(hc)
        if code != got: n_code_ne += 1
        f = fam(tok_all[s + 1]); nrow += f == 'ROW'; nvdot += (f == 'V') and False
        if got != want:
            bad.append(dict(step=s, want=nm(tok_all[s + 1]), got=nm(got - v0) if got >= v0 else f'text {got}', margin=mg, code_decode=None if code is None else int(code), ctx=[nm(x) for x in tok_all[max(0, s - 5):s + 1]]))
            if len(bad) >= a.maxbad: break
        if (s + 1) % a.every == 0:
            print(f'step {s+1}/{N} mismatches {len(bad)} ROW so far {nrow} {time.time()-t0:.0f}s ({(s+1)/(time.time()-t0):.0f} step/s)', flush=True)
            json.dump(dict(partial=True, steps_done=s + 1, mismatches=bad, rows=nrow), open(a.out, 'w'))
done = s + 1 if not bad else bad[-1]['step']
fams = {}
for t in tok_all[1:done + 1]: fams[fam(t)] = fams.get(fam(t), 0) + 1
res = dict(model=a.model, neg=a.neg, steps_requested=N, steps_checked=done, mismatches=bad, n_mismatch=len(bad), first_mismatch=bad[0] if bad else None, full_vocab=int(lg.numel()),
           table_rows=meta['row_rows'], hidden=meta['hidden'], family_counts_checked=fams, n_row_tokens_checked=fams.get('ROW', 0), n_vdot_tokens_checked=fams.get('V', 0),
           min_top1_top2_logit_margin=margin_min, steps_where_code_decode_ne_fullvocab_argmax=n_code_ne, injections_applied=int(sum(len(v) for k, v in inj_by.items() if k < done)),
           seconds=time.time() - t0)
json.dump(res, open(a.out, 'w'), indent=1); print('RESULT ' + json.dumps({k: res[k] for k in res if k != 'mismatches'}), flush=True); sys.exit(0 if not bad else 1)
