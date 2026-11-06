"""PURE FREE-RUN variant of circuit_run.py (PURE=1 semantics, always on). ZERO host writes into circuit state during the run: no memory injections (RAM/blob/WROM), no token override/forcing,
no injection schedule on the GPU. Allowed: step-0 state snapshot (registers/pc/n/weight base/latch/optional RAM image), program, parameter tables (WF row table). The reference trace is read
ONLY for after-the-fact comparison and for accounting of what the unpure run would have injected. The first free mismatch is reported; after it the run is free (not corrected).
Per-step kernel `_record` only READS the circuit token. Result json: pure/host_writes/token_overrides/injections_applied = true/0/0/0."""
"""Run segments of the full-model reference trace on the circuit executor (fused Triton FreeRun3, tuned warps), BRING-UP MODE (host-served memory, injected into the circuit's tables).

INJECTION MODE (default): before the step that produces the first token of a LOAD / SB / SH / VDOT record the host writes the true memory word (RAM/blob word from the reference
machine's memory) into the aliased slot of the circuit's 32 KB RAM head state, or the 8 true int4 cells (from the host weight table) into the circuit's WROM head state at WP mod 2^15.
The circuit then does the instruction itself (load/merge/VDOT arithmetic, register write, everything). Only VDOTs whose WP mod 2^15 is in the 128 unaddressable WROM words are forced (token override).
Legacy token-override mode: --kinds 1,2,3 (does not work: the circuit's register write for a load uses its own load value, not the fed token).


Per row (= one trace segment starting at a chunk boundary of ref_big.py): the circuit is snapshotted to the machine state at the boundary (regs, pc anchor, n, weight base),
then free-runs: every step the circuit computes the next token from its residual; for tokens of LOAD / VDOT / STORE records the circuit's token is REPLACED (inside the CUDA graph,
before the next embedding) by the reference token, because the circuit's RAM (32 KB), DROM (128 KB) and WROM (32,640 words) cannot hold the full model's memories (host-served bring-up).
Every other token (ALU, MUL/DIV, branches/jumps, WB, R refresh, ecall) is the circuit's own and is compared to the reference trace; raw circuit predictions at host-served tokens are logged too.
usage: circuit_run.py ELF REFDIR OUTDIR --chunks 0,1,2 [--steps N] [--block K] [--kinds 1,2,3]"""
import argparse, json, os, sys, time, pickle
WF = bool(os.environ.get('WF'))   # WEIGHT-FETCH MODE: firmware built with WF=1 (wrow before every 8-word VDOT window), circuit = core.build(wrow=8): the VDOT weights come from the outer model's own
# embedding rows (ROW tokens; table rows are parameters), NO weight word is injected, NO VDOT record is forced. Still host-assisted: RAM/blob words (type 1/3 injections), chunk-start snapshot
# (regs, pc anchor, n, weight base low 15 bits, latch window), the chunk schedule. Needs PYTHONPATH to a tree patched with weight_fetch/wf_npu.patch.
if WF: os.environ.setdefault('WF_ROW_CELLS', '3840')
import numpy as np, torch, triton, triton.language as tl
import types
from chipformer.rv import core as _core0, elf, ref, fusedexec
# BRING-UP circuit variant: the RAM address-range check is removed (in_ram := 1) so that loads/stores at the full model's addresses do not fault and still
# commit (the circuit's 32 KB RAM cannot hold the model's data; host-served values are forced in). Nothing else in the compiled core changes.
_src = open(_core0.__file__).read(); _old = 'in_ram = c.and_([1 - b for b in EA[15:32]], "in_ram")'
REALMAP = bool(os.environ.get('REALMAP'))   # REALMAP=1: firmware linked for the real 32 KB RAM map (ramown/fw_rom.ld). in_ram stays REAL (no aliasing, no fault-masking); only in_drom is widened so that
# DROM-class loads (real 128 KB DROM, plus the 0x10000000 weight/table blob) are answered by the circuit's DROM table, whose slot (EA[16:2]) the host fills for blob words (constants, not RAM).
if REALMAP:
    _old2 = 'in_drom = c.and_([1 - b for b in EA[18:32]] + [EA[17]], "in_drom")'; assert _old2 in _src; _src = _src.replace(_old2, 'in_drom = 1 - in_ram')
else:
    assert _old in _src; _src = _src.replace(_old, 'in_ram = one')
# >>> WB27 (ramown/wb27/wb27.py, WB27=1): the circuit holds the 27-bit weight pointer itself (7-nibble WB adder + wbase_hi latch head); WB token, WROM path unchanged;
# the --chain boundary readback then compares the circuit's wbase|wbase_hi<<15 with the reference's full 27-bit wb.
WB27 = bool(os.environ.get('WB27'))
if WB27:
    for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ramown', 'wb27')): sys.path.insert(0, _p)
    import wb27 as _wb27; _src = _wb27.patch_core_src(_src)
# <<< WB27
core = types.ModuleType('chipformer.rv.core_bringup'); core.__file__ = _core0.__file__; exec(compile(_src, 'core_bringup.py', 'exec'), core.__dict__)
from chipformer.rv import spec as S
import autotune_layers as AL
from tune_warps import W

ap = argparse.ArgumentParser(); ap.add_argument('elf'); ap.add_argument('refdir'); ap.add_argument('out'); ap.add_argument('--chunks', required=True)
ap.add_argument('--steps', type=int, default=0); ap.add_argument('--block', type=int, default=50000); ap.add_argument('--kinds', default=''); ap.add_argument('--tuned', default='autotune.json'); ap.add_argument('--forcewin', default=''); ap.add_argument('--wrom', default='')
ap.add_argument('--ramimg', action='store_true', help='restore the snapshot RAM image (refdir/ramimg_NNNN.npy, 8192 aliased words) into the circuit RAM head state at chunk start')
ap.add_argument('--rowtab', default='/data/rowtab.npy', help='WF mode: full row table (npy uint8 [rows, 3840])')
ap.add_argument('--inj', default='inj', help='injection file prefix in refdir: inj (all, legacy), injF (filtered), injB (blob-only), injN (no RAM injection)')
ap.add_argument('--chain', action='store_true', help='ONE continuous circuit run over consecutive chunks (no snapshot restore between chunks); compare circuit state to the reference snapshot at every chunk boundary')
ap.add_argument('--handoff', action='store_true', help='like --chain, but at each boundary READ the circuit state back (regs, pc, n, wb, bn, 32KB RAM) and re-snapshot the circuit from that read-back state (hidden lag history reset, as a chunk restart)')
ap.add_argument('--reset0', action='store_true', help='chunk 0 starts from the PROGRAM ONLY (ref.Machine.from_program: reset registers/pc/n/wb, initial .data RAM), asserted equal to the reference state_0000 pickle; no host snapshot of state')
ap.add_argument('--noforce', action='store_true', help='do NOT force (override) the tokens of VDOT records whose weights are in the 128 unaddressable WROM words: the circuit free-runs through them (first divergence is then measured)')
ap.add_argument('--nodrom', action='store_true', help='drop all type-3 (DROM-class constant) injections')
ap.add_argument('--novdot', action='store_true', help='drop all VDOT weight injections (type 2) and zone forcing')
a = ap.parse_args()
if WF: a.novdot = True   # no weight injection, no forcing
os.makedirs(a.out, exist_ok=True); CHAIN = a.chain or a.handoff; assert not CHAIN, 'PURE variant: single-chunk rows only (no chain/handoff)'
chunks = [int(x) for x in a.chunks.split(',')]; B = len(chunks); forced_kinds = [int(x) for x in a.kinds.split(',') if x]; assert not forced_kinds and not a.forcewin, 'PURE: no forced kinds / forcewin'

@triton.jit
def _record(tok, ctr, raw, N):
    """PURE: the ONLY per-step host-side kernel. It READS the circuit's predicted token and stores it for after-the-fact comparison. It writes nothing into circuit state:
    no fid/p/pay store (no token override), no head-state store (no injection)."""
    bat = tl.program_id(0)
    c = tl.load(ctr + bat)
    tl.store(raw + bat * N + c, tl.load(tok + bat).to(tl.int32))
    tl.store(ctr + bat, c + 1)

cfg = dict(ffn_warps={}, head_warps={}) if (WF or a.tuned == 'none') else json.load(open(a.tuned))   # tuned warps belong to the 11-layer circuit
for n, tab in (('_ffn3', 'wf'), ('_ffn4', 'wf'), ('_head3', 'wh'), ('_head4', 'wh')):
    setattr(fusedexec, n, AL.PL(getattr(fusedexec, n), tab, 'd' + tab))
fusedexec._decode_embed = W(fusedexec._decode_embed, num_warps=cfg.get('decode_embed_warps', 4))
AL.CUR.update(wf={int(k): v for k, v in cfg['ffn_warps'].items()}, wh={int(k): v for k, v in cfg['head_warps'].items()})

# reference segments
toks, kinds, starts, states = [], [], [], []
for ch in chunks:
    t = np.load(f'{a.refdir}/tok_{ch:04d}.npy'); k = np.load(f'{a.refdir}/kind_{ch:04d}.npy')
    if ch == 0: t, k = t[1:], k[1:]   # drop the GO token: the circuit is fed GO and predicts the next one
    if a.steps: t, k = t[:a.steps], k[:a.steps]
    toks.append(t.astype(np.int64)); kinds.append(k); states.append(pickle.load(open(f'{a.refdir}/state_{ch:04d}.pkl', 'rb')))
N = max(len(t) for t in toks)
if CHAIN:
    chunk_list, chunk_lens = list(chunks), [len(t) for t in toks]; assert chunk_list == list(range(chunk_list[0], chunk_list[0] + len(chunk_list))), 'chain needs consecutive chunks'
    INJ, off = [], 0
    for ci, ch in enumerate(chunk_list):
        inj_ = np.load(f'{a.refdir}/{a.inj}_{ch:04d}.npy'); sh_ = 1 if ch == 0 else 0
        inj_ = inj_[inj_[:, 0] - sh_ < chunk_lens[ci]].copy(); inj_[:, 0] = inj_[:, 0] - sh_ + off; INJ.append(inj_); off += chunk_lens[ci]
    INJ = np.concatenate(INJ); bounds = np.cumsum(chunk_lens).tolist()
    nxt_states = []
    for ch in chunk_list:   # reference snapshot at the END of each chunk = start of chunk ch+1
        p_ = f'{a.refdir}/state_{ch + 1:04d}.pkl'; nxt_states.append(pickle.load(open(p_, 'rb')) if os.path.exists(p_) else None)
    toks, kinds, states, chunks = [np.concatenate(toks)], [np.concatenate(kinds)], [states[0]], [chunk_list[0]]; B = 1; N = len(toks[0]) + 1   # +1: the step that feeds the chunk's last token
if WF:
    from chipformer.rv import fastexec as _fe
    class SliceTab:   # the ROW family's table rows [lo, hi] (all rows a chunk can touch) of the full table file (155,205 x 3840 uint8)
        def __init__(s_, full, lo, hi): s_.lo, s_.hi, s_.n = lo, hi, len(full); s_.a = np.ascontiguousarray(full[lo:hi + 1]); s_.a = (s_.a + 1) & 15 if os.environ.get('WF_NEG') else s_.a   # WF_NEG=1: NEGATIVE CONTROL, every table entry +1, the circuit must then diverge
        def __len__(s_): return s_.n
        def __getitem__(s_, p): return s_.a[p - s_.lo] if s_.lo <= p <= s_.hi else np.zeros(s_.a.shape[1], np.uint8)
    _tab = np.load(a.rowtab, mmap_mode='r'); _lo = _hi = None; _b0, _sz = S.FAMILIES['ROW']
    for r_ in range(B):
        tt_ = toks[r_][(toks[r_] >= _b0) & (toks[r_] < _b0 + _sz)] - _b0
        if len(tt_): _lo = int(tt_.min()) if _lo is None else min(_lo, int(tt_.min())); _hi = int(tt_.max()) if _hi is None else max(_hi, int(tt_.max()))
    _lo, _hi = max((_lo or 0) - 2, 0), (_hi or 0) + 2
    S.ROW_TABLE = SliceTab(_tab, _lo, _hi); print(f'ROW table slice materialised for this run: rows {_lo}..{_hi} of {len(_tab)}', flush=True)
    _fs = open(_fe.__file__).read(); _o = "            if size <= 4096:\n"
    assert _o in _fs
    _fs = _fs.replace(_o, "            if nm == 'ROW':\n                small[f], row0[f] = True, len(table) - ROWLO\n                table += list(self.ex.embed(list(range(t0 + ROWLO, t0 + ROWHI + 1))).cpu().numpy())\n            elif size <= 4096:\n")
    _m = types.ModuleType('chipformer.rv.fastexec_wf'); _m.__file__ = _fe.__file__; _m.__dict__.update(ROWLO=_lo, ROWHI=_hi); exec(compile(_fs, 'fastexec_wf.py', 'exec'), _m.__dict__); _fe.FreeRun = _m.FreeRun
c = core.build(wrow=8) if WF else core.build()
entry, segs = elf.segments(a.elf)
text = [s for s in segs if s[0] == S.ROM_BASE][0][1]
from chipformer.rv.asm import Program
prog = elf.program(a.elf) if REALMAP else Program([int.from_bytes(text[i:i + 4], 'little') for i in range(0, len(text), 4)], b'', {'_start': entry}, data_base=S.RAM_BASE, rodata=b'', wrom=b'')
def machine(stt, ch=None):
    m = ref.Machine.from_program(prog);
    if a.ramimg and not (a.reset0 and ch == 0): m.ram = [int(v) for v in np.load(f'{a.refdir}/ramimg_{ch:04d}.npy')]
    if a.reset0 and ch == chunks[0] and ch == 0:   # step-0 state comes from the program alone
        assert list(stt['x']) == list(m.x) and stt['anchor'] == m.anchor and stt['n'] == m.n == 0 and stt['wb'] == 0, 'reference state_0000 != program reset state'
        if a.ramimg: assert all(int(v) == int(w) for v, w in zip(np.load(f'{a.refdir}/ramimg_{ch:04d}.npy'), m.ram)), 'ramimg_0000 != program initial RAM'
        if os.environ.get('INITWROM'): m.wrom = bytes(np.memmap(a.wrom, dtype=np.uint8, mode='r')[:8 * 32640])
        return m
    if WF: m.wrow_w = 8; m.latch = [list(w) for w in stt.get('latch', [])]   # chunk-start snapshot of the latched 8-word window (host-seeded here; a chained run carries it itself)
    m.x = list(stt['x']); m.anchor = stt['anchor']; m.n = stt['n']; m.wb = stt['wb'] & 0x7FFF; m.wb27 = stt.get('wb27', stt['wb'])   # wb27: only read by the WB27 snapshot (wbase_hi init)
    if os.environ.get('INITWROM'): m.wrom = bytes(np.memmap(a.wrom, dtype=np.uint8, mode='r')[:8 * 32640])
    return m
core.snapshot(c, machine(states[0], chunks[0]))
fr = fusedexec.FreeRun3(c, B, 'cuda'); fr.load(0, S.tok('GO'))
AL.setup(fr)
# force tables
ids = fr.fr2.ids.cpu().numpy().astype(np.int64); order = np.argsort(ids); sids = ids[order]
dev = 'cuda'
F_fid = torch.zeros(B, N, dtype=torch.int32); F_p = torch.zeros(B, N, dtype=torch.int32); MASK = torch.zeros(B, N, dtype=torch.uint8); TREF = torch.zeros(B, N, dtype=torch.int32)
for r in range(B):
    t = toks[r]; i = np.searchsorted(sids, t, side='right') - 1; fid = order[i]; p = t - ids[fid]
    F_fid[r, :len(t)] = torch.from_numpy(fid.astype(np.int32)); F_p[r, :len(t)] = torch.from_numpy(p.astype(np.int32)); TREF[r, :len(t)] = torch.from_numpy(t.astype(np.int32))
    MASK[r, :len(t)] = torch.from_numpy(np.isin(kinds[r], forced_kinds).astype(np.uint8))
# ---- PURE: NO injection tables are built. The reference injection files are read ONLY to report what the host WOULD have had to write (accounting; never uploaded to the GPU). ----
n_inj = np.zeros(B, np.int64); n_ramin = np.zeros(B, np.int64); would = []
for r in range(B):
    inj = np.load(f'{a.refdir}/{a.inj}_{chunks[r]:04d}.npy'); sh = 1 if chunks[r] == 0 else 0
    inj = inj[inj[:, 0] - sh < len(toks[r])]; to = inj[:, 0] - sh
    w = {f'type{t}': int((inj[:, 1] == t).sum()) for t in (1, 2, 3)}
    w['first_step'] = {f'type{t}': (int(to[inj[:, 1] == t].min()) if (inj[:, 1] == t).any() else None) for t in (1, 2, 3)}
    would.append(w); n_inj[r] = len(inj); n_ramin[r] = w['type1']
print('PURE: host writes performed = 0 (no injection tables, no override). Would-be injections the reference schedule contains (NOT applied):', would, flush=True)
HOST_WRITES = 0
assert not MASK.any(), "PURE: token override mask must be empty"
TREF = TREF.to(dev); MASK = MASK.to(dev)
CTR = torch.zeros(B, dtype=torch.int32, device=dev); RAW = torch.zeros(B, N, dtype=torch.int32, device=dev)
orig = fusedexec._decode_embed
class OV:
    def __getitem__(self, grid):
        inner = orig[grid]
        def call(*args, **kw):
            _record[(B,)](fr.tok, CTR, RAW, N)
            return inner(*args, **kw)
        return call
fusedexec._decode_embed = OV()
fr.capture()
def load_rows():
    for r in range(B):
        core.snapshot(c, machine(states[r], chunks[r])); fr.load(r, S.tok('GO'))
    CTR.zero_()
load_rows(); torch.cuda.synchronize()
print(f'rows {B}, steps/row {N:,}, forced kinds {forced_kinds}, forced fraction {[float(MASK[r,:len(toks[r])].float().mean()) for r in range(B)][:4]}', flush=True)
lens = torch.tensor([len(t) for t in toks], device=dev)
first_bad = [None] * B; nbad_free = np.zeros(B, np.int64); nbad_forced = np.zeros(B, np.int64); n_free = np.zeros(B, np.int64); n_forced = np.zeros(B, np.int64)
done = 0; t0 = time.time(); log = open(a.out + '/progress.log', 'a')
while done < N:
    k = min(a.block, N - done)
    for _ in range(k): fr.graph.replay()
    torch.cuda.synchronize()
    sl = slice(done, done + k); idx = torch.arange(done, done + k, device=dev)[None, :]
    valid = idx < lens[:, None]; m = MASK[:, sl] != 0; neq = RAW[:, sl] != TREF[:, sl]
    for r in range(B):
        v = valid[r]; fm = (v & ~m[r]); mm = (v & m[r])
        n_free[r] += int(fm.sum()); n_forced[r] += int(mm.sum()); bf = neq[r] & fm; nbad_free[r] += int(bf.sum()); nbad_forced[r] += int((neq[r] & mm).sum())
        if first_bad[r] is None and bf.any(): first_bad[r] = done + int(bf.nonzero()[0, 0])
    done += k; el = time.time() - t0
    msg = f'step {done:,}/{N:,} {el:.0f}s {done*B/el:,.0f} agg tok/s; free-token mismatches {int(nbad_free.sum())} (rows with any: {sum(x is not None for x in first_bad)}), forced-token raw disagreements {int(nbad_forced.sum())} of {int(n_forced.sum()):,}'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    if all(x is not None for x in first_bad): print('all rows diverged; stop', flush=True); break
for r in range(B):
    if first_bad[r] is not None:
        fb = first_bad[r]; print(f'row {r} (chunk {chunks[r]}) first free mismatch at step {fb}: ' + ' | '.join(f'{j}:{S.name(int(TREF[r,j]))}/{S.name(int(RAW[r,j]))}/f{int(MASK[r,j])}' for j in range(max(0, fb - 16), fb + 2)), flush=True)
assert not MASK.any() and HOST_WRITES == 0 and int(CTR.max()) <= N
res = dict(pure=True, host_writes=0, token_overrides=0, injections_applied=0, would_be_injections=would, wf_mode=WF, inj_file=a.inj, ramimg=a.ramimg, chunks=chunks, rows=B, steps_per_row=[len(t) for t in toks], steps_run=done, seconds=time.time() - t0, agg_tok_per_s=done * B / (time.time() - t0), 
           free_tokens=n_free.tolist(), forced_tokens=n_forced.tolist(), free_mismatches=nbad_free.tolist(), first_free_mismatch=first_bad, forced_raw_disagreements=nbad_forced.tolist(),
           gpu=torch.cuda.get_device_name())
json.dump(res, open(a.out + '/result_' + '_'.join(map(str, chunks[:3])) + f'_n{len(chunks)}.json', 'w'), indent=1); print(json.dumps(res), flush=True)
