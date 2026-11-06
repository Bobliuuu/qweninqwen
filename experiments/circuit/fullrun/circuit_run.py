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
import numpy as np, torch, triton, triton.language as tl
import types
from chipformer.rv import core as _core0, elf, ref, fusedexec
# BRING-UP circuit variant: the RAM address-range check is removed (in_ram := 1) so that loads/stores at the full model's addresses do not fault and still
# commit (the circuit's 32 KB RAM cannot hold the model's data; host-served values are forced in). Nothing else in the compiled core changes.
_src = open(_core0.__file__).read(); _old = 'in_ram = c.and_([1 - b for b in EA[15:32]], "in_ram")'
assert _old in _src; _src = _src.replace(_old, 'in_ram = one')
core = types.ModuleType('chipformer.rv.core_bringup'); core.__file__ = _core0.__file__; exec(compile(_src, 'core_bringup.py', 'exec'), core.__dict__)
from chipformer.rv import spec as S
import autotune_layers as AL
from tune_warps import W

ap = argparse.ArgumentParser(); ap.add_argument('elf'); ap.add_argument('refdir'); ap.add_argument('out'); ap.add_argument('--chunks', required=True)
ap.add_argument('--steps', type=int, default=0); ap.add_argument('--block', type=int, default=50000); ap.add_argument('--kinds', default=''); ap.add_argument('--tuned', default='autotune.json'); ap.add_argument('--forcewin', default=''); ap.add_argument('--wrom', default='')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
chunks = [int(x) for x in a.chunks.split(',')]; B = len(chunks); forced_kinds = [int(x) for x in a.kinds.split(',') if x]

@triton.jit
def _override(tok, fid, p, pay, f_fid, f_p, mask, ctr, raw, N, NPB: tl.constexpr):
    bat = tl.program_id(0)
    c = tl.load(ctr + bat)
    tl.store(raw + bat * N + c, tl.load(tok + bat).to(tl.int32))
    if tl.load(mask + bat * N + c) != 0:
        nf = tl.load(f_fid + bat * N + c); npay = tl.load(f_p + bat * N + c)
        tl.store(fid + bat, nf); tl.store(p + bat, npay.to(tl.int64))
        j = tl.arange(0, 32)
        tl.store(pay + bat * 32 + j, ((npay >> j) & 1).to(tl.float32))
    tl.store(ctr + bat, c + 1)

@triton.jit
def _put(S, SV: tl.constexpr, bat, ty, b0, cc0, d):
    base = S + bat * SV * 16384
    if ty == 1:      # RAM word: dims 0,1 ballast, 2..33 bits, 34.. zero
        dim = tl.arange(0, 128)
        bit = ((d >> tl.maximum(dim - 2, 0)) & 1).to(tl.float32) * 128.0
        val = tl.where(dim < 2, 16777216.0, tl.where(dim < 34, bit, 0.0))
        tl.store(base + b0 + dim, val)
    else:            # WROM word: 8 cells u in dims 2+cc (cc >= 63 in the row 64 below)
        lane = tl.arange(0, 8); cc = cc0 + lane
        off = b0 + tl.where(cc >= 63, 8192, 0) + 2 + cc
        tl.store(base + off, ((d >> (4 * lane)) & 15).to(tl.float32))

@triton.jit
def _inject(S0, S1, S2, S3, SV0: tl.constexpr, SV1: tl.constexpr, SV2: tl.constexpr, SV3: tl.constexpr, kind, base0, cc0, data, ictr, N):
    bat = tl.program_id(0)
    c = tl.load(ictr + bat)
    kd = tl.load(kind + bat * N + c).to(tl.int32)
    if kd != 0:
        ty = kd >> 3; sid = kd & 7
        b0 = tl.load(base0 + bat * N + c); cc = tl.load(cc0 + bat * N + c).to(tl.int32); d = tl.load(data + bat * N + c)
        if sid == 0: _put(S0, SV0, bat, ty, b0, cc, d)
        elif sid == 1: _put(S1, SV1, bat, ty, b0, cc, d)
        elif sid == 2: _put(S2, SV2, bat, ty, b0, cc, d)
        else: _put(S3, SV3, bat, ty, b0, cc, d)
    tl.store(ictr + bat, c + 1)

cfg = json.load(open(a.tuned))
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
c = core.build()
entry, segs = elf.segments(a.elf)
text = [s for s in segs if s[0] == S.ROM_BASE][0][1]
from chipformer.rv.asm import Program
prog = Program([int.from_bytes(text[i:i + 4], 'little') for i in range(0, len(text), 4)], b'', {'_start': entry}, data_base=S.RAM_BASE, rodata=b'', wrom=b'')
def machine(stt):
    m = ref.Machine.from_program(prog); m.x = list(stt['x']); m.anchor = stt['anchor']; m.n = stt['n']; m.wb = stt['wb'] & 0x7FFF
    if os.environ.get('INITWROM'): m.wrom = bytes(np.memmap(a.wrom, dtype=np.uint8, mode='r')[:8 * 32640])
    return m
core.snapshot(c, machine(states[0]))
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
# ---- injection tables ----
tabid = {id(v): k for k, v in c.tables.items() if k.startswith('ram') or k.startswith('wrom')}
loc = {}                                                  # table name -> (sublayer step idx, first value-head index j)
for si, st in enumerate(fr.ex.steps):
    if st['kind'] != 'head': continue
    j = 0
    for nd in st['heads']:
        if id(nd) in tabid: loc[tabid[id(nd)]] = (si, j)
        j += len(nd.vheads())
used = sorted({si for si, _ in loc.values()}); assert len(used) <= 4, used
sidx = {si: n for n, si in enumerate(used)}
Ss = [fr.ex.steps[si]['S'] for si in used] + [fr.ex.steps[used[0]]['S']] * (4 - len(used))
SVs = [int(x.shape[1]) for x in Ss]
I_kind = np.zeros((B, N), np.int8); I_base = np.zeros((B, N), np.int32); I_cc = np.zeros((B, N), np.int16); I_dat = np.zeros((B, N), np.int32)
wromf = np.memmap(a.wrom, dtype=np.uint8, mode='r') if a.wrom else None
isfinal = np.zeros(S.VOCAB, bool)
for t_ in range(S.VOCAB): isfinal[t_] = S.is_final(t_)
rl = lambda name, h, x: loc[f'{name}{h}'][1] + x
lut_ram = np.zeros((8192, 3), np.int64)                   # slot -> (sublayer id, j, row)
for wi in range(8192):
    h = (wi & 15) | ((wi >> 12) << 4); x = wi >> 4 & 1; si, j = loc[f'ram{h}']; lut_ram[wi] = (sidx[si], j + x, wi >> 5 & 127)
n_inj = np.zeros(B, np.int64); n_zone = np.zeros(B, np.int64)
for r in range(B):
    inj = np.load(f'{a.refdir}/inj_{chunks[r]:04d}.npy'); sh = 1 if chunks[r] == 0 else 0
    inj = inj[inj[:, 0] - sh < len(toks[r])]; to = inj[:, 0] - sh
    m1 = inj[:, 1] == 1; q = inj[m1]; i1 = to[m1]; sl = lut_ram[q[:, 2]]
    I_kind[r, i1] = (1 << 3) | sl[:, 0]; I_base[r, i1] = sl[:, 1] * 16384 + sl[:, 2] * 128; I_dat[r, i1] = q[:, 3].astype(np.uint32).view(np.int32)
    m2 = inj[:, 1] == 2; q = inj[m2]; i2 = to[m2]; wp27 = q[:, 2]; wp = wp27 & 0x7FFF
    cells = wromf[(wp27[:, None] * 8 + np.arange(8)[None, :])].astype(np.uint32)
    packed = (cells << (4 * np.arange(8, dtype=np.uint32))[None, :]).sum(1).astype(np.uint32).view(np.int32)
    sgrp = wp >> 11; ok = (sgrp < 15) | (((wp >> 7) & 15) < 15)
    hh = np.where(sgrp < 15, (wp >> 7) & 15, 16); xx = (wp >> 6) & 1; pp = wp & 63; slot = np.where(sgrp < 15, sgrp, (wp >> 7) & 15)
    for n_ in np.nonzero(ok)[0]:
        si, j = loc[f'wrom{hh[n_]}']
        I_kind[r, i2[n_]] = (2 << 3) | sidx[si]; I_base[r, i2[n_]] = (j + xx[n_]) * 16384 + pp[n_] * 128; I_cc[r, i2[n_]] = 8 * slot[n_]; I_dat[r, i2[n_]] = packed[n_]
    n_inj[r] = len(inj); n_zone[r] = int((~ok).sum())
    for n_ in np.nonzero(~ok)[0]:      # unaddressable WROM words (WP mod 2^15 >= 32640): force the whole record
        st0 = int(i2[n_]); en0 = int(np.searchsorted(np.nonzero(isfinal[toks[r]])[0], st0)); MASK[r, st0:int(np.nonzero(isfinal[toks[r]])[0][en0]) + 1] = 1
if os.environ.get('NOVD'): I_kind[I_kind >> 3 == 2] = 0
I_kind, I_base, I_cc, I_dat = (torch.from_numpy(x).to('cuda') for x in (I_kind, I_base, I_cc, I_dat)); ICTR = torch.zeros(B, dtype=torch.int32, device='cuda')
_orig_step = fr._step
def _step_inj():
    _inject[(B,)](Ss[0], Ss[1], Ss[2], Ss[3], SV0=SVs[0], SV1=SVs[1], SV2=SVs[2], SV3=SVs[3], kind=I_kind, base0=I_base, cc0=I_cc, data=I_dat, ictr=ICTR, N=N)
    _orig_step()
fr._step = _step_inj
print(f'injections per row {n_inj.tolist()[:4]}..., unaddressable-WROM VDOTs forced {n_zone.tolist()[:4]}...', flush=True)
if a.forcewin:
    lo, hi = (int(x) for x in a.forcewin.split(',')); MASK[:, lo:hi] = 1
F_fid, F_p, MASK, TREF = (x.to(dev) for x in (F_fid, F_p, MASK, TREF))
CTR = torch.zeros(B, dtype=torch.int32, device=dev); RAW = torch.zeros(B, N, dtype=torch.int32, device=dev)
orig = fusedexec._decode_embed
class OV:
    def __getitem__(self, grid):
        inner = orig[grid]
        def call(*args, **kw):
            _override[(B,)](fr.tok, args[5], args[6], args[7], F_fid, F_p, MASK, CTR, RAW, N, NPB=32)
            return inner(*args, **kw)
        return call
fusedexec._decode_embed = OV()
fr.capture()
def load_rows():
    for r in range(B):
        core.snapshot(c, machine(states[r])); fr.load(r, S.tok('GO'))
    CTR.zero_(); ICTR.zero_()
load_rows(); torch.cuda.synchronize()
print(f'rows {B}, steps/row {N:,}, forced kinds {forced_kinds}, forced fraction {[float(MASK[r,:len(toks[r])].float().mean()) for r in range(B)][:4]}', flush=True)
lens = torch.tensor([len(t) for t in toks], device=dev)
first_bad = [None] * B; nbad_free = np.zeros(B, np.int64); nbad_forced = np.zeros(B, np.int64); n_free = np.zeros(B, np.int64); n_forced = np.zeros(B, np.int64)
done = 0; t0 = time.time(); log = open(a.out + '/progress.log', 'a')
if os.environ.get('DBG'):
    fr.graph.replay(); torch.cuda.synchronize()
    kd, b0, cc, dt = (int(x[0, 0]) for x in (I_kind, I_base, I_cc, I_dat)); sid = kd & 7
    print('DBG step0 inject', dict(kind=kd, base=b0, cc=cc, data=hex(dt & 0xFFFFFFFF)), 'loc', loc.get('wrom0'), 'used', used, flush=True)
    Sx = Ss[sid]; print('S shape', tuple(Sx.shape), 'row dims', Sx[0].reshape(-1)[b0 + 0: b0 + 128].tolist()[:20], 'row+64', Sx[0].reshape(-1)[b0 + 8192: b0 + 8192 + 8].tolist(), flush=True)
    print('raw step0', int(RAW[0, 0]), S.name(int(RAW[0, 0])), 'ref', S.name(int(TREF[0, 0])), flush=True); sys.exit()
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
res = dict(chunks=chunks, rows=B, steps_per_row=[len(t) for t in toks], steps_run=done, seconds=time.time() - t0, agg_tok_per_s=done * B / (time.time() - t0), forced_kinds=forced_kinds,
           free_tokens=n_free.tolist(), forced_tokens=n_forced.tolist(), free_mismatches=nbad_free.tolist(), first_free_mismatch=first_bad, forced_raw_disagreements=nbad_forced.tolist(),
           gpu=torch.cuda.get_device_name())
json.dump(res, open(a.out + '/result_' + '_'.join(map(str, chunks[:3])) + f'_n{len(chunks)}.json', 'w'), indent=1); print(json.dumps(res), flush=True)
