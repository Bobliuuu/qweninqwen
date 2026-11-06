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
REALMAP = bool(os.environ.get('REALMAP'))   # REALMAP=1: firmware linked for the real 32 KB RAM map (ramown/fw_rom.ld). in_ram stays REAL (no aliasing, no fault-masking); only in_drom is widened so that
# DROM-class loads (real 128 KB DROM, plus the 0x10000000 weight/table blob) are answered by the circuit's DROM table, whose slot (EA[16:2]) the host fills for blob words (constants, not RAM).
if REALMAP:
    _old2 = 'in_drom = c.and_([1 - b for b in EA[18:32]] + [EA[17]], "in_drom")'; assert _old2 in _src; _src = _src.replace(_old2, 'in_drom = 1 - in_ram')
else:
    assert _old in _src; _src = _src.replace(_old, 'in_ram = one')
core = types.ModuleType('chipformer.rv.core_bringup'); core.__file__ = _core0.__file__; exec(compile(_src, 'core_bringup.py', 'exec'), core.__dict__)
from chipformer.rv import spec as S
import autotune_layers as AL
from tune_warps import W

ap = argparse.ArgumentParser(); ap.add_argument('elf'); ap.add_argument('refdir'); ap.add_argument('out'); ap.add_argument('--chunks', required=True)
ap.add_argument('--steps', type=int, default=0); ap.add_argument('--block', type=int, default=50000); ap.add_argument('--kinds', default=''); ap.add_argument('--tuned', default='autotune.json'); ap.add_argument('--forcewin', default=''); ap.add_argument('--wrom', default='')
ap.add_argument('--ramimg', action='store_true', help='restore the snapshot RAM image (refdir/ramimg_NNNN.npy, 8192 aliased words) into the circuit RAM head state at chunk start')
ap.add_argument('--inj', default='inj', help='injection file prefix in refdir: inj (all, legacy), injF (filtered), injB (blob-only), injN (no RAM injection)')
ap.add_argument('--chain', action='store_true', help='ONE continuous circuit run over consecutive chunks (no snapshot restore between chunks); compare circuit state to the reference snapshot at every chunk boundary')
ap.add_argument('--handoff', action='store_true', help='like --chain, but at each boundary READ the circuit state back (regs, pc, n, wb, bn, 32KB RAM) and re-snapshot the circuit from that read-back state (hidden lag history reset, as a chunk restart)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); CHAIN = a.chain or a.handoff
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
    elif ty == 3:    # DROM word: 4 byte values in dims cc0..cc0+3 (cc0 = 2 + 32*half + 4*k)
        ln4 = tl.arange(0, 4)
        tl.store(base + b0 + cc0 + ln4, ((d >> (8 * ln4)) & 255).to(tl.float32))
    else:            # WROM word: 8 cells u in dims 2+cc (cc >= 63 in the row 64 below)
        lane = tl.arange(0, 8); cc = cc0 + lane
        off = b0 + tl.where(cc >= 63, 8192, 0) + 2 + cc
        tl.store(base + off, ((d >> (4 * lane)) & 15).to(tl.float32))

@triton.jit
def _inject(S0, S1, S2, S3, S4, S5, S6, S7, SV0: tl.constexpr, SV1: tl.constexpr, SV2: tl.constexpr, SV3: tl.constexpr, SV4: tl.constexpr, SV5: tl.constexpr, SV6: tl.constexpr, SV7: tl.constexpr, kind, base0, cc0, data, ictr, N):
    bat = tl.program_id(0)
    c = tl.load(ictr + bat)
    kd = tl.load(kind + bat * N + c).to(tl.int32)
    if kd != 0:
        ty = kd >> 3; sid = kd & 7
        b0 = tl.load(base0 + bat * N + c); cc = tl.load(cc0 + bat * N + c).to(tl.int32); d = tl.load(data + bat * N + c)
        if sid == 0: _put(S0, SV0, bat, ty, b0, cc, d)
        elif sid == 1: _put(S1, SV1, bat, ty, b0, cc, d)
        elif sid == 2: _put(S2, SV2, bat, ty, b0, cc, d)
        elif sid == 3: _put(S3, SV3, bat, ty, b0, cc, d)
        elif sid == 4: _put(S4, SV4, bat, ty, b0, cc, d)
        elif sid == 5: _put(S5, SV5, bat, ty, b0, cc, d)
        elif sid == 6: _put(S6, SV6, bat, ty, b0, cc, d)
        else: _put(S7, SV7, bat, ty, b0, cc, d)
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
if CHAIN:
    chunk_list, chunk_lens = list(chunks), [len(t) for t in toks]; assert chunk_list == list(range(chunk_list[0], chunk_list[0] + len(chunk_list))), 'chain needs consecutive chunks'
    INJ, off = [], 0
    for ci, ch in enumerate(chunk_list):
        inj_ = np.load(f'{a.refdir}/inj_{ch:04d}.npy'); sh_ = 1 if ch == 0 else 0
        inj_ = inj_[inj_[:, 0] - sh_ < chunk_lens[ci]].copy(); inj_[:, 0] = inj_[:, 0] - sh_ + off; INJ.append(inj_); off += chunk_lens[ci]
    INJ = np.concatenate(INJ); bounds = np.cumsum(chunk_lens).tolist()
    nxt_states = []
    for ch in chunk_list:   # reference snapshot at the END of each chunk = start of chunk ch+1
        p_ = f'{a.refdir}/state_{ch + 1:04d}.pkl'; nxt_states.append(pickle.load(open(p_, 'rb')) if os.path.exists(p_) else None)
    toks, kinds, states, chunks = [np.concatenate(toks)], [np.concatenate(kinds)], [states[0]], [chunk_list[0]]; B = 1; N = len(toks[0]) + 1   # +1: the step that feeds the chunk's last token
c = core.build()
entry, segs = elf.segments(a.elf)
text = [s for s in segs if s[0] == S.ROM_BASE][0][1]
from chipformer.rv.asm import Program
prog = elf.program(a.elf) if REALMAP else Program([int.from_bytes(text[i:i + 4], 'little') for i in range(0, len(text), 4)], b'', {'_start': entry}, data_base=S.RAM_BASE, rodata=b'', wrom=b'')
def machine(stt, ch=None):
    m = ref.Machine.from_program(prog);
    if a.ramimg: m.ram = [int(v) for v in np.load(f'{a.refdir}/ramimg_{ch:04d}.npy')]
    m.x = list(stt['x']); m.anchor = stt['anchor']; m.n = stt['n']; m.wb = stt['wb'] & 0x7FFF
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
# ---- injection tables ----
tabid = {id(v): k for k, v in c.tables.items() if k.startswith('ram') or k.startswith('wrom') or (REALMAP and k.startswith('drom')) or (bool(os.environ.get('DUMPREG')) and k.startswith('regs.'))}
loc = {}                                                  # table name -> (sublayer step idx, first value-head index j)
for si, st in enumerate(fr.ex.steps):
    if st['kind'] != 'head': continue
    j = 0
    for nd in st['heads']:
        if id(nd) in tabid: loc[tabid[id(nd)]] = (si, j)
        j += len(nd.vheads())
used = sorted({si for si, _ in loc.values()}); assert len(used) <= 8, used
sidx = {si: n for n, si in enumerate(used)}
Ss = [fr.ex.steps[si]['S'] for si in used] + [fr.ex.steps[used[0]]['S']] * (8 - len(used))
SVs = [int(x.shape[1]) for x in Ss]
I_kind = np.zeros((B, N), np.int8); I_base = np.zeros((B, N), np.int32); I_cc = np.zeros((B, N), np.int16); I_dat = np.zeros((B, N), np.int32)
wromf = np.memmap(a.wrom, dtype=np.uint8, mode='r') if a.wrom else None
isfinal = np.zeros(S.VOCAB, bool)
for t_ in range(S.VOCAB): isfinal[t_] = S.is_final(t_)
rl = lambda name, h, x: loc[f'{name}{h}'][1] + x
lut_ram = np.zeros((8192, 3), np.int64)                   # slot -> (sublayer id, j, row)
for wi in range(8192):
    h = (wi & 15) | ((wi >> 12) << 4); x = wi >> 4 & 1; si, j = loc[f'ram{h}']; lut_ram[wi] = (sidx[si], j + x, wi >> 5 & 127)
lut_drom = np.zeros((32768, 4), np.int64)                 # DROM slot -> (sublayer id, base offset, row, dim offset)
if REALMAP:
    for ds in range(32768):
        h = ds >> 11; x = (ds & 2047) >> 10; wl = ds & 1023; pp = wl >> 4; half = wl >> 3 & 1; kk = wl & 7
        si, j = loc[f'drom{h}']; lut_drom[ds] = (sidx[si], (j + x) * 16384 + (pp + 64 * half) * 128, 0, 2 + 32 * half + 4 * kk)
n_ramin = np.zeros(B, np.int64); n_inj = np.zeros(B, np.int64); n_zone = np.zeros(B, np.int64)
for r in range(B):
    if CHAIN: inj = INJ; to = inj[:, 0]
    else:
        inj = np.load(f'{a.refdir}/{a.inj}_{chunks[r]:04d}.npy'); sh = 1 if chunks[r] == 0 else 0
        inj = inj[inj[:, 0] - sh < len(toks[r])]; to = inj[:, 0] - sh
    m1 = inj[:, 1] == 1; q = inj[m1]; i1 = to[m1]; sl = lut_ram[q[:, 2]]
    I_kind[r, i1] = (1 << 3) | sl[:, 0]; I_base[r, i1] = sl[:, 1] * 16384 + sl[:, 2] * 128; I_dat[r, i1] = q[:, 3].astype(np.uint32).view(np.int32)
    m3 = inj[:, 1] == 3; q = inj[m3]; i3 = to[m3]
    if len(q):
        sl = lut_drom[q[:, 2]]; I_kind[r, i3] = (3 << 3) | sl[:, 0]; I_base[r, i3] = sl[:, 1]; I_cc[r, i3] = sl[:, 3]; I_dat[r, i3] = q[:, 3].astype(np.uint32).view(np.int32)
    m2 = inj[:, 1] == 2; q = inj[m2]; i2 = to[m2]; wp27 = q[:, 2]; wp = wp27 & 0x7FFF
    cells = wromf[(wp27[:, None] * 8 + np.arange(8)[None, :])].astype(np.uint32)
    packed = (cells << (4 * np.arange(8, dtype=np.uint32))[None, :]).sum(1).astype(np.uint32).view(np.int32)
    sgrp = wp >> 11; ok = (sgrp < 15) | (((wp >> 7) & 15) < 15)
    hh = np.where(sgrp < 15, (wp >> 7) & 15, 16); xx = (wp >> 6) & 1; pp = wp & 63; slot = np.where(sgrp < 15, sgrp, (wp >> 7) & 15)
    for n_ in np.nonzero(ok)[0]:
        si, j = loc[f'wrom{hh[n_]}']
        I_kind[r, i2[n_]] = (2 << 3) | sidx[si]; I_base[r, i2[n_]] = (j + xx[n_]) * 16384 + pp[n_] * 128; I_cc[r, i2[n_]] = 8 * slot[n_]; I_dat[r, i2[n_]] = packed[n_]
    n_inj[r] = len(inj); n_ramin[r] = int((inj[:, 1] == 1).sum()); n_zone[r] = int((~ok).sum())
    fin_idx = np.nonzero(isfinal[toks[r]])[0]      # (computed once per row: was recomputed per forced word, quadratic for long chained rows)
    for n_ in np.nonzero(~ok)[0]:      # unaddressable WROM words (WP mod 2^15 >= 32640): force the whole record
        st0 = int(i2[n_]); en0 = int(np.searchsorted(fin_idx, st0)); MASK[r, st0:int(fin_idx[en0]) + 1] = 1
if os.environ.get('NOVD'): I_kind[I_kind >> 3 == 2] = 0
I_kind, I_base, I_cc, I_dat = (torch.from_numpy(x).to('cuda') for x in (I_kind, I_base, I_cc, I_dat)); ICTR = torch.zeros(B, dtype=torch.int32, device='cuda')
_orig_step = fr._step
def _step_inj():
    _inject[(B,)](*Ss, **{f'SV{i}': SVs[i] for i in range(8)}, kind=I_kind, base0=I_base, cc0=I_cc, data=I_dat, ictr=ICTR, N=N)
    _orig_step()
fr._step = _step_inj
print(f'injections per row {n_inj.tolist()[:4]}... (RAM-type {n_ramin.tolist()[:4]}), inj file {a.inj}, ramimg {a.ramimg}, unaddressable-WROM VDOTs forced {n_zone.tolist()[:4]}...', flush=True)
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
        core.snapshot(c, machine(states[r], chunks[r])); fr.load(r, S.tok('GO'))
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

if CHAIN:
    # ================= chained run: one continuous circuit run over consecutive chunks; circuit state compared with the reference snapshot at each boundary =================
    from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS
    THR = A_BIT / 2; idname = {id(v): k for k, v in c.tables.items()}; locall = {}
    for si, st in enumerate(fr.ex.steps):
        if st['kind'] != 'head': continue
        j = 0
        for nd in st['heads']:
            nv = len(nd.vheads())
            if id(nd) in idname: locall[idname[id(nd)]] = (si, j, nv)
            j += nv
    grab = lambda name: fr.ex.steps[locall[name][0]]['S'][0, locall[name][1]:locall[name][1] + locall[name][2]].float().cpu().numpy()
    bit = lambda row, cols: sum(int(row[cc] > THR) << i for i, cc in enumerate(cols))
    def extract():
        """Everything the circuit itself holds at a record boundary, read back from its head states (no host involvement)."""
        o = {}
        o['anchor'] = bit(grab('pc')[0, 0], [2 + i for i in range(16)]); o['wb'] = bit(grab('wbase')[0, 0], [2 + i for i in range(S.WP_BITS)])
        cnt = grab('count')[0, 0, 2]; o['n'] = int(round(-np.log2(cnt / A_BIT))) if cnt > 0 else None
        o['j'] = float(grab('j')[0, 0, 2]); o['vcnt'] = float(grab('vcnt')[0, 0, 2])
        for p in ('rs1', 'rs2', 'rd'):
            R = grab(f'regs.{p}')[0]; o['x_' + p] = [0] + [sum(int(R[r, c.reg_cols[i]].sum() > THR) << i for i in range(32)) for r in range(1, 32)]   # bit i = SUM of its channel columns (a write spreads a bit over F0/F1/F2/V/link channels)
        if os.environ.get('DBGREG'):
            R = grab('regs.rd')[0]
            for r in (1, 2, 4):
                nz = np.nonzero(R[r])[0]; print('DBGREG reg', r, 'nonzero cols/vals', [(int(k), float(R[r, k])) for k in nz][:80], flush=True)
            print('DBGREG reg_cols[0..40]', {i: c.reg_cols[i] for i in range(0, 32)}, flush=True)
        o['bn'] = []
        for h in range(4):
            for x in range(2): R = grab(f'bn1.{h}')[x]; o['bn'].append([bit(R[r], [2, 3, 4, 5]) for r in range(128)])
        cache = {}; ram = np.zeros(S.RAM_WORDS, np.int64)
        for wi in range(S.RAM_WORDS):
            sd, jx, rw = (int(v) for v in lut_ram[wi])
            if sd not in cache: cache[sd] = Ss[sd][0].float().cpu().numpy()
            rowv = cache[sd][jx, rw]; ram[wi] = sum(int(rowv[2 + i] + rowv[34 + i] > THR) << i for i in range(32))   # word = block A (SW data, cols 2..33) + block B (SB/SH merged word, cols 34..65)
        o['ram'] = ram
        return o
    def compare(o, rs):
        d = {}
        d['regs_rd_bad'] = [(r, o['x_rd'][r], rs['x'][r] & 0xFFFFFFFF) for r in range(1, 32) if o['x_rd'][r] != rs['x'][r] & 0xFFFFFFFF]
        d['regs_rs1_bad'] = sum(o['x_rs1'][r] != rs['x'][r] & 0xFFFFFFFF for r in range(1, 32)); d['regs_rs2_bad'] = sum(o['x_rs2'][r] != rs['x'][r] & 0xFFFFFFFF for r in range(1, 32))
        d['anchor'] = (o['anchor'], rs['anchor']); d['n'] = (o['n'], rs['n']); d['wb15'] = (o['wb'], rs['wb'] & 0x7FFF); d['wb_ref_full27'] = rs['wb']
        d['wb_high12_lost'] = bool(rs['wb'] >> 15); d['pc_ok'] = o['anchor'] == rs['anchor'] and o['n'] == rs['n']; d['wb15_ok'] = o['wb'] == rs['wb'] & 0x7FFF
        d['j_vcnt'] = (o['j'], o['vcnt'])
        if 'bn' in rs: d['bn_bad'] = sum(o['bn'][k][r] != rs['bn'][k][r] for k in range(8) for r in range(128))
        if 'touch_ea' in rs:
            te, tv = rs['touch_ea'], rs['touch_val']; m_ = te >= 0; bad = np.nonzero(m_ & (o['ram'] != tv))[0]
            d['ram_slots_touched'] = int(m_.sum()); d['ram_bad'] = int(len(bad)); d['ram_bad_first'] = [(int(i), hex(int(o['ram'][i])), hex(int(tv[i])), hex(int(te[i]))) for i in bad[:6]]
            if 'touch_op' in rs:
                to_ = rs['touch_op']; d['ram_touched_by_op'] = {k: int((m_ & (to_ == v)).sum()) for k, v in (('load', 0), ('sb_sh', 1), ('sw', 2))}
                d['ram_bad_by_op'] = {k: int(len(bad[to_[bad] == v])) for k, v in (('load', 0), ('sb_sh', 1), ('sw', 2))}
            d['ram_untouched_nonzero'] = int(((~m_) & (o['ram'] != 0)).sum())
        d['ok'] = (not d['regs_rd_bad']) and d['pc_ok'] and d['wb15_ok'] and d.get('ram_bad', 0) == 0
        return d
    bl = []; done = 0; cmp_ = 0; t0 = time.time(); log = open(a.out + '/chain_progress.log', 'a'); diverged = False
    n_free0 = n_forced0 = nbad_free0 = nbad_forced0 = 0; first_bad0 = None
    TT = len(toks[0])
    def cmp_slice(lo, hi):
        global n_free0, n_forced0, nbad_free0, nbad_forced0, first_bad0
        if hi <= lo: return
        m = MASK[0, lo:hi] != 0; neq = RAW[0, lo:hi] != TREF[0, lo:hi]; bf = neq & ~m
        n_free0 += int((~m).sum()); n_forced0 += int(m.sum()); nbad_free0 += int(bf.sum()); nbad_forced0 += int((neq & m).sum())
        if first_bad0 is None and bf.any(): first_bad0 = lo + int(bf.nonzero()[0, 0])
    for bi, bnd in enumerate(bounds):
        tgt = bnd + 1
        while done < tgt:
            k = min(a.block, tgt - done)
            for _ in range(k): fr.graph.replay()
            torch.cuda.synchronize(); done += k; cmp_slice(cmp_, min(done, bnd)); cmp_ = max(cmp_, min(done, bnd)); el = time.time() - t0
            msg = f'[{"handoff" if a.handoff else "chain"}] step {done:,}/{tgt:,} (boundary {bi + 1}/{len(bounds)}, chunk {chunk_list[bi]}) {el:.0f}s {done/el:,.0f} tok/s; free {n_free0:,} mismatches {nbad_free0}, forced {n_forced0:,} raw-disagree {nbad_forced0:,}'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
            if first_bad0 is not None: break
        if first_bad0 is not None:
            fb = first_bad0; print(f'FIRST FREE-TOKEN MISMATCH at step {fb} (chunk {chunk_list[min(np.searchsorted(bounds, fb, side="right"), len(bounds) - 1)]}): ' + ' | '.join(f'{j}:{S.name(int(TREF[0,j]))}/{S.name(int(RAW[0,j]))}/f{int(MASK[0,j])}' for j in range(max(0, fb - 16), fb + 2)), flush=True); diverged = True; break
        o = extract(); rstate = nxt_states[bi]
        if rstate is None: print(f'boundary {bi + 1}: no reference snapshot (end of trace), circuit state: anchor {o["anchor"]} n {o["n"]} wb {o["wb"]}', flush=True); continue
        d = compare(o, rstate); d['boundary_after_chunk'] = chunk_list[bi]; d['step'] = bnd; bl.append(d)
        msg = f'BOUNDARY after chunk {chunk_list[bi]} (step {bnd:,}): ok={d["ok"]} regs_rd_bad={len(d["regs_rd_bad"])} (rs1 {d["regs_rs1_bad"]}, rs2 {d["regs_rs2_bad"]}) anchor/n {d["anchor"]}/{d["n"]} wb15 {d["wb15"]} ref_wb27 {d["wb_ref_full27"]} high12_lost={d["wb_high12_lost"]} j,vcnt {d["j_vcnt"]} bn_bad {d.get("bn_bad")} ram touched {d.get("ram_slots_touched")} bad {d.get("ram_bad")} untouched-nonzero {d.get("ram_untouched_nonzero")} touched-by-op {d.get("ram_touched_by_op")} bad-by-op {d.get("ram_bad_by_op")}'
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if d['regs_rd_bad']: print('  rd mismatches (reg, circuit, ref):', d['regs_rd_bad'][:8], flush=True)
        if d.get('ram_bad'): print('  ram mismatches (slot, circuit, ref, ref_ea):', d['ram_bad_first'], flush=True)
        if a.handoff and bi + 1 < len(bounds):   # re-snapshot the circuit FROM ITS OWN READ-BACK STATE (rd copy of the regfile) and restart the chunk
            mm = machine(dict(x=o['x_rd'], anchor=o['anchor'], n=o['n'] or 0, wb=o['wb'])); mm.ram = [int(v) for v in o['ram']]; mm.bn = [list(r) for r in o['bn']]
            core.snapshot(c, mm); fr.load(0, S.tok('GO')); CTR.fill_(bnd); ICTR.fill_(bnd); done = bnd; torch.cuda.synchronize()
    res = dict(mode='handoff' if a.handoff else 'chain', chunks=chunk_list, steps=TT, steps_run=done, seconds=time.time() - t0, tok_per_s=done / (time.time() - t0), free_tokens=n_free0, free_mismatches=nbad_free0,
               first_free_mismatch=first_bad0, forced_tokens=n_forced0, forced_raw_disagreements=nbad_forced0, diverged=diverged, boundaries=bl, gpu=torch.cuda.get_device_name())
    json.dump(res, open(a.out + f'/{res["mode"]}_{chunk_list[0]}_to_{chunk_list[-1]}.json', 'w'), indent=1, default=str); print(json.dumps(res, default=str), flush=True); sys.exit(0)
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
if os.environ.get('DUMPREG'):   # debug: the three register-file ports' view of registers (comma list)
    for rg in (int(x) for x in os.environ['DUMPREG'].split(',')):
        out_ = []
        for pn in ('rs1', 'rs2', 'rd'):
            si_, j_ = loc[f'regs.{pn}']; vec = Ss[sidx[si_]][0].reshape(-1)[j_ * 16384 + rg * 128: j_ * 16384 + rg * 128 + 128].cpu().numpy()
            v_ = sum((1 << i) for i in range(32) if vec[c.reg_cols[i][0]] > 64); out_.append((pn, v_ - (1 << 32) if v_ >> 31 else v_))
        print('REG x%d' % rg, out_, flush=True)
if os.environ.get('DUMPSLOT'):   # debug: print the circuit's current RAM head state for slots (comma list), row 0
    for sl_ in (int(x) for x in os.environ['DUMPSLOT'].split(',')):
        sid_, jb, row_ = (int(v) for v in lut_ram[sl_]); vec = Ss[sid_][0].reshape(-1)[jb: jb + 128] if False else Ss[sid_][0].reshape(-1)[jb * 16384 + row_ * 128: jb * 16384 + row_ * 128 + 128].cpu().numpy()
        print('SLOT', sl_, 'ballast', vec[:2].tolist(), 'blockA', hex(sum(int(round(vec[2 + i] / 128)) << i for i in range(32)) & 0xFFFFFFFF), 'blockB', hex(sum(int(round(vec[34 + i] / 128)) << i for i in range(32)) & 0xFFFFFFFF), 'raw A[0:6]', vec[2:8].tolist(), flush=True)
for r in range(B):
    if first_bad[r] is not None:
        fb = first_bad[r]; print(f'row {r} (chunk {chunks[r]}) first free mismatch at step {fb}: ' + ' | '.join(f'{j}:{S.name(int(TREF[r,j]))}/{S.name(int(RAW[r,j]))}/f{int(MASK[r,j])}' for j in range(max(0, fb - 16), fb + 2)), flush=True)
res = dict(inj_file=a.inj, ramimg=a.ramimg, injections=n_inj.tolist(), ram_injections=n_ramin.tolist(), chunks=chunks, rows=B, steps_per_row=[len(t) for t in toks], steps_run=done, seconds=time.time() - t0, agg_tok_per_s=done * B / (time.time() - t0), forced_kinds=forced_kinds,
           free_tokens=n_free.tolist(), forced_tokens=n_forced.tolist(), free_mismatches=nbad_free.tolist(), first_free_mismatch=first_bad, forced_raw_disagreements=nbad_forced.tolist(),
           gpu=torch.cuda.get_device_name())
json.dump(res, open(a.out + '/result_' + '_'.join(map(str, chunks[:3])) + f'_n{len(chunks)}.json', 'w'), indent=1); print(json.dumps(res), flush=True)
