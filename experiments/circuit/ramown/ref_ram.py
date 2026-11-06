"""ref_ram.py ELF BLOB WROM OUTDIR LASTCHUNK [QP_RISCV_DIR]
Like fullrun/ref_big.py (same enlarged-memory reference machine, same trace), but it also
  * dumps the aliased RAM image at each chunk start (ramimg_NNNN.npy, 8192 words: the 4 MB model RAM folded into the circuit's 8192 slots, slot = word index & 0x1FFF, higher index wins),
  * runs three software models of the CIRCUIT's own 8192-slot RAM next to the reference, one per injection policy, and writes one injection file per policy:
      inj_NNNN.npy   legacy: every LOAD/SB/SH injected (identical to ref_big)
      injF_NNNN.npy  FILTERED: inject a LOAD (or the old word before a SB/SH) only if the circuit's slot does not already hold the true word
      injB_NNNN.npy  BLOB-ONLY: like F but only for blob (DROM-region) addresses; stack/activation RAM is never injected
      injN_NNNN.npy  NONE: no RAM injection at all
    (VDOT rows, type 2, are identical in all files.)  Per-chunk stats with a cause breakdown in stats_NNNN.json.
Circuit RAM model: SW writes the slot; SB/SH merge into the slot's current (possibly wrong) word; an injection overwrites the slot with the true word.
Stops after writing chunk LASTCHUNK (so chunks 0..LASTCHUNK are produced)."""
import sys, os, time, json, numpy as np
elf_path, blob_path, wrom_path, out, last = sys.argv[1:6]; last = int(last)
sys.path.insert(0, sys.argv[6] if len(sys.argv) > 6 else os.path.expanduser('~/qp-npu/experimental/riscv'))
from chipformer.rv import spec as S, ref, elf as E
os.makedirs(out, exist_ok=True)
blob = open(blob_path, 'rb').read(); wrom = open(wrom_path, 'rb').read()
S.RAM_BASE, S.RAM_WORDS = 0x1000000, 1 << 20; S.SP_INIT = S.RAM_BASE + 4 * S.RAM_WORDS
S.DROM_BASE, S.DROM_WORDS = 0x10000000, (len(blob) + 3) // 4
S.WP_BITS = 27; S.WROM_WORDS = (len(wrom) + 7) // 8; S.WROM_BASE = 0x80000000
M27 = (1 << 27) - 1; M32 = 0xFFFFFFFF
MEMLOAD = ('LB', 'LBU', 'LH', 'LHU', 'LW')
KIND = dict(other=0, load=1, vdot=2, store=3, wb=4, ecall=5, branch=6, alu=7, div=8, smaqa=9, custom=10)
SRC = ['zero', 'image', 'ownstore', 'injected']
CAUSES = ['cold', 'alias', 'merge']

class Sim:
    """software model of the circuit's 32 KB RAM under one injection policy"""
    def __init__(self, name, policy): self.name, self.policy = name, policy; self.inj = []; self.reset(None)
    def reset(self, ram):
        self.cs = [0] * 8192; self.own = [-1] * 8192; self.src = [0] * 8192
        if ram is not None:
            a = np.array(ram, dtype=np.int64); idx = np.nonzero(a)[0]
            for wi in idx.tolist(): s = wi & 0x1FFF; self.cs[s] = int(a[wi]); self.own[s] = wi; self.src[s] = 1
        self.st = dict(loads=0, sbsh=0, hit_zero=0, hit_image=0, hit_ownstore=0, hit_injected=0, hit_coincidence=0, need=0, need_blob=0, need_ram=0, injected=0,
                       need_cold=0, need_alias=0, need_merge=0, need_sbsh=0, merge_err=0)
    def cause(self, s, wi):
        if self.own[s] < 0: return 'cold'
        return 'alias' if self.own[s] != wi else 'merge'
    def load(self, nb, ea, w):
        st = self.st; wi = ea >> 2; s = wi & 0x1FFF; st['loads'] += 1
        if self.cs[s] == w:
            if self.own[s] == wi: st['hit_' + SRC[self.src[s]]] += 1
            elif self.own[s] < 0: st['hit_zero'] += 1
            else: st['hit_coincidence'] += 1
            return
        st['need'] += 1; blob = ea >= 0x10000000; st['need_blob' if blob else 'need_ram'] += 1; st['need_' + self.cause(s, wi)] += 1
        if self.policy == 'all' or (self.policy == 'blob' and blob):
            self.inj.append((nb, 1, s, w)); st['injected'] += 1; self.cs[s] = w; self.own[s] = wi; self.src[s] = 3
    def store(self, nb, ea, op, b, trueold):
        st = self.st; wi = ea >> 2; s = wi & 0x1FFF
        if op == 'SW': self.cs[s] = b & M32; self.own[s] = wi; self.src[s] = 2; return
        st['sbsh'] += 1
        if self.cs[s] != trueold:
            st['need'] += 1; st['need_sbsh'] += 1; st['need_ram'] += 1; st['need_' + self.cause(s, wi)] += 1
            if self.policy in ('all', 'blob'):   # blob policy never injects RAM-region words; 'all' does
                if self.policy == 'all': self.inj.append((nb, 1, s, trueold)); st['injected'] += 1; self.cs[s] = trueold; self.own[s] = wi
            if self.cs[s] != trueold: st['merge_err'] += 1
        n = 1 if op == 'SB' else 2; sh, mask = 8 * (ea & 3), (1 << 8 * n) - 1
        self.cs[s] = self.cs[s] & ~(mask << sh) & M32 | (b & mask) << sh; self.own[s] = wi; self.src[s] = 2

class Big(ref.Machine):
    def setup(self):
        self.wbset = set(); self.kind = [0] * len(self.rom); self.dec = [None] * len(self.rom); self.inj = []; self.inj2 = []; self.nbuf = lambda: 0
        self.sims = [Sim('F', 'all'), Sim('B', 'blob'), Sim('N', 'none')]
        for i, w in enumerate(self.rom):
            if w == 0: continue
            self.dec[i] = ref.decode(w); op = self.dec[i].op
            if op == 'WB': self.wbset.add(i)
            self.kind[i] = (KIND['load'] if op in ref.LOAD_OPS else KIND['store'] if op in ref.STORE_OPS else KIND['vdot'] if op == 'VDOT' else KIND['smaqa'] if op == 'SMAQA'
                            else KIND['wb'] if op == 'WB' else KIND['ecall'] if op == 'ECALL' else KIND['branch'] if op in ref.BRANCH_OPS or op in ('JAL', 'JALR')
                            else KIND['div'] if op in ref.DIV_OPS else KIND['custom'] if op in ref.CUSTOM else KIND['alu'])
    def record(self):
        i = self.anchor + self.n
        if self.n != S.N_MAX:
            d = self.dec[i]
            if d is not None:
                op = d.op
                if op in MEMLOAD or op in ('SB', 'SH', 'SW'):
                    ea = (self.x[d.rs1] + d.imm) & M32
                    ri = self.ram_index(ea)
                    if ri is not None: w = self.ram[ri]
                    else:
                        w = self.drom_word(ea); w = 0 if w is None else w
                    if op != 'SW':
                        self.inj.append((self.nbuf(), 1, (ea >> 2) & 0x1FFF, w)) if op != 'SW' else None
                    for sm in self.sims:
                        if op in MEMLOAD: sm.load(self.nbuf(), ea, w)
                        else: sm.store(self.nbuf(), ea, op, self.x[d.rs2], w)
                elif op == 'VDOT':
                    self.inj2.append((self.nbuf(), 2, (self.wb + i) & M27, 0))
        if self.n != S.N_MAX and i in self.wbset:
            a = self.x[ref.decode(self.rom[i]).rs1]
            self.wb = (a - i - 1) & M27; self.n += 1
            return [S.tok('WB', self.wb & 0x7FFF)]
        return super().record()

prog = E.program(elf_path)
m = Big.from_program(prog); m.drom = blob; m.wrom = wrom; m.setup()
chunk = 0; ntok = 1; t0 = time.time()
def dump_state():
    pickle_dump = __import__('pickle').dump
    pickle_dump(dict(x=list(m.x), anchor=m.anchor, n=m.n, wb=m.wb, steps=m.steps, tok_start=ntok, halted=m.halted), open(f'{out}/state_{chunk:04d}.pkl', 'wb'))
    a = np.zeros(8192, np.int64); r = np.array(m.ram, dtype=np.int64); idx = np.nonzero(r)[0]; a[idx & 0x1FFF] = r[idx]
    np.save(f'{out}/ramimg_{chunk:04d}.npy', a.astype(np.uint32))
    for sm in m.sims: sm.reset(m.ram)
CH = 1 << 21; buf = []; kbuf = []; nb = 0
def flush():
    global chunk, buf, kbuf, nb
    np.save(f'{out}/tok_{chunk:04d}.npy', np.array(buf, dtype=np.uint32)); np.save(f'{out}/kind_{chunk:04d}.npy', np.array(kbuf, dtype=np.uint8))
    r2 = np.array(m.inj2, dtype=np.int64).reshape(-1, 4)
    def mk(rows): 
        a = np.concatenate([np.array(rows, dtype=np.int64).reshape(-1, 4), r2]); return a[np.argsort(a[:, 0], kind='stable')]
    np.save(f'{out}/inj_{chunk:04d}.npy', mk(m.inj))
    for sm in m.sims: np.save(f'{out}/inj{sm.name}_{chunk:04d}.npy', mk(sm.inj))
    json.dump(dict(chunk=chunk, steps_tokens=len(buf), ram_loads_sbsh_legacy_injections=len(m.inj), vdot_injections=len(m.inj2), policies={sm.name: dict(policy=sm.policy, **sm.st) for sm in m.sims}),
              open(f'{out}/stats_{chunk:04d}.json', 'w'), indent=1)
    print(f'chunk {chunk}: legacy RAM injections {len(m.inj)}, ' + ', '.join(f"{sm.name}: inj {sm.st['injected']} need {sm.st['need']}" for sm in m.sims), flush=True)
    m.inj = []; m.inj2 = []; chunk += 1; buf = []; kbuf = []; nb = 0
    dump_state()
buf.append(S.tok('GO')); kbuf.append(0); nb = 1; dump_state()
m.nbuf = lambda: nb
rom_kind = m.kind; rec = m.record
while not m.halted and chunk <= last:
    i = m.anchor + m.n; k = rom_kind[i] if m.n != S.N_MAX else 0
    r = rec(); L = len(r); buf += r; kbuf += [k] * L; nb += L; ntok += L
    if nb >= CH: flush()
print('done', chunk, f'{time.time()-t0:.0f}s', flush=True)
