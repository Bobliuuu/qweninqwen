"""Full Qwen3-0.6B int4 firmware on Qwenputer's software reference machine (ref.Machine) with ENLARGED memories (software model only):
  RAM 4 MB, RAM relocated to 0x1000000 (ROM stays at the real 0x40000 so link values match the circuit's; code is 9 KB), DROM = the 624.6 MB weight/table blob at 0x10000000,
  WROM = 74.5M words (27-bit weight pointer; the WB *token* still carries only the low 15 bits of the base, as in the real vocabulary; the full pointer is machine state).
Writes the exact trace (uint32 tokens, uint8 op-kind per token) in chunks + stats.  Resumable (ckpt.pkl).
usage: ref_big.py ELF BLOB WROM OUTDIR [QP_RISCV_DIR]"""
import sys, os, time, json, pickle, numpy as np
elf_path, blob_path, wrom_path, out = sys.argv[1:5]
sys.path.insert(0, sys.argv[5] if len(sys.argv) > 5 else os.path.expanduser('~/qp/experimental/riscv'))
from chipformer.rv import spec as S, ref, elf as E
os.makedirs(out, exist_ok=True)
blob = open(blob_path, 'rb').read(); wrom = open(wrom_path, 'rb').read()
S.RAM_BASE, S.RAM_WORDS = 0x1000000, 1 << 20; S.SP_INIT = S.RAM_BASE + 4 * S.RAM_WORDS
S.DROM_BASE, S.DROM_WORDS = 0x10000000, (len(blob) + 3) // 4
S.WP_BITS = 27; S.WROM_WORDS = (len(wrom) + 7) // 8; S.WROM_BASE = 0x80000000
M27 = (1 << 27) - 1
KIND = dict(other=0, load=1, vdot=2, store=3, wb=4, ecall=5, branch=6, alu=7, div=8, smaqa=9, custom=10)

class Big(ref.Machine):
    def setup(self):
        self.wbset = set(); self.kind = [0] * len(self.rom)
        for i, w in enumerate(self.rom):
            if w == 0: continue
            op = ref.decode(w).op
            if op == 'WB': self.wbset.add(i)
            self.kind[i] = (KIND['load'] if op in ref.LOAD_OPS else KIND['store'] if op in ref.STORE_OPS else KIND['vdot'] if op == 'VDOT' else KIND['smaqa'] if op == 'SMAQA'
                            else KIND['wb'] if op == 'WB' else KIND['ecall'] if op == 'ECALL' else KIND['branch'] if op in ref.BRANCH_OPS or op in ('JAL', 'JALR')
                            else KIND['div'] if op in ref.DIV_OPS else KIND['custom'] if op in ref.CUSTOM else KIND['alu'])
    def record(self):
        i = self.anchor + self.n
        if self.n != S.N_MAX and i in self.wbset:
            a = self.x[ref.decode(self.rom[i]).rs1]
            self.wb = (a - i - 1) & M27; self.n += 1
            return [S.tok('WB', self.wb & 0x7FFF)]
        return super().record()
    def wrom_cells(self, wp):
        o = 8 * (wp & M27); c = self.wrom[o:o + 8]
        return list(c) if len(c) == 8 else list(c) + [0] * (8 - len(c))

prog = E.program(elf_path)
m = Big.from_program(prog); m.drom = blob; m.wrom = wrom; m.setup()
ckp = out + '/ckpt.pkl'; chunk = 0; ntok = 0; t0 = time.time(); fam = {}
if os.path.exists(ckp):
    st = pickle.load(open(ckp, 'rb')); chunk, ntok, fam = st.pop('chunk'), st.pop('ntok'), st.pop('fam')
    for k, v in st.items(): setattr(m, k, v)
    print(f'resumed at chunk {chunk} steps {m.steps}', flush=True)
else:
    first = np.array([S.tok('GO')], dtype=np.uint32); ntok = 1
CH = 1 << 24; buf = []; kbuf = []; nb = 0
def flush():
    global chunk, buf, kbuf, nb
    np.save(f'{out}/tok_{chunk:04d}.npy', np.array(buf, dtype=np.uint32)); np.save(f'{out}/kind_{chunk:04d}.npy', np.array(kbuf, dtype=np.uint8))
    chunk += 1; buf = []; kbuf = []; nb = 0
    st = {k: getattr(m, k) for k in ('x', 'ram', 'bn', 'anchor', 'n', 'halted', 'steps', 'wb', 'out', 'fault', 'exit_code')}
    st.update(chunk=chunk, ntok=ntok, fam=fam); pickle.dump(st, open(ckp + '.tmp', 'wb')); os.replace(ckp + '.tmp', ckp)
if chunk == 0 and not buf: buf.append(S.tok('GO')); kbuf.append(0); nb = 1; ntok = 1
rom_kind = m.kind; rec = m.record; last = time.time()
while not m.halted:
    i = m.anchor + m.n; k = rom_kind[i] if m.n != S.N_MAX else 0
    r = rec(); L = len(r); buf += r; kbuf += [k] * L; nb += L; ntok += L
    if nb >= CH:
        flush(); print(f'{ntok:,} tokens {m.steps:,} instr {time.time()-t0:.0f}s ({m.steps/(time.time()-t0):,.0f} instr/s)', flush=True)
if buf: flush()
fams = {}
for c in range(chunk):
    t = np.load(f'{out}/tok_{c:04d}.npy'); b = np.zeros(S.VOCAB, dtype=np.int64)
    for f, (base, size) in S.FAMILIES.items(): fams[f] = fams.get(f, 0) + int(((t >= base) & (t < base + size)).sum())
    for nme, v in S.CONTROLS.items(): fams[nme] = fams.get(nme, 0) + int((t == v).sum())
instr = sum(fams.get(f, 0) for f in ('F0', 'F1', 'F2', 'J', 'BYTE', 'WB')) + sum(fams.get(f, 0) for f in ('NT', 'ST0', 'ST1', 'SS0', 'SS1', 'HALT'))
res = dict(trace_tokens=ntok, instructions_machine=m.steps, instructions_by_final_tokens=instr, refreshes=fams.get('R', 0), families=fams, out_hex=bytes(m.out).hex(),
           argmax=int.from_bytes(bytes(m.out[:4]), 'little', signed=True), fault=m.fault, exit_code=m.exit_code, chunks=chunk, wall_s=time.time() - t0)
json.dump(res, open(out + '/summary.json', 'w'), indent=1); print(json.dumps(res), flush=True)
