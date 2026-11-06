"""pcat.py REFDIR CHUNK STEP [ELF]: the ROM word index / instruction at trace step STEP of a chunk (follows anchor/n through the records)."""
import sys, pickle, numpy as np
sys.path.insert(0, '.')
from chipformer.rv import spec as S, ref, elf as E
d, ch, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = np.load(f'{d}/tok_{ch:04d}.npy'); st = pickle.load(open(f'{d}/state_{ch:04d}.pkl', 'rb'))
if ch == 0: t = t[1:]
anchor, n = st['anchor'], st['n']; i = 0; rec_start = 0
entry, segs = E.segments(sys.argv[4]); text = [s for s in segs if s[0] == S.ROM_BASE][0][1]
while i <= step:
    pc = anchor + n; rs = i
    while True:   # consume one record
        fam, pay = S.family(int(t[i])); i += 1
        if fam in ('F0', 'F1', 'F2', 'NT', 'ST0', 'ST1', 'SS0', 'SS1', 'BYTE', 'WB', 'HALT', 'J', 'R'):
            if fam in ('J', 'R'): anchor, n = pay, 0
            else: n += 1
            break
    if rs <= step < i:
        w = int.from_bytes(text[4 * pc:4 * pc + 4], 'little'); print('record tokens', rs, i - 1, 'pc word', pc, hex(w), ref.decode(w), [S.name(int(x)) for x in t[rs:i]]); break
if len(sys.argv) > 5:   # also list every record up to STEP whose rd == argv[5]
    rd = int(sys.argv[5]); anchor, n = st['anchor'], st['n']; i = 0
    while i <= step:
        pc = anchor + n; rs = i
        while True:
            fam, pay = S.family(int(t[i])); i += 1
            if fam in ('F0', 'F1', 'F2', 'NT', 'ST0', 'ST1', 'SS0', 'SS1', 'BYTE', 'WB', 'HALT', 'J', 'R'):
                if fam in ('J', 'R'): anchor, n = pay, 0
                else: n += 1
                break
        w = int.from_bytes(text[4 * pc:4 * pc + 4], 'little'); ins = ref.decode(w)
        if ins.rd == rd and fam in ('F0', 'F1', 'F2'): print(rs, 'pc', pc, ins.op, ins.rs1, ins.rs2, ins.imm, [S.name(int(x)) for x in t[rs:i]])
