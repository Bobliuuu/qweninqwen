"""Constants-fetch (CF) generator: the ~625 MB read-only blob that the Qwen3-0.6B firmware reads at 0x10000000 (embedding row, norm gains, per-row matvec scales, rope row)
becomes extra ROWS of the outer model's embedding table, fetched with WROW exactly like the int4 weights (no host injection, no DROM table).
A 32-bit constant c is stored in one VDOT word (8 nibbles u_k) as the nibbles of (c + 0x88888888) mod 2^32; the firmware reads it back with 4 VDOTs whose int8 'activations' are
(1,16) on cell pair q (cells 2q,2q+1): dot_q = (u_2q-8) + 16 (u_2q+1-8) = byte_q - 136, and acc = ((d3<<8 + d2)<<8 + d1)<<8 + d0 = c (mod 2^32). Exact, integer only, no new circuit logic.
Outputs in OUT: cf_cfg.h (row/offset macros), cfetch.S (kernel), rowtab_cf.npy (uint8 [rows,3840], weight rows at their remapped ids + constant rows), cf_layout.json.
Row-id layout (so every 2.1 M-step chunk touches a contiguous ~5 K-row slice, as in the WF run): [head 3 rows: emb row(785) + rope(pos 0)] then per layer l: 4096 weight rows
(layer stride 1,966,080 words = 4096 rows exactly) followed by KL=31 constant rows (that layer's norms + scales), then 3 rows final-norm gains, LM weights rows, 317 rows LM scales.
usage: cf_gen.py CFGDIR(cfg.h blob.bin wrom.bin) OUT"""
import sys, os, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/..')
from wf_table import build_table
cfgd, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
cfg = open(cfgd + '/cfg.h').read()
def num(n): return int(re.search(r'#define %s (\d+)' % n, cfg).group(1))
def arr(n): return [[int(v) for v in row.split(',')] for row in re.search(r'%s\[[^=]*=\s*\{(.*?)\};' % n, cfg, re.S).group(1).replace('\n', '').strip()[1:-1].split('},{')]
DIM, NL, NH, NKV, HD, HID, VOCAB, NTOK = (num(k) for k in ('DIM', 'NL', 'NH', 'NKV', 'HD', 'HID', 'VOCAB', 'NTOK'))
EMB_OFF, ROPE_OFF, FN_OFF, LM_SOFF, LM_WB = (num(k) for k in ('EMB_OFF', 'ROPE_OFF', 'FINAL_NORM_OFF', 'LM_SOFF', 'LM_WB'))
tok = int(re.search(r'#define PROMPT \{(\d+)\}', cfg).group(1)); assert NTOK == 1 and num('MAXPOS') == 1, 'CF firmware: one inner token (ctx 1) only, embedding row of that token'
NORM = arr('NORM_OFF'); SOFF = arr('SOFF'); QD, KVD = NH * HD, NKV * HD
blob = np.memmap(cfgd + '/blob.bin', dtype='<u4', mode='r')
def words(byteoff, n): assert byteoff % 4 == 0; return np.asarray(blob[byteoff // 4: byteoff // 4 + n]).astype(np.uint32)
RL = 480                                                  # words per table row (3840 cells / 8)
wrom_words = (os.path.getsize(cfgd + '/wrom.bin') + 7) // 8; NWR = -(-wrom_words // RL); LSTR = 4096
# ---- regions -------------------------------------------------------------------------------------------------------------------------------------------
head = [('emb', words(EMB_OFF + tok * DIM * 4, DIM)), ('rope', words(ROPE_OFF, HD // 2))]
def layer_regions(l):
    return [('n0', words(NORM[l][0], DIM)), ('s0', words(SOFF[l][0], QD)), ('s1', words(SOFF[l][1], KVD)), ('s2', words(SOFF[l][2], KVD)), ('qn', words(NORM[l][1], HD)), ('kn', words(NORM[l][2], HD)),
            ('s3', words(SOFF[l][3], DIM)), ('n3', words(NORM[l][3], DIM)), ('s4', words(SOFF[l][4], HID)), ('s5', words(SOFF[l][5], HID)), ('s6', words(SOFF[l][6], DIM))]
def pack(regs):
    offs, cur = {}, 0
    for nme, w in regs: assert len(w) % 8 == 0; offs[nme] = cur; cur += len(w)
    return offs, np.concatenate([w for _, w in regs]), cur
hoff, hw, hn = pack(head); loff, _, ln = pack(layer_regions(0))
KH = -(-hn // RL); KL = -(-ln // RL); KF = -(-DIM // RL); KS = -(-VOCAB // RL)
assert VOCAB % 8 == 0 and all(len(w) % 8 == 0 for _, w in head)
LBLK = LSTR + KL
def lbase(l): return KH + l * LBLK                       # first weight row of layer l
LM_ROW0 = KH + NL * LBLK + KF                            # first LM weight row; LM_WB / 480 = NL * 4096 exactly
assert LM_WB % RL == 0 and LM_WB // RL == NL * LSTR, (LM_WB, NL)
LS_ROW = LM_ROW0 + (NWR - NL * LSTR)                     # LM scales rows follow the LM weight rows
NR = LS_ROW + KS
# ---- table ---------------------------------------------------------------------------------------------------------------------------------------------
wt = build_table(open(cfgd + '/wrom.bin', 'rb').read()); assert wt.shape[0] == NWR
tab = np.zeros((NR, 3840), np.uint8)
r = np.arange(NWR); new = np.where(r < NL * LSTR, r + KH + (r // LSTR) * KL, r - NL * LSTR + LM_ROW0)
tab[new] = wt; del wt
def put(row0, w):
    w = ((w.astype(np.uint64) + 0x88888888) & 0xFFFFFFFF).astype(np.uint32); n = len(w); nr = -(-n // RL)
    p = np.zeros(nr * RL, np.uint32); p[:n] = w
    nib = ((p[:, None] >> (4 * np.arange(8, dtype=np.uint32))[None, :]) & 15).astype(np.uint8)       # [word, cell m]
    tab[row0:row0 + nr] = nib.reshape(nr, 3840)
put(0, hw)
for l in range(NL): put(lbase(l) + LSTR, np.concatenate([w for _, w in layer_regions(l)]))
put(KH + NL * LBLK, words(FN_OFF, DIM)); put(LS_ROW, words(LM_SOFF, VOCAB))
np.save(out + '/rowtab_cf.npy', tab)
# ---- header --------------------------------------------------------------------------------------------------------------------------------------------
h = ['#pragma once', '// generated by cf_gen.py: constant-fetch row layout (see docs/CONST_FETCH_PATH.md)', f'#define CF_RL {RL}', f'#define CF_TOK {tok}',
     f'#define CF_HEAD_ROW 0', f'#define CFO_EMB {hoff["emb"]}', f'#define CFO_ROPE {hoff["rope"]}',
     f'#define CF_LBASE(l) ({KH + LSTR} + (l) * {LBLK})   /* first constant row of layer l */', f'#define CFW_SHIFT(l) ({KH} + (l) * {KL})        /* weight row id = g/480 + shift */',
     f'#define CF_FN_ROW ({KH + NL * LBLK})', f'#define CFW_SHIFT_LM ({LM_ROW0 - NL * LSTR})', f'#define CF_LS_ROW {LS_ROW}', f'#define CF_NROWS {NR}']
for nme, o in loff.items(): h.append(f'#define CFO_{nme.upper()} {o}')
open(out + '/cf_cfg.h', 'w').write('\n'.join(h) + '\n')
json.dump(dict(rows=NR, weight_rows=NWR, KH=KH, KL=KL, KF=KF, KS=KS, LM_ROW0=LM_ROW0, LS_ROW=LS_ROW, token=tok, const_words=hn + NL * ln + DIM + VOCAB, head=hoff, layer=loff,
               const_rows=KH + NL * KL + KF + KS, table_bytes_bf16_hidden4096=NR * 4096 * 2), open(out + '/cf_layout.json', 'w'), indent=1)
# ---- kernel --------------------------------------------------------------------------------------------------------------------------------------------
ACC = ['t0', 't1', 't2', 't3', 't4', 'a5', 'a6', 'a7']
s = ['  .text', '  .globl cf_fetch_raw', '  .type cf_fetch_raw,@function',
     '# cf_fetch_raw(a0 = dst int32*, a1 = nwin (8 constants per window), a2 = first window in row (0..59), a3 = row): dst[8 w + i] = constant (row, window, word i) for nwin windows (rows wrap at 60 windows)',
     'cf_fetch_raw:', '  lui t5, 1', '  addi t5, t5, 1', '  lui t6, 0x10010', '1:']
s += [f'  li {a}, 0' for a in ACC]
s += ['  slli a4, a2, 3', '  addi a4, a4, 511', '  .insn r 0x0B, 6, 0, zero, a4, zero', '  .insn r 0x0B, 6, 1, zero, a3, zero']   # WB (wrow at WP = 8 win + 511), WROW a3
for rnd, (r1, r2) in enumerate((('zero', 't6'), ('zero', 't5'), ('t6', 'zero'), ('t5', 'zero'))):   # byte 3, 2, 1, 0: VDOT acc, rs1, rs2
    s += [f'  .insn r 0x0B, 5, 0, {a}, {r1}, {r2}' for a in ACC]
    if rnd < 3: s += [f'  slli {a}, {a}, 8' for a in ACC]
s += [f'  sw {a}, {4 * i}(a0)' for i, a in enumerate(ACC)]
s += ['  addi a0, a0, 32', '  addi a2, a2, 1', '  li a4, 60', '  bne a2, a4, 2f', '  li a2, 0', '  addi a3, a3, 1', '2:', '  addi a1, a1, -1', '  bnez a1, 1b', '  ret', '  .size cf_fetch_raw, .-cf_fetch_raw']
open(out + '/cfetch.S', 'w').write('\n'.join(s) + '\n')
print(json.dumps(json.load(open(out + '/cf_layout.json'))))
