"""Run every kernel on the Qwenputer reference Machine, verify bit-exact vs numpy, print the results table, write res_kernels.json.
usage: QWENPUTER_NPU=<qwenputer npu-branch>/experimental/riscv python run_all.py"""
import json, time
from kernels import *
from kernels_ext import *

K = 1024
rows = []
def add(group, k, W, x, y, dist, note=''):
    t = time.time(); r = k.run(W, x, y)
    assert r['ok'], ('NOT BIT-EXACT', k.name)
    d = dict(group=group, name=k.name, dist=dist, instr_per_mac=round(r['ipm'], 4), tok_per_mac=round(r['tpm'], 4), macs=r['macs'],
             instr=r['instr'], tokens=r['tokens'], by_op_tokens={o: r['res'].by_op[o] for o in r['res'].by_op}, note=note)
    rows.append(d); print(f"[{dist:7s}] {k.name:68s} instr/MAC {d['instr_per_mac']:7.4f}  tok/MAC {d['tok_per_mac']:7.4f}  ({time.time()-t:.0f}s)", flush=True)

for dist in ('real', 'uniform'):
    Wp, xp, yp = problem(64, K, dist=dist)               # plain kernels: int8 weights must fit the 128 KB DROM
    add('plain', k0_naive(Wp, xp), Wp, xp, yp, dist)
    add('plain', k1_unroll(Wp, xp), Wp, xp, yp, dist)
    add('plain', k1b_lane_pair(Wp, xp), Wp, xp, yp, dist)
    add('smaqa', k2_smaqa(Wp, xp), Wp, xp, yp, dist)
    W, x, y = problem(220, K, dist=dist)
    add('vdot', k3_vdot_row(W, x), W, x, y, dist)
    add('vdot', k5_vdk_like(W, x), W, x, y, dist, 'structure of experiments/circuit/compiler/fw (existing)')
    for RB, C in ((8, 4), (16, 4), (20, 2), (22, 2), (23, 1)):
        if 220 % RB == 0 or True:
            Wb, xb, yb = problem(RB * -(-220 // RB), K, dist=dist)
            add('vdot', k4_vdot_blocked(Wb, xb, RB, C), Wb, xb, yb, dist)
    add('vdot', k4_vdot_blocked(W, x, 22, 2, wb_per_chunk=True), W, x, y, dist, 'one WB per chunk (no gap constraint)')
    rng = np.random.default_rng(7); m = rng.integers(1, 128, 220)
    k6 = k6_vdot_requant(W, x, m, 12, RB=20, C=2)
    r = run(k6.src, k6.ram, k6.rodata, k6.wrom)
    got = np.array([(r.byte(0x1800 + i) ^ 128) - 128 for i in range(220)]); assert (got == q_ref(y, m, 12)).all()
    rows.append(dict(group='vdot', name='K6 K4(RB=20,C=2) + requant/clamp/pack epilogue (int8 out, bit-exact)', dist=dist,
                     instr_per_mac=round(r.instr / (220 * K), 4), tok_per_mac=round(r.ntok / (220 * K), 4), macs=220 * K, instr=r.instr, tokens=r.ntok,
                     by_op_tokens=dict(r.by_op), note='see README'))
    print(f"[{dist:7s}] {k6.name:68s} instr/MAC {rows[-1]['instr_per_mac']:7.4f}  tok/MAC {rows[-1]['tok_per_mac']:7.4f}", flush=True)
    k7 = k7_vdot_argmax(W, x, m, RB=22, C=2); r7 = run(k7.src, k7.ram, k7.rodata, k7.wrom); assert r7.sword(OUT) == int(np.argmax(y * m))
    rows.append(dict(group='vdot', name='K7 LM-head: K4(RB=22,C=2) + fused scale+argmax (no store, no softmax)', dist=dist,
                     instr_per_mac=round(r7.instr / (220 * K), 4), tok_per_mac=round(r7.ntok / (220 * K), 4), macs=220 * K, instr=r7.instr, tokens=r7.ntok,
                     by_op_tokens=dict(r7.by_op), note=''))
    print(f"[{dist:7s}] {k7.name:68s} instr/MAC {rows[-1]['instr_per_mac']:7.4f}  tok/MAC {rows[-1]['tok_per_mac']:7.4f}", flush=True)
    # proposed instructions
    add('proposed', k8_vdota_row(W, x), W, x, y, dist)
    add('proposed', k8_vdota_row(W, x, wide=True), W, x, y, dist)
    x4 = np.clip(np.rint(x / 127 * 7), -8, 7).astype(np.int64); y4 = W @ x4
    add('proposed', k9_vdot16_blocked(W, x4, 22, 2), W, x4, y4, dist, 'activations quantised to int4 (A4)')
json.dump(rows, open('res_kernels.json', 'w'), indent=1)
