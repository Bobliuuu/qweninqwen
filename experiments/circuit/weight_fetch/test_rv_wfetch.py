"""Weight fetch: VDOT weights from rows of the outer model's own table (ROW tokens), teacher-forced.

The table rows (uint8 cells 0..15, ROW_CELLS per row) are the ROW tokens' embeddings; the circuit
selects the slot's cells, unpacks the int4 bits, latches them in a DeltaNet head and feeds the VDOT
multiplier; the reference machine reads the same table. WROM_BIN=<wrom.bin> takes the rows from
the real Qwen3-0.6B int4 stream (one nibble per byte), else random.
"""
import os
import random
import sys
import time

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import core, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

sys.path.insert(0, os.path.dirname(__file__))
from test_rv_core import EXIT  # noqa: E402
from test_rv_vdot import sx32, word  # noqa: E402

NROWS = 6


def make_table(seed=0, nrows=NROWS):
    """Table entries [rows, ROW_CELLS]: int4 cells (the real Qwen3-0.6B stream if WROM_BIN is set,
    else random) packed ROW_EPW entries per word of 8 cells (S.pack_word)."""
    nw = nrows * S.ROW_WORDS * 8
    p = os.environ.get("WROM_BIN")
    if p and os.path.exists(p):
        off = int(os.environ.get("WROM_OFF", 7_000_003))
        raw = (np.fromfile(p, dtype=np.uint8, count=off + nw)[off:] & 15)
    else:
        raw = np.random.default_rng(seed).integers(0, 16, nw, dtype=np.uint8)
    words = raw.reshape(nrows * S.ROW_WORDS, 8)
    ents = np.array([S.pack_word(w) for w in words], dtype=np.uint8)
    return ents.reshape(nrows, S.ROW_CELLS)


def wf_program(rng, tab, W, n_blocks, per_block):
    """Blocks of `wb; wrow; vdot x per_block` (other instructions between some), W-aligned."""
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {sx32(word(rng))}" for r in regs]
    for _ in range(n_blocks):
        row = rng.randrange(len(tab) + 1)  # one past the table too: reads as zeros
        win = rng.randrange(-(-S.ROW_WORDS // W))
        x = (win * W - 1) & 0x7FFF  # wb: the next word (wrow) is at WP = x, the first vdot at win * W
        acc = rng.choice(["s2", "s3"])
        init = rng.choice([0, rng.randint(-2**14, 2**14), rng.randint(-2**31, 2**31 - 1)])
        lines += [f"li s4, {x & 0x7FFF}", f"li s5, {row}", f"li {acc}, {init}", "wb s4", "wrow s5"]
        for i in range(per_block):
            if i and rng.random() < 0.0:
                lines.append("nop")
            lines.append(f"vdot {acc}, {rng.choice(regs)}, {rng.choice(regs)}")
    return "\n".join(lines) + EXIT


def check(c, prog, tab, W):
    m = ref.Machine.from_program(prog)
    m.rowtab, m.wrow_w = tab, W
    core.snapshot(c, m)
    m2 = ref.Machine.from_program(prog)
    m2.rowtab, m2.wrow_w = tab, W
    toks = m2.run()
    t0 = time.time()
    mat = ir.evaluate(c, toks, check=True)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    return toks, time.time() - t0


@pytest.fixture(scope="module", params=[1, 8])
def circ_tab(request):
    W = request.param
    tab = make_table()
    S.ROW_TABLE = tab
    return W, tab, core.build(wrow=W)


@pytest.mark.parametrize("seed", range(3))
def test_weight_fetch(circ_tab, seed):
    W, tab, c = circ_tab
    rng = random.Random(seed)
    toks, dt = check(c, assemble(wf_program(rng, tab, W, 4, 1 if W == 1 else min(W, 5))), tab, W)
    fams = [S.family(t)[0] for t in toks]
    assert "ROW" in fams and fams[-1] == "HALT"


def test_negative_control(circ_tab):
    """The reference reads a table with every cell shifted by one: the circuit must disagree."""
    W, tab, c = circ_tab
    rng = random.Random(7)
    prog = assemble(wf_program(rng, tab, W, 4, 1 if W == 1 else min(W, 5)))
    bad_tab = (tab + 1) & ((1 << S.ROW_ENT_BITS) - 1)
    with pytest.raises(AssertionError):
        check(c, prog, bad_tab, W)


def test_stats(circ_tab):
    W, tab, c = circ_tab
    st = c.stats()
    print(f"\nWF_STATS W={W} {st}")
    assert st["max_heads"] <= c.target.key_heads and st["max_units"] <= c.target.intermediate


# ---- a chunk of N VDOTs, weights chosen first and the table laid out for them (PC-linear) -------
REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]


def chunk(rng, W, n_vdot, rows=True):
    """n_vdot VDOTs with random weights (cells 0..15). rows=True: one `wrow` before every W VDOTs
    (the latch window), the table laid out so the circuit reads exactly these weights, with D
    constant (no WB per window: the window is WP[7:logW] at the ROW, the word is WP mod W);
    rows=False: the same VDOTs on the WROM (the baseline). Returns
    (source, table, wrom_bytes, expected accumulator, token-free info)."""
    vals = [word(rng) for _ in REGS]
    pick = [(rng.randrange(10), rng.randrange(10)) for _ in range(n_vdot)]
    want = [[rng.randrange(16) for _ in range(8)] for _ in range(n_vdot)]
    pre = [f"li {r}, {sx32(v)}" for r, v in zip(REGS, vals)] + ["li s2, 0", "li s5, 0", "wb zero"]
    n0 = len(assemble("\n".join(pre) + EXIT).text) - len(assemble(EXIT).text)  # words before body
    w_wb = n0 - 1
    D = (-(w_wb + 1)) & 0x7FFF  # WB zero: D = -w - 1, so wp(w) = w - w_wb - 1
    body, w = [], n0
    used: dict = {}  # row -> {col: u}
    cur_row, tab_rows = 0, {}
    wrom = bytearray()
    exp = 0
    i = 0
    while i < n_vdot:
        if rows:
            wp_next = (D + w + 1) & 0x7FFF
            win = (wp_next & ((1 << S.ROW_SLOT_BITS) - 1)) // W
            if win * W >= S.ROW_WORDS:  # this PC maps past the row's last window: skip a word
                body.append("nop")
                w += 1
                continue
            grp = list(range(i, min(i + W, n_vdot)))
            need = {}
            for j, v in enumerate(grp):
                wiw = (D + w + 1 + j) & (W - 1)
                for m_, e_ in enumerate(S.pack_word(want[v])):
                    need[win * S.ROW_EPW * W + S.ROW_EPW * wiw + m_] = e_
            r = cur_row
            while any(used.get(r, {}).get(c_, u) != u for c_, u in need.items()):
                r += 1
            if r != cur_row:
                body.append(f"li s5, {r}")
                w += 1
                cur_row = r
                continue  # the li moved the PC: recompute the window and the word slots
            used.setdefault(r, {}).update(need)
            body.append("wrow s5")
            w += 1
        else:
            grp = [i]
        for v in grp:
            a, b = REGS[pick[v][0]], REGS[pick[v][1]]
            body.append(f"vdot s2, {a}, {b}")
            if not rows:
                wrom += bytes(want[v])
            w += 1
            exp = (exp + RFvdot(vals[pick[v][0]], vals[pick[v][1]], want[v])) & 0xFFFFFFFF
        i = grp[-1] + 1
    src = "\n".join(pre + body) + EXIT
    tab = np.zeros((max(used) + 1 if used else 1, S.ROW_CELLS), dtype=np.uint8)
    for r, d in used.items():
        for c_, u in d.items():
            tab[r][c_] = u
    return src, tab, bytes(wrom), exp, len(body)


def RFvdot(a, b, cells):
    return ref.vdot(a, b, cells) & 0xFFFFFFFF


@pytest.mark.parametrize("W,n", [(1, 20), (1, 300), (8, 64), (8, 300), (16, 96), (16, 600)])
def test_chunk(W, n):
    S.ROW_TABLE = None
    rng = random.Random(100 + W)
    src, tab, _, exp, _ = chunk(rng, W, n)
    S.ROW_TABLE = tab
    c = core.build(wrow=W)
    prog = assemble(src)
    m = ref.Machine.from_program(prog)
    m.rowtab, m.wrow_w = tab, W
    m.run()
    assert m.x[18] == exp  # s2: the independent oracle (the chosen weights, not the table code)
    toks, dt = check(c, prog, tab, W)
    base_src, _, wrom, exp2, nb = chunk(random.Random(100 + W), W, n, rows=False)
    assert exp2 == exp
    mb = ref.Machine.from_program(assemble(base_src + "\n.wrom\n" + "\n".join(
        f".byte {', '.join(str(b) for b in wrom[i:i + 64])}" for i in range(0, len(wrom), 64))))
    tb = mb.run()
    assert mb.x[18] == exp
    fam = [S.family(t)[0] for t in toks]
    print(f"\nCHUNK W={W} N={n} VDOTs: rows used {len(tab)}, trace tokens {len(toks)} (ROW {fam.count('ROW')}), "
          f"WROM baseline {len(tb)}; extra tokens/VDOT {(len(toks) - len(tb)) / n:.3f}; "
          f"circuit teacher-forced exact over {len(toks)} positions")
