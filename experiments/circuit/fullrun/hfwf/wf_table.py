"""Row table of the outer model's own parameters from the int4 weight stream (wrom.bin: one cell u=w+8 per byte, 8 cells per VDOT word).
WF_ENT_BITS=4 (default): row r = stream words [480 r, 480 r + 480), 3840 entries of 4 bits (one uint8 per entry here), 1.27 GB at hidden 4096 bf16.
WF_ENT_BITS=6: row r = stream words [640 r, 640 r + 640), 3840 entries of 6 bits, 6 entries per word (the word's 32 bits = 8 nibbles, low cell first, cut into 6-bit entries; the 6th entry carries 2 bits), 0.92 GB at hidden 4096.
Layout is set by the weight-fetch firmware (compiler/fw/gen_vdk_wf.py): window win = slot // 8; latch word position p = (1 + k + (win & 3)) & 7, hence
table[r, (8 win + p) * EPW + e] = entry e of stream word ROWW r + 8 win + k (EPW entries per word)."""
import os
import numpy as np
def build_table(wrom: bytes, bits: int = None) -> np.ndarray:
    bits = bits or int(os.environ.get('WF_ENT_BITS', 4)); epw = -(-32 // bits); roww = 3840 // epw; nwin = roww // 8
    words = (len(wrom) + 7) // 8; nrows = -(-words // roww)
    w = np.zeros(nrows * roww * 8, np.uint8); w[:len(wrom)] = np.frombuffer(wrom, np.uint8) & 15
    w = w.reshape(nrows, nwin, 8, 8)                    # [row, window, k, cell]
    p = np.arange(8)[None, :]; g = (np.arange(nwin) & 3)[:, None]
    kidx = (p - 1 - g) & 7                               # [nwin, 8]: stream k stored at latch position p
    w = w[:, np.arange(nwin)[:, None], kidx, :]          # [row, window, p, cell]
    if bits == 4: return np.ascontiguousarray(w).reshape(nrows, 3840)
    v = np.zeros(w.shape[:-1], np.uint64)
    for k in range(8): v |= w[..., k].astype(np.uint64) << np.uint64(4 * k)
    ent = np.stack([((v >> np.uint64(bits * e)) & np.uint64((1 << bits) - 1)).astype(np.uint8) for e in range(epw)], -1)   # [row, window, p, e]
    return np.ascontiguousarray(ent).reshape(nrows, nwin * 8 * epw)
