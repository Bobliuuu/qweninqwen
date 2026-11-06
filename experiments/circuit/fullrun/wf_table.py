"""Row table of the outer model's own parameters from the int4 weight stream (wrom.bin: one cell u=w+8 per byte, 8 cells per VDOT word).
Row r = stream words [480 r, 480 r + 480), ROW_CELLS = 3840 entries (4 bits each, one uint8 per entry here; packed 2/byte = 1.27 GB x 0.5... see docs).
Layout is set by the weight-fetch firmware (compiler/fw/gen_vdk_wf.py): window win = slot // 8 (slot = g % 480); the latch word position p of the VDOT with in-window index k is
p = (1 + k + (win & 3)) & 7 (WP = slot0 + 1 + 9 group + k), hence table[r, win*64 + 8*p + m] = cell m of stream word 480 r + 8 win + k."""
import numpy as np
def build_table(wrom: bytes) -> np.ndarray:
    words = (len(wrom) + 7) // 8; nrows = -(-words // 480)
    w = np.zeros(nrows * 480 * 8, np.uint8); w[:len(wrom)] = np.frombuffer(wrom, np.uint8) & 15
    w = w.reshape(nrows, 60, 8, 8)                      # [row, window, k, cell]
    p = np.arange(8)[None, :]; g = (np.arange(60) & 3)[:, None]
    kidx = (p - 1 - g) & 7                               # [60, 8]: stream k stored at latch position p
    return np.ascontiguousarray(w[:, np.arange(60)[:, None], kidx, :]).reshape(nrows, 3840)
