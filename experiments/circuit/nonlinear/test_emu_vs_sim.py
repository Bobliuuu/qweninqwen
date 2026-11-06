"""Bit-exactness: nl_fixed.h compiled for the host with -DNL_EXT_EMU (C reference of the instructions) vs the same header compiled for RV32 with -DNL_EXT and run in the interpreter."""
import ctypes, numpy as np, sys
from rvsim import Sim, elf_text, s32
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "/tmp/libnl_emu.dylib"); text, syms, _ = elf_text("k2_ext.o"); rng = np.random.default_rng(3)
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
bad = 0
def sim(): return Sim(text)
for trial in range(40):
    n = 64 + trial * 8
    sc = rng.integers(-12 * 65536, 3 * 65536, n).astype(np.int32); o = np.zeros(n, np.int32)
    lib.t_softmax(P(sc), P(o), n); s = sim(); s.put_i32(0x10000, [int(v) for v in sc]); s.call(syms["t_softmax"], [0x10000, 0x40000, n]); bad += o.tolist() != s.get_i32(0x40000, n)
    g = rng.integers(-14 * 65536, 14 * 65536, n).astype(np.int32); u = rng.integers(-300 * 65536, 300 * 65536, n).astype(np.int32); o = np.zeros(n, np.int32)
    lib.t_swiglu(P(g), P(u), P(o), n); s = sim(); s.put_i32(0x10000, [int(v) for v in g]); s.put_i32(0x40000, [int(v) for v in u]); s.call(syms["t_swiglu"], [0x10000, 0x40000, 0x70000, n]); bad += o.tolist() != s.get_i32(0x70000, n)
    x = rng.integers(-2**25, 2**25, 128).astype(np.int32); gg = rng.integers(0, 2**17, 128).astype(np.int32); o = np.zeros(128, np.int32)
    lib.t_rms_inv.restype = ctypes.c_uint32; lib.t_rms_scale.restype = None
    y = lib.t_rms_inv(P(x), 128, 7); lib.t_rms_scale(P(x), P(gg), P(o), 128, ctypes.c_uint32(y))
    s = sim(); s.put_i32(0x10000, [int(v) for v in x]); s.put_i32(0x40000, [int(v) for v in gg]); y2 = s.call(syms["t_rms_inv"], [0x10000, 128, 7]) & 0xFFFFFFFF
    s.call(syms["t_rms_scale"], [0x10000, 0x40000, 0x70000, 128, y2]); bad += (y != y2) or (o.tolist() != s.get_i32(0x70000, 128))
    xp = rng.integers(-2**31, 2**31, 64).astype(np.int32); cs = rng.integers(-2**31, 2**31, 64).astype(np.int32); o = np.zeros(64, np.int32)
    lib.t_rope(P(xp.view(np.uint32)), P(cs.view(np.uint32)), P(o.view(np.uint32)), 64); s = sim(); s.put_i32(0x10000, [int(v) for v in xp]); s.put_i32(0x40000, [int(v) for v in cs]); s.call(syms["t_rope"], [0x10000, 0x40000, 0x70000, 64]); bad += o.tolist() != s.get_i32(0x70000, 64)
    a = rng.integers(-2**31, 2**31, 300).astype(np.int32); lib.t_argmax.restype = ctypes.c_int; r1 = lib.t_argmax(P(a), 300)
    s = sim(); s.put_i32(0x10000, [int(v) for v in a]); r2 = s.call(syms["t_argmax"], [0x10000, 300]); bad += (r1 != r2) or (r1 != int(np.argmax(a)))
print("trials 40 x 5 kernels, mismatches:", bad); sys.exit(bad != 0)
