"""Tiny gate-level netlist builder + bitsliced GPU/CPU simulator.
Gates: AND, OR, XOR (2-input, each = one product gadget = 2 SwiGLU neurons in the outer MLP) and NOT (affine: free in the outer, absorbed
into the consumer's linear form).  Constants 0/1 are wires 0/1 and are folded.  Wires are python ints; a bus is a list of wires, LSB first."""
import numpy as np, torch
AND, OR, XOR, NOT = 0, 1, 2, 3

class Net:
    def __init__(self):
        self.ops = []; self.a = []; self.b = []        # gate g produces wire (n_in + 2 + g)... we number wires globally
        self.nw = 2                                      # wire 0 = const0, wire 1 = const1
        self.gop = {}; self.cache = {}; self.inputs = []; self.level = {0: 0, 1: 0}
        self.gate_of = {}                                # wire -> (op,a,b)
    def inp(self, n):
        ws = list(range(self.nw, self.nw + n)); self.nw += n
        for w in ws: self.level[w] = 0
        self.inputs += ws; return ws
    def _new(self, op, a, b):
        key = (op, a, b)
        if key in self.cache: return self.cache[key]
        w = self.nw; self.nw += 1
        self.gate_of[w] = key; self.cache[key] = w
        self.level[w] = max(self.level[a], self.level[b]) + (0 if op == NOT else 1)
        return w
    def NOT(self, a):
        if a == 0: return 1
        if a == 1: return 0
        g = self.gate_of.get(a)
        if g and g[0] == NOT: return g[1]
        return self._new(NOT, a, a)
    def AND(self, a, b):
        if a == 0 or b == 0: return 0
        if a == 1: return b
        if b == 1: return a
        if a == b: return a
        return self._new(AND, min(a, b), max(a, b))
    def OR(self, a, b):
        if a == 1 or b == 1: return 1
        if a == 0: return b
        if b == 0: return a
        if a == b: return a
        return self._new(OR, min(a, b), max(a, b))
    def XOR(self, a, b):
        if a == 0: return b
        if b == 0: return a
        if a == 1: return self.NOT(b)
        if b == 1: return self.NOT(a)
        if a == b: return 0
        return self._new(XOR, min(a, b), max(a, b))
    def MUX(self, s, a, b):          # s ? a : b   = b ^ (s & (a ^ b))   (2 gates + 1 xor)
        return self.XOR(b, self.AND(s, self.XOR(a, b)))
    # ------------------------------------------------------------------ stats
    def stats(self, outs=None):
        """gate counts restricted to the cone of `outs` (default: all)"""
        live = self._cone(outs)
        c = {AND: 0, OR: 0, XOR: 0, NOT: 0}
        for w in live:
            g = self.gate_of.get(w)
            if g: c[g[0]] += 1
        two = c[AND] + c[OR] + c[XOR]
        depth = max([self.level[w] for w in (outs if outs is not None else live)] + [0])
        return dict(AND=c[AND], OR=c[OR], XOR=c[XOR], NOT=c[NOT], gates2=two, depth=depth, neurons=2 * two, wires=len(live))
    def _cone(self, outs):
        if outs is None: return sorted(self.gate_of.keys())
        seen = set(); st = list(outs)
        while st:
            w = st.pop()
            if w in seen or w < 2: continue
            seen.add(w); g = self.gate_of.get(w)
            if g: st.append(g[1]); st.append(g[2])
        return sorted(w for w in seen if w in self.gate_of)

class Sim:
    """levelized bitsliced evaluation of the cone of `outs`; 64 samples per int64 word."""
    def __init__(self, net, outs, dev="cpu"):
        self.net, self.outs, self.dev = net, outs, dev
        live = net._cone(outs); self.live = live
        ids = sorted(set(net.inputs) | set(live) | {0, 1}); self.idx = {w: i for i, w in enumerate(ids)}; self.nids = len(ids)
        lv = {}
        for w in live: lv.setdefault(net.level[w], []).append(w)
        # NOT gates sit at the level of their input; evaluate them right after their input's level -> process in topological order per level
        self.steps = []
        order = sorted(lv)
        done = set(net.inputs) | {0, 1}
        for L in order:
            ws = lv[L]
            # NOTs may depend on gates of the same level (not of NOTs of same level beyond one hop): iterate until all done
            pend = list(ws)
            while pend:
                ready = [w for w in pend if net.gate_of[w][1] in done and net.gate_of[w][2] in done]
                assert ready, "cycle"
                by = {}
                for w in ready: by.setdefault(net.gate_of[w][0], []).append(w)
                for op, wl in by.items():
                    o = torch.tensor([self.idx[w] for w in wl], device=dev)
                    a = torch.tensor([self.idx[net.gate_of[w][1]] for w in wl], device=dev)
                    b = torch.tensor([self.idx[net.gate_of[w][2]] for w in wl], device=dev)
                    self.steps.append((op, o, a, b))
                done |= set(ready); rs = set(ready); pend = [w for w in pend if w not in rs]
    def run(self, in_bits):
        """in_bits: dict wire -> int64 tensor [words]; returns dict wire -> tensor"""
        words = next(iter(in_bits.values())).shape[0]
        W = torch.zeros(self.nids, words, dtype=torch.int64, device=self.dev)
        W[self.idx[1]] = -1
        for w, v in in_bits.items():
            if w in self.idx: W[self.idx[w]] = v
        for op, o, a, b in self.steps:
            A = W[a]
            if op == NOT: W[o] = ~A
            else:
                B = W[b]
                W[o] = (A & B) if op == AND else ((A | B) if op == OR else (A ^ B))
        return {w: W[self.idx[w]] for w in self.outs if w in self.idx}, W

def pack(bits):
    """bits: bool array [nbits, N] -> int64 [nbits, words] bitsliced"""
    nb, N = bits.shape; pad = (-N) % 64
    if pad: bits = np.concatenate([bits, np.zeros((nb, pad), bool)], 1)
    p = np.packbits(bits.reshape(nb, -1, 8)[..., ::-1], axis=2).reshape(nb, -1)   # little-endian bit order inside bytes
    return p.view(np.uint64).view(np.int64)
def unpack(words, N):
    w = np.ascontiguousarray(words.cpu().numpy().astype(np.int64)).view(np.uint8)
    return np.unpackbits(w.reshape(-1, 8)[:, ::-1] if False else w, bitorder="little")[:N].astype(bool)
