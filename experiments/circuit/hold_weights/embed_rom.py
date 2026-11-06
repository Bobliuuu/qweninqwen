"""Prototype: hold ALL of Qwen3-0.6B's int4 weights (596 M nibbles, the compiler's wrom.bin, one nibble per byte) as extra EMBEDDING ROWS of a Qwen3.5-4B-geometry
(hidden 2560, bf16) table, then read them back through a stock-style Qwen3.5 RMSNorm and check the whole stream bit-for-bit.
Row = D data symbols of BITS bits + FILL filler dims chosen so sum(x^2) is exactly C = 2560 * 2^(2m); then fp32 mean = 2^(2m) exactly, rsqrt = 2^-m exactly,
so the norm is a pure power-of-two scale and the data come back exact (bf16 holds integers <= 256 exactly).
usage: python embed_rom.py WROM_BIN BITS(4|6) [HIDDEN=2560] [DATA_DIMS=2300]"""
import sys, time, json, torch
wrom, BITS = sys.argv[1], int(sys.argv[2]); H = int(sys.argv[3]) if len(sys.argv) > 3 else 2560; D = int(sys.argv[4]) if len(sys.argv) > 4 else 2300
dev = 'cuda'; t0 = time.time()
raw = torch.from_file(wrom, shared=False, size=__import__('os').path.getsize(wrom), dtype=torch.uint8)
N = raw.numel(); assert int(raw.max()) < 16
nib = raw.to(dev)                                                   # one nibble per byte: 596,000,000-ish
if BITS == 4: sym = nib
else:                                                               # repack the bit stream into BITS-bit symbols (chunked; 12 bits = 3 nibbles = 2 symbols)
    assert BITS == 6; padn = (-N) % 3; nn = torch.cat([nib, torch.zeros(padn, dtype=torch.uint8, device=dev)]) if padn else nib; parts = []
    for s0 in range(0, nn.numel(), 3 * 10_000_000):
        c = nn[s0:s0 + 3 * 10_000_000].reshape(-1, 3).to(torch.int16); w = c[:, 0] | (c[:, 1] << 4) | (c[:, 2] << 8)
        parts.append(torch.stack([w & 63, w >> 6], 1).reshape(-1).to(torch.uint8))
    sym = torch.cat(parts); del parts, nn
M = sym.numel(); R = -(-M // D); sym = torch.cat([sym, torch.zeros(R * D - M, dtype=torch.uint8, device=dev)]).reshape(R, D)
smax = (1 << BITS) - 1; Smax = D * smax * smax
m = 0
while H * 4 ** m < Smax: m += 1                                      # C = H * 4^m  >= max data energy; var = 4^m, scale = 2^-m
C = H * 4 ** m; S = (sym.to(torch.int64) ** 2).sum(1); rem = C - S; assert int(rem.min()) >= 0
F = H - D; fill = torch.zeros(R, F, dtype=torch.int64, device=dev)
for j in range(F):                                                  # greedy sum-of-squares, entries <= 255 (exact in bf16)
    v = torch.sqrt(rem.double()).floor().clamp(max=255).long(); v = torch.where((v + 1) ** 2 <= rem, (v + 1).clamp(max=255), v)
    fill[:, j] = v; rem = rem - v * v
    if j % 8 == 7 and int(rem.max()) == 0: break
assert int(rem.max()) == 0, 'filler could not reach C'
nfill = int((fill > 0).sum(1).max())
table = torch.cat([sym.to(torch.bfloat16), fill.to(torch.bfloat16)], 1); del fill
print(f'bits/entry {BITS}: {N:,} nibbles -> {M:,} symbols -> {R:,} rows x {H} dims, filler dims used (max) {nfill}, C={C} var={4**m}, table {table.numel()*2/1e9:.3f} GB bf16, build {time.time()-t0:.1f}s', flush=True)
emb = torch.nn.Embedding(R, H, dtype=torch.bfloat16, device=dev); emb.weight.data.copy_(table); del table
gain = torch.zeros(H, dtype=torch.bfloat16, device=dev)             # Qwen3.5 RMSNorm: weight init 0, out = x*rsqrt(mean(x^2)+eps)*(1+w)
def qwen_rmsnorm(x, eps=1e-6):
    xf = x.float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps); return (xf * (1.0 + gain.float())).type_as(x)
bad_norm = 0; bad_sym = 0; out_sym = torch.empty(R, D, dtype=torch.uint8, device=dev); B = 8192
for s in range(0, R, B):
    ids = torch.arange(s, min(R, s + B), device=dev); x = emb(ids)                  # the lookup = the 'row fetch' (one trace token per row)
    y = qwen_rmsnorm(x); r = (y.float() * (2.0 ** m))                                  # undo the constant power-of-two scale
    bad_norm += int((r != x.float()).sum()); out_sym[s:s + B] = r[:, :D].to(torch.uint8)
bad_sym = int((out_sym != sym).sum())
rec = out_sym.reshape(-1)[:M]
if BITS == 4: back = rec
else:
    padn = (-N) % 3; r2 = rec[: (N + padn) // 3 * 2].reshape(-1, 2).to(torch.int16); w = r2[:, 0] | (r2[:, 1] << 6)
    back = torch.stack([w & 15, (w >> 4) & 15, w >> 8], 1).reshape(-1)[:N].to(torch.uint8)
ok = bool(torch.equal(back, nib))
res = dict(bits_per_entry=BITS, nibbles=N, rows=R, hidden=H, data_dims=D, filler_dims_max=nfill, var=4 ** m, bf16_table_bytes=R * H * 2, payload_bytes=N // 2,
           overhead_x=R * H * 2 / (N / 2), norm_entries_inexact=bad_norm, symbol_mismatches=bad_sym, nibble_stream_identical=ok, seconds=time.time() - t0, gpu=torch.cuda.get_device_name())
print(json.dumps(res)); assert ok
