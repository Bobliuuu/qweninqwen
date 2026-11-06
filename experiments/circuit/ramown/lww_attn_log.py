"""Prototype: exact latest-write-wins address lookup over an append-only attention log (the qwenFS copy/lookup idea applied to RV32 RAM stores).
Log entry t = (address a_t, value v_t). Key = dual-rail address bits (+-1) and recency digits; query = address bits. Value = 32 value bits (+-1).
Three score paths: eager bf16 (q@k^T rounded to bf16 as HF eager does), SDPA bf16 (torch CPU kernel), fp32 reference path.
Small CPU job (safe on a laptop). usage: python lww_attn_log.py [Nmax]"""
import sys, math, torch
torch.manual_seed(0)
AB, VBITS, BASE = 13, 32, 128           # 8192 words (the core's RAM), 32 value bits, recency digits base 128

def build(N, nwords, ndig):
    addr = torch.randint(0, nwords, (N,)); val = torch.randint(0, 2**31 - 1, (N,), dtype=torch.int64)
    sgn = lambda x, n: (((x[:, None] >> torch.arange(n)) & 1) * 2 - 1).float()
    digs = torch.stack([(torch.arange(N) // BASE**j) % BASE for j in range(ndig)], 1).float()   # exact in bf16 (<=128)
    return addr, val, sgn, digs

def run(N, nwords, mode):
    ndig = max(1, math.ceil(math.log(N + 1, BASE)))
    addr, val, sgn, digs = build(N, nwords, ndig)
    A = AB
    M = 8.0 * N                                   # one address bit off (score drop 2M) beats the whole recency range (D*N), D=8 margin
    D = 8.0                                       # score gap per recency step
    K = torch.cat([sgn(addr, A), digs], 1)        # key
    # query for address a: weights M on bits, recency weights D*BASE^j (<= bf16-exact: 8*128^j)
    ids = torch.randint(0, nwords, (min(512, nwords),))
    Q = torch.cat([sgn(ids, A) * M, (D * BASE ** torch.arange(ndig)).float().expand(len(ids), -1)], 1)
    V = sgn(val, VBITS)
    # reference: latest write per address
    last = {}
    for t in range(N): last[int(addr[t])] = t
    ok = tot = 0; miss = 0
    for qi, a in enumerate(ids.tolist()):
        if a not in last: miss += 1; continue
        tot += 1
        if mode == 'fp32':
            s = K @ Q[qi]
        elif mode == 'eager_bf16':
            s = (K.bfloat16() @ Q[qi].bfloat16()).float()      # score rounded to bf16, as an eager bf16 layer does
        s = s - s.max(); p = torch.softmax(s, 0)
        out = (p[:, None] * V).sum(0) > 0
        got = int((out.long() << torch.arange(VBITS)).sum()); ok += int(got == int(val[last[a]]))
    return ok, tot

if __name__ == '__main__':
    Nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    print('N stores | words | path | exact lookups / live lookups')
    for N in [64, 256, 1024, 4096, Nmax]:
        for nwords in [256, 8192]:
            for mode in ['fp32', 'eager_bf16']:
                ok, tot = run(N, nwords, mode); print(N, nwords, mode, f'{ok}/{tot}', flush=True)
