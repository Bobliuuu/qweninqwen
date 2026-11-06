"""Join the rvemu_mt trace (PREFIX.words, PREFIX.events) with the ELF symbols: per-region / per-buffer histogram of LOAD/STORE words and the exact
live-word peak (a word is live from a store to its last read before the next store; never-stored words that are read are read-only data).
usage: fw_addr_analysis.py ELF PREFIX [OUT.json]"""
import sys, subprocess, json, numpy as np
elf, pre = sys.argv[1:3]; outj = sys.argv[3] if len(sys.argv) > 3 else None
syms = []
for l in subprocess.check_output(['riscv64-unknown-elf-nm', '-S', '-n', elf], text=True).splitlines():
    p = l.split()
    if len(p) == 4 and p[2].lower() in 'rbdv': syms.append((int(p[0], 16), int(p[1], 16), p[3], p[2]))
W = np.fromfile(pre + '.words', dtype=np.uint64).reshape(-1, 6)
addr, nr, nw, first, lastr, lastw = W.T
ev = np.fromfile(pre + '.events', dtype=np.uint64).reshape(-1, 2).astype(np.int64)
def name(a):
    if a >= 0xFFE00000: return 'STACK'
    for s, z, n, t in syms:
        if s <= a < s + z: return n
    return 'other'
names = np.array([name(int(a)) for a in addr])
rw = nw > 0
rows = {}
for n in sorted(set(names)):
    m = names == n
    rows[n] = dict(words=int(m.sum()), words_stored=int((m & rw).sum()), words_ro=int((m & ~rw).sum()), loads=int(nr[m].sum()), stores=int(nw[m].sum()),
                   bytes_declared=next((z for s, z, nn, t in syms if nn == n), None))
# live-word sweep over instruction time
s, e = ev[:, 0], ev[:, 1]
times = np.concatenate([s, e + 1]); delta = np.concatenate([np.ones(len(s), np.int64), -np.ones(len(e), np.int64)])
# stored-values only: events with start 0 are values never stored (initial/zero read) -> report separately as 'read-before-write'
o = np.argsort(times, kind='stable'); t = times[o]; live = np.cumsum(delta[o])
rwmask = s > 0
t2 = np.concatenate([s[rwmask], e[rwmask] + 1]); d2 = np.concatenate([np.ones(rwmask.sum(), np.int64), -np.ones(rwmask.sum(), np.int64)])
o2 = np.argsort(t2, kind='stable'); live2 = np.cumsum(d2[o2])
res = dict(total_loads_data=int(nr.sum()), total_stores=int(nw.sum()), distinct_words_touched=int(len(addr)), distinct_words_stored=int(rw.sum()),
           distinct_words_read_never_stored=int(((nr > 0) & ~rw).sum()), peak_live_stored_words=int(live2.max()), peak_live_incl_read_before_write=int(live.max()),
           by_buffer=rows)
# stored words that are read-before-first-write (bss zero assumption)
if outj: json.dump(res, open(outj, 'w'), indent=1)
print(f"touched {res['distinct_words_touched']} words; stored {res['distinct_words_stored']}; read-only-touched {res['distinct_words_read_never_stored']}; loads {res['total_loads_data']:,} stores {res['total_stores']:,}")
print(f"PEAK live (stored values, exact) = {res['peak_live_stored_words']} words = {res['peak_live_stored_words']*4/1024:.1f} KB ; incl. read-before-write = {res['peak_live_incl_read_before_write']}")
print(f"{'buffer':14s} {'decl B':>9s} {'words':>8s} {'stored':>8s} {'RO':>7s} {'loads':>12s} {'stores':>12s}")
for n, r in sorted(rows.items(), key=lambda kv: -kv[1]['loads'] - kv[1]['stores']):
    print(f"{n:14s} {str(r['bytes_declared']):>9s} {r['words']:8d} {r['words_stored']:8d} {r['words_ro']:7d} {r['loads']:12,d} {r['stores']:12,d}")
