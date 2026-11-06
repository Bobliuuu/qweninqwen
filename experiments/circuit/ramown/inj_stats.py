"""inj_stats.py REFDIR OUT.json : per-chunk host-role counts of the remapped-firmware reference (ref_chain.py output): VDOT weight words, DROM-class constants,
unaddressable-WROM VDOTs (WP mod 2^15 >= 32640 -> record forced), upper-12-bit usage (WP27 >> 15 distinct values), first global step of a zone VDOT."""
import sys, glob, json, numpy as np
d, o = sys.argv[1:3]; res = []; off = 0; first_zone = None; first_vdot = None; first_drom = None
for f in sorted(glob.glob(f'{d}/inj_[0-9]*.npy')):
    ch = int(f[-8:-4]); inj = np.load(f); n = len(np.load(f'{d}/tok_{ch:04d}.npy'))
    v = inj[inj[:, 1] == 2]; r = inj[inj[:, 1] == 3]; wp = v[:, 2]; zone = (wp & 0x7FFF) >= 32640
    if zone.any() and first_zone is None: first_zone = int(off + v[zone][0, 0])
    if len(v) and first_vdot is None: first_vdot = int(off + v[0, 0])
    if len(r) and first_drom is None: first_drom = int(off + r[0, 0])
    res.append(dict(chunk=ch, tokens=n, vdot=int(len(v)), drom=int(len(r)), ram_inj=int((inj[:, 1] == 1).sum()), zone_vdot=int(zone.sum()), wp_hi_distinct=int(len(np.unique(wp >> 15))), wp_hi_max=int(wp.max() >> 15) if len(wp) else 0)); off += n
tot = {k: sum(x[k] for x in res) for k in ('tokens', 'vdot', 'drom', 'ram_inj', 'zone_vdot')}
json.dump(dict(per_chunk=res, total=tot, first_zone_vdot_step=first_zone, first_vdot_step=first_vdot, first_drom_step=first_drom), open(o, 'w'), indent=1); print(tot, 'first zone', first_zone, 'first vdot', first_vdot, 'first drom', first_drom)
