#!/usr/bin/env python3
"""Static, white-background SVG visuals from the REAL run data (see ../PROVENANCE.md):
  circuit_heatstrip.svg   per-chunk job time (96 cells) coloured by minutes, machine marker below
  circuit_metrics.svg     metrics dashboard tiles
  circuit_tiers.svg       tier ladder (instructions per token, log scale; measured vs projected)
  circuit_fleet.svg       fleet timeline from cfram_q_vast_out/status.txt
usage: python3 make_static.py   (writes ../*.svg and ../*.png via rsvg-convert)"""
import json, os, re, subprocess, datetime, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.abspath(os.path.join(HERE, '..')); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FR = os.path.join(REPO, 'experiments', 'circuit', 'fullrun'); DEMO = os.path.join(FR, 'logs', 'cfram_q_vast')
FONT = "-apple-system, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif"; MONO = "SFMono-Regular, Menlo, Consolas, monospace"
INK, G1, G2, G3, G4, ACC, ACCD, ACCL, ACCM = '#111827', '#6B7280', '#9CA3AF', '#E5E7EB', '#F3F4F6', '#4F46E5', '#4338CA', '#EEF2FF', '#C7D2FE'
def J(p): return json.load(open(p))
seams = J(os.path.join(FR, 'logs', 'cfram_q_vast', 'seams_0-95_20261107_004845.json')); rows = {r['chunk']: r for r in seams['rows']}
verdict = J(os.path.join(DEMO, 'FINAL_VERDICT.json')); fleet = J(os.path.join(DEMO, 'fleet_result.json'))
assert seams['ok'] and len(rows) == 96
def machine(c):
    if c in verdict['modal_chunks']: return 'Modal A100'
    g = (fleet['done'].get(str(c)) or {}).get('gpu') or ''
    return 'RTX 4090' if '4090' in g else 'A100 PCIE' if 'A100' in g else 'L40S' if 'L40S' in g else '?'
def head(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}"><title>{title}</title><desc>{desc}</desc>'
            f'<defs><filter id="sh" x="-5%" y="-5%" width="110%" height="120%"><feDropShadow dx="0" dy="5" stdDeviation="8" flood-color="#1b1b3a" flood-opacity="0.07"/></filter>'
            f'<linearGradient id="acc" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366F1"/><stop offset="1" stop-color="#4338CA"/></linearGradient>'
            f'<pattern id="hat" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="7" stroke="{ACCM}" stroke-width="3"/></pattern></defs>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>')
def txt(x, y, s, size=14, fill=INK, w=400, anchor='start', mono=False, ls=None):
    f = f' font-family="{MONO}"' if mono else ''; l = f' letter-spacing="{ls}"' if ls else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{f}{l}>{s}</text>'
def save(name, svg):
    p = os.path.join(OUT, name + '.svg'); open(p, 'w').write(svg + '</svg>')
    subprocess.run(['rsvg-convert', '-w', '2400', p, '-o', os.path.join(OUT, name + '.png')], check=True)
    print('wrote', name, os.path.getsize(p), 'B svg')

# ---------------- heat strip ----------------
def lerp(a, b, t): return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
def col(t):
    lo, hi = (238, 242, 255), (67, 56, 202)
    return '#%02x%02x%02x' % lerp(lo, hi, max(0, min(1, t)))
def heatstrip():
    W, H = 1280, 360; sec = [rows[c]['seconds'] / 60.0 for c in range(96)]
    mx = max(m for c, m in enumerate(sec) if c != 95); mn = min(sec[:95])
    s = head(W, H, 'Job time per chunk', 'Per-chunk job time in minutes for the 96 chunks, coloured by minutes, with the machine that ran each chunk')
    s += txt(48, 50, 'Job time per chunk', 24, INK, 700) + txt(48, 74, 'minutes per chunk job, chunks 0 to 95 (chunk 95 is the short HALT chunk: 57,707 steps)', 14, G1)
    x0, y0, cw, gap, ch = 48, 104, 11.5, 0.9, 108
    for c in range(96):
        x = x0 + c * (cw + gap); v = sec[c]; t = (v - mn) / (mx - mn) if c != 95 else 0
        s += f'<rect x="{x:.1f}" y="{y0}" width="{cw}" height="{ch}" rx="3" fill="{col(t) if c != 95 else "#FFFFFF"}"' + (f' stroke="{ACC}" stroke-width="1.5"' if c == 95 else '') + '/>'
    # machine strip
    MC = {'RTX 4090': ACC, 'A100 PCIE': '#818CF8', 'L40S': '#A5B4FC', 'Modal A100': '#9CA3AF', '?': '#E5E7EB'}
    for c in range(96):
        x = x0 + c * (cw + gap); s += f'<rect x="{x:.1f}" y="{y0 + ch + 10}" width="{cw}" height="8" rx="2" fill="{MC[machine(c)]}"/>'
    # tick labels
    for c in (0, 20, 40, 60, 80, 95):
        x = x0 + c * (cw + gap) + cw / 2; s += txt(x, y0 + ch + 40, str(c), 12, G2, 400, 'middle')
    s += txt(x0, y0 + ch + 62, 'chunk', 12, G2)
    # legend
    lx = 48; ly = 292
    for k in ('RTX 4090', 'A100 PCIE', 'L40S', 'Modal A100'):
        s += f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="3" fill="{MC[k]}"/>' + txt(lx + 20, ly, f'{k}  ({(n_ := sum(1 for c in range(96) if machine(c) == k))} chunk' + ('s' if n_ != 1 else '') + ')', 13, G1); lx += 200
    # colour scale
    sx = 920
    for i in range(10): s += f'<rect x="{sx + i * 18}" y="{ly - 10}" width="18" height="12" fill="{col(i / 9)}"/>'
    s += txt(sx - 8, ly, f'{mn:.1f}', 12, G1, 400, 'end') + txt(sx + 190, ly, f'{mx:.1f} min', 12, G1)
    by = collections.defaultdict(list)
    for c in range(95): by[machine(c)].append(rows[c]['seconds'] / 60)
    s += txt(48, 332, 'median minutes: ' + '   ·   '.join(f'{k} {statistics.median(v):.1f}' for k, v in sorted(by.items())) + '   (job time, not billed time)', 13, G2)
    save('circuit_heatstrip', s)

# ---------------- metrics dashboard ----------------
def metrics():
    W, H = 1280, 560
    steps = sum(r['steps_run'] for r in rows.values()); secs = sum(r['seconds'] for r in rows.values())
    inj = sum(r['injections'] for r in rows.values()); forced = sum(r['forced_tokens'] for r in rows.values()); mism = sum(r['free_mismatches'] for r in rows.values())
    seam = sum(1 for r in rows.values() if r.get('seam', {}).get('exact'))
    assert (inj, forced, mism, seam) == (0, 0, 0, 95)
    s = head(W, H, 'Run metrics', 'Dashboard of the headline numbers of the 96-chunk run')
    s += txt(48, 54, 'The run in numbers', 26, INK, 700) + txt(48, 78, 'one Qwen3-0.6B token (prompt 785, position 0) on the circuit', 14, G1)
    tiles = [('15846', "final word · ' Question'", 'accent'), ('96 / 96', 'chunk jobs clean', ''), ('95 / 95', 'seams exact', ''), ('0 · 0 · 0', 'mismatches · forced · host injections', ''),
             ('199,287,158', 'trace tokens', ''), ('178,066,316', 'instructions', ''), (f'{secs / 3600:.1f} h', 'total GPU job time (sum over 96 jobs)', ''), ('≈ $10', 'rented Vast.ai GPUs (fleet-log estimate)', ''),
             ('168,684', 'rows in the weight table (3,840 int4 cells)', ''), ('2.59 GB', 'table in the outer checkpoint (bf16)', ''), ('6.0 min', 'per chunk on an RTX 4090 (≈ $0.05)', ''), ('57,705', 'step of the circuit\'s own HALT, chunk 95', '')]
    cw, chh, gx, gy, x0, y0 = 284, 120, 20, 20, 48, 108
    for i, (big, small, kind) in enumerate(tiles):
        r_, c_ = divmod(i, 4); x = x0 + c_ * (cw + gx); y = y0 + r_ * (chh + gy)
        if kind == 'accent':
            s += f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{cw}" height="{chh}" rx="16" fill="url(#acc)"/></g>' + txt(x + 22, y + 62, big, 38, '#FFFFFF', 700) + txt(x + 22, y + 92, small, 13.5, '#E0E7FF')
        else:
            s += f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{cw}" height="{chh}" rx="16" fill="#FFFFFF" stroke="{G3}"/></g>' + txt(x + 22, y + 62, big, 32 if len(big) < 12 else 28, INK, 700) + txt(x + 22, y + 92, small, 13.5, G1)
    s += txt(48, 536, 'Sources: seams_0-95_20261107_004845.json, FINAL_VERDICT.json, quantfix/STEP2_REPORT.md. See docs/assets/PROVENANCE.md. Cost = fleet-log estimate, not billing.', 12.5, G2)
    save('circuit_metrics', s)

# ---------------- tier ladder ----------------
def tiers():
    import math
    W, H = 1280, 470
    # roadmap tiers (docs/TIERS.md: derived/projected instruction counts) + the one measured result (summary.json instructions_machine)
    T = [('T0', 'literal fp32 soft-float baseline', 1.10e11, False), ('T1', '+ fused executor', 4.0e9, False), ('T2', '+ SMAQA int8 (matvec only)', 5.4e8, False),
         ('T3', '+ VDOT int4 (matvec only)', 1.05e8, False), ('Measured', 'this run: CF + RAM, 189,109,345 per token', 1.78066316e8, True)]
    s = head(W, H, 'Tier ladder', 'Roadmap tiers (hatched, projected) and the one measured result, as instructions per token on a log scale')
    s += txt(48, 54, 'The tier ladder', 26, INK, 700) + txt(48, 78, 'instructions per generated token (log scale): roadmap tiers hatched, the measured result solid', 14, G1)
    x0, x1 = 470, 1220; lo, hi = 7.5, 11.5
    def X(v): return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    for e in (8, 9, 10, 11):
        s += f'<line x1="{X(10 ** e):.1f}" y1="100" x2="{X(10 ** e):.1f}" y2="378" stroke="{G4}"/>' + txt(X(10 ** e), 400, f'1e{e}', 12, G2, 400, 'middle')
    for i, (n, sub, v, meas) in enumerate(T):
        y = 108 + i * 54; bw = X(v) - x0
        s += txt(48, y + 24, n, 18, INK, 700) + txt(168, y + 24, sub, 13.5, G1)
        if meas: s += f'<rect x="{x0}" y="{y + 4}" width="{bw:.1f}" height="30" rx="8" fill="url(#acc)"/>'
        else: s += f'<rect x="{x0}" y="{y + 4}" width="{bw:.1f}" height="30" rx="8" fill="url(#hat)" stroke="{ACCM}"/>'
        s += txt(x0 + bw + 10, y + 25, f'{v:.3g}'.replace('e+', 'e').replace('e0', 'e') + ('  measured' if meas else '  roadmap'), 13, ACC if meas else G1, 600 if meas else 400)
    s += txt(48, 440, 'Roadmap rows: docs/TIERS.md (derived or projected; T2 and T3 count the matvec only). Measured row: this run, 178,066,316 machine instructions over the whole trace (summary.json), one prompt token at ctx 1.', 12, G2)
    save('circuit_tiers', s)

# ---------------- fleet timeline ----------------
def fleetplot():
    st = []
    for ln in open(os.path.join(FR, 'logs', 'cfram_q_vast', 'status.txt'), errors='ignore'):
        m = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) boxes up=(\d+) computing=(\d+) instances_rented=(\d+) chunks done=(\d+)/96', ln)
        if m: st.append((datetime.datetime.fromisoformat(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5))))
    assert len(st) > 20
    events = []
    t0, t1 = st[0][0], st[-1][0]; W, H = 1280, 470; x0, x1, y0, y1 = 80, 1230, 110, 360
    def X(t): return x0 + (t - t0).total_seconds() / (t1 - t0).total_seconds() * (x1 - x0)
    mxc = max(d for *_, d in st) or 1
    def Y(v): return y1 - v / 100 * (y1 - y0)
    s = head(W, H, 'Fleet timeline', 'Chunks completed by the Vast.ai fleet manager and boxes up over time')
    s += txt(48, 54, 'Fleet timeline', 26, INK, 700) + txt(48, 78, f'{t0:%b %d %H:%M} to {t1:%H:%M} (local time): chunks completed by the Vast.ai fleet manager, and boxes up', 14, G1)
    for v in (0, 25, 50, 75, 100):
        s += f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{G4}"/>' + txt(x0 - 10, Y(v) + 4, str(v), 12, G2, 400, 'end')
    pts = ' '.join(f'{X(t):.1f},{Y(d):.1f}' for t, _, _, d in st)
    s += f'<polygon points="{x0},{y1} {pts} {X(t1):.1f},{y1}" fill="{ACCL}"/><polyline points="{pts}" fill="none" stroke="{ACC}" stroke-width="2.5" stroke-linejoin="round"/>'
    bp = ' '.join(f'{X(t):.1f},{y1 - b / 14 * 70:.1f}' for t, b, _, _ in st)
    s += f'<polyline points="{bp}" fill="none" stroke="{G2}" stroke-width="1.6" stroke-dasharray="4 4"/>'
    tt = t0.replace(minute=0 if t0.minute < 30 else 30, second=0) + datetime.timedelta(minutes=30)
    while tt < t1:
        s += f'<line x1="{X(tt):.1f}" y1="{y1}" x2="{X(tt):.1f}" y2="{y1 + 5}" stroke="{G2}"/>' + txt(X(tt), y1 + 24, f'{tt:%H:%M}', 12, G2, 400, 'middle'); tt += datetime.timedelta(minutes=30)
    for t, k in events:
        s += f'<line x1="{X(t):.1f}" y1="{y0 - 6}" x2="{X(t):.1f}" y2="{y1}" stroke="{ACCM}" stroke-dasharray="2 4"/>' + txt(X(t), y0 - 12, 'restart' if k == 'restart' else 'boxes vanished', 11.5, G1, 400, 'middle')
    s += txt(x0, 412, 'solid: chunks done (of 96, Vast fleet only: chunks 32 to 70 were imported from Modal at the end)   ·   dashed: boxes up (scale 0 to 14)', 12.5, G2)
    s += txt(x0, 432, 'Source: logs/cfram_q_vast/status.txt (fleet manager status lines).', 12.5, G2)
    save('circuit_fleet', s)

if __name__ == '__main__':
    heatstrip(); metrics(); tiers(); fleetplot()
