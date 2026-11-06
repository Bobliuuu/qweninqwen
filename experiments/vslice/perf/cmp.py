"""compare two PERFJSON results: max |bestblk_a - bestblk_b| / max|bestblk| per position (outer-model block maxima, in-model values)."""
import json, re, sys
def load(p):
    t = open(p, errors='ignore').read(); m = re.search(r'^PERFJSON (.*)$', t, re.M); return json.loads(m.group(1)) if m else json.load(open(p))
a, b = load(sys.argv[1]), load(sys.argv[2])
for sa, sb in zip(a['steps'], b['steps']):
    x, y = sa['bestblk'], sb['bestblk']; mx = max(abs(v) for v in x)
    print('step', sa['step'], 'max|diff|/max|blk| = %.3e' % (max(abs(p - q) for p, q in zip(x, y)) / mx), 'maxrel', sa['maxrel'], sb['maxrel'], 'argmax', sa['outer_argmax'], sb['outer_argmax'])
