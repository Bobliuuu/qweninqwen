import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = True
for x, y in zip(a['steps'], b['steps']):
    same = all(x[k] == y[k] for k in ('ctrl_digest_noEMB', 'jobs_digest', 'cpos_digest'))
    print(f"step {x['step']}: tokens {x['token']} vs {y['token']}: program identical except EMBX: {same}")
    ok &= same
print('PROGRAM TOKEN-INDEPENDENT (only EMBX differs)' if ok else 'PROGRAM DEPENDS ON TOKENS BEYOND EMBX')
sys.exit(0 if ok else 1)
