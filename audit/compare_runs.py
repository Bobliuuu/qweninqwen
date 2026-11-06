"""Black-box token-independence check.  Run the driver twice with different host randomness and compare the streams:
    python run_audit.py --variant 0 --out a.json drv.py ; python run_audit.py --variant 1 --out b.json drv.py
    python compare_runs.py a.json b.json
A legitimate stream (weights from safetensors + flags/addresses/token embeddings + position ids) must be BIT-IDENTICAL across
variants unless the prompt/tokens themselves differ.  Any differing row = the host put variant-dependent data (activations!)
into the stream.  Exit 1 if any row differs."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
A, B = a["round_digests"], b["round_digests"]
print(f"rounds: {len(A)} vs {len(B)}")
bad_rows = tot = 0; bad_rounds = []
for i, (x, y) in enumerate(zip(A, B)):
    for k in sorted(set(x) | set(y)):
        hx, hy = x.get(k, []), y.get(k, [])
        n = max(len(hx), len(hy)); tot += n
        diff = [j for j in range(n) if j >= len(hx) or j >= len(hy) or hx[j] != hy[j]]
        if diff:
            bad_rows += len(diff); bad_rounds.append((i, k, len(diff), n, diff[:6]))
print(f"stream rows compared: {tot}; differing: {bad_rows}")
for i, k, d, n, ex in bad_rounds[:20]: print(f"  round {i} arg {k}: {d}/{n} rows differ (e.g. rows {ex})")
if len(A) != len(B): print("round count differs: control flow depends on host data")
ok = bad_rows == 0 and len(A) == len(B)
print("STREAM TOKEN-INDEPENDENT" if ok else "STREAM DEPENDS ON HOST-RANDOM DATA  -> host-injected data in the stream (FAIL)")
sys.exit(0 if ok else 1)
