# Run log

| time (UTC) | box | command | outcome |
|---|---|---|---|
| 2026-11-06 | Lambda A100 (lambda-final) | `python -u test_vs2.py real151936_L28 1` (generation, 6 positions) | done: argmax == HF at all 6 positions, see experiments/vslice/logs/real_L28_gen.log |
| 2026-11-06 | Modal A100-80GB (app ap-ntHuJQgVmL3y18TjdZ7Rv0) | `modal run --detach experiments/vslice/perf/modal_perf.py::main --job micro` (sparse vs dense linear microbench, D=40775) | done: F.linear with a COO sparse weight 1.4 ms vs dense fp32 110 ms vs dense TF32 30 ms (TF32 rel err 4e-4), sparse rel err 5e-8; logs in experiments/vslice/perf/logs |
| 2026-11-06 | Modal A100-80GB (ap-O28RH3xpSJs8HcOHLXQTvr) | `modal_perf.py::main --job run --args '{"variant":"base","positions":2}'` (stock dense baseline, positions 0-1) | done: 236.3 s / 237.1 s per position, maxrel 5.83e-5 / 6.95e-6, argmax == HF; log+json in experiments/vslice/perf/ |
| 2026-11-06 | Modal A100-80GB (ap-59IpCwghQOfhY8VoVKD02T) | `... {"variant":"base_prof","positions":1,"prof":true,"kernprof":true}` (baseline per-phase profile, cuda-event hooks) | done: 261.1 s with profiling; breakdown in PERF_NOTES.md |
| 2026-11-06 | Modal A100-80GB (ap-Jsm30acNQH1lY9FvIvEjuI) | `... {"variant":"sparse_all_prof","positions":2,"prof":true,"kernprof":true,"fastbuild":true,"sparse":true,"cache_layers":true,"cache_pay":true}` | done: 39.7 s (pos 0, fills caches) / 28.8 s (pos 1) with profiling hooks; maxrel 5.83e-5 / 2.79e-5 |
| 2026-11-06 | Modal A100-80GB (ap-DfeKgSezYZwSe2FCjB1eTQ) | `... sparse_graph (COO weights + CUDA graphs)` | failed: COO sparse F.linear syncs (coalesce) inside graph capture |
| 2026-11-06 | Modal A100-80GB (ap-Rn6rncWxOLi3Z8ME0JIUy2) | `... sparse_csr_graph (CSR weights + CUDA graphs, 3 positions)` | queued behind workspace GPU limit |
