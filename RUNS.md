# Run log

| time (UTC) | box | command | outcome |
|---|---|---|---|
| 2026-11-06 | Lambda A100 (lambda-final) | `python -u test_vs2.py real151936_L28 1` (generation, 6 positions) | done: argmax == HF at all 6 positions, see experiments/vslice/logs/real_L28_gen.log |
| 2026-11-06 | Modal A100-80GB (app ap-ntHuJQgVmL3y18TjdZ7Rv0) | `modal run --detach experiments/vslice/perf/modal_perf.py::main --job micro` (sparse vs dense linear microbench, D=40775) | done: F.linear with a COO sparse weight 1.4 ms vs dense fp32 110 ms vs dense TF32 30 ms (TF32 rel err 4e-4), sparse rel err 5e-8; logs in experiments/vslice/perf/logs |
