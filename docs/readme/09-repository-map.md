```
experiments/circuit/
  fullrun/           the circuit runs: circuit_run.py, firmware (fw/, cf/, quantfix/), hfwf/ (checkpoint merge + stock-module check),
                     cf/ (parallel runner, seam collector, Vast fleet), diag_quant/, logs/, RUNS.md
  ramown/            circuit-owned RAM, real address decode (realdecode/)
  engine/ kernels/   the fused executor and Triton kernels, ISA and trace-token cost rules
  compiler/ algo/    firmware generator, quantisation studies
  multicore/ scaleout/   projections for the speed tiers
docs/
  HOST_ROLE_AUDIT_CFRAM.md   the host-role audit of the final path
  HOST_ROLE_AUDIT.md         the baseline host-role audit
  CIRCUIT_ROUTE_PLAN.md  TIERS.md  WEIGHT_FETCH_PATH.md  CONST_FETCH_PATH.md  RAM_FIT.md
  RELEASE_NOTES.md           draft release notes
  demo/                      the demo video and its provenance
  history/                   background notes
audit/               HostGuard harness
reference/ older/    vendored Qwenputer lab and background experiments
RUNS.md              log of every launched run
```
