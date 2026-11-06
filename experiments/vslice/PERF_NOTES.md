# Perf notes: real 28-layer Qwen3-0.6B inside Qwen3.5 (vs2 compiled route)

Harness: `perf/modal_perf.py` (Modal, A100-80GB PCIe, torch 2.15, transformers 5.22.1), executor `vs2_fast.py`, driver `test_vs2_fast.py`.
Raw logs/JSON: `perf/logs/`, `perf/results/`; run list in `/RUNS.md`. vs2.py / test_vs2.py are untouched.

## Baseline breakdown (stock vs2.Runner semantics, cuda-event profiled, position 0, 261 s)
| phase | s |
|---|---|
| layer build (1006 layers, random init + compile) | 13.2 |
| payload tiles + dense frame assembly (weight prep, per position) | 20.7 |
| layer forward | 223.2 |
| everything else (prog 0.7, HF ref 0.3, timeline scan 1.0, D2H) | ~3 |

Layer forward by submodule (s, summed over layers): GDN in_proj_qkv 51.7, in_proj_z 29.3, out_proj 24.6, MLP gate/up/down of the NG/S/TS/TM layers ~96;
dense matmuls = ~200 of 223 s. Norms, conv, torch chunk-GDN core, KV attention are each only a few s. The matmuls are dense fp32 over D=40775 even though
every compiled weight matrix has 1-3 nonzeros per row. Unprofiled baseline in the same harness: 236.3 s (pos 0, maxrel 5.83e-5), 237.1 s (pos 1, 6.95e-6); argmax == HF.
The "~10 s unattributed" is prog build + HF reference + the O(timeline x layers) python scan + per-layer cache flush/D2H.

## Measured speedups
| variant | pos0 s | pos1 s | maxrel vs HF (pos0/pos1) | notes |
|---|---|---|---|---|
| baseline dense fp32 | 236.3 | 237.1 | 5.83e-5 / 6.95e-6 | |
| sparse COO weights (exact zero skipping) + layers cached on GPU + payload cache + meta-device build (profiling hooks on) | 39.7 (fills caches) | 28.8 | 5.83e-5 / 2.79e-5 | 9.3x steady; argmax == HF; block-best differs from dense by 2.6e-5 rel (summation order, ballast amplification) |
| TF32 dense (micro) | - | - | 4e-4 rel error on the matmul alone | rejected: exceeds 1e-4 |
| COO sparse + CUDA graphs | failed | | | COO F.linear coalesces (sync) inside capture |
| CSR sparse + CUDA graphs | 8.1 (incl. capture) | 1.75 / 1.08 | WRONG (all-zero output) | replay runs but output is zero; bug not yet found (gcheck job queued to localise the layer) |

Findings: the sparse run is CPU/launch-bound (kernel profile: ~1.7 ms GPU per layer vs ~28 ms wall), so CUDA graphs are the next lever (potentially ~1-3 s/token)
once the zero-output bug is fixed. The sparse path keeps stock `nn.Linear` classes (sparse Parameter) so layer classes are unchanged, but `outer_check.py` param
inventory expects dense params; flag if the audit is run on the fast executor. Caching payloads/layers across positions is simulation-side only.
fla (flash-linear-attention) not yet re-checked: GDN core is minor after sparsification (~10 s of 29 s, launch-bound).
