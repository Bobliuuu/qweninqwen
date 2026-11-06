<div align="center">

# qweninqwen

**A real Qwen, executed by the weights and recurrent state of another Qwen.**

A homage to [Qwenputer](https://github.com/divagr18/qwenputer), in which a CPU lives inside Qwen's weights. Here the machine runs a language model.

![architecture](docs/assets/architecture.svg)

</div>

> **Status in one paragraph.** Every ingredient has been built and measured separately on real Qwen3-0.6B / Qwen2.5-0.5B weights: all 197 weight matrices of Qwen3-0.6B are paged through DeltaNet state and contracted in-model (relative error 5.6e-7, 41.95 s on an A10G), and the non-matvec operations (RMSNorm, SwiGLU, RoPE + softmax attention with KV, an argmax over the full 151,936-token vocabulary) are each verified at full inner dimensions. **These pieces have not yet been chained into one inner token.** Every current driver is teacher-forced: the activations that enter the machine are written into frames by the test harness (or come from the Hugging Face inner model), not produced by an earlier round of the machine. The paging run used harness-fed random activations, not a real token. Nothing in this repository claims otherwise; see [What is verified today](#what-is-verified-today) and [Status by tier](#status-by-tier).

## The idea in 30 seconds

- The **outer** model is an unmodified `Qwen3_5ForCausalLM` from `transformers` (GatedDeltaNet layers, SwiGLU MLPs, RMSNorm, full-attention layers). Only its weights are written; its code is not patched.
- The **inner** model is the smallest real Qwen (Qwen3-0.6B or Qwen2.5-0.5B) with its real weights. No scaled-down variant is ever presented as a result.
- A DeltaNet layer is an associative memory: the delta rule writes a value `v` at key `k`, and a query `q` reads back `S^T q`. With one-hot keys, **writing a weight row and reading with an activation is a matrix-vector product**. Inner weights are paged through that state.
- RMSNorm, SwiGLU, RoPE and softmax attention are compiled into the outer model's norm, MLP and attention layers. The outer KV cache is the inner KV cache.
- The host only streams weights, tokens, embeddings, addresses and position ids, and performs the outer argmax and append. It never computes an inner activation.
- Speed is allowed to be absurd. The honest accounting is in [the cost model](#cost-model). Correctness and the host rules are not negotiable.

## Architecture

The host feeds frames (vectors of the outer model's hidden size) into the outer stack. Three kinds of outer layers do three different jobs:

| Outer component | Role for the inner model | Where |
|---|---|---|
| GatedDeltaNet layers | Paged weight memory. A WRITE frame sets one key row of the state to a weight row; a READ frame contracts an activation against all written rows. K-tile accumulation happens in the out-projection and the residual sum. | `experiments/state/engine.py` |
| Compiled MLP / norm layers | Inner RMSNorm (exact for any `d`, `eps`), SwiGLU `silu(gate)*up`, residual and copy gadgets, elementwise products. | `experiments/ops/ops.py` |
| Full-attention layers | Inner RoPE (rotate-half, theta), GQA, causal softmax with the inner KV held in the outer KV cache; also the in-model argmax over the vocabulary. | `experiments/ops/ops.py` |
| Outer `lm_head` | Decodes the 18-bit address code produced by the in-model argmax into the outer token id. | `experiments/ops/ops.py` |

The outer layers are strictly feed-forward, so data produced at depth `d` of a step can only feed shallower layers in a **later** outer forward call, through state or KV cache. One inner token is therefore a sequence of outer forward calls ("rounds"), each a chunked prefill of `[WRITE frames..., READ frame(s)]`. Interface contract and hazards (L2-normalised q/k, conv SiLU, gated RMSNorm, ballast trick): [`docs/INTERFACE.md`](docs/INTERFACE.md).

### DeltaNet state as weight memory

![state as memory](docs/assets/state_memory.svg)

### How a token is computed, round by round

For each inner layer the schedule has four dependent rounds (six for Qwen3, which adds per-head q/k norm and two gain-product rounds), because every op consumes a matvec result from an earlier outer call:

![layer timeline](docs/assets/layer_timeline.svg)

1. **Embed.** The host places the embedding row of the current token in the stream (a table lookup: token to row, no arithmetic).
2. **Round 1.** The host writes the `W_qkv` tiles into the state. In the READ frame the outer norm layer produces the normalised hidden state and the state read returns `q, k, v`.
3. **Round 2.** RoPE and softmax attention run in an outer attention layer, appending the new `k, v` to the KV cache and reading the whole cache; the `W_o` read and the residual add follow.
4. **Round 3.** The second RMSNorm and the `W_gate, W_up` read.
5. **Round 4.** SwiGLU in the MLP layer, then the `W_down` read and the residual add.
6. **After the last layer.** Final RMSNorm, the LM-head contraction against the (paged) embedding matrix, and an in-model hard-max attention over all 151,936 positions whose value is the address code of the winner. The outer `lm_head` decodes that code.
7. **The host does the outer argmax and appends the token.** That is the only host-side decision.

This schedule is the design from the depth rule, with each step verified in isolation. It has not yet been executed end to end.

## What is verified today

Only claims with a result file or a test in this repository are listed.

| Claim | Evidence |
|---|---|
| All 197 weight matrices of Qwen3-0.6B (28 layers + LM head, 595,984,384 real weights, 519,430 outer frames) are written into DeltaNet state and contracted in-model on an A10G in 41.95 s; worst relative error vs an fp64 reference 5.6e-7; argmax identical on all 197 matrices. **Activations were harness-fed (random) and the matrices were not chained: this is not a real token.** | [`experiments/run/token_results.jsonl`](experiments/run/token_results.jsonl), [`experiments/run/README.md`](experiments/run/README.md), `experiments/run/run_token.py` |
| Whole Qwen3-0.6B LM head (151,936 x 1024) through the state bank: 3.70 s on an A100 (nL=2), error 3.8e-7 | [`experiments/run/a100_results.jsonl`](experiments/run/a100_results.jsonl) |
| RMSNorm at full dims on real activations: 1.0e-7 to 1.7e-7 (max error / max abs), `eps` handled exactly | [`experiments/ops/res_norm_swiglu_fp32_1e+08.json`](experiments/ops/res_norm_swiglu_fp32_1e+08.json) |
| SwiGLU (4864 / 3072 neurons): 1.8e-8 to 1.9e-7 | same file |
| Softmax attention with RoPE, GQA, causal mask and KV (T=25, four layers each of Qwen2.5-0.5B and Qwen3-0.6B): 6.6e-7 to 1.3e-5 | [`experiments/ops/res_attn_fp32.json`](experiments/ops/res_attn_fp32.json) |
| Qwen3 per-head q/k norm plus streamed gain: 2.9e-7 to 3.1e-7 | [`experiments/ops/res_misc.json`](experiments/ops/res_misc.json) |
| In-model argmax over the full 151,936 vocabulary through attention and an outer `lm_head`, correct on all 4 tested positions for each of Qwen2.5-0.5B and Qwen3-0.6B (smallest margin 0.14 logits); 18 to 44 s per argmax on a Mac CPU | [`experiments/ops/res_argmax_q25.json`](experiments/ops/res_argmax_q25.json), [`res_argmax_q3.json`](experiments/ops/res_argmax_q3.json) |
| bf16 outer models are only at noise level (0.1% to 3.6%, Qwen2.5 layer 0 13%); fp32 is the supported mode | [`experiments/ops/res_attn_bf16.json`](experiments/ops/res_attn_bf16.json), [`res_norm_swiglu_bf16_1e+08.json`](experiments/ops/res_norm_swiglu_bf16_1e+08.json) |
| HostGuard catches host-side inner math (bad matmul, softmax, silu, add, RMS, numpy/math calls, hidden readbacks, forward hooks, feeding outer output back) and passes a clean driver; the outer-model integrity check rejects patched forwards, subclasses and modifying hooks | `audit/test_hostguard.py`, `audit/test_outer_check.py` (both print `ALL OK`; re-run them yourself, see [How to run](#how-to-run)) |
| Circuit-route baseline (bit-level arithmetic in GDN state, CPU bf16): 14.7 MAC/s on a real 896-element row, 458 to 488 MAC/s on small synthetic tiles | [`experiments/bench/circuit_baseline/`](experiments/bench/circuit_baseline/), [`experiments/bench/RESULTS.md`](experiments/bench/RESULTS.md) |
| Outer-stack step throughput (A100, random 3.4 B-parameter outer, torch reference GDN): 2219 steps/s fp32 chunked prefill vs 25 steps/s single-step decode | [`experiments/bench/RESULTS.md`](experiments/bench/RESULTS.md) |
| Earlier hosted-nesting experiment: Qwen3-0.6B inside Qwen3.5-0.8B with exact fp32 agreement | [`older/hosted-nesting/README.md`](older/hosted-nesting/README.md) |

Reported in a findings file but without a result file in the repository (treat as reported, not independently reproducible from committed outputs): an alternative exact matvec built from SwiGLU as a multiplier, `silu(w)x + silu(-w)(-x) = wx`, about 1e-6 relative error on real Qwen2.5-0.5B `q_proj` and `down_proj`, and 0.64 to 1.14 s per 494 M weights of MAC throughput on an A100 (this figure assumes the host copies the activation into every weight step, which the audit flags as outside the rules until replaced by an in-model read) ([`experiments/alt/FINDINGS.md`](experiments/alt/FINDINGS.md), code in `experiments/alt/`). The same file records that using the KV cache as weight storage was ruled out.

## Status by tier

| Tier | What runs inside the outer Qwen's weights | Status |
|---|---|---|
| T0 | A single signed MAC / gadget (state write + read, SwiGLU multiplier, norm, product) | **Verified** in isolation, teacher-forced inputs (errors in the table above) |
| T1 | A full projection on real weights, accumulation inside the model | **Verified for all 197 matrices of Qwen3-0.6B**, teacher-forced (harness-fed activations), rel. err 5.6e-7. Reading the activation from an earlier round's output (round hand-off) is open. |
| T2 | A real decoder op chain: norm, QKV, RoPE, attention with KV, SwiGLU | **Partial.** Each op is verified at full inner dims on real activations; the ops are not yet chained across rounds with the matvecs. |
| T3 | The full model produces a token entirely inside the outer weights | **In progress; not yet achieved.** |
| T4 | Measured cost model for the full model | **Partial.** Weight streaming is measured (41.95 s per token's worth of weights on an A10G); the op-layer rounds are estimated, not measured. |

Known open items, from the agents' notes and the audit ([`audit/AUDIT.md`](audit/AUDIT.md)): the chain from one round's output to the next round's input does not exist yet (the audit measured that streamed READ frames differ when host randomness changes, so current drivers are not token-independent); the BALLAST constant differs between modules (`machine.py` 2^27, the state engine 1e6) and one value must be chosen for a shared model; the outer argmax/append loop has only been exercised in the argmax test, with logits host-injected; per-layer RMSNorm gain vectors need a vector write path (or folding into the streamed weights, which is exact); the large `XS` scale of the norm trick must end a round; K accumulation across tiles for small banks; bf16 is not exact; no long-context attention test (T=25 only); no longrope test.

## Cost model

![cost model](docs/assets/cost_model.svg)

Measured: **41.95 s** for the weight-streaming part of one Qwen3-0.6B token on an A10G with the torch reference GDN kernel and an `nL=4` bank (12.4 k frames/s). Estimates, labelled as such in the figure: smaller banks are faster per frame, and the extrapolated 7 to 19 s come from the runner's `plan()` function, not from a run. Not included in any measured number: roughly 113 dependent rounds of op layers (an estimate), the in-model argmax (18 to 44 s on a Mac CPU, not yet measured on GPU), and per-round launch overhead. Frame cost scales roughly with the square of the bank size and the path is fp32 compute-bound, not bandwidth-bound.

### Circuit route versus state-as-memory route

![comparison](docs/assets/comparison.svg)

The circuit route in [`reference/`](reference/nested-qwen-lab) emulates arithmetic bit by bit in GDN state. Extrapolating its measured rates to 494 M projection MACs gives between about 12 days (488 MAC/s, small synthetic tiles) and about a year (14.7 to 15 MAC/s, real-weight rows) per token, before any non-matvec operation. The state-as-memory route moves the same contraction into the delta-rule read, which is why the weight-streaming measurement above is about six orders of magnitude faster than the slower circuit number. That comparison is matvec against matvec; a full chained token for the state route is not yet measured.

## The honesty rules

What the host **may** do: write BALLAST (a constant), control flags, one-hot address fields, inner weight rows (transport and reformatting only: transpose, pad, tile, quantise and dequantise, per-row shifts and the SiLU pre-inversion of the payload, which depend on weights alone), token ids and embedding rows, `position_ids` (the inner token clock, needed for RoPE continuity), the outer argmax and append, and read outputs for validation only.

Weight-only host arithmetic is allowed but must be disclosed (it never depends on a token): SiLU pre-inversion of the payload, compensation of the outer RMSNorm scale on weight/flag frames, folding RMSNorm gains and biases into streamed weights, and compile-time constants. The audit lists each of these.

What the host **may not** do: compute any inner activation, matmul, norm, softmax, RoPE rotation, nonlinearity or sampling; feed a readback value into the next input; modify the outer model's code (hooks that compute, patched `forward`, subclasses); or use the HF inner model anywhere except as a reference for checks.

### How to audit

[`audit/`](audit/) is the HostGuard harness. It does not trust the drivers:

- `hostguard.py` intercepts every ATen op with a dispatch mode. Outside the outer forward, matmul-like, softmax-like and norm-like ops are violations; elementwise arithmetic is a violation unless declared as `transport(...)`; unknown ops are violations (strict). Argmax is only permitted on tensors derived from outer logits. Readbacks of anything but outer logits are violations unless declared `validation_read(...)`. Numpy and `math` compute calls from user code are caught on a best-effort basis.
- `outer_check.py` checks that the outer model really is the stock `Qwen3_5ForCausalLM`: no class `forward` monkeypatch, no subclass, no modifying hooks, no swapped `gdn_impl` without being listed.
- `static_scan.py` is an AST scan for host-side math the dynamic guard cannot see (for example ndarray `@`).
- `run_audit.py` runs any driver under the guard and exits non-zero on violations. Two demo drivers are included: `demo_good_driver.py` passes, `demo_cheat_driver.py` (a host `bmm`) fails.
- `compare.py` / `compare_runs.py` compare in-model traces against an HF inner reference per op (T1 op, T2 layer, T3 model).

Limits: the guard is a strong check, not a proof. Declared scopes (`transport`, `weight_prep`, `reference`) are listed in its report so a reviewer can read them, and a human still has to read those. **The chained drivers (T2, T3) do not exist yet, so they have not been audited.** The audit so far ([`audit/AUDIT.md`](audit/AUDIT.md)) ran HostGuard on the CPU-sized state-engine and glue tests (pass, with declared scopes) and read the GPU drivers without executing them; it found no host arithmetic on activations inside what is claimed today, and found that all current results are teacher-forced. The GPU paging runner (`experiments/run/run_token.py`) was read, not run under HostGuard.

## Repository layout

```
README.md
docs/
  INTERFACE.md           outer-machine interface: layout, frames, what the host may do, rounds
  assets/                SVG figures (architecture, state memory, layer timeline, cost, comparison)
  make_figures.py        regenerates the SVGs from the numbers quoted in this README
qweninqwen/
  machine.py             Layout, make_outer (zeroed stock Qwen3_5ForCausalLM), BALLAST
experiments/
  state/                 matvec in DeltaNet state: engine.py (frame protocol), tiled.py, tests
  ops/                   non-matvec ops in outer layers: ops.py, tests, res_*.json results
  alt/                   alternative matvec: SwiGLU-as-multiplier (mlpmul.py), findings
  run/                   full-model weight-streaming runner and its measured results
  bench/                 outer step throughput, circuit-route baseline results
audit/                   HostGuard, outer-model check, static scan, demo drivers, AUDIT.md findings
reference/              the original circuit-based Qwen-in-Qwen lab (bit-level arithmetic in GDN state)
older/
  hosted-nesting/        earlier hosted-nesting experiments: Qwen3-0.6B hosted in Qwen3.5-0.8B (exact)
  hosted-multi-model/    earlier prototype: several models hosted in one checkpoint
```

The `experiments/` files are working code copied from the agents' benches. Modules import each other by sibling directory (see the `sys.path` lines at the top of each file), so run scripts from inside their own directory.

## How to run

Requirements: Python 3.11, `torch`, `transformers` 5.x (tested with 5.20.2), `safetensors`, and the Qwen checkpoints in the Hugging Face cache (`Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B`). GPU scripts expect CUDA and fp32 with TF32 off. Setting `NOFLA=1` forces the torch reference GDN path, which was the fastest correct one measured.

```sh
# audit harness self-tests and demos (CPU, seconds)
cd audit && python test_hostguard.py && python test_outer_check.py
python run_audit.py demo_good_driver.py      # AUDIT RESULT: PASS
python run_audit.py demo_cheat_driver.py     # AUDIT RESULT: FAIL

# state engine
cd experiments/state && python test_silu_inv.py && python test_tiled.py

# ops at full inner dims (writes res_*.json)
cd experiments/ops && python test_norm_swiglu.py fp32 1e8 && python test_attn.py fp32

# the full-token weight-streaming measurement (GPU; args: model nL layers lm_head)
cd experiments/run && NOFLA=1 python run_token.py q3 4

# regenerate figures
python docs/make_figures.py
```

## Credits

- **Qwenputer** ([divagr18/qwenputer](https://github.com/divagr18/qwenputer)) is the idea this repository is a homage to: a computer whose circuits are Qwen's own weights. Its Qwen-in-weights compiler is vendored under `reference/nested-qwen-lab/qwentpu` with its license.
- **The qwentpu reference** (`reference/nested-qwen-lab`) is the circuit-based Qwen-in-Qwen lab whose measurements form the baseline above.
- **Qwen** models by the Qwen team at Alibaba (Qwen3.5 as the outer model; Qwen3-0.6B and Qwen2.5-0.5B as the inner models), Apache-2.0.
- Built on Hugging Face `transformers`.

## License

Code in this repository is MIT (see `LICENSE`); vendored reference code keeps its own license file. Qwen model weights are Apache-2.0, licensed by Alibaba.
