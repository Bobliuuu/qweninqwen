<div align="center">

# qweninqwen

**A real Qwen3-0.6B token, computed by the weights and recurrent state of another, stock Qwen.**

![Qwen inside Qwen](docs/assets/hero.gif)

| outer argmax == Hugging Face | max relative logit error | per token |
|:---:|:---:|:---:|
| **15846 == 15846**, and 6 of 6 positions | **5.8e-5** | **268.6 s** (shared A100, fp32, untuned) |

[Explainer video](docs/launch/video/explainer_16x9.mp4) | [Demo video](docs/launch/video/demo_16x9.mp4) | [Tweet draft](docs/launch/TWEET.md) | [Provenance of every number](docs/launch/PROVENANCE.md)

A homage to [Qwenputer](https://github.com/divagr18/qwenputer), in which a CPU lives inside Qwen's weights. Here the machine runs a language model.

</div>

> **Scope (read before the numbers).** This repo has TWO routes to "Qwen inside Qwen". The **circuit route** (every inner operation, bit by bit, inside the outer model's weights and DeltaNet state, as in Qwenputer; the actual goal) is estimated at about **3.4e7 s (about 1.1 years) per token** and has **not been run end to end**; only its pieces were benchmarked (14.7 MAC/s on a real 896-element row, [`experiments/bench`](experiments/bench/RESULTS.md)). Everything measured below, including the 268.6 s token, is the faster **compiled route** (each inner op gets outer layers doing real fp32 matvecs; the host streams the inner weights in). It is a verified stepping stone, not the circuit-level result.

> **Read this first.** One full token of the **real, unscaled Qwen3-0.6B** (28 layers, real weights, the full 151,936-token LM head, argmax computed in-model) runs inside a stock `Qwen3_5ForCausalLM` (1006 outer layers, D=40775, 293,068 frames) and matches Hugging Face. A 6-position run matches at every position. **Those are teacher-forced positions of one prompt, not free-running generation, and the host still does embedding lookup, flags, one-hot addresses, position ids, weight payloads, KV scratch truncation and the lm_head evaluation through 36 columns.** Free-running generation and the audit of the real driver are not done yet (status table below). The whole thing is ~4.5 minutes per token.

## Status board

Single source of truth for what is done and what is not. Pending rows are flipped by the maintainers when results land (see [`docs/launch/UPDATING.md`](docs/launch/UPDATING.md)).

<!-- STATUS:BEGIN -->
| Item | Status | Evidence |
|---|---|---|
| One real token, 28 layers, full vocab head, in-model argmax == HF | **done** | [`logs/real_L28.log`](experiments/vslice/logs/real_L28.log) |
| 6-position teacher-forced run (KV used at positions 1-5) == HF at every position | **done** | [`logs/real_L28_gen.log`](experiments/vslice/logs/real_L28_gen.log), [`res_vs2_real151936_L28.json`](experiments/vslice/logs/res_vs2_real151936_L28.json) |
| Free-running generation vs HF greedy (outer token fed back) | **in progress** | pending: result path `TBD` |
| HostGuard audit of the real driver | **in progress** | pending: result path `TBD` |
| Speedup work (tuned executor) | **in progress** | pending, see [`PERF_NOTES.md`](PERF_NOTES.md) / [`RUNS.md`](RUNS.md); the 268.6 s headline is the untuned baseline |
<!-- STATUS:END -->

## Watch it

| Explainer (what it means, how it works) | Demo (the real 6-position run) |
|---|---|
| [`explainer_16x9.mp4`](docs/launch/video/explainer_16x9.mp4) ([square cut](docs/launch/video/explainer_square.mp4)) | [`demo_16x9.mp4`](docs/launch/video/demo_16x9.mp4) ([square cut](docs/launch/video/demo_square.mp4)) |

The mechanism animations are **illustrations** and say so on screen; every number shown comes from the committed logs. The demo is a time-lapse replay of the recorded run, not a live one.

![idea](docs/assets/idea.gif)

## The tier ladder

The project is organised as tiers: each tier is a harder claim than the one below it and needs the previous one. The staircase shows where the evidence stands today.

![tier climb](docs/assets/tiers_climb.gif)

| Tier | What it means | What is needed to reach it | Status and evidence |
|---|---|---|---|
| T0 | A single signed MAC or gadget (state write + read, SwiGLU multiplier, norm, product) runs in the outer weights | write/read protocol, one op at a time, teacher-forced inputs | **Verified.** [`experiments/state`](experiments/state/), [`experiments/ops`](experiments/ops/) `res_*.json` |
| T1 | A full projection on real weights, with accumulation inside the model | page every weight matrix through DeltaNet state | **Verified for all 197 matrices** of Qwen3-0.6B, rel. err 5.6e-7, harness-fed activations: [`token_results.jsonl`](experiments/run/token_results.jsonl) |
| T2 | A real decoder op chain (norm, QKV, RoPE, attention with KV, SwiGLU) at real dimensions | chain the ops with the matvec banks at real dims | **Partial.** Each op verified at full dims; chained only in the toy |
| T3-toy | A complete toy inner model (random weights, d=128) generates tokens entirely inside the outer weights | every op chained, host limited to the allowed list | **Achieved** (validation vehicle): [`VSLICE_RESULTS.md`](experiments/vslice/VSLICE_RESULTS.md) |
| T3 real | Real Qwen3-0.6B produces tokens entirely inside the outer weights | T3-toy at real size (1006 outer layers), layer-major executor | **One token achieved; 6 teacher-forced positions match HF.** Pending: free-running generation (**in progress**), HostGuard audit of the real driver (**in progress**) |
| T4 | Measured cost model for the full model | timings of the real run, tuned kernels | **Measured for one token:** 268.6 s untuned. Speedup work: **in progress** |

![tier ladder](docs/assets/tiers_ladder.png)

```mermaid
flowchart LR
  T0["T0 gadget<br/>verified"]
  T1["T1 projection<br/>verified"]
  T2["T2 op chain<br/>partial"]
  T3t["T3-toy<br/>achieved"]
  T3r["T3 real<br/>partial"]
  T4["T4 cost model<br/>partial"]
  FR["free-running vs HF<br/>in progress"]
  HG["HostGuard audit<br/>in progress"]
  SP["speedup<br/>in progress"]
  T0 --> T1
  T1 --> T2
  T2 --> T3t
  T3t --> T3r
  T3r --> T4
  FR -->|completes| T3r
  HG -->|completes| T3r
  SP -->|improves| T4
```

(The same graph as a picture, for viewers without mermaid: [`flow_tiers.png`](docs/assets/flow_tiers.png).)

## How it works at a glance

```mermaid
flowchart LR
  H["Host streamer<br/>weights, ids, addresses,<br/>position ids"]
  G["DeltaNet layers<br/>paged weight memory<br/>(matvec banks, registers)"]
  O["Compiled norm / MLP layers<br/>RMSNorm, SwiGLU,<br/>residual, gadgets"]
  A["Full-attention layers<br/>RoPE, softmax, KV<br/>(outer KV = inner KV)"]
  L["In-model argmax<br/>+ outer lm_head<br/>36 rails -> token id"]
  X["Host: outer argmax<br/>+ append"]
  H --> G
  G --> O
  O --> A
  A --> L
  L --> X
  X -->|next token| H
```

![who does what](docs/assets/flow_architecture.png)

```mermaid
flowchart LR
  E["Embed (host row)"]
  L["Inner layers x28<br/>norm, QKV bank, RoPE+attn,<br/>SwiGLU, residuals"]
  N["Final RMSNorm"]
  B["LM head<br/>594 blocks of 256"]
  M["Block max/argmax<br/>(in model)"]
  C["Hard-max attention<br/>over 594 winners"]
  R["36 rails<br/>outer lm_head"]
  T["Host argmax<br/>+ append token"]
  E --> L
  L --> N
  N --> B
  B --> M
  M --> C
  C --> R
  R --> T
```

![one token](docs/assets/flow_token.png)

### Results

![six positions](docs/assets/results_table.png)

![cost](docs/assets/cost_chart.png)

The 3.4e7 s circuit-route figure in the cost chart is an extrapolation (estimate), not a run; see [Cost model](#cost-model).

### The argmax tournament

![tournament](docs/assets/tournament.gif)

(Bar heights are an illustration; the winning block and the 18-bit code are derived from the real token id 15846.)

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

The outer layers are strictly feed-forward: layer `l` at step `t` sees only the outputs of earlier layers at step `t` and the state of layer `l` from earlier steps, and that state only contains functions of layers shallower than `l`. State and KV therefore carry data **deeper** in the stack, never back to shallower layers. The consequence, and the design basis of the toy, is an **unrolled pipeline in depth**: every inner op gets its own outer layers in dependency order, and a register is a DeltaNet layer placed between the op that writes it and the op that reads it. One inner token is one outer forward call (a chunked prefill of the WRITE frames followed by the READ frame of each bank). The real-dimension run needs 1006 outer layers (D=40775) for the 28 inner layers plus the LM head and argmax, and every frame flows through all of them. Contract and hazards (L2-normalised q/k, conv SiLU, gated RMSNorm, ballast trick): [`docs/INTERFACE.md`](docs/INTERFACE.md) (section 5b supersedes section 5, which assumed per-round state hand-off).

### DeltaNet state as weight memory

![state as memory](docs/assets/state_memory.gif)

Illustration with a toy 4x5 matrix; static version: [`state_memory.png`](docs/assets/state_memory.png) (hand-drawn figure) and [`state_memory_toy.png`](docs/assets/state_memory_toy.png).

### How a token is computed

![unrolled pipeline](docs/assets/pipeline.gif)

![unrolled pipeline, static](docs/assets/unrolled_pipeline.png)

1. **Embed.** The host places the embedding row of the current token in the stream (a lookup, no arithmetic) together with position id (the inner token clock) and flags.
2. **Per inner layer, in depth order.** RMSNorm (a GDN gated-RMSNorm head, or a norm layer); the `W_qkv` bank (WRITE frames load weight rows into DeltaNet state, the READ frame contracts the activation against them); a register layer; for Qwen3, q/k norm plus a gain product in MLP gadgets; RoPE and softmax attention in an outer full-attention layer that appends to and reads the KV cache; `W_o` and the residual add; the second RMSNorm; `W_gate, W_up`; SwiGLU in an MLP layer; `W_down` and the residual add.
3. **After the last layer.** Final RMSNorm, the LM-head contraction, and an in-model hard-max attention over all vocabulary positions whose value is the address code of the winner; the outer `lm_head` decodes the code.
4. **The host does the outer argmax and appends the token.** That is the only host-side decision and the only deep-to-shallow channel.

Steps 1 to 4 have been executed for the real Qwen3-0.6B at one position, and for a toy model over several tokens. Because the stock config is uniform per layer, the real-dimension outer (1006 layers, D=40775) is far too large to hold at once; the runner uses a **layer-major executor** that builds each outer layer, pushes all frames through it and frees it. This is an execution strategy for the same stock layers, not a change to the model.

### How the full-vocabulary LM head works in-model

![LM head](docs/assets/lm_head.png)

The LM head is a 151,936 x 1024 matrix, so its logits are produced in 594 blocks of 256 by the same state-bank matvec as every other projection. Each block then reduces itself in the model: a per-block max/argmax tournament yields the block's best logit and the local index (8 bits). A hard-max attention over the 594 block winners (key = block best logit, value = a dual-rail code of the local index plus the block index, 8 + 10 = 18 bits, 36 rails) selects the global winner across blocks. The outer `lm_head` has only 36 non-zero columns and decodes the rails into the outer token id, and the host takes the outer argmax and appends it. In the full-token run the winning rails were clean: top code score 18.0007 against an ideal 18.

## What is verified today

Only claims with a result file or a test in this repository are listed.

| Claim | Evidence |
|---|---|
| **One full token of the real Qwen3-0.6B, one position.** 28 layers, real unscaled weights (d=1024, 16 q / 8 kv heads x 128, inter 3072), full 151,936-token LM head and in-model argmax, prompt "The capital of France is" (position-0 token): outer argmax 15846 == Hugging Face 15846; block-best logit max error 5.8e-5 relative to max\|logit\|; winning 36 rails clean (top code score 18.0007 of ideal 18). Stock Qwen3_5 decoder layers, fp32, layer-major executor, 1006 outer layers, D=40775, 293,068 outer frames, **268.6 s on a shared A100** (13.9 s building layers, about 242 s running; HF torch chunk GDN, untuned). All norms, 197 matvecs, registers, RoPE, softmax attention and KV, SwiGLU, residuals and the block/cross-block argmax run in the outer model. Host: embedding row, flags, one-hot addresses, position ids, weight payloads (SiLU pre-inversion, gains folded into W, padding), per-token truncation of the argmax-attention scratch KV, evaluation of the outer lm_head through its 36 non-zero columns. **Headline number is position 0; a 6-position teacher-forced run (KV used at positions 1-5) also matches HF at every position (`logs/real_L28_gen.log`). HostGuard audit pending.** The raw run log and result JSON are in `experiments/vslice/logs/`; the layer, frame and weight counts are reproducible without a GPU. | [`experiments/vslice/FULL_TOKEN_RESULT.md`](experiments/vslice/FULL_TOKEN_RESULT.md), [`logs/real_L28.log`](experiments/vslice/logs/real_L28.log), `experiments/vslice/vs2.py`, `test_vs2.py`, `cost_model.py` |
| Smaller checks with the same code: 1 real layer + full vocab, argmax == HF at 5/5 positions (error 2e-6..4e-5); 2 real layers, 4/4 positions (error up to 4e-4, cause not isolated) | same file |
| **Toy end to end.** A random-weight Qwen3-architecture inner model (d=128, 4 q / 2 kv heads of 32, inter 256, 2 layers, vocab 512, q/k-norm) generates 4 prompt + 4 generated tokens inside a stock `Qwen3_5ForCausalLM` (34 layers, D=5641, 2.02 B fp32 parameters): outer logits vs Hugging Face 5e-7..1.9e-6 at every step, argmax identical. All norms, matvecs (GDN state banks), registers, SwiGLU, q/k-norm gain products, RoPE/GQA/KV attention and the LM head run in the outer model; the host writes flags, one-hot addresses, position ids, SiLU-pre-inverted gain-folded weight payloads and the embedding row, and does argmax + append. 1530 outer steps and about 12 to 14 s per inner token on an A100. **Toy only: tiny and random, not Qwen3-0.6B.** Reported from an A100 run; not re-run on the machine that built this repository. | [`experiments/vslice/VSLICE_RESULTS.md`](experiments/vslice/VSLICE_RESULTS.md), `experiments/vslice/vslice.py`, `experiments/vslice/toy_test.py` |
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

## Status by tier (detail)

| Tier | What runs inside the outer Qwen's weights | Status |
|---|---|---|
| T0 | A single signed MAC / gadget (state write + read, SwiGLU multiplier, norm, product) | **Verified** in isolation, teacher-forced inputs (errors in the table above) |
| T1 | A full projection on real weights, accumulation inside the model | **Verified for all 197 matrices of Qwen3-0.6B**, teacher-forced (harness-fed activations), rel. err 5.6e-7. Chaining the output of one matrix into the next is demonstrated only in the toy. |
| T2 | A real decoder op chain: norm, QKV, RoPE, attention with KV, SwiGLU | **Partial.** Each op is verified at full inner dims on real activations; the ops are chained with the matvecs only in the toy, not at real dimensions. |
| T3-toy | A complete toy inner model (random weights, 2 layers, d=128) generates tokens entirely inside the outer weights | **Achieved** (validation vehicle, not the real model) |
| T3 real | Real Qwen3-0.6B produces a token entirely inside the outer weights | **One full token achieved; 6-position teacher-forced run with KV matches HF at every position.** Free-running generation and the HostGuard audit are pending. |
| T4 | Measured cost model for the full model | **Measured for one token:** 268.6 s on a shared A100 (293,068 frames, 1006 outer layers, untuned fp32, torch GDN). Not yet measured at a KV-using position or on tuned kernels. |

Known open items, from the agents' notes and the audit ([`audit/AUDIT.md`](audit/AUDIT.md)): the HostGuard audit of the full-token driver is pending; the earlier audit (written before the chained drivers) measured that streamed READ frames differ when host randomness changes in the per-matrix drivers, so those are not token-independent, and neither the toy nor the full-token driver has been audited; the BALLAST constant differs between modules (`machine.py` 2^27, the state engine 1e6) and one value must be chosen for a shared model; the L=2 error bump (up to 4e-4, cause not isolated); free-running generation (only teacher-forced positions 0-5 tested) for the real model; per-layer RMSNorm gain vectors need a vector write path (or folding into the streamed weights, which is exact); the large `XS` scale of the norm trick must end a round; K accumulation across tiles for small banks; bf16 is not exact; no long-context attention test (T=25 only); no longrope test.

## Cost model

![cost model](docs/assets/cost_model.png)

Measured, full token of the real Qwen3-0.6B, single position: **268.6 s** on a shared A100 (fp32, torch-reference GDN, layer-major executor, untuned): 13.9 s building layers plus about 242 s running 293,068 outer frames (about 1.2 k frames/s) through 1006 outer layers at D=40775. The frame, layer and weight counts come from the planner and are reproducible without a GPU (`experiments/vslice/cost_model.py`). Measured, weight streaming alone (an `nL=4` bank on an A10G, 12.4 k frames/s, harness-fed activations): 41.95 s; this is a different, much shallower stack, so the two numbers do not add up or compare directly. Estimates, labelled as such in the figure: the 7 to 19 s smaller-bank figures come from the runner's `plan()` function, not from a run, and the 110 s paging bound is an untested design estimate. Not yet measured: a KV-using position, tuned kernels (the flash-linear-attention path was slower or inexact in the agents' tests), and the in-model argmax on its own (18 to 44 s on a Mac CPU in the op test). Frame cost scales with depth and `D`, and the path is fp32 compute-bound.

### Circuit route versus state-as-memory route

![comparison](docs/assets/comparison.png)

The circuit route in [`reference/`](reference/nested-qwen-lab) emulates arithmetic bit by bit in GDN state. Extrapolating its measured rates to 494 M projection MACs gives between about 12 days (488 MAC/s, small synthetic tiles) and about a year (14.7 to 15 MAC/s, real-weight rows) per token, before any non-matvec operation. The state-as-memory route moves the same contraction into the delta-rule read, which is why even the full chained token (268.6 s) is more than five orders of magnitude faster than the slower circuit number (3.4e7 s, matvecs only). That comparison is matvec against matvec; the full chained token for the state route (268.6 s) is a single-position measurement.

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

Limits: the guard is a strong check, not a proof. Declared scopes (`transport`, `weight_prep`, `reference`) are listed in its report so a reviewer can read them, and a human still has to read those. **The chained toy driver (`experiments/vslice`) has not been run under HostGuard yet; no audit result for it exists.** The audit so far ([`audit/AUDIT.md`](audit/AUDIT.md)) ran HostGuard on the CPU-sized state-engine and glue tests (pass, with declared scopes) and read the GPU drivers without executing them; it found no host arithmetic on activations inside what is claimed today, and found that all current results are teacher-forced. The GPU paging runner (`experiments/run/run_token.py`) was read, not run under HostGuard. The audit predates the toy.

## Repository layout

```
README.md
docs/
  INTERFACE.md           outer-machine interface: layout, frames, what the host may do, depth ordering
  assets/                PNG and GIF figures (no SVGs), mermaid sources (*.mmd)
  make_figures.py        regenerates the static PNG figures from the numbers quoted in this README (needs rsvg-convert)
  launch/                make_media.py (GIFs, PNGs, mp4s), TWEET.md, PROVENANCE.md, UPDATING.md, video/
qweninqwen/
  machine.py             Layout, make_outer (zeroed stock Qwen3_5ForCausalLM), BALLAST
experiments/
  state/                 matvec in DeltaNet state: engine.py (frame protocol), tiled.py, tests
  ops/                   non-matvec ops in outer layers: ops.py, tests, res_*.json results
  alt/                   alternative matvec: SwiGLU-as-multiplier (mlpmul.py), findings
  run/                   full-model weight-streaming runner and its measured results
  vslice/                end-to-end slice: toy model (vslice.py) and real Qwen3-0.6B one-token run (vs2.py), planner, results
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

# end-to-end toy (A100 reported; needs ~8 GB for the 2 B-parameter outer)
cd experiments/vslice && python toy_test.py 4
# planner counts for the real model (CPU, seconds): layers, D, frames
python -c "import vs2, cost_model as c; print(c.counts(vs2.Cfg(1024,16,8,128,3072,28,151936)))"

# the full-token weight-streaming measurement (GPU; args: model nL layers lm_head)
cd experiments/run && NOFLA=1 python run_token.py q3 4

# regenerate figures and launch media (Pillow, numpy, matplotlib fonts, ffmpeg)
python docs/make_figures.py && python docs/launch/make_media.py
```

## Credits

- **Qwenputer** ([divagr18/qwenputer](https://github.com/divagr18/qwenputer)) is the idea this repository is a homage to: a computer whose circuits are Qwen's own weights. Its Qwen-in-weights compiler is vendored under `reference/nested-qwen-lab/qwentpu` with its license.
- **The qwentpu reference** (`reference/nested-qwen-lab`) is the circuit-based Qwen-in-Qwen lab whose measurements form the baseline above.
- **Qwen** models by the Qwen team at Alibaba (Qwen3.5 as the outer model; Qwen3-0.6B and Qwen2.5-0.5B as the inner models), Apache-2.0.
- Built on Hugging Face `transformers`.

## License

Code in this repository is MIT (see `LICENSE`); vendored reference code keeps its own license file. Qwen model weights are Apache-2.0, licensed by Alibaba.
