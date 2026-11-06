# Circuit-route plan: Qwen3-0.6B as a program on the RV32IM CPU inside a real Qwen3.5

Status: version 2 (rewritten on the user's definition). Tags: **[M]** measured (source cited), **[E]** estimated (derivation shown), **[?]** unknown. Projected and measured columns in the tier table are refreshed as the build agents' results land under `experiments/circuit/`.

**This is not the compiled route.** The README's 268.6 s/token result runs a custom, giant outer model (1006 layers, D=40775) in which inner ops are real fp32 matvecs and the host streams weights. That is a different thing. The circuit route here uses a **real, fixed-size, stock Qwen3.5** (0.8B primary, 4B alternative) exactly as Qwenputer does, and Qwen3-0.6B runs on it as a program.

## 1. Definition

**Outer model.** The stock-architecture Qwen3.5-0.8B (primary) or Qwen3.5-4B at its real size, with the Qwenputer RV32IM CPU compiled into its weights (Qwenputer: `divagr1925/Qwen3.5-4B-Qwenputer`, code `~/qwenos-work/qwenputer`, `experimental/riscv`). Registers, RAM, ROM and program counter live in Gated DeltaNet state; each emitted **trace token** is one CPU step (1.0-1.55 tokens per instruction [M, `rv32.md`]).

**Inner model.** The real Qwen3-0.6B, run as a C program on that CPU, the way Karpathy's stories260K already runs there (weights as int4 data, VDOT + SMAQA instructions) [M, Qwenputer README]. Every arithmetic op of Qwen3-0.6B is then executed by the compiled CPU circuit: multiply, add, norm, softmax, argmax.

**Host is allowed to:** boot the CPU (load firmware and data into state), embed token ids, run the outer argmax and append (the autoregressive loop), and the README's other transport items. Not allowed: any inner arithmetic, or feeding a readback activation into the next input.

**Exactness bar.** A pinned inner numeric contract (int4 weights, int8 activations, fixed-point nonlinear ops) with a bit-exact reference emulator; the trace must equal the emulator's. Agreement with bf16 Hugging Face argmax is reported separately (quantisation study: `experiments/circuit/algo/quant_study.py`, results pending).

## 2. What exists today (Qwenputer, all [M] from its repo unless noted)

| Piece | Measured |
|---|---|
| RV32I + MUL, 32 KB RAM, 128 KB DROM, 256 KB ROM; riscv-tests pass; 4.05 M instructions verified free-running (`rv32.md`) | exact in bf16 on GPU with fla |
| stories260K as a C program, plain int8 | 2,430,000 trace tokens per story token |
| same with VDOT + SMAQA, int4 weights | 225,000 trace tokens per story token; about 41 s at 5,500 tok/s on an L4 |
| same with the QwenPU vector unit | 53,000 trace tokens per story token; about 16 s on an L4 (work in progress, checkpoint 26.7 GB) |
| weight memory | 261,120 int4 weights in DeltaNet heads |
| executor speed (L4, batch 1) | HF stock modules 59-91 tok/s dense; 112-609 tok/s compiled/compacted; custom fused executor 5,562 tok/s (3,594 instr/s) |
| executor batch scaling (independent sessions) | batch 8: 18.2k tok/s; batch 32: 21.3k (Triton) to 28-38k tok/s (sparse torch) |
| executor equivalence | equals the IR on every dim; free-running traces equal the reference (`execcheck`) |

The executor is "same arithmetic, not the same kernels" as the HF modules (`rv32.md`). Any figure that depends on it must say so and be backed by lock-step equivalence checks.

## 3. Work per Qwen3-0.6B token

Matvec MACs [E, from config]: 28 x 15.73 M = 440.4 M in the layers + 155.6 M LM head = **596.0 M**. Attention adds 114,688 x T MACs (under 3 % at T=128). Instruction count per token will come from the compiler agent (`experiments/circuit/compiler`, only `isa.py` is pushed so far); until then I scale Qwenputer's measured stories260K cost per parameter, which assumes cost proportional to parameters:

| Build | Trace tokens per param (stories260K [M]) | Trace tokens per Qwen3-0.6B token [E] | Implied instr/MAC at 1.4 tok/instr [E] |
|---|---|---|---|
| plain C, int8 | 9.35 | 5.6e9 | about 6.7 |
| VDOT + SMAQA, int4 | 0.865 | 5.2e8 | about 0.6 |
| QwenPU vector unit | 0.204 | 1.2e8 | about 0.15 |

Caveats on this scaling: (a) the LM head is 26 % of the MACs here but tiny in stories260K (vocab 512) and d=1024 amortises norms better than d=64, so linear-in-params is rough, probably within 2x; (b) nonlinear ops are not proportional: about 1e6 float ops per token (RMSNorm 2x28x1024 + q/k norm, softmax exps 28x16xT, SiLU 86 K, RoPE) at about 100-200 soft-float instructions each is 1e8-2e8 instructions [E], 30-50 % on top of the VDOT build unless the nonlinear agent's fixed-point or fp ISA extensions cut it (the gate-level netlists in `experiments/circuit/nonlinear` are fp32 mul with 4,332 neurons and depth 124, add 2,670 neurons depth 176 [M, `res_fp.json`], which says dedicated instructions beat soft-float); (c) LLM int4 weights are 298 MB, see section 4.

## 4. The weight-memory problem

Qwenputer's weight memory holds 261,120 int4 weights. Qwen3-0.6B needs 596 M nibbles (298 MB), about 2,300x more. The DeltaNet state of the outer models is tiny: Qwen3.5-4B has about 24 DeltaNet layers x 32 heads x 128 x 128 = 12.6 M float entries [E from config]; 0.8B has about 4.7 M [E]. Even at one byte per entry the state cannot hold the weights. Options:

| Option | Mechanism | Cost | Honesty issue |
|---|---|---|---|
| M1 paging | the host rewrites weight pages into state between trace tokens on a static, data-independent schedule (like a DMA from disk) | 298 MB of transport per token; negligible against minutes of compute; page-fault style stalls | needs a ruling: bulk state writes mid-run are a stronger host capability than prompt-time boot, though they depend on weights only |
| M2 embedding table as ROM | the embedding matrix is a lookup table of real parameters (248 K rows; 4B: 2560 dims, about 636 M entries, just enough for 596 M nibbles; 0.8B: about 254 M entries, not enough). The CPU emits a ROM-row token, the next step embeds the whole row (thousands of nibbles per token) | weights live in the outer's real parameters; bandwidth up to about 2,500 weights per trace token | 4B only; the 18-bit token id is the address funnel; needs the circuit width to consume a row; unproven [?] |
| M3 filesystem model | Qwenputer-Filesystem keeps files in DeltaNet state; same capacity limit | no gain | n/a |

Plan: M1 for the first end-to-end runs (clearly labelled), investigate M2 as the end-state. Owner: algo agent.

## 5. Tier table (faster builds of the same run)

Projected s/token = tokens per token / rate [E]. The measured column stays empty until a run exists. Costs use L4 about 1 USD/h on a cloud GPU [E, assumption; check current prices].

| Tier | Build | Tokens per token | Rate | Projected s/token | Measured | Hardware, cost |
|---|---|---|---|---|---|---|
| T0 | literal: plain C int8, stock HF modules, Qwenputer circuit | 5.6e9 | 91-250 tok/s (HF path [M]) | **2.2e7 - 6.1e7 (0.7 - 1.9 years)** | none (extrapolated) | L4, about 6e3 - 1.7e4 GPU-hours, not run |
| T1 | fused executor (Triton + CUDA graph), same program | 5.6e9 | 5,562 tok/s [M] | 1.0e6 (11.6 days) | pending (engine agent) | 1 L4, about 280 h, about 280 USD |
| T2 | VDOT + SMAQA int4 build of Qwen3-0.6B | 5.2e8 | 5,562 tok/s | 9.3e4 (26 h) | pending | 1 L4, about 26 h, about 26 USD |
| T3 | + QwenPU vector unit, nonlinear ISA extensions, algorithmic cuts (skip zero activations, tiling, fewer instructions per MAC) | 1.2e8 or below | about 3,300-5,562 tok/s | 2.2e4 - 3.7e4 (6-10 h) | pending (algo, nonlinear, compiler) | 1 L4/A10G, about 10 h |
| T4 | persistent single-kernel step, faster GPU (L40S estimated 16 k tok/s upper bound [E, Qwenputer]) | 1.2e8 | 1.0e4 - 1.6e4 | 7.5e3 - 1.2e4 (2-3 h) | pending | L40S/A100/H100 |
| T5 | parallel instances (scale-out, `experiments/circuit/scaleout`): 8-32 CPUs each on a slice of rows, aggregate 21-38 k tok/s measured for independent sessions at batch 32 | 1.2e8 | 2.1e4 - 3.8e4 aggregate | 3e3 - 6e3, 1-2 h | pending | 1 GPU, a few hours |
| T6 | T5 on several GPUs with layer pipelining | 1.2e8 | 1e5 or more aggregate | about 1e3 (about 15 min) [speculative] | pending | 4-8 GPUs |

Honest cautions on the table: T1 onward replaces the HF kernels by an equivalent executor; T3 numbers depend on the nonlinear extensions and on quantisation fidelity; **T5/T6 need a ruling**: instances are independent sessions, and splitting a matvec across them requires moving activations or partial sums between sessions (host copy of emitted tokens). That is data transport, not arithmetic, but it breaks the "no readback fed into input" rule as written. If not allowed, T5 is capped at what one session's width can do and the table stops at T4. The scaling from stories260K may be off by 2x in either direction, and the batch speedup (about 4-7x over batch 1) is measured for tiny programs.

## 6. Verification ladder (same program, growing size)

| Step | Claim | Cost [E] |
|---|---|---|
| V0 | one Qwen3-0.6B-shaped int4 matvec row (K=1024) on the CPU, trace equals emulator | tens of thousands of tokens, seconds on the executor |
| V1 | one projection (1024 x 3072) | 3e6 MACs: about 2e6 tokens at VDOT, minutes at 5.5 k tok/s |
| V2 | one decoder layer incl. norms, RoPE, attention (T=8), SwiGLU | 15.7 M MACs: 1e7-1e8 tokens, 0.5-5 h at T1/T2 speed |
| V3 | tiny e2e (`experiments/circuit/e2e`): small Llama/Qwen-architecture checkpoint, whole token loop in the CPU | hours |
| V4 | full Qwen3-0.6B, one token, vs the pinned integer emulator and vs HF | per tier table |

## 7. Hardware

AWS G-family quota is 16 vCPUs and both g5 boxes are in use, so no new AWS GPU without a quota increase. Lambda A100 (`lambda-final`) is shared and was at 100 % utilisation when checked; use `flock -w 7200 ~/.gpu.lock`. Modal (`modal run --detach`) offers L4/L40S/A100 but a workspace GPU limit queued one earlier job. The steps are launch-latency-bound, so an L4 or A10G is nearly as good as an A100 for batch 1.

## 8. Recommended first experiment

On a spare GPU within hours: take the Qwenputer executor and measure it on a program shaped like the real inner workload. Two parts.
1. (CPU, minutes) With the Qwenputer reference emulator, write the Qwen3-0.6B inner kernel (int4 weights, int8 x, K=1024 row, then a 1024 x 3072 projection) as a C program with VDOT/SMAQA and count instructions and trace tokens per MAC. This replaces the stories260K scaling in section 3 with a measured number. Owner: compiler agent.
2. (GPU) Run that program under `fastexec.py` and report tok/s at batch 1, 8, 32, plus the state size at 0.8B and 4B width:
   `python -m chipformer.rv.timing --hidden 1024` (0.8B width; `--hidden 2560` for 4B), then the same with the new program.
Decision rule: if instr/MAC at VDOT is at or below 1 and executor tok/s at 0.8B width is at least 5 k, T2 (about 1 day per token) is on track; if instr/MAC is above 3 the nonlinear and loop overhead dominates and the ISA work moves first.

## 9. Open items needing the user

1. Ruling on mid-run weight paging (M1) and on cross-session data movement (T5/T6).
2. Numeric contract for Qwen3-0.6B (W4A8 vs W8A8, fixed-point nonlinear ops) and what "equals" means against bf16 HF.
3. Outer size: 0.8B cannot hold the embedding-ROM option; 4B can.
