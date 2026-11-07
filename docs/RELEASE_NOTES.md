# Release notes (DRAFT): v1.0-token15846

Draft tag name: `v1.0-token15846`. Not created, not pushed, not published.

## 🧠 Headline

A real Qwen3-0.6B token, produced by a RISC-V circuit compiled into the weights of a Qwen3.5-4B-geometry model. Prompt token `[785]`, position 0. The circuit halts on its own at step 57,705 of chunk 95 and its final word is **15846**, the token Hugging Face's Qwen3-0.6B (fp32 and bf16) produces.

## ✅ What was verified

- 96 chunks, 199,287,158 trace tokens, 178,066,316 instructions.
- 96 / 96 jobs clean: 0 free-token mismatches, 0 forced tokens, 0 host injections.
- 95 / 95 seams exact (registers, pc, anchor, `n`, `wb`, latch, all 8,192 RAM words, `bn`).
- Chunk 0 boots from the firmware ELF alone.
- Weights and blob constants come from the outer checkpoint's own `embed_tokens.weight` (168,684 rows, hidden size 7680, 2.59 GB bf16) via `ROW` tokens.
- Circuit-owned 32 KB RAM with the real `in_ram` / `in_drom` decode (no bring-up patch).
- Stock `Qwen3_5` modules agree with the circuit on chunk 0 for 130,000 steps over the full 680,128-token vocabulary (2,320 ROW tokens); the same check with every table entry +1 diverges at step 41.

## 🎯 What changed since the earlier result

The earlier firmware (int4 weights, int8 activations everywhere) reproduced its emulator exactly but answered 13. Layers 2, 26 and 27 carry the position-0 massive activations; running those three layers at 8-bit weights with 16-bit activations gives 15846 (about +17% instructions per token (compiler-level count), +7.9% weight words). Details: `experiments/circuit/fullrun/diag_quant/REPORT.md`, `quantfix/REPORT.md`, `quantfix/STEP2_REPORT.md`.

## 🧑‍💻 Host roles

Removed: weight, RAM and blob injections; forced tokens; the `in_ram`/`in_drom` patch; reference-derived table window, run length and stop point; the table outside the model; the chunk-0 reference state. Allowed by the design: boot, the `GO` token, the embed-and-append loop, `ecall` I/O. The reference comparison is the test. Later chunk starts are seeded from the reference and verified by exact end-state equality at every seam. Audit: `docs/HOST_ROLE_AUDIT_CFRAM.md`.

## ⏱️ Speed and cost

6.0 min per chunk on an RTX 4090 (about $0.05), 4.9 to 9.7 min on an A100 PCIE, about 9.2 min on a Modal A100-40GB. About $10 of rented Vast.ai GPUs for chunks 0 to 31 and 71 to 95 (fleet-log estimate), Modal A100s for chunks 32 to 70.

## 📌 Scope

One prompt token at position 0 (ctx=1 firmware). Chunks ran in parallel, each seeded from the reference and checked at all 95 seams. The executor is the fused Triton kernel; the stock-module check covers chunk 0. The outer checkpoint carries the 2.59 GB table.

## 🔁 Reproduce

See the README (Reproduce) and `experiments/circuit/fullrun/cf/README_parallel.md`.

## Before publishing (checklist)

- [ ] Demo video linked from the README (`docs/demo/`), replacing the README's marked placeholder line.
- [ ] Tag `v1.0-token15846` created by the maintainer.
- [ ] Repo visibility and announcement decided by the maintainer.
- [ ] Secret scan re-run on the final commit (the scan of the working tree and all history found no key values, passwords or private addresses on 2026-11-07).
