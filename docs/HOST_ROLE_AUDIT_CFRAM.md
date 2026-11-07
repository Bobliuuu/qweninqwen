# Host-role audit of the CF+RAM chained run (strict, 2026-11-06)

Scope: `experiments/circuit/fullrun/circuit_run.py` (399 lines, read in full), `cf/modal_cfram_chain.py`, `cf/fw_qwen3_ram_cf.c`, `chipformer/rv/fusedexec.py`, reference `/data/cfram/refRC` (75 chunks, 0-74, verified complete; state_0075 = final state). Read-only; nothing launched. Line numbers are `circuit_run.py` at commit `4937235` unless stated. Tags: **[R]** read from code, **[M]** measured (a file), **[?]** not verified.

Verdict: **this is not yet Qwen in Qwen.** Chunks 0-2 of the per-chunk CF+RAM replay have no weight, RAM or blob injections [M, see 1.1]. Everything below the "computed by the circuit" line is still supplied or decided by the host/harness. Agents A-E cover most of it. **Agents F and G (sampled HF check, final token vs real Qwen) were never started**: only five forks (A seams, B chunk-0 boot, C table-in-checkpoint, D in_drom/in_ram patch, E 6-bit) exist. Roles 6, 7, 8, 9, 10 below have no owner.

## 0. What the circuit does compute (the part that is real)
Every token the free-run feeds back is the circuit's own prediction (`fr.tok`) unless `MASK=1`; with `--kinds ''` (the CF launcher passes none) `MASK` is all zero [R: 59, 182, 250], so no token is overridden. ALU/MUL/DIV/branch/PC/register write-back, load merge, VDOT/SMAQA arithmetic, the ROW-token weight reads, softmax/RMSNorm/SwiGLU as firmware instructions, and the emitted trace tokens are the circuit's. 0 mismatches over a chunk means the circuit's tokens equal the reference trace.

## 1. Host/harness inputs, role by role

| # | Role | Where | Used in CF+RAM chunks 0-2? | Owner | Residual after owner |
|---|---|---|---|---|---|
| 1 | Type-1 RAM injections (`_inject`, `_put`) | 62-105, 211-218 | **No**: `refRC/inj_0000`, `inj_0001` are empty (0 rows) [M]. 73 inj files exist; only chunks 0-1 inspected here [?] | none needed | Verify all 73 inj files are empty before the final run (cheap) |
| 2 | Type-3 DROM/blob injections | 218-221, `lut_drom` 205-207 | No (same empty files). CF firmware reads nothing at 0x10000000 [R: fw line 6] | none | same check as 1 |
| 3 | Type-2 VDOT weight injections and 128-word zone forcing | 222-231 | No: `if WF: a.novdot = True` (57) and asserted absent (222) | done | none |
| 4 | Forced tokens (`MASK`, `_override`) | 61-72, 182, 249-253 | No (`forced_kinds` empty), but the machinery stays in the graph | none | Keep `--kinds ''` and report `forced_tokens=0` in the final JSON |
| 5 | **Chunk-start snapshot**: regs `x`, pc `anchor`, `n`, `wb` (15 bit), latch window, RAM image (`ramimg_NNNN.npy`) | 158-171, 172 | **Yes**, every chunk in a per-chunk run; in a chain only the first chunk of each segment | **A** (own-state resume across segments), **B** (chunk 0 from program) | Until A and B are validated on a run, each segment seam is host-seeded. `bn` is not seeded from the reference at all (only on RESUME, 169): it takes the program default [R] |
| 6 | **Look-ahead from the reference trace**: `toks`/`kinds` sizing (`N`), the ROW-table window `[_lo,_hi]` chosen from the reference's ROW tokens (143-148), run length, chunk list | 114-120, 143-148 | **Yes** | **none** | The harness decides which table rows exist and when to stop from the reference. Outside the window `SliceTab` returns zeros (140), which would show up as a mismatch, not hide one, but it is still look-ahead. A final run should materialise the full table and stop on the circuit's own `exit` ecall |
| 7 | **Table lives in a harness LUT, not model parameters**: `S.ROW_TABLE = SliceTab(np.load(rowtab_cf.npy))`, rows pushed through `ex.embed(...)` into `FreeRun`'s `table` by an `exec`-patched copy of `fastexec.py` (149-153) | 139-153 | **Yes** | **C** builds the merged checkpoint and a HF check (WF_NROWS=156396 commit `05889d1`) | C's check is HF modules on a sample (chunk 0); the run itself still reads a harness LUT. See role 9 |
| 8 | **Bring-up patches to the circuit** | 22-30 | REALMAP: `in_drom` rewritten to `1 - in_ram` (28), `in_ram` stays real. Legacy path `in_ram := one` (30) is not used with REALMAP | **D** (commit `4937235`, "real in_drom decode launcher", CF has 0 blob loads) | Not yet validated on a GPU run. The monkey-patched `core_bringup` module is still what runs until D is merged into the launcher (`modal_cfram_chain.py:23` sets `REALMAP=1`, so it still uses the patched core) |
| 9 | **Executor is not the HF forward**: `fusedexec.FreeRun3` (Triton), token decode is a per-family argmax plus payload-bit threshold (`_decode_read`, `fusedexec.py:254-267`), candidate set `fr2.ids`, not a full-vocab logit argmax; the embedding is a precomputed LUT (`_decode_embed`, 274) | `fusedexec.py` | **Yes**, always | **C** partly (HF check, one chunk, 130k steps on WF) | **No agent owns the equivalence for the final run.** Needs agent F: HF-module check on sampled windows across all 75 chunks incl. ROW tokens, plus a negative control |
| 10 | **Verification oracle**: pass/fail = circuit token vs `TREF` from the reference (`RAW != TREF`, 261-270, 338-345) | 338-345 | Yes | none | Cannot be removed; it is the test, not an input. But "correct" is defined as "equals the Python emulator `ref.Machine`", not "equals Qwen3-0.6B". **No agent owns agent G: final token vs HF Qwen3-0.6B/bf16 argmax** (the audit's own "exactness bar" says report this separately). `argmax 13` comes from the emulator [M: `logs/wf/ref_summary.json`] |
| 11 | Firmware/ROM/DROM contents loaded into the circuit tables (`elf.program`, `core.snapshot`) | 154-157, 172 | Yes | none | Boot. Allowed by plan ("boot the CPU: load firmware and data into state") |
| 12 | First token `GO`, embedding, autoregressive loop, CUDA-graph replay | 173, 257, 264-266 | Yes | none | Allowed by plan (transport items). Cannot be zero |
| 13 | Boundary readback `extract()` and `compare()` | 284-336 | Yes (chain mode) | none | Read-only unless `--handoff`; not fed back. `--save-state` writes it for resume (A) |
| 14 | WB27: upper 12 bits of the 27-bit weight pointer | 31-37 | Not set (`modal_cfram_chain.py:23` has no `WB27`), compare only checks 15 bits (`wb15_ok`) | none | Low risk: weights come from ROW rows in WF/CF (row id is a firmware register). Say so in the final report |
| 15 | Table size/packing: 4-bit entries, bf16 table 2.38 GB at hidden 7680 | build | Yes | **E** (6-bit, optional), user waived the size | none |
| 16 | **Scope of the inner run**: firmware is ctx=1, compile-time single prompt (`#error "CF port: ctx 1 only"`, `PROMPT` array, `cf_get(x, CF_HEAD_ROW, CFO_EMB, DIM)`) | `fw_qwen3_ram_cf.c:289, 359-361` | Yes | none | The inner token id is baked in at compile time, attention runs over one position (no KV use). This is a scope limit of the claim ("one token of Qwen3-0.6B at position 0"), not a host role. State it |
| 17 | Halting/output: `exit`/`write` ecalls are serviced by the harness (`MARK()` ecall, `wr`) | fw 11-20 | Yes | none | Allowed as transport; the answer token is read from the trace |

## 2. Not covered by any agent (gaps)
1. **Role 6**: reference-derived table window, run length, stop condition. Small change: materialise the full table, stop on circuit `exit`.
2. **Role 9 / F**: executor-vs-HF equivalence across the whole run (only chunk 0 of the WF run was checked).
3. **Role 10 / G**: final token against real Qwen3-0.6B (bf16 HF), not just the emulator.
4. **Role 1/2 hygiene**: confirm all 73 `inj_*.npy` of `refRC` are empty (only 0 and 1 checked).
5. **Role 8 integration**: D's real-decode launcher is not what `modal_cfram_chain.py` runs; someone must switch it and re-run the chunk 0-2 smoke test.
6. The smoke test (chain 0-2, app `ap-TXUqekl6gResdrpJFP4d4c`) tests chaining with `REALMAP=1` and the patched core; it says nothing about A, B, C or D.

## 3. What cannot be removed, and why
- **Boot, `GO` token, embedding of the circuit's own token, the autoregressive replay loop, ecall I/O** (roles 11, 12, 17): something outside the weights has to start and step a transformer. The plan allows exactly these.
- **The reference comparison** (role 10): a run cannot certify itself. It stays as the check. What can change is what it is compared against (add G).
- **Running the full 174.8M-step run through HF modules** for role 9: the HF modules do 59-91 tok/s on an L4 (`CIRCUIT_ROUTE_PLAN.md`), so 174.8M steps is about 22-34 days on one GPU. Impractical; a sampled check with a negative control is the achievable form. "Same arithmetic, not same kernels" stays true for the bulk of the run.
- **The table in the outer parameters** (roles 7, 15): nothing at the released Qwen3.5-4B size can hold 596 M int4 weights (state is 24x too small). The 2.38 GB embedding table is a parameter-count increase and is ruled acceptable by the user; it is not a stock Qwen3.5-4B.
- **A zero-seam single chain**: possible but about 11-13 h on one GPU, limit 24 h. Feasible, not impossible; A addresses it.

## 4. Reading the final JSON
A claim of "Qwen in Qwen, one token" requires, together: `injections=0`, `forced_tokens=0`, `free_mismatches=0` over all 75 chunks, one unbroken chain from a program-only start (A+B), full table materialised and halting on the circuit's own exit (role 6), the table-in-checkpoint check on the CF firmware (C), D's real decode, the sampled HF check with negative control (F), and the final token equal to real Qwen3-0.6B (G). Today A-D are in flight, F and G do not exist, and role 6 has no owner.
