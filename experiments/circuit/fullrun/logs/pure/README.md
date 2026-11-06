PURE free-run (circuit_run_pure.py, modal_pure.py), 2026-11-06, A100 on Modal. Zero host writes, zero token overrides, zero injections applied (asserted in result json).
wf_c012: WF firmware, chunks 0,1,2 -> first free mismatch at steps 35 / 11 / 31 = exactly the first type-1 (RAM/blob) injection step of each chunk's reference schedule (0 type-2 weight injections exist in WF).
realmap_c012: remapped-RAM firmware, chunks 0,1,2 (--ramimg step-0 RAM image) -> first free mismatch at steps 32 / 0 / 26 = first type-3 (blob) / type-2 (VDOT weight) injection step of each chunk.
Before the first needed injection the circuit free-runs bit-exactly; the first host-served word (blob/RAM in WF, blob/weights in REALMAP) is where it diverges; the run was stopped after 100,000 steps (all rows diverged).
