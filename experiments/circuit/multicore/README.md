# Multi-core lanes inside one forward pass

Question: can N CPU lanes (own registers/RAM in different DeltaNet heads, shared read-only weights) run slices of a matvec inside the same trace-token step, with a join?

Files: `lanes_sim.py` (N lockstep lanes = N real Qwenputer `ref.Machine`, exact vs numpy), `model.py` (capacity + projection), `res_tiny.json`, `res_model.json`.

## Findings
1. **Exactness [M]**: row-split int8 matvec (16 rows x 64, plain RV32IM, 7.1 instr/MAC) on N=1,2,4,8,16 lanes equals numpy; lanes stay PC-lockstep (SPMD), so fetch/decode/PC can be shared.
2. **The limit is the token bus, not compute [M]**. A joint step must leave the model as ONE token of the stock vocabulary (248,320 rows = 17.92 bits). State writes can only use values from tokens or dims computed *below* the writing layer, so every lane-divergent result (mul product, accumulator) must go through the token. Joint token cost per step = ceil(bits/17.92). Speedup of the matvec phase vs the current 1-lane format (1.2 tok/instr): N=2 1.50x, N=4 1.94x, N=8 2.3x, N=16 2.5x (asymptote ~2.9x). If the bus were unbounded (needs an N-way tuple vocabulary, impossible) it would be 1.95/3.8/7.5/14.7x. Of that, 1.55x is just tighter token coding available to a single lane; the pure lane gain is <=1.7x.
3. **Capacity [E]** (`model.py:capacity`): 4B (32 layers, 24 DeltaNet, MLP 9216, core uses 8,149/9,216 units per datapath sublayer) cannot host two lanes in the same layers; lanes stack in depth: control L1-L4 shared, datapath L5-L11 per lane => 4 SIMT lanes (or 2 independent cores). 0.8B (24 layers, 18 DeltaNet, MLP 3584, 1 value head per key head while the core needs 2): 1 lane at best; even one core needs a re-schedule (~2.3x MLP sublayers).
4. **Executor cost [E]**: stock HF layers run sequentially, so N lanes in depth cost (4+7N)/11 x step time (N=2: 1.64x slower) and the net is a LOSS; only a fused executor that runs independent lane sublayers concurrently keeps the step time flat.
5. **Projection (Qwen3-0.6B, 596M MACs, compiler-measured instr/MAC, 1e8 nonlinear instr unsplit, 5,562 tok/s L4)**: VDOT int4 build 1 lane 12.3 h/token; 4 lanes 8.7 h (divergent bits d=7 per lane-instr, optimistic), 10.1 h (d=12), 12.4 h (d=20, no gain); 16 lanes 8.3/9.6/12.0 h. Plain int8: 256 h -> 113 h (4 lanes) -> 96 h (16). Fork/join (all-gather of 344k int8 outputs, 2 per token) adds only 1.7e5 tokens (~30 s), negligible; the bottleneck after ~4 lanes is the bus and the serial nonlinear work, not fork/join.
6. **Better direction**: reduce lane-divergent bits per MAC instead of adding lanes (in-circuit width = vector unit / VDOT, fused row-dot with one result token), since bus bits/MAC is what bounds everything.

Not done: no lane-replicated circuit was compiled and nothing was measured on a GPU box; all GPU-side numbers use the upstream 5,562 tok/s and are projections.
