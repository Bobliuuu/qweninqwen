# trueqiq_alt (agent B2) FINDINGS  (newest first)

## F3 RANKING + RECOMMENDATION (A100 fp32, real/random Qwen2.5-0.5B-sized ops, max|err|/max|ref| vs fp64)
| rank | candidate | exactness | MAC/outer step | steps/s | MAC/s measured | s per inner token (494M) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | (b) SwiGLU odd-silu multiplier, weights streamed raw, reduction in down_proj (F1; bare layer) | 5e-7..2e-6 (= torch fp32) | 896 (R*Ct, tunable) | 1.0e5-1.5e5 (T=4k-16k/forward) | 0.43-0.77 G (MLP alone 1.5 G) | 0.64-1.14 (0.33 MLP-only) | WINNER for the multiply-accumulate itself |
| 2 | (b)+(a-hard) full op incl. attention x-broadcast / K-tile sum / gather (F2, untuned glue) | 4e-7..1e-6 | 896-1024 | 4e4 | 36-49 M (6.8 M for 4864-wide gather) | ~10-14 est | exact end-to-end; glue-bound, see tuning below |
| 3 | (c) GDN chunked write stream (stock layer, torch reference kernel; B owns exactness) | n/a here | 2048 floats written/step/layer (hv=16) | 1.3e5 (T=8k) | ~0.27 G floats/s/layer load rate (+ read) | ~1.8 floats-load alone for 494M with 1 layer | same order as (b); B's state engine has no 1/I dense waste but needs paging+norm hacks |
| 4 | (a-linear) KV-as-weights, near-uniform softmax, odd pairs | best 8e-5 (eps=3e-3), 2e-4..2e-2 elsewhere | K*O/step | - | - | - | KILLED: ~100x worse than (b); normaliser x-dependent, fp32 cancellation floor |
| 5 | bit-circuit (ref) | exact | - | - | ~500 | ~1e6 s | too slow |
(a-hard) hard-select attention is NOT a matvec engine but IS an exact addressable copy/gather/sum (used in F2): keep as glue.
Single-unit variant (gate=silu^-1(w+s), one unit per MAC + shared -s*sum x correction) would halve dense flops (~2x on F1); NOT measured.
Tuning projected (not measured) for F2: (i) gather per x-tile (several query steps each returning one Ct'-wide tile; heads = Ct'/R instead of O/R: round-B q_proj
 dominates GPU time: 11 of 16.5 ms are the T-head q_proj mm) -> round B ~4 ms; next op's L0 stores these tile positions directly as its x-entries, so no single-frame vector is needed;
 (ii) batch q|k|v and gate|up into single ops (shared x), 96 instead of 168 ops/token; (iii) T=thousands of steps/forward amortises launch cost. Realistic target 2-4 s/token.
RECOMMENDATION to integration owner / agent B: if B's GDN-state matvec is exact but slower or needs ballast/norm hacks, adopt the MLP multiplier (mlpmul.py) as the multiply unit with
 attention-code glue (glue.py); it needs no paging of state, host streams raw fp32 weights (zero reformatting), error = torch fp32. Layer budget: 2 layers per op (L0=attn+MLP, L1=attn),
 chained so op n+1's L0 is deeper than op n's L1; every op = 3 rounds (store x, stream weights, gather). Hybrid is natural: B for registers/KV state, (b) for big matvecs.
Caveats: x must be present in the frame of its own earlier round (as in INTERFACE sec.5); RoPE: all key/code dims sit in the non-rotary tail (hd*0.75), values unrotated; position_ids supplied.

## F2 [measured] COMPLETE exact y=Wx on stock Qwen3_5 layers: attention does x-broadcast, K-tile reduce and gather; host streams ONLY weights  (glue.py, t_glue.py, bench_glue.py)
2 stock full_attention layers, only weights compiled (model dims: heads=T=O/R, kv=1, hd~C+1).  Rounds (separate forward calls with DynamicCache):
  A: nt frames carry x tiles (stands for an earlier-computed activation; in integration it sits in the residual of that step) -> L0 KV entries (key = +-1 binary code of tile a, valid=+).
  B: T*nt weight steps (step = R rows x Ct cols raw weights + code(a) + code(row-group t)).  L0 attention: query = code(a) (read from a frame field), weight-step keys carry valid=-1;
     q/k RMSNorm gains chosen so a 1-bit code mismatch costs >=40 in score => softmax weight exactly 1.0/0.0 => EXACT copy of x_a (v/o_proj are 0/1 matrices, gate sigmoid(30)=1-1e-13).
     Same layer's SwiGLU multiplier (F1) -> partial dots y_a[R] in slots Y.  L1 stores v=Y with key=code(t) (the SAME code for all nt tiles of row group t).
  C: one query step; L1 head t attends equally (equal scores => softmax 1/nt each, exact) to its nt partials -> mean; o_proj gain nt => exact K-tile REDUCTION + gather into X2[O].
  => attention is an exact reducer/gatherer/broadcaster (hard select by +-1 code; equal-score sum).  Everything numerically exact bar fp32 rounding.
Measured vs fp64 (random + REAL layer3 q_proj weights on Mac CPU: 5.7e-7 of max):  A100 fp32 (TF32 off), keyed on max|err|/max|ref|:
  q_proj 896x896  Ct=64  R=14 nt=14 (896 steps): err 6.4e-7, 22 ms/op = 36 MMAC/s ;  Ct=64 R=16 : 6.5e-7, 20 ms = 40 MMAC/s
  down 896x4864   Ct=64  R=16 nt=76 (4256 steps): err 9.2e-7, 89 ms = 49 MMAC/s
  gate 4864x896   Ct=64  R=16 nt=14 (4256 steps): err 9.9e-7, 644 ms = 6.8 MMAC/s  (D=5988 because the gather target X2 has O slots, and T=304 heads: DENSE-waste; fix below)
=> NOTE end-to-end per-op throughput is 40-50 MMAC/s, ~25x below the bare multiplier (F1: 0.4-0.8 GMAC/s): the glue (3 forward calls, attention over all entries with T heads
   in BOTH layers, D inflated by X2) dominates. Whole 494M-weight inner token ~ 10-14 s if ops are batched (qkv as one op, gate+up as one op) with this untuned glue;
   untuned: only 1 of T heads in L0 is useful, X2 can alias X, L0/L1 could be split into sparser configs. Expected floor ~2-4 s/token (F1 0.6-1.1 s + per-round latency of ~100-170 ops x 3 rounds).
Depth cost: each op consumes 2 layers (+ its rounds) because x'(gather output, depth 2) can only be consumed by layers deeper than L1 (INTERFACE sec.5). ~100-170 ops/token => 200-340 outer layers or
   per-op layer reuse is impossible in a stock model (separate weights); parameters: q_proj of the gather layer D x 2*T*hd (~10^8 for 896-wide ops) per op.

## F1 [measured] (b) SwiGLU MLP as STREAMED-WEIGHT MULTIPLIER: exact, ~1e9 MAC/s on A100 fp32  -> 0.6-1.1 s per inner token of MACs
Code: trueqiq_alt/mlpmul.py (MulMLP), t_mlp.py (real-weight exactness), bench_mlp*.py. Stock Qwen3_5 layer, only weights compiled.
Trick (exact, no inverse/shift/approximation): odd part of silu is exactly z/2:  silu(z)-silu(-z) = z.
 Unit pair for each MAC (w_rj, x_j):  gate=+w, up=+x  and  gate=-w, up=-x  =>  silu(w)x + silu(-w)(-x) = w*x  EXACTLY (even part cancels).
 gate_proj/up_proj are 0/+-1 selection matrices (up selects the x slot, gate selects the streamed weight slot), down_proj sums the 2C units of
 output row r into slot Y[r].  So y_r = sum_j W[r,j] x_j is ONE outer step (position) for C columns: reduction inside the step, no cross-step state.
 Frame = [BALLAST | X(C) | W(R*C) | Y(R)], intermediate I = 2*R*C.  Host streams only W (raw fp32 weights, no reformatting!) + BALLAST; x must
 be present in the frame (integration: copied into every weight-step by an attention/state read, see F2 TODO).
Exactness (stock fp32 forward incl. transparent RMSNorms, BALLAST=1e6, x with a 300 outlier, REAL Qwen2.5-0.5B weights):
  layer3 q_proj C=896: max |err|/|ref|max = 1.7e-6 (max elementwise rel 1e-5 on near-zero outputs) ; torch fp32 matmul vs fp64 itself: 3e-6 rel.
  layer3 down_proj C=4864: 2.7e-6 of max (6e-5 worst rel on tiny outputs); torch fp32 itself 1.8e-5.  I.e. same as torch fp32 matmul. Random: 5e-7..1.4e-6.
Throughput, A100 (shared, fp32, TF32 OFF), one full stock decoder layer (full_attention zero-weights + this MLP), T positions per forward (chunked prefill):
  C=896 R=1  (896 MAC/step)   T=4096: 0.43 GMAC/s   -> 1.14 s per 494M weights
  C=224 R=4  (896)            T=4096: 0.61 GMAC/s   -> 0.82 s
  C=128 R=7  (896)            T=8192: 0.71 GMAC/s   -> 0.69 s
  C=64  R=14 (896)            T=16384: 0.77 GMAC/s  -> 0.64 s      (MLP matmuls alone: 1.5 GMAC/s, 17 TFLOP/s ~ fp32 peak; 0.33 s/token)
  (a GDN layer with torch-reference chunk kernel as the host layer costs 5x more per forward: use a full_attention layer or install flash-linear-attention)
  => compute-bound by the DENSE gate/up (D x 2*Rc) matmuls: flops/MAC = 12*I/... ~ 8*C*(R+..), so SMALLER C (tile K) and larger R is faster.
  Rank vs bit-circuit (~500 MAC/s): ~1e6x. vs B's GDN state engine: B to report its own number; this one needs NO state paging at all.
Open items (next): x-broadcast + result-gather glue via attention (F2), single-unit variant (silu^-1) ~2x fewer flops, GDN chunked/state variant, KV-as-weights.
