# Parallel GDN projection engine

Decision: replace scalar instruction/token execution with vector frames and
banked GDN bit-plane matrix reads. Keep the CPU only for control. Use QwenTPU's
compiled integer reduction machinery for signed output formation.

## Tested change

`parallel_gdn.py` runs the actual Transformers Qwen3.5 recurrent gated-delta
primitive with BF16 inputs and models the following BF16 RMSNorm boundary.
A 32x31 INT4 weight tile is stored as four bits per output column, plus two
ballast columns, in GDN state. Eight heads query the eight INT8 activation bits.
Each query chooses one of two rows per input: a real weight row when the
activation bit is 1, or a zero-data dummy row when it is 0. Both carry ballast.
Exactly 32 rows are selected regardless of input, fixing query normalization
and the norm denominator. This avoids input-dependent renormalization errors.

One read returns 8x31x4 small integer counts. All 672 output dot products across
random and extreme tests matched the signed reference after reconstruction.
The measured native-read-plus-norm rate was about 2852 calls/s on this CPU.
Each read represents 992 logical MACs, but 2.83M logical MAC/s is ONLY an
equivalent primitive rate: reconstruction, query formation, weight loading,
scaling, accumulation, and full model layers are excluded. It must not be
compared directly to v3's end-to-end measured projection throughput.

## Execution contract

1. Load packed weight tiles from external storage/VRAM into banked recurrent
   state. The host may transport and unpack bit representations. It must not
   compute inner-model products, sums, nonlinearities, or attention.
2. Keep activation vectors as bits/limbs in recurrent state or residual vector
   frames. Form complementary queries inside compiled projections.
3. Read all output-column bit-plane counts in parallel through native GDN.
4. Compile signed reconstruction using carry-save trees and prefix adders;
   retain 32-bit partial sums in state across K tiles.
5. Apply quantization-group scales at the correct group boundary, then write
   the completed output vector directly into the next operation's state.
6. Emit ordinary language tokens only at the end of inner-model decoding.

This changes the interface: vectors travel as state/vector frames, not one
sampled token per scalar. Bulk cache initialization is a deliberate driver
capability, not a property of unmodified generate(). If all state updates must
instead be serialized through vocabulary tokens, the old loading bottleneck
returns and this design loses much of its benefit.

## Memory and paging

Current state is 8x128x128 FP32 = 524288 bytes per tile, versus 496 bytes for
the packed INT4 codes. Replicating all model weights this way is impractical:
roughly 243 GiB for Qwen2.5-0.5B's projection weights, before other state.
Keep only a small number of tiles resident and double-buffer their loading.
Benchmark expansion, transfer, and compute together. Loading 243 GiB of expanded
state per token could dominate execution even if raw input codes are small.

Two alternatives to test:
- Shared read-only state for the eight bit queries, or eight serial reads of
  one state bank. This trades parallelism for up to 8x lower resident state.
- More compact ballast/data packing or direct low-precision weight state,
  provided actual Qwen normalization remains correct. Do not assume arbitrary
  signed matrix data survives the existing normalization unchanged.

## Next decisive benchmark

Compile vector-to-query formation and signed count reduction around this
primitive; run a complete 32x31 projection in actual Qwen layers. Compare
against independent INT8xINT4 arithmetic, including negative/overflow cases.
Then run a real 896x31 Qwen2.5 weight block with group scales and state paging.
Report cold load, recurrent-state bytes, warm execution, whole-block latency,
and numerical error separately. Only then extrapolate to an inner token.

## Full-model work still required

RMSNorm, QKV bias, RoPE, attention/softmax, KV storage, SiLU/gating, residuals,
weight scales, embedding lookup, final vocabulary projection and argmax, and
an integrated sequencer. The primitive experiment does not establish any of
these or demonstrate a pretrained Qwen model running inside another one.

Target remains Qwen2.5-0.5B first. A stronger outer language model is not the
main lever; the accelerator's parallelism and state traffic are.
