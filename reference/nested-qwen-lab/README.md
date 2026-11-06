# Complete parallel integer tile

Run:

    OPENBLAS_NUM_THREADS=1 python parallel_complete.py /path/to/model.safetensors

Omit the checkpoint path for synthetic tests only. The recorded real-weight
run uses Qwen2.5-0.5B revision 060db6499f32faf8b98477b0a26969ef7d8b9987.

`parallel_complete.py` integrates:
- a fixed BF16 neural projection that forms complementary GDN queries;
- native Qwen3.5 recurrent GDN and actual gated RMSNorm;
- a fixed count-output projection;
- a shared compiled Qwen3.5 signed reduction model, batched over 31 outputs.

The host loads/encodes weights and activations, transfers count vectors into
reducer input frames, and decodes final result bits. It performs no dot products
or signed reduction except the independent validation assertion. This is a
composed circuit engine, not one merged pretrained language checkpoint.

Results include cold tile loading and input encoding, then compute and output
decoding. One-time circuit compilation/model construction is reported separately.
Only raw integer projection is tested: quantization scales are defined by the
host export but not applied to outputs here. Group scale arithmetic, persistent
cross-tile sums, integrated sequencing, and all other transformer operations
remain outside this milestone. Prior persistent-sum tests remain in the archive.

The real test covers a 32-input by 31-output submatrix of the first gate
projection, using per-output scales calculated over each full 896-weight row.
It does not test the entire 896x31 matrix. Mutation zeroes the query projection
and must change the output. See parallel_complete_results.json for timings,
parameter count, circuit size, and pass flags.

# Version 3 readout optimization

The streaming implementation now reads the final circuit code bits directly.
For the first two calls in each session, it checks those token IDs against the
actual vocabulary argmax on the same hidden state. All projection results also
remain checked against an independent integer dot product. The computation
still runs the actual Qwen3.5 transformer layers in CPU BF16.

On the 128-MAC synthetic benchmark, direct code readout reduced execution from
8.725 seconds to 6.329 seconds (~1.38x). This is far below the improvement needed
for practical full inner-model inference. The test is host-scheduled and uses
circuit-only outer models; it is not a full nested pretrained Qwen demonstration.

The complete 896-weight real projection row passed the earlier vocabulary-head
path in 60.900 seconds, output -883. The direct-readout repeat is recorded in
real_weight_fast_results.json. Activations are synthetic INT8 and the weight
row uses symmetric INT4 quantization. Scale application and all remaining
transformer operations are outside this integer projection test.

# Version 3: cached execution and real checkpoint experiment

Run the streaming component benchmark:

    OPENBLAS_NUM_THREADS=1 python streaming_projection.py 2
    OPENBLAS_NUM_THREADS=1 python streaming_projection.py 8

To test an actual Qwen2.5-0.5B first-layer gate projection row:

    OPENBLAS_NUM_THREADS=1 python streaming_projection.py 8 /path/to/model.safetensors

The tested checkpoint revision is 060db6499f32faf8b98477b0a26969ef7d8b9987.
Only one complete output row is tested, with synthetic signed INT8 activations
and a symmetric per-output-row INT4 quantizer. This is not a full matrix, a
calibrated quantized model, or inner language generation. Checkpoint weights
are downloaded separately and are not redistributed in this archive.

Unlike the v2 replay test, these runs use persistent Hugging Face DynamicCache
in actual Qwen3.5 CPU BF16 modules. The tile and adder are separate circuit-only
models. Host scheduling and bit/byte encoding connect them; multiplication and
cross-tile accumulation occur in those models. The only native dot product is
an independent assertion at the end of the test.

Measured synthetic throughput is roughly 6 MAC/s for two-row tiles and 15 MAC/s
for eight-row tiles on this CPU using reference PyTorch GDN kernels. Extrapolating
that implementation unchanged to 494 million projection MACs implies roughly
a year per inner token, before other transformer operations. This is evidence
AGAINST practical feasibility of the current execution path, not a general
lower bound on every possible Qwen-based accelerator. GPU throughput, parallel
columns, scale handling and complete-model correctness remain unverified.

A feasibility guarantee is not supported. Next decision gate: demonstrate
orders-of-magnitude better projection throughput before implementing all inner
transformer operations on this execution path.

# Version 2: persistent state and actual Qwen layers

New commands (install torch and transformers for the layer tests):

    python persistent.py
    python tiled_ir.py
    python check_qwen_layers.py
    python check_tile_layers.py

The persistent 32-bit accumulator stores its bytes in GDN state. All additions
are circuit operations. A host driver schedules QUERY/COMMIT and copies emitted
bytes; it does not add tile results. Thirty-two additions, cancellation, and
signed wraparound pass the IR evaluator. A 64x4 projection now performs both
tile multiplication and cross-tile addition in IR, with four accumulator sessions.

The accumulator is also compiled into actual stock Qwen3.5 transformer modules
and tested in CPU BF16. It emits the expected result tokens for four sequential
updates: 1778, 0, 3145728, 0. This is a circuit-only model with 108,515,744
parameters and ten layers. It contains no pretrained language-model weights.
The signed tile has a separate CPU BF16 layer test; see its result JSON.

This is NOT yet Qwen2.5 inference inside pretrained Qwen. Still missing:
real checkpoint quantization and loading, group-scale arithmetic, transformer
operations and KV memory, an integrated sequencer, and full inner-token validation.
The host-scheduled component tests are meaningful intermediate evidence, not an
end-to-end nested-model claim. Original CPU-only results below are historical.

# Nested Qwen experiment lab

Run:

```bash
python -m pip install numpy
OPENBLAS_NUM_THREADS=1 python experiments.py
OPENBLAS_NUM_THREADS=1 python compare_references.py
```

This package contains CPU tests, a new signed projection circuit expressed in
qwenTPU's compiler IR, and an implementation plan. It does not contain model
weights or demonstrate stock Qwen inference through another Qwen checkpoint.
The initial tests used NumPy only. The v2 tests below also run PyTorch CPU BF16. No CUDA GPU or pretrained language checkpoint was used.

## Which backend?

| Route | What was tested here | Role in the project |
|---|---|---|
| Qwenputer | Upstream RV32 traced reference, assembled signed dot-product loop | Control and correctness baseline; use its QwenPU vector extension for transformer primitives |
| QwenGPU | Upstream PTX reference, existing signed dot-product kernel | Useful independent reference; current compiled memory/lane limits make it a poor first full-model target |
| QwenTPU | Existing GDN/SwiGLU circuit IR plus new signed INT8 x INT4 tile | Best place to develop and validate projection acceleration |

The CPU and GPU runs are reference-interpreter tests. Only the TPU experiments
evaluate compiler IR nodes with modeled BF16 checks. None are physical-GPU
throughput benchmarks, and their wall times must not be compared as backend speed.

## Implemented here

`signed_tile.py` builds a generic runtime-weight-loaded, signed INT8 x INT4 tile.
One-hot GDN writes load weight bits; ACT tokens read weights and write product
bits into accumulator rows. A multi-hot READ obtains bit-plane counts.
Count bits then enter a carry-save reduction and a parallel-prefix final adder.
The result is signed and unclamped, avoiding v2's loss of partial-sum information.
The IR output is a bit vector, not yet an emitted-token ISA.

The first implementation used serial addition and needed 80 layers for 8x2.
Carry-save reduction with a final ripple adder reduced this to 18. A tested
parallel-prefix alternative reduces the 32x2 tile from 21 to 18 layers. The
default selects ripple for 8 rows and prefix otherwise. Measured compiler
statistics for the selected variants are in `results.json`.
These layer counts use the existing compiler's target capacities; they do not
include model merge overhead, checkpoint padding, or a token readout head.

Tests include:

- 512 upstream unsigned/clamped readouts across four configurations.
- 480 signed readouts over 2,4,8,16,32,64-row tiles, random and extreme values.
- All 4096 signed INT8 x INT4 scalar pairs through the circuit IR.
- A 23-bit adder gadget, including cancellation and large partial sums.
- A mutation test which deletes reduction gates and must break the answer.
- Cancellation showing why clamping individual K tiles is invalid.
- A synthetic full 1024-to-3072 projection using a separate NumPy bit-plane
  oracle. This full projection is NOT run through the new circuit IR.
- 16 dot-product cases across the RV32 and PTX reference interpreters.

The upstream evaluator asserts its modeled BF16 exactness conditions. Passing
it is necessary evidence, but cannot establish behavior on actual hardware.
Earlier projects found hardware-specific failures despite mathematical proofs.

## First stock model: Qwen2.5-0.5B, then Qwen3-0.6B

Calculated from official architecture configs, at batch one and short context:

| Requirement | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---:|---:|
| Dense projection MACs/output token, including LM head | 493,961,216 | 595,984,384 |
| Packed INT4 matrix codes, before scales | 246,980,608 bytes | 297,992,192 bytes |
| BF16 KV bytes at context 128 | 1,572,864 | 14,680,064 |
| Q/K per-head normalization | No | Yes |

The 0.5B target has about 17% fewer dense MACs and 9.3x less KV state in this
comparison. It also avoids Qwen3's additional Q/K normalization. Its QKV biases
must be implemented: a Llama guest is not automatically a Qwen guest.
Its FFN reduction dimension is 4864, so allow at least signed 24-bit INT8 x
INT4 accumulation (32-bit limbs are a sensible common implementation).

## The bottleneck is loading, not just multiplication

An optimistic hypothetical 32x32 protocol uses 1024 one-weight load tokens,
32 activation tokens, and 64 READ/feedback tokens per tile. Reloading weights
per output token gives 540,270,080 trace tokens for 0.5B or 651,857,920 for 0.6B.
At an ASSUMED 100k trace tokens/sec, the latter takes 109 minutes. This is an
analytical protocol estimate, not measured performance. Scales, control, limb
arithmetic, attention, and nonlinearities add work.

The planner includes hypothetical weight packing sweeps, but packing eight
weights into one token is not automatically implementable: a Cartesian token
vocabulary grows exponentially. Prefer bulk state initialization/paging or
parallel write ports with an explicit data path. Measure their costs.

Packed INT4 bytes are not GDN-state memory requirements. Storing four bits as
four BF16 scalars alone expands code storage 16x; ballast, head padding, and
FP32 recurrent state where used add further overhead. A resident full-model
design needs measured state and checkpoint sizes before claiming it fits.

## Concrete implementation sequence

1. **Emit this signed tile into actual Qwen layers.** Add compact readout tokens
   for result limbs. Compare IR, whole-model, and chip-only paths; test A100/L4
   BF16 and mutate weights to ensure outputs depend on the arithmetic.
2. **Persistent accumulation.** Store signed 32-bit limbs in GDN state, implement
   carry propagation, and preserve them across K tiles. The current 23-bit
   gadget and host reference sum do NOT implement this persistent machine.
3. **Group scaling.** Accumulate within quantization groups, apply each group's
   scale before combining groups, then requantize at defined boundaries.
   Do not sum differently scaled integer groups and apply one global scale.
4. **Real projection.** Export a real checkpoint matrix with a pinned quantizer.
   Validate output against that quantized oracle. Separately measure cold load,
   warm execution, recurring paging, state memory, and autonomous throughput.
5. **One Qwen2.5 block.** Adapt QwenPU vector primitives for RMSNorm, QKV biases,
   RoPE, grouped-query attention, softmax, SiLU, residuals, and scale handling.
6. **All blocks and LM head.** Generate one verified greedy token, then eight
   tokens with a 32-position cache. Keep the full vocabulary projection and
   argmax inside the compiled computation. Tokenizer/readback can be external.
7. **Scale and optimize.** Only after end-to-end validation, widen/bank the
   projection engine and target Qwen3-0.6B with its Q/K normalization.

All reference arithmetic is validation only. Host GEMM, attention, normalization,
or sampling must not be hidden inside an alleged in-model inference backend.
A driver may transport weights/data and run argmax/append for chip trace tokens;
the inner language model's vocabulary argmax belongs to the compiled machine.

## Files and provenance

`qwentpu/` contains unmodified minimal upstream compiler files and baseline
circuit, under the included upstream MIT license. New code is in `signed_tile.py`,
`experiments.py`, and `compare_references.py`. RV32/PTX references are vendored
under `cpu_reference/` and `gpu_reference/`, with their licenses.
See `sources.json` for pinned revisions and `results.json`,
`reference_results.json`, and logs for executed evidence.

Architecture configs:
- https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json

Relevant vector-unit implementation: https://github.com/divagr18/QwenPU
