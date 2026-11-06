# Weight memory for Qwen3-0.6B inside Qwen3.5 (0.8B / 4B)

Tags: **[M]** measured or read from a config/source, **[E]** estimate with the derivation shown, **[?]** unknown.
Code: `geometry.py` (all capacity arithmetic), `rom_mlp.py` (ROM experiment, results in `rom_mlp.log`, `rom_mlp_results.json`).

## 1. What has to be stored

Qwen3-0.6B (28 layers, H=1024, 16 q heads, 8 kv heads x 128, MLP 3072, vocab 151,936, tied head) [M, HF config]:

| Item | Count |
|---|---|
| per layer: q,o 2 x 2.10 M, k,v 2 x 1.05 M, MLP 3 x 3.15 M | 15.73 M |
| 28 layers | 440.4 M |
| tied embedding / LM head | 155.6 M |
| **total weights** | **596.0 M = 298 MB int4 = 596 MB int8** |
| KV cache per context token (28 x 8 x 128 x K,V) | 57,344 values = 56 KB int8 (T=128: 7.3 MB; T=1024: 59 MB) |
| activations / scratch (H=1024, MLP 3072, logits are streamed to argmax) | under 32 KB, fits Qwenputer's 32 KB RAM |

## 2. What the outer model can hold (stock Qwen3.5 geometry)

| | 0.8B | 4B |
|---|---|---|
| layers (DeltaNet / full attention) [M] | 24 (18 / 6) | 32 (24 / 8) |
| hidden H, MLP I | 1024, 3584 | 2560, 9216 |
| DeltaNet key / value heads, dk x dv | 16 / 16, 128 x 128 | 16 / 32, 128 x 128 |
| value heads in total | 288 | 768 |
| **DeltaNet state entries** (fp32 matrices) | **4.72 M (18.9 MB as fp32)** | **12.58 M (50.3 MB)** |
| one-hot addressable (key head, row) pairs | 36,864 | 49,152 |
| full-attn KV per context position (K+V, bf16) | 6,144 values = 12 KB | 16,384 values = 32 KB |
| MLP entries per layer (I x H) | 3.67 M | 23.6 M |
| tied embedding entries (248,320 rows) | 254 M | 636 M |

DeltaNet state capacity depends on bits per entry:

| Bits stored per state entry | 0.8B | short of 298 MB | 4B | short of 298 MB |
|---|---|---|---|---|
| 1 (bit per column: how Qwenputer's RAM/ROM and the store use it) | 0.59 MB | 505x | 1.57 MB | 189x |
| 4 (nibble per entry) | 2.4 MB | 126x | 6.3 MB | 47x |
| 8 (a bf16 integer; the ceiling, since writes pass through bf16) | 4.7 MB | 63x | 12.6 MB | 24x |
| realised Qwenputer density (ROM 128 KB in 32 key x 2 value heads = 0.125 B/entry; store 0.06 B/entry) | 0.3-0.6 MB | 500-1000x | 0.7-1.6 MB | 190-420x |

So **the stock-size DeltaNet state cannot hold the weights at any density**; the best case is 24x short. It can hold the
inner model's *KV cache* only for short contexts: 82 tokens (0.8B) or 219 (4B) at 8 bits/entry, 10-30 tokens at realised density.
Adding heads is a config change (store v1 goes 104 to 204 heads), but each key head costs about 2 MB of dense q/k/v
projection weights; holding 298 MB needs 18 K value heads at 8 bits (36 GB checkpoint) or 146 K at 1 bit.

## 3. The two channels

Everything below follows from one fact about what can write into the outer model:

- **Token channel.** One trace position carries one token id = log2(248,320) = 17.9 bits = at most 4 int4 weights. Anything that reaches state or KV *from tokens* is bounded by 18 bits per position. Loading 298 MB this way is 134 M positions; the KV cache for them is 4.4 TB (4B) or 1.6 TB (0.8B).
- **Parameter channel.** A position's MLP can read a whole H-wide word out of its parameters in one layer. Measured below: 8 bits per entry, 2,560 entries per layer per position at 4B, 100 % exact.

## 4. Experiment: ROM in an MLP down_proj (option d), measured

`rom_mlp.py` builds a stock-shaped Qwen3.5 MLP sublayer (RMSNorm in fp32 with the (1+w) gain, SwiGLU, down_proj; bf16 as HF runs it).
The address sits in the residual as dual-rail bits (so the norm statistic is constant) plus constant dims chosen so that mean(x^2) = 1 (norm is the identity up to eps).
Gate row j gives silu(+16) iff the address is j and at most silu(-16) otherwise; down_proj column j holds table row j. Run on the A100, all I addresses, random tables:

| Geometry | Entries | Bits per entry | Exact entries (round to nearest) | Max abs error |
|---|---|---|---|---|
| 0.8B, H=1024, I=3584 | 3.67 M | 1 / 4 / 6 / 7 / 8 | 100 % at every width | 1e-6 / 2e-5 / 7e-5 / 1.4e-4 / 2.8e-4 |
| 4B, H=2560, I=9216 | 23.6 M | 1 / 4 / 6 / 7 / 8 | 100 % at every width | 1.6e-6 / 2e-5 / 8e-5 / 1.6e-4 / 3.1e-4 |

So one MLP sublayer reads an **entire 1,024-wide (0.8B) or 2,560-wide (4B) word of 8-bit entries in one position, with no extra trace token for the read**, exactly.
(The ceiling of 8 bits is bf16's mantissa: 9-bit integers above 256 are not representable.)
A first version without dual-rail/exact-norm failed at 4B (rms wobble with popcount, and bf16 rounding of (1+w) near 0), which is why these two details matter.
What this does not yet test: the downstream multiply-accumulate on the read word, and the address arriving from DeltaNet registers; both are circuit work [?].

## 5. Options, ranked

"Tokens per weight" is the read-side cost in trace tokens and multiplies the 596 M MACs. The measured reference is Qwenputer's VDOT build at 0.865 trace tokens per weight (stories260K) and 0.204 with the QwenPU vector unit.

| Rank | Option | Capacity vs 298 MB | Trace tokens per weight (read) | Per-inner-token total [E] | GPU memory | In-Qwen? |
|---|---|---|---|---|---|---|
| **1** | **(d) ROM in appended layers' MLP down_proj**, address from state, wide MAC unit consuming the word | 4B: 13 layers at 8 bit/entry hold it all (2 nibbles per entry); 0.8B: 82 layers | about 1/1000 to 1/100 (a position consumes up to about 1,150 MACs per MAC layer at 4B if an exact int4 x int8 product costs 8 MLP units [E, assumption]; 8 ROM+MAC layer pairs give about 9 K MACs per position) | 6e4-6e5 positions for the MACs; then nonlinear ops (about 1e8 soft-float instructions [E, plan section 3]) dominate unless given ISA support | 4B: 8.4 GB + 13 x about 226 MB = 11.3 GB bf16 (A10G/L4 ok); 0.8B: 1.6 GB + 82 x about 47 MB = 5.5 GB | **Yes, with a stated caveat**: the weights are parameters of the outer model, exactly like a program ROM; the CPU still issues every address and the arithmetic is by layers. Host only boots, embeds, argmaxes. |
| 2 | (d-emb) weights as appended embedding rows ("M2"); the CPU prints the row's token id (store v1 can print any token), the next position embeds a full H-wide row | 4B: 636 M entries, 2 nibbles per entry gives 2.1x margin; 0.8B: 254 M entries = 508 M nibbles at 2 per entry, 85 % of the 596 M needed, so it does not fit alone | 1 trace token per row = about 1/2560 (4B) or 1/1024 (0.8B), plus the layers that consume it | about 2.3e5 token reads for the row fetches alone at 4B | 4B +1.2 GB (233 K appended rows x 2560 x 2 B), no extra depth | **Yes**, same standing as (d); costs a trace token per row, but needs no extra layers. Row fetch is exact by construction (bf16 parameter). Good for the LM head (155 M entries = 61 K rows). |
| 3 | (e) hybrid: ROM layers for matrices, embedding rows for the LM head and tied embedding, DeltaNet state for activations / scratch / CPU RAM, outer full-attention KV for the inner KV cache | all weights, scratch and KV | as (d) | as (d) | as (d) | Yes |
| 4 | (a) host streams weights as data tokens | unlimited | 0.25 minimum (18 bits per token = 4 int4 weights), 0.5-1 realistic with address/instruction tokens | at least 1.5e8 positions just for the weights, 2.7 h at 5.5 K tok/s [E] | model only, no weight storage | **No.** The host holds all 596 M weights and re-supplies them every token; Qwen only does arithmetic. Same footing as the earlier 268.6 s/token route. Acceptable only as a labelled bring-up path. |
| 5 | (b) bulk-load into DeltaNet state before the run | 24x (4B, 8 bit/entry) to 505x (0.8B, 1 bit/entry) too small; reaching 298 MB needs 18 K-146 K value heads | 0.2-0.9 (the VDOT/QwenPU read path, [M]) | 1.2e8-5e8 | 36 GB-290 GB checkpoint, state 1-10 GB | Yes in principle (it is the Qwenputer way) but **infeasible at stock size**: not stock geometry any more, and not loadable. Use for small tables (norm gains, rope tables, the program) only. |
| 6 | (c) full-attention KV as read-only weight store | token-bound: 18 bits per prefilled position, so 134 M positions for 298 MB | lookup 0 extra tokens (softmax with a one-hot key); the lookup is fine, the *fill* is the problem | n/a | 1.6 TB (0.8B) to 4.4 TB (4B) of KV | No use for weights. **Use it for the inner KV cache** (section 6). |

Verdict on what counts: in this setting "the model's weights" are the only place a fixed, data-independent table of 596 M numbers can live without the host.
(d) puts the 0.6B's weights into the outer's parameters, a ROM the CPU addresses itself. That is the same relation a program has to its ROM in Qwenputer
(code in state, loaded at boot), only moved into the parameters. If the rule is that the outer checkpoint must remain *exactly* a stock Qwen3.5 with only a CPU in it,
then no option satisfies it: nothing at stock size has 298 MB of room, and the 0.6B weights have to be added to the checkpoint somewhere. (a) keeps the checkpoint
smallest but puts the weights on the host. This is the honest crux and the owner should state which they accept.

## 6. Everything else the program needs fits

- **Inner KV cache in the outer's full-attention KV.** Per inner token 57,344 int8 values; each outer position stores 16,384 (4B) or 6,144 (0.8B) bf16 values per K and V, so one inner token takes 3.5 (4B) or 9.3 (0.8B) outer positions. T=128 costs about 450 positions (4B). A read is a softmax with a one-hot positional key (no extra trace token, peaked-softmax exactness is the filesystem's assumption, not yet tested here [?]); reading the whole cache per token is about 900 trace tokens at T=128, negligible. Alternatively DeltaNet state holds only 82-219 inner tokens at the 8-bit ceiling, 10-30 at realised density, so state is the wrong place for KV.
- **Activations and scratch** (under 32 KB) stay in the CPU's 32 KB RAM (DeltaNet state).
- **Small constants** (norm gains 57 K, rope table, program text) fit the ROM or the existing 128 KB program ROM.

## 7. Recommended design

1. **Target the 4B.** Its per-layer ROM is 23.6 M entries (6.4x the 0.8B) and MLP I=9216 gives about 2.6x more MAC units per layer. The 0.8B needs 82 ROM layers (5.5 GB, deep) and 448 MACs per layer, so it works as a bring-up target, not the speed target.
2. **Weights as ROM in about 13 appended layers** (int4 weights, two nibbles per 8-bit entry, packed so one word is a row segment of one matrix), address = 14 dual-rail bits written into the residual by the CPU (from a register in DeltaNet state, with an auto-incrementing stream pointer like `WB`). The LM head's 155 M entries can instead live in appended embedding rows (option 2) if depth is the constraint.
3. **One wide MAC unit** (appended MLP layers) multiplies the ROM word with an activation slice read from DeltaNet state and sums it into the CPU's accumulator. This is the QwenPU vector-unit idea with the weights coming from parameters instead of state. It makes the MAC share about 1e5-1e6 trace tokens per inner token; **the nonlinear ops (RMSNorm, softmax, SiLU, RoPE) then dominate at about 1e8 soft-float instructions**, so give them ISA support next [E].
4. **Inner KV in the outer's full-attention KV**, scratch in DeltaNet RAM.
5. Bring-up before any of that exists: stream weights (option a), clearly labelled; it validates the program against the integer emulator.

## 8. Open risks, in order

1. Exact int4 x int8 products in an MLP layer at about 8 units per product, and 1,150 products per layer at 4B, is an estimate. Build one MAC layer next, in the same harness as `rom_mlp.py`.
2. The ROM read assumes the address is already in the residual as dual-rail bits at that position; writing it there from a DeltaNet register read costs a layer or two before each ROM layer. Fine, but unmeasured.
3. The per-position GPU cost of the stock-module path grows with checkpoint size (about 12 GB of weights streamed per position at 4B + ROM, roughly 120 tok/s on an A100 [E]); the fused executor should treat a ROM layer as a one-column gather. At 6e5 positions for the MACs this is minutes either way; at 1e8 positions (nonlinear-bound) it is the fused executor that matters.
4. Overwriting vs appending vocab rows (option 2): append, as store v1 does for its chip rows, so text behaviour is unchanged.
