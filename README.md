<div align="center">

# qweninqwen

**Qwen, living inside Qwen's weights.**
A homage to [Qwenputer](https://github.com/divagr18/qwenputer): the outer model is an ordinary Qwen3.5 whose own DeltaNet weights and recurrent state are the machine, and the inner model is a real Qwen executed by that machine.

</div>

## The goal (v3)

- **Outer:** a stock `Qwen3_5ForCausalLM` (GatedDeltaNet layers, SwiGLU MLPs, RMSNorm, full-attention layers). No code patches: only weights, config and the tokens/embeddings we feed it.
- **Inner:** the smallest real Qwen (Qwen2.5-0.5B / Qwen3-0.6B), real weights.
- **The inner model's arithmetic runs in the outer model's forward pass.** Inner weights are paged through the outer DeltaNet state (delta-rule writes in, query reads out give `W x`), norms, SwiGLU, RoPE and softmax attention are executed by compiled outer layers, and the outer's own KV cache is the inner KV cache.
- **Rules:** the host may only stream weights, tokens and embeddings and do the outer argmax/append. It never does inner matmul, attention, normalisation or sampling. Speed is allowed to be absurd (the original circuit-based reference, in `reference/`, measures about a year per token); correctness and honesty are not.

## Status

v3 is under active construction, validated in tiers. Nothing below is claimed until it has a passing test in this repo.

| Tier | What runs inside the outer Qwen's weights | Status |
|---|---|---|
| T0 | single signed MAC / gadget | in progress |
| T1 | one full projection row on a real weight matrix, accumulation in state | in progress |
| T2 | a real decoder op chain (norm, QKV, RoPE, attention with KV, SwiGLU) | in progress |
| T3 | smallest real Qwen end to end, one token produced entirely inside outer weights | planned |
| T4 | cost model for the full 0.5B / 0.6B model (measured per-op cost x op count) | planned |

## Repository

```
reference/              the original nested-qwen lab: circuit-based Qwen-in-Qwen (bit-level arithmetic in GDN state)
older/v1_hosted/        an earlier, fast version: Qwen3-0.6B hosted block-diagonally in the real Qwen3.5-0.8B (exact, tested, with demo video)
older/v2_multihost/     an earlier prototype: several models hosted in the same checkpoint
```

## Older versions

The earlier versions are kept for reference. They run a complete Qwen inside a Qwen3.5 checkpoint using stock transformers code, with outputs verified against the standalone models (see `older/v1_hosted/README.md` for its numbers and demo), but the two models share a forward pass instead of the inner one being executed by the outer one's weights. v3 is that second thing.

Qwen models are Apache-2.0 licensed by Alibaba. Code here is MIT.
