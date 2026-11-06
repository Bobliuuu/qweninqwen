# One full token of real Qwen3-0.6B inside the outer model (vs2.py)

Status: measured on the shared A100; the raw run log and result JSON are committed in `logs/` (`real_L28.log`, `res_vs2_real151936_L28.json`) and match the numbers below. The planner numbers below are reproducible here
without a GPU: `python -c "import vs2, cost_model as c; print(c.counts(vs2.Cfg(1024,16,8,128,3072,28,151936)))"`
(gives D=40775, 1006 outer layers, 291,072 write + 1,996 control = 293,068 frames, 595,984,384 weights).

Inner model: real, unscaled Qwen3-0.6B (28 layers, d=1024, 16 q / 8 kv heads x 128, inter 3072, vocab 151,936, q/k-norm), real weights.
Prompt "The capital of France is", the position-0 token as input (a single position; no KV-using position yet).

Result: outer argmax 15846 == Hugging Face argmax 15846. Block-best logit max error 5.8e-5 relative to max|logit|. Winning 36 rails clean
(top code score 18.0007 of an ideal 18).

Outer model: stock Qwen3_5 decoder layers, fp32, layer-major executor (layers built and freed one at a time), 1006 outer layers, D=40775,
293,068 outer frames per token, HF torch chunk GDN (no fla). Time: 268.6 s per token on a shared A100 (13.9 s layer building + about 242 s running; the
rest is unattributed overhead), untuned.

Earlier smaller checks (same code): L=1 real layer + full vocab: argmax == HF at 5/5 prompt positions (max error 2e-6..4e-5); L=2: 4/4 positions
(35 s per token, max error up to 4e-4, cause not isolated).

Host: embedding row lookup, flags, one-hot addresses, position ids, weight payloads (SiLU pre-inversion, norm gains folded into W, padding),
per-token truncation of the argmax-attention scratch KV, evaluating the outer lm_head through its 36 non-zero columns.
In the outer weights and state: all norms, 197 matvecs, registers, RoPE, softmax attention and KV, SwiGLU, residuals, block max/argmax, cross-block argmax.

Pending: multi-position generation (6-position run including KV-using positions in progress), HostGuard audit of this driver, the L=2 error bump.
