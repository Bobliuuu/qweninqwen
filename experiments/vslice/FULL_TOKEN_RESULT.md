# One full token of real Qwen3-0.6B inside the outer model (vs2.py)

Status: measured on the shared A100; the raw run log and result JSON are committed in `logs/` (`real_L28.log`, `res_vs2_real151936_L28.json`) and match the numbers below. The planner numbers below are reproducible here
without a GPU: `python -c "import vs2, cost_model as c; print(c.counts(vs2.Cfg(1024,16,8,128,3072,28,151936)))"`
(gives D=40775, 1006 outer layers, 291,072 write + 1,996 control = 293,068 frames, 595,984,384 weights).

Inner model: real, unscaled Qwen3-0.6B (28 layers, d=1024, 16 q / 8 kv heads x 128, inter 3072, vocab 151,936, q/k-norm), real weights.
Prompt "The capital of France is", the position-0 token as input (position 0 for the headline number; see the multi-position run at the end).

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

Multi-position run (teacher-forced prompt-plus-continuation tokens [785, 6722, 315, 9625, 374, 12095, 13]; positions 0-5, so positions 1-5 use the KV cache): outer argmax == HF argmax at all 6 positions (15846, 315, 279, 374, 12095, 13).
Block-best logit max relative error per position: 5.8e-5, 6.9e-6, 1.8e-5, 1.4e-5, 3.2e-5, 1.05e-5. Each position took 264-301 s (first 300.7 s incl. 26 s build). Log: `logs/real_L28_gen.log`. These are teacher-forced positions of one prompt, not free-running generation.

Free-running mode: `FREE=1 python test_vs2.py real151936_L28 6` (added to `test_vs2.py`, commit bc1315c) feeds back only the outer argmax and compares 6 new tokens with `generate()` of HF greedy. The compiled-route run was NOT completed (OOM from a co-tenant, then stopped on request while queued/early); no free-running result exists, see `logs/real_L28_free_*.log`. The only feedback so far is the single appended token (position 5) in the run above.

Pending: HostGuard audit of this driver (running), the L=2 error bump.
