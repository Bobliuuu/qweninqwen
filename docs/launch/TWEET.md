# Tweet thread (DRAFT ONLY, not posted)

Nothing here has been posted or sent anywhere. Pending items are marked `[PENDING]`: do not claim them until the README status board says **done**. Update lines per [`UPDATING.md`](UPDATING.md).

## 1/ Main post (277 characters, attach `video/explainer_square.mp4` or `assets/hero.gif`)

> A real Qwen3-0.6B token, computed inside another Qwen. The outer model's weights and DeltaNet state do the matmuls, norms, attention and the 151,936-way argmax. Matches Hugging Face at 6/6 positions, error 5.8e-5, 268.6 s/token on an A100. Teacher-forced, not free-running yet.

Alt text (hero.gif / explainer): "Animation titled Qwen inside Qwen. A row of colored slivers representing 1006 stacked outer layers lights up from left to right, then the text 'outer argmax 15846 == Hugging Face 15846' appears with three numbers: max relative logit error 5.8e-5, 6 of 6 positions matching Hugging Face, 268.6 seconds per token on a shared A100 in fp32. Illustration; numbers from the repository logs."

## 2/ The idea (attach `assets/idea.gif`)

> The program is a real Qwen3-0.6B (28 layers, d=1024). The machine is a stock Qwen3.5 model from transformers, code unpatched, only its weights are written. A DeltaNet layer is an associative memory, so writing a weight row and reading with an activation is a matrix-vector product.

Alt text: "Two boxes. Left: INNER, the program: real Qwen3-0.6B weights, 28 layers, d 1024, vocab 151,936. Right: OUTER, the machine: stock Qwen3_5ForCausalLM, fp32, 1006 layers, D 40775. A green arrow labelled 'compiled into weights plus state' goes from left to right. A bottom strip lists what the host is allowed to do and states that it never computes an inner activation."

## 3/ How it works (attach `assets/pipeline.gif`)

> Every inner op gets its own outer layers, in depth order: RMSNorm, QKV bank, register, RoPE + attention over the outer KV cache, SwiGLU, residuals. One token = 1006 outer layers and 293,068 frames. State only flows deeper, so it is an unrolled pipeline. (Animation is an illustration.)

Alt text: "A row of nine blocks (RMSNorm, W_qkv bank, register, RoPE plus attention, W_o plus residual, RMSNorm, gate and up, SwiGLU, down plus residual) is lit one after another by a moving highlight. Below, a chain from embed through 28 inner layers, final norm, LM head, argmax, outer lm_head, token, with a progress bar counting outer layers up to 1006. Illustration."

## 4/ The argmax (attach `assets/tournament.gif`)

> Even the 151,936-way argmax is in-model: 594 blocks of 256 reduce themselves, then hard-max attention over the 594 winners picks the global best as an 18-bit code (36 rails). The outer lm_head only decodes it. Winning code score 18.0007 vs ideal 18. (Bars in the animation are illustrative.)

Alt text: "Bar chart of 594 grey bars (illustrative heights) turning amber as each block finds its own best, then one bar, block 61, turns green as an attention arrow selects it. A panel shows block 61 plus local index 230 equals token 15846, the first eight rails 0 1 1 0 1 0 0 1, top code score 18.0007, and outer argmax 15846 equal to Hugging Face 15846."

## 5/ Results (attach `video/demo_square.mp4` or `assets/results_table.png`)

> Prompt "The capital of France is", real run, positions 0-5 (KV used at 1-5): outer argmax == HF at all six (15846, 315, 279, 374, 12095, 13). Max rel. logit error 6.9e-6 to 5.8e-5. 265-300 s per position on one shared A100, fp32, untuned torch GDN.

Alt text: "Table with six rows, one per position 0 to 5. Columns: input token id, outer argmax, Hugging Face argmax, match, max relative logit error, time. All six rows show a match; errors range from 6.9e-6 to 5.8e-5 and times from 264.6 to 300.7 seconds."

## 6/ Honest caveats (attach `assets/tiers_ladder.png`)

> What this is NOT yet: (1) the 6 positions are teacher-forced on one prompt, not free-running generation [PENDING]. (2) The HostGuard audit of this driver is not done [PENDING]. (3) The host still does embedding lookup, flags, one-hot addresses, position ids, weight payloads, KV scratch truncation and the lm_head eval through 36 columns. (4) ~4.5 minutes per token; speedup work is in progress [PENDING].

Alt text: "A staircase of six tiers T0, T1, T2, T3-toy, T3 real, T4. T0, T1 and T3-toy are green (verified or achieved); T2, T3 real and T4 are amber (partial). A marker labelled 'we are here' sits on T3 real. A checklist at top lists free-running generation, HostGuard audit and speedup work, each marked in progress."

## 7/ Repo (no attachment)

> Code, logs, the audit harness, a provenance ledger for every number in these clips, and a tier-by-tier status board: https://github.com/Bobliuuu/qweninqwen . Homage to Qwenputer (divagr18/qwenputer). Qwen models by Alibaba, Apache-2.0.

## Media inventory

| File | Use |
|---|---|
| `docs/assets/hero.gif` | post 1 (GIF alternative to video) |
| `docs/launch/video/explainer_square.mp4` (1080x1080) / `explainer_16x9.mp4` (1280x720) | post 1 |
| `docs/launch/video/demo_square.mp4` / `demo_16x9.mp4` | post 5 |
| `docs/assets/idea.gif`, `pipeline.gif`, `tournament.gif`, `tiers_climb.gif`, `state_memory.gif` | posts 2, 3, 4, 6 |
| `docs/assets/results_table.png`, `tiers_ladder.png`, `cost_chart.png`, `flow_*.png` | static fallbacks |
