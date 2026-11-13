**What does "Qwen in Qwen" mean here, precisely?**
A Qwen3-0.6B inference (the inner model) runs as a program on a CPU whose circuit is implemented by the weights of a Qwen3_5-architecture model (HF `Qwen3_5ForCausalLM` classes, widened to hidden 7680 with 12 added chip layers and an extended 680,128-token vocabulary; derived from the Qwen3.5-4B layer design) (the outer model). The outer model's own layers decode each instruction, do the arithmetic, hold the registers and RAM in recurrent state, and fetch the inner model's weights from its own embedding rows. The inner model's answer token appears as the circuit's output before `HALT`.

**Why 15846?**
15846 is what Hugging Face's Qwen3-0.6B produces for this prompt (fp32 and bf16). The firmware uses int4 weights and int8 activations, with layers 2, 26 and 27 at 8-bit weights and 16-bit activations because they carry the large position-0 activations ([precision design](#precision-design)).

**Why were the chunks run in parallel, and is that the same as one continuous run?**
A single GPU would need about 10 to 15 hours for the 199M steps. Each of the 96 jobs started from the reference's start state for its chunk and saved the circuit's own end state; the collector verified exact equality with the next job's start state at every seam. That is a seam-checked parallel run. A serial run through all 96 chunks was not done.

**Is this free-running generation?**
No. It is one token at position 0 of a single compile-time prompt token. The comparison against the reference trace is how the run is checked, token by token, and it is not an input to the run.

**What does the host still do?**
Boot, the `GO` token, the embed-and-append loop and `ecall` I/O, and it seeds each later chunk's start state ([table](#-what-the-host-does)).

**Is the outer model the released Qwen3.5-4B?**
No. It is a Qwen3_5-architecture model (HF `Qwen3_5ForCausalLM` classes, widened to hidden 7680 with 12 added chip layers and an extended 680,128-token vocabulary; derived from the Qwen3.5-4B layer design). It uses the HF module classes with only its weights written, but it is wider than the real Qwen3.5-4B (hidden 2560, 32 layers) and carries the 2.59 GB table, so it is not the released checkpoint.

**Why layers 2, 26 and 27?**
They carry the position-0 massive activations. Quantising any one of them alone flips the answer; quantising any one of the other 25 does not ([`diag_quant`](experiments/circuit/fullrun/diag_quant/REPORT.md)).
