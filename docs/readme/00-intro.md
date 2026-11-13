<p align="center">
  <img src="docs/assets/circuit_hero.gif" alt="A real Qwen3-0.6B token, computed inside another Qwen: the prompt token enters a Qwen3.5-style outer model, the circuit inside runs the firmware while the weights are fetched from the model's own embedding table, then halts with 15846" width="900">
  <br><sub>🎞️ <a href="docs/assets/circuit_hero.mp4">mp4</a> · a visualization of the real run (layer widths, counts and the answer are real data)</sub>
</p>

A homage to [Qwenputer](https://github.com/divagr18/qwenputer), where a RISC-V CPU is compiled into Qwen's weights and runs as the model generates tokens. Here that CPU runs **another language model**: a C firmware of Qwen3-0.6B (int4 weights, three layers at higher precision) executes on the circuit, one instruction per generated trace token, fetches its weights from the outer model's own embedding rows, keeps its working memory in circuit-owned RAM, and finishes by emitting its answer word.
