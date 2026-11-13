Everything is logged in [`RUNS.md`](RUNS.md). Pipeline, in order (all under [`experiments/circuit/fullrun`](experiments/circuit/fullrun)):

1. **Firmware and table:** `quantfix/gen_fx_q.py`, `quantfix/cf_gen_q.py`, `quantfix/fw_qwen3_ram_cf_q.c` build the final-contract CF+RAM firmware and the 168,684-row table ([`STEP2_REPORT.md`](experiments/circuit/fullrun/quantfix/STEP2_REPORT.md) has the exact commands and paths).
2. **Reference trace:** `quantfix/modal_refq.py` generates the 96-chunk reference (argmax 15846, 0 injections).
3. **Checkpoint:** `hfwf/merge_wf.py` merges the table into a Qwen3_5-architecture checkpoint (hidden 7680, 12 added chip layers); `hfwf/check_wf.py` runs the stock-module check.
4. **Parallel run and verdict** (full details and every flag in [`cf/README_parallel.md`](experiments/circuit/fullrun/cf/README_parallel.md)):

```sh
cd experiments/circuit/fullrun
CFRAM_MAX_CONTAINERS=25 modal deploy cf/modal_cfram_par.py
python cf/spawn_cfram_parallel.py --build-cache --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB   # once per table
python cf/spawn_cfram_parallel.py --n 96 --root R --elf E --refdir D --ckpt C --rowtab T --rowemb EMB          # detached
python cf/collect_seams.py --n 96 --root R --refvol-dir <D relative to qfr-data> --expect-word 15846           # one verdict
modal app stop -y <app id>                                                                                      # when done
```

For this run: `--elf /data/cfram_q/ramcf_q.elf --refdir /data/cfram_q/refQ --ckpt /wf/ckpt_cf_q --rowtab /data/cfram_q/cf/rowtab_cf.npy --rowemb /data/cfram_q/rowemb_q.npy --root cfram_q_par_full`. The same job can be run on rented GPUs with the scripts in [`cf/vast`](experiments/circuit/fullrun/cf/vast).

Requirements: Python 3.11, `torch`, `triton`, `transformers` 5.x, `safetensors`, [Modal](https://modal.com) (or any CUDA GPU with the same scripts), and the Qwen checkpoints in the Hugging Face cache.
