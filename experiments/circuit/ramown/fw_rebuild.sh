#!/bin/bash
# fw_rebuild.sh WORKDIR [PY] [QP_RISCV_DIR]  -- rebuild + measure + verify the RAM-minimal 0.6B firmware from scratch (needs: riscv64-unknown-elf-gcc+picolibc, HF Qwen3-0.6B cached, torch+transformers python PY).
#  1 gen cfg.h/blob/wrom for prompt [785] (the ref785b token) and for the 4-token prompt;  2 build ELF on the real 32 KB map;  3 address trace + live-word analysis on the rvemu_mt emulator;
#  4 bit-exact verification vs the original compiler/fw/qwen3_fx.c (all logits);  5 (optional, ~30 min, run by hand) full ref-machine run with the REAL 8192-word RAM: fw_ref32.py
set -e
R=$(cd "$(dirname "$0")" && pwd); W=${1:-$HOME/ramown}; PY=${2:-python3}; QP=${3:-$HOME/qp-npu/experimental/riscv}; mkdir -p $W; L=$R/logs
$PY $R/../fullrun/gen_full.py $W/g785 785 4 0 | tail -1
$PY $R/../fullrun/gen_full.py $W/g4e1 785,6722,315,9625 4 1 | tail -1
cc -O2 -w -o $W/rvemu_mt $R/fw_rvemu_mt.c -lm; export RVEMU=$W/rvemu_mt
for v in "" "-DTWOPASS"; do
  n=$(echo "new785$v" | tr -d ' -'); $R/fw_build.sh $W/g785 $W/$n.elf $v
  $W/rvemu_mt $W/$n.elf -s 0x8000 -b 0x10000000 $W/g785/blob.bin -w $W/g785/wrom.bin -o $W/$n.out -T $W/$n.tr | head -1
  python3 $R/fw_addr_analysis.py $W/$n.elf $W/$n.tr $L/addr_$n.json | tee $L/addr_$n.txt
done
python3 $R/fw_verify.py $W/g785 | tee $L/verify_ctx1.json
python3 $R/fw_verify.py $W/g785 -DEXT=1 | tee -a $L/verify_ctx1.json
python3 $R/fw_verify.py $W/g785 -DTWOPASS | tee -a $L/verify_ctx1.json
python3 $R/fw_verify.py $W/g4e1 | tee $L/verify_ctx4_generic.json
python3 $R/fw_verify.py $W/g4e1 -DTWOPASS | tee -a $L/verify_ctx4_generic.json
python3 $R/fw_verify.py $W/g4e1 -DTILE=128 | tee -a $L/verify_ctx4_generic.json
echo "full ref run:  $PY $R/fw_ref32.py $W/new785.elf $W/g785/blob.bin $W/g785/wrom.bin $W/ref32_785 $QP"
