#!/bin/bash
# spark2 (GB10, aarch64) single-chunk runner: EXACT flags of cf/modal_cfram_par.py chunk_job / cf/vast/run_chunks.sh, paths under ~/qiq-spark/data. usage: run_chunk.sh K [extra circuit_run args]
D=$HOME/qiq-spark/data; K=$1; shift
. $HOME/qiq-spark/venv/bin/activate
O=$HOME/qiq-spark/out/cfram_q_par_full/chunk_$(printf %04d $K); mkdir -p $O
T0=$(date +%s)
( cd $D/qpriscv_wf && env PYTHONPATH=.:$D/circuit_wf/engine WF=1 REALMAP=1 REALDECODE=1 WF_ROW_CELLS=3840 TRITON_DEFAULT_FP_FUSION=1 ROWEMB_CACHE=$D/cfram_q/rowemb_q.npy ROWEMB_STRICT=1 \
  python -u $D/circuit_wfram/fullrun/circuit_run.py $D/cfram_q/ramcf_q.elf $D/cfram_q/refQ $O --chunks $K --ramimg --block 100000 --rowtab $D/cfram_q/cf/rowtab_cf.npy \
  --chain --save-state --fulltable --stop-on-halt --no-host-writes "$@" ) > $O/job.stdout 2>&1
RC=$?
echo "{\"chunk\": $K, \"rc\": $RC, \"seconds\": $(( $(date +%s) - T0 )), \"host\": \"$(hostname)\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"}" > $O/job_done.json
echo "chunk $K rc=$RC $(( $(date +%s) - T0 ))s"
