#!/bin/bash
# Runs ON a Vast box: run the given chunks one after another with EXACTLY the Modal chunk_job flags (cf/modal_cfram_par.py), ckpt='' (table from /data/cfram_q/cf/rowtab_cf.npy).
# usage: run_chunks.sh K1 K2 ...     results: /out/cfram_q_par_full/chunk_NNNN/{job.stdout,job_done.json,circuit_state_*.pkl,...}
. /root/venv/bin/activate
for K in "$@"; do
  O=/out/cfram_q_par_full/chunk_$(printf %04d $K); mkdir -p $O
  EXTRA=""; [ "$K" = "0" ] && EXTRA="--reset0"
  T0=$(date +%s)
  ( cd /data/qpriscv_wf && env PYTHONPATH=.:/data/circuit_wf/engine WF=1 REALMAP=1 REALDECODE=1 WF_ROW_CELLS=3840 ROWEMB_CACHE=/data/cfram_q/rowemb_q.npy ROWEMB_STRICT=1 \
    python -u /data/circuit_wfram/fullrun/circuit_run.py /data/cfram_q/ramcf_q.elf /data/cfram_q/refQ $O --chunks $K --ramimg --block 100000 --rowtab /data/cfram_q/cf/rowtab_cf.npy \
    --chain --save-state --fulltable --stop-on-halt --no-host-writes $EXTRA $STEPS ) > $O/job.stdout 2>&1
  RC=$?
  echo "{\"chunk\": $K, \"rc\": $RC, \"seconds\": $(( $(date +%s) - T0 )), \"host\": \"$(hostname)\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"}" > $O/job_done.json
  echo "chunk $K rc=$RC $(( $(date +%s) - T0 ))s"
done
echo ALLDONE
