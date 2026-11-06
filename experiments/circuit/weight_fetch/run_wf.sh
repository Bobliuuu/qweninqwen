#!/bin/bash
# runs on lambda-final: tests at 2048 and 3840 cells/row, then the executor bench (GPU lock)
cd ~/qiq-wf/qp/experimental/riscv
L=~/qiq-wf/logs
export WROM_BIN=/home/ubuntu/rown/fw0/wrom.bin
.venv/bin/python -m pytest tests/test_rv_wfetch.py -q -s > $L/test_rv_wfetch_cells2048.log 2>&1
WF_ROW_CELLS=3840 .venv/bin/python -m pytest tests/test_rv_wfetch.py -q -s > $L/test_rv_wfetch_cells3840.log 2>&1
flock -w 7200 ~/.gpu.lock env PYTHONPATH=. .venv/bin/python -u tests/bench_wf_exec.py --steps 3000 --out $L/bench_wf_exec_cells2048.json > $L/bench_wf_exec_cells2048.log 2>&1
WF_ROW_CELLS=3840 flock -w 7200 ~/.gpu.lock env PYTHONPATH=. .venv/bin/python -u tests/bench_wf_exec.py --steps 3000 --out $L/bench_wf_exec_cells3840.json > $L/bench_wf_exec_cells3840.log 2>&1
echo DONE > $L/run_wf.done
