cd ~/qiq-wf/qp/experimental/riscv; L=~/qiq-wf/logs
WF_ROW_CELLS=3840 flock -w 7200 ~/.gpu.lock env PYTHONPATH=. .venv/bin/python -u tests/bench_wf_exec.py --steps 3000 --out $L/bench_wf_exec_cells3840.json > $L/bench_wf_exec_cells3840.log 2>&1
echo DONE > $L/bench.done
