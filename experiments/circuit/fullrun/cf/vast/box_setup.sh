#!/bin/bash
# Runs ON a rented Vast box: python 3.11 venv with torch/numpy/triton/safetensors (same package set as the Modal image), dirs mirroring the Modal volumes.
set -e
mkdir -p /data/cfram_q/cf /data/cfram_q/refQ /out/cfram_q_par_full /root/stage
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
[ -d /root/venv ] || uv venv --python 3.11 /root/venv
. /root/venv/bin/activate
uv pip install -q torch numpy triton safetensors
command -v gcc >/dev/null || (apt-get update -qq && apt-get install -y -qq gcc build-essential rsync)
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
python -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo SETUP_OK
