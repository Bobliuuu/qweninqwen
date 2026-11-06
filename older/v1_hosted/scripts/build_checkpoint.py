"""Build the merged checkpoint (Qwen3-0.6B inside Qwen3.5-0.8B) and save it.
  python scripts/build_checkpoint.py OUT_DIR [--device cuda|cpu] [--dtype bf16|fp32]
"""
import argparse, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qweninqwen.merge import build, load_inner

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--dtype', default='bf16'); a = ap.parse_args()
dt = {'bf16': torch.bfloat16, 'fp32': torch.float32}[a.dtype]
m, cfg = build(load_inner(dt), dt, a.device)
m.save_pretrained(a.out, safe_serialization=True); print('saved', a.out)
