"""Run the inner Qwen3-0.6B through the merged Qwen3.5 checkpoint.
  python scripts/generate.py CKPT "prompt" [--tokens 64]        (inner mode)
  python scripts/generate.py CKPT "prompt" --outer               (outer Qwen3.5-0.8B mode)
"""
import argparse, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from qweninqwen.merge import NO, INNER_REPO, OUTER_DIR

ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('prompt'); ap.add_argument('--tokens', type=int, default=64)
ap.add_argument('--outer', action='store_true'); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); a = ap.parse_args()
m = Qwen3_5ForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).to(a.device).eval()
tok = AutoTokenizer.from_pretrained(OUTER_DIR if a.outer else INNER_REPO)
x = tok(a.prompt, return_tensors='pt').input_ids.to(a.device); off = 0 if a.outer else NO
with torch.no_grad(): y = m.generate(x + off, max_new_tokens=a.tokens, do_sample=False)
print(tok.decode(y[0] - off))
