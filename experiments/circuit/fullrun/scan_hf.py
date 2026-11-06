import sys, json, torch
from transformers import AutoModelForCausalLM
toks = [int(t) for t in sys.argv[1:]]
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float32).eval()
with torch.no_grad(): a = m(torch.tensor(toks)[:, None]).logits[:, 0].argmax(-1).tolist()
print(json.dumps(dict(zip(map(str, toks), a))))
