import sys, torch, time
sys.path.insert(0, 'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from nest import compile_nested
import os
I = [l.split('=')[1].strip() for l in open('env.sh') if l.startswith('I=')][0]
tok = AutoTokenizer.from_pretrained(I)
inner = AutoModelForCausalLM.from_pretrained(I, torch_dtype=torch.float32).eval()
outer, cfg = compile_nested(inner)
outer.eval()
print(type(outer).__name__, 'layers', {t for t in cfg.layer_types}, 'params', sum(p.numel() for p in outer.parameters())/1e6)
ids = tok(["The capital of France is", "def fibonacci(n):\n    if n < 2:"], return_tensors='pt', padding=True)
x = tok("The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is", return_tensors='pt').input_ids
with torch.no_grad():
    a = inner(x).logits; b = outer(x).logits
print('max |dlogit|', (a-b).abs().max().item(), 'argmax match', (a.argmax(-1)==b.argmax(-1)).float().mean().item())
g1 = inner.generate(x, max_new_tokens=24, do_sample=False)
g2 = outer.generate(x, max_new_tokens=24, do_sample=False)
print(tok.decode(g1[0])); print(tok.decode(g2[0])); print('IDENTICAL', torch.equal(g1, g2))
