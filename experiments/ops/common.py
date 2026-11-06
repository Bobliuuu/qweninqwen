import sys, torch, json, glob
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', 'qweninqwen')); sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '.'))
from transformers import AutoModelForCausalLM, AutoTokenizer
PROMPT = "The capital of France is Paris. In quantum mechanics, the wave function describes the state of a system, and the transformer architecture"
def load_inner(name, dtype=torch.float32):
    r = {'q25': 'Qwen/Qwen2.5-0.5B', 'q3': 'Qwen/Qwen3-0.6B'}[name]
    m = AutoModelForCausalLM.from_pretrained(r, dtype=dtype, attn_implementation='eager').eval()
    return m, AutoTokenizer.from_pretrained(r)
def ids(tok, n=None):
    t = tok(PROMPT, return_tensors='pt').input_ids
    return t if n is None else t[:, :n]
def err(a, b):
    a = a.double(); b = b.double(); d = (a - b).abs()
    return dict(max_abs=d.max().item(), max_rel_to_maxabs=(d.max() / b.abs().max()).item(), rmse_rel=(d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item())
