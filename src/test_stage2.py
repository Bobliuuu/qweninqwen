import sys, time, torch, glob, os
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest2
from nest2 import *
dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[sys.argv[1]]
dev = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
I3 = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*'))[0]
tin = AutoTokenizer.from_pretrained(I3)
tout = AutoTokenizer.from_pretrained(OUTER_DIR)
inner = AutoModelForCausalLM.from_pretrained(I3, dtype=dtype).eval()
t = time.time(); merged, cfg = build(inner, dtype, dev); print('built', time.time() - t, 's  params(B)=', sum(p.numel() for p in merged.parameters()) / 1e9, flush=True)
outer = reference_outer(dtype)
inner.to(dev); outer.to(dev)
with torch.no_grad():
    # ---- inner mode
    s = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is"
    x = tin(s, return_tensors='pt').input_ids.to(dev)
    a = inner(x).logits.float(); b = merged(x + NO).logits.float()
    print('INNER  max|dlogit| vs Qwen3-0.6B:', (a - b[..., NO:]).abs().max().item(), ' argmax equal:', (a.argmax(-1) == b[..., NO:].argmax(-1)).all().item(),
          ' outer-vocab logits max in inner mode:', b[..., :NO].max().item(), ' inner top logit min:', b[..., NO:].max(-1).values.min().item())
    # ---- outer mode
    s2 = "The quick brown fox jumps over the lazy dog. In 1969, humans first walked on the"
    y = tout(s2, return_tensors='pt').input_ids.to(dev)
    c = outer(y).logits.float(); d = merged(y).logits.float()
    print('OUTER  max|dlogit| vs Qwen3.5-0.8B:', (c - d[..., :NO]).abs().max().item(), ' argmax equal:', (c.argmax(-1) == d[..., :NO].argmax(-1)).all().item(),
          ' inner-vocab logit absmax in outer mode:', d[..., NO:].abs().max().item())
    # ---- generation
    g1 = inner.generate(x, max_new_tokens=20, do_sample=False)
    g2 = merged.generate(x + NO, max_new_tokens=20, do_sample=False)
    print('inner  :', tin.decode(g1[0]))
    print('nested :', tin.decode(g2[0] - NO), '\nIDENTICAL:', torch.equal(g1, g2 - NO))
    h1 = outer.generate(y, max_new_tokens=20, do_sample=False)
    h2 = merged.generate(y, max_new_tokens=20, do_sample=False)
    print('outer  :', tout.decode(h1[0])); print('merged :', tout.decode(h2[0])); print('IDENTICAL:', torch.equal(h1, h2))
