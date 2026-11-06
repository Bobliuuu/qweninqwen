"""Exactness tests. Heavy: merged fp32 model is ~30GB RAM (CPU) ; bf16 test needs a ~16GB GPU.
  pytest -x tests/test_merged.py -k fp32      # strict
  pytest -x tests/test_merged.py -k bf16      # statistical
"""
import os, sys, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoTokenizer
from qweninqwen.merge import build, load_inner, reference_outer, NO, OUTER_DIR, INNER_REPO

TEXTS = ["The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
         "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[0]\n",
         "Explain in one sentence why the sky is blue:", "1, 2, 3, 5, 8, 13, 21,",
         "Q: What is 17 * 3?\nA:", "La capitale de l'Italie est", "东京是日本的首都。北京是",
         "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
         "import numpy as np\nx = np.arange(10)\nprint(x.", "Once upon a time, in a small village by the sea, there lived"]


def run(dtype, dev, tol, margin, gen_exact):
    inner = load_inner(dtype).to(dev); outer = reference_outer(dtype).to(dev)
    merged, _ = build(inner, dtype, dev)
    ti, to = AutoTokenizer.from_pretrained(INNER_REPO), AutoTokenizer.from_pretrained(OUTER_DIR)
    stats = dict(inner_agree=0, inner_n=0, outer_agree=0, outer_n=0, maxd_i=0., maxd_o=0.)
    with torch.no_grad():
        for t in TEXTS:
            for side, tok, ref, off in (('i', ti, inner, NO), ('o', to, outer, 0)):
                x = tok(t, return_tensors='pt').input_ids.to(dev)
                a = ref(x).logits.float(); b = merged(x + off).logits.float()
                b = b[..., NO:] if side == 'i' else b[..., :NO]
                top2 = a.topk(2, -1).values; solid = (top2[..., 0] - top2[..., 1]) > margin
                agree = (a.argmax(-1) == b.argmax(-1)) | ~solid
                stats[side == 'i' and 'inner_agree' or 'outer_agree'] += int(agree.sum()); stats[side == 'i' and 'inner_n' or 'outer_n'] += agree.numel()
                stats['maxd_' + side] = max(stats['maxd_' + side], (a - b).abs().max().item())
                assert (agree.all() if gen_exact else True), (side, t)
                if gen_exact:
                    g1 = ref.generate(x, max_new_tokens=32, do_sample=False); g2 = merged.generate(x + off, max_new_tokens=32, do_sample=False) - off
                    assert torch.equal(g1, g2), (side, t)
    assert stats['maxd_i'] < tol and stats['maxd_o'] < tol * 4, stats
    print(stats); return stats


@pytest.mark.skipif(os.environ.get('QIQ_FP32') is None, reason='set QIQ_FP32=1 (needs ~40GB RAM)')
def test_fp32_exact():
    run(torch.float32, 'cpu', 1e-3, 0.05, True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_bf16_gpu():
    s = run(torch.bfloat16, 'cuda', 1e9, 0.5, False)
    assert s['inner_agree'] / s['inner_n'] > 0.99 and s['outer_agree'] / s['outer_n'] > 0.99
