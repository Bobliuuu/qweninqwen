"""Throughput of a stock Qwen3.5 stack sized like the compiled qwenTPU tile circuit (layers/dims/heads/ffn from the upstream IR).
Random weights (timing only).  Reports chunked-prefill tokens/s (what batched trace tokens cost) and single-step decode latency (one dependent round)."""
import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def bench(layers, dims, heads, ffn, vocab, dtype='bfloat16', N=2048, reps=3):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, DynamicCache
    dev = 'cuda'; dt = getattr(torch, dtype)
    types = ['linear_attention'] * layers
    cfg = Qwen3_5TextConfig(hidden_size=dims, intermediate_size=max(ffn, 64), num_hidden_layers=layers, layer_types=types, vocab_size=vocab,
                            linear_num_key_heads=heads, linear_num_value_heads=heads, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=64)
    m = Qwen3_5ForCausalLM(cfg).to(dev).to(dt).eval(); npar = sum(p.numel() for p in m.parameters())
    res = dict(layers=layers, dims=dims, heads=heads, ffn=ffn, vocab=vocab, dtype=dtype, params=npar)
    ids = torch.randint(0, vocab, (1, N), device=dev)
    with torch.no_grad():
        for name, n in (('prefill', N), ('decode1', 1)):
            c = DynamicCache(config=cfg); kw = lambda: dict(past_key_values=c, use_cache=True, position_ids=torch.arange(n, device=dev)[None], cache_position=torch.arange(n, device=dev))
            m(ids[:, :n], **kw()); torch.cuda.synchronize(); r = reps if n > 1 else 30; t = time.time()
            for _ in range(r): m.model(ids[:, :n], **kw())          # trunk only: the lm head is read at designated steps
            torch.cuda.synchronize(); s = (time.time() - t) / r; res[name] = dict(n=n, sec=s, tokens_per_s=n / s)
    return res

def run(configs=None, dtypes=('bfloat16', 'float32')):
    from depth_probe import probe
    out = []
    for c in configs or [(32, 32, 4, 4, 1, 1), (32, 8, 4, 4, 2, 2), (32, 32, 3, 3, 1, 1), (32, 32, 2, 2, 1, 1)]:
        st = probe(*c)
        for dt in dtypes:
            r = bench(st['layers'], st['dims'], st['heads'], st['ffn_units'], st['vocab'], dt); r['probe'] = st; print('TPS ' + json.dumps(r), flush=True); out.append(r)
            torch.cuda.empty_cache()
    return dict(tps=out)
