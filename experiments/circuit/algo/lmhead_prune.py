"""Exact argmax over the 151,936-way vocabulary with fewer MACs (branch and bound with rigorous Cauchy-Schwarz tail bounds), on real Qwen3-0.6B final hidden
states.  Stage s computes the partial dot over dims [0, r_s) (fixed offline dim order), keeps only rows whose upper bound >= best lower bound.
No softmax is ever needed (argmax only).  Exactness: rows are dropped only if they provably cannot beat the best row -> identical argmax to the full matvec."""
import torch, json
from quant_study import PROMPTS

@torch.no_grad()
def run(stages=((64,), (128,), (64, 256), (128, 512), (32, 128, 512), (64, 192, 512)), nprompt=12):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); m = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32).cuda().eval()
    E = m.lm_head.weight.data.float(); N, D = E.shape; H = []
    for p in PROMPTS[:nprompt]:
        ids = torch.tensor(tok(p)['input_ids']).cuda()[None]; H.append(m.model(ids).last_hidden_state[0])
    H = torch.cat(H)                                      # [P, D] final-norm hidden states
    order = (H.pow(2).mean(0) * E.pow(2).sum(0)).argsort(descending=True)   # fixed offline dim order (calibration prompts; not per token)
    Eo = E[:, order]; Ho = H[:, order]; out = []
    for st in stages:
        cuts = list(st) + [D]; tails = [Eo[:, c:].norm(dim=1) for c in cuts[:-1]]
        macs = []; ok = 0
        for t in range(H.shape[0]):
            h = Ho[t]; cand = torch.arange(N, device='cuda'); part = torch.zeros(N, device='cuda'); prev = 0; mac = 0
            for ci, c in enumerate(cuts):
                part[cand] += Eo[cand, prev:c] @ h[prev:c]; mac += cand.numel() * (c - prev); prev = c
                if c == D: break
                hn = h[c:].norm(); tl = tails[ci][cand] * hn
                lo = (part[cand] - tl).max(); cand = cand[(part[cand] + tl) >= lo]
            ok += int(cand[part[cand].argmax()].item() == (E @ H[t]).argmax().item()); macs.append(mac)
        r = dict(stages=list(st), mean_macs=sum(macs) / len(macs), full_macs=N * D, fraction=sum(macs) / len(macs) / (N * D), argmax_ok=ok, positions=H.shape[0])
        print('LMH ' + json.dumps(r), flush=True); out.append(r)
    return dict(lmhead=out)


@torch.no_grad()
def shortlist(ranks=(32, 64, 128, 256, 512), nprompt=12):
    """NOT rigorous: sketch logits with a rank-r SVD of the (tied) head, keep top-k candidates, evaluate them exactly.  Reports the worst rank of the true argmax under
    the sketch over all calibration positions (k must exceed it for argmax==HF on these positions) and the resulting MACs; also leave-one-prompt-out generalisation."""
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); m = Qwen3ForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32).cuda().eval()
    E = m.lm_head.weight.data.float(); N, D = E.shape; Hs = []; owner = []
    for pi, p in enumerate(PROMPTS[:nprompt]):
        ids = torch.tensor(tok(p)['input_ids']).cuda()[None]; h = m.model(ids).last_hidden_state[0]; Hs.append(h); owner += [pi] * h.shape[0]
    H = torch.cat(Hs); true = (H @ E.T).argmax(-1)
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)       # E = U S Vh ; sketch uses top-r right singular vectors
    out = []
    for r in ranks:
        Er = E @ Vh[:r].T                                       # [N, r] precomputed offline
        sk = (H @ Vh[:r].T) @ Er.T                              # [P, N] approx logits (cost N*r MACs per token)
        rk = (sk > sk.gather(1, true[:, None])).sum(1)          # rank of the true argmax under the sketch (0 = first)
        k99 = int(rk.max().item()) + 1
        rec = dict(rank=r, worst_rank_of_true_argmax=k99, median_rank=int(rk.median().item()), p99_rank=int(rk.float().quantile(0.99).item()),
                   macs_if_k_is_worst=N * r + k99 * D, full=N * D, fraction=(N * r + k99 * D) / (N * D))
        print('LMS ' + json.dumps(rec), flush=True); out.append(rec)
    return dict(lmhead_shortlist=out)
