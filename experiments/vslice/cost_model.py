"""T4 cost model: measured outer-step counts/costs x op counts => seconds/token for real Qwen3-0.6B (and Qwen2.5-0.5B).
Frame counts are exact (derived from the same planner as vs2.py); times use measured rates logged in res_vs2_*.json / real_L*.log."""
import os
import math, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs2 import Cfg, build_layout, make_plan, op_dims

def counts(c):
    S = build_layout(c); plan = make_plan(S, c)
    nblk = lambda op: -(-op_dims(c, op)[0] // c.O)
    writes = sum(nblk(op) * S.nbl[op] * 2 * c.P for op in range(4 * c.L + 1))
    ctrl = c.L * (nblk(0) + 1 + nblk(1) + nblk(2) + 1 + nblk(3)) + nblk(4 * c.L) + 2
    return dict(D=S.D, outer_layers=len(plan), write_frames=writes, control_frames=ctrl, weights=sum(op_dims(c, op)[0] * op_dims(c, op)[1] for op in range(4 * c.L + 1)))

if __name__ == '__main__':
    q3 = Cfg(1024, 16, 8, 128, 3072, 28, 151936, theta=1e6, I_MLP=8192)
    q25 = Cfg(896, 14, 2, 64, 4864, 24, 151936, theta=1e6, I_MLP=8192)
    for n, c in (('Qwen3-0.6B', q3), ('Qwen2.5-0.5B (no qk-norm; approx)', q25)):
        try: print(n, counts(c))
        except Exception as e: print(n, 'planner failed:', repr(e))

# ---- measured calibration (A100 40GB, fp32, HF torch GDN path, vs2.py layer-major executor) -------------------------------------
# run_s = ALPHA * write_frames + BETA * control_frames * outer_layers   (two measured points, solved exactly)
#   Qwen3-0.6B L=2 (real151936_L2):  92,088 frames (91,392 writes, 696 ctrl) x 64 layers: run_s ~ 30.9 s
#   Qwen3-0.6B L=28 (real151936_L28): 293,068 frames (291,072 writes, ~1996 ctrl) x 1006 layers: run_s = 241.9 s, build_s = 13.9 s
ALPHA = 3.0e-4      # s per weight-write frame (one bank layer pass; D~41k)
BETA = 7.7e-5       # s per (control frame x outer layer)
def predict(cnt, skip_idle_layers=False, tokens_per_pass=1, layers_per_ctrl=40):
    w = cnt['write_frames'] / tokens_per_pass          # weights streamed once per pass, shared by all tokens in the pass
    ctrl = cnt['control_frames']
    lay = layers_per_ctrl if skip_idle_layers else cnt['outer_layers']
    return ALPHA * w + BETA * ctrl * lay

if __name__ == '__main__':
    print('--- T4 predictions (seconds / token, A100 fp32) ---')
    for n, c in (('Qwen3-0.6B', q3), ('Qwen2.5-0.5B (approx: qk-norm layers kept)', q25)):
        cnt = counts(c)
        print(n, cnt)
        print('  as measured design        : %.0f s' % predict(cnt))
        print('  + skip idle layers per ctrl frame (data-independent schedule): %.0f s' % predict(cnt, True))
        print('  + 16-token prompt batch sharing weight writes                : %.0f s/token' % predict(cnt, True, 16))
    print('measured: Qwen3-0.6B 28 layers one token = 268.6 s (see real_L28 log / RESULTS_VS2.md)')
