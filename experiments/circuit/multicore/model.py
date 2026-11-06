"""Capacity + projection model for N CPU lanes inside one forward pass (see README_multicore in docs/MULTICORE.md).
All [E] = estimate from documented budgets; [M] = measured (lanes_sim.py on the Qwenputer reference machine, or compiler instr/MAC).
Run: python model.py > res_model.json"""
import json, math
BUS = math.log2(248320)                      # bits per trace token (stock vocab 248,320 for 0.8B and 4B)
MACS_LAYER = 15.73e6; MACS = 28 * MACS_LAYER + 155.6e6          # Qwen3-0.6B matvec MACs per token = 596.0M
Y_ELEMS = 28 * (2048 + 1024 + 1024 + 1024 + 3072 + 3072 + 1024)  # matvec outputs gathered per token (excl. lm_head) = 344,064
TOK_PER_INSTR = 1.2                          # [M] sim: 8781 tokens / 7309 instr
A_CTL, D_DIV = 7.5, 7.3                      # [M] fit of lanes_sim joint bits/step = A + D*N (plain int8 kernel, uniform values shared)
IPM = dict(plain=7.0, smaqa=0.913, vdot=0.176)   # [M] compiler agent measured instr/MAC
RATE_L4 = 5562.0                             # [M upstream] fused executor trace tok/s, 11-layer core, L4

def capacity():
    """Lanes that fit.  Core P2: 11 layers, chain L1-L4 control (fetch/decode: 3 stateful sublayers L1,L3,L4 + 2-sublayer ROM),
    datapath L5-L11 (stateful: regs L5, digit file L6, RAM L7-L8, DROM L9 = 5 stateful sublayers, <=8,149 of 9,216 MLP units)."""
    out = {}
    # 4B: 32 layers = 24 DeltaNet + 8 full; 16 key heads, 32 value heads, MLP 9216.  [E from config/rv32.md]
    ctl_depth, lane_depth, lane_state, ctl_state = 4, 7, 5, 3
    d4 = (32 - ctl_depth) // lane_depth; s4 = (24 - ctl_state) // lane_state
    out['4B'] = dict(layers=32, deltanet_layers=24, simt_lanes_by_depth=d4, simt_lanes_by_stateful_layers=s4,
                     simt_lanes=min(d4, s4), independent_cores=min(32 // 11, 24 // 8),
                     note='MLP 8,149/9,216 units used per datapath sublayer, so lanes cannot share a layer (width); they stack in depth')
    # 0.8B: 24 layers = 18 DeltaNet + 6 full; 16 key heads = 16 value heads (core wants 2 value heads per key head), MLP 3584, hidden 1024.
    # Re-schedule: ROM 4+RAM 4+DROM 2 sublayers at 1 value head per key head, and 8,149 -> <=3,584 units per sublayer (>=2.3x more MLP sublayers).
    out['0.8B'] = dict(layers=24, deltanet_layers=18, simt_lanes=1, independent_cores=1 if True else 0,
                       note='core itself needs ~2.3x MLP sublayers and ~2x the stateful sublayers (1 value head per key head); a 2nd lane does not fit, even the 1st is unproven [E]')
    return out

def lane_tokens(I, N, d=D_DIV):
    """Trace tokens for I single-lane instructions run on N lockstep lanes, token bus bound."""
    return (I / N) * max(1.0, (A_CTL + d * N) / BUS) if N > 1 else I * TOK_PER_INSTR

def project(build, N, d=D_DIV, nonlin_instr=0.0, nonlin_parallel=False, step_growth=1.0, rate=RATE_L4):
    I_mv = MACS * IPM[build]
    t_mv = lane_tokens(I_mv, N, d)
    gather = 0.0 if N == 1 else Y_ELEMS / math.floor(BUS / 8)       # int8 outputs, 2 per token, all-gather over the one bus
    t_nl = lane_tokens(nonlin_instr, N, d) if nonlin_parallel else nonlin_instr * TOK_PER_INSTR
    toks = t_mv + gather + t_nl
    return dict(build=build, lanes=N, d_bits=d, tokens_matvec=t_mv, tokens_gather=gather, tokens_nonlinear=t_nl, tokens_total=toks,
                trace_tok_per_s=rate / step_growth, s_per_token=toks / (rate / step_growth))

if __name__ == '__main__':
    res = dict(capacity=capacity(), proj=[])
    for build in ('plain', 'smaqa', 'vdot'):
        for N in (1, 2, 3, 4, 8, 16, 32, 64):
            for d in ((D_DIV,) if build == 'plain' else (D_DIV, 12.0, 20.0)):
                res['proj'].append(project(build, N, d, nonlin_instr=1e8))
    # HF-sequential (non-fused) executor: depth grows with lanes: (4 + 7N)/11
    res['proj_seq_depth'] = [project('vdot', N, 12.0, nonlin_instr=1e8, step_growth=(4 + 7 * N) / 11) for N in (1, 2, 3, 4)]
    print(json.dumps(res, indent=1))
