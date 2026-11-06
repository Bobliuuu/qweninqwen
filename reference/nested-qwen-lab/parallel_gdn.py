"""Parallel bit-plane GDN experiment. Native Qwen primitive, not full model.
Host constructs input/state and reconstructs signed outputs for validation.
Reconstruction, loading protocol, scaling, and sequencing are NOT compiled here.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

def setup(x,w):
    k,n=w.shape
    assert k==32 and n<=31
    state=torch.zeros(1,8,128,128,dtype=torch.float32)
    # Real rows + complementary zero-data rows; every query selects 32 rows.
    state[:,:,:64,:2]=2**24
    for c in range(n):
        for b in range(4):
            state[:,:,0:32,2+4*c+b]=torch.tensor((w[:,c]>>b)&1)*128
    q=torch.zeros(1,1,8,128,dtype=torch.bfloat16)
    for a in range(8):
        bits=(x>>a)&1
        q[0,0,a,:32]=torch.tensor(bits)*128
        q[0,0,a,32:64]=torch.tensor(1-bits)*128
    return q,state

def run(q,state,n):
    z=torch.zeros_like(q);g=torch.zeros(1,1,8);beta=torch.zeros_like(g)
    y,_=torch_recurrent_gated_delta_rule(q,z,z,g,beta,initial_state=state,
                output_final_state=False,use_qk_l2norm_in_kernel=True)
    # Model the actual per-head RMS normalization and its BF16 output boundary.
    normalized=(y.float()*torch.rsqrt(y.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
    # z gate SiLU(128)=128; output projection factor=4096.
    counts=(normalized.float()*128*4096)[0,0,:,2:2+4*n].reshape(8,n,4)
    return counts

def main():
    torch.set_num_threads(2);rng=np.random.default_rng(82);checks=0
    for n in [1,8,16,31]:
        for case in range(12):
            x=rng.integers(-128,128,32);w=rng.integers(-8,8,(32,n))
            if case==0:x[:]=-128;w[:]=-8
            if case==1:x[:]=127;w[:]=7
            q,state=setup(x,w);counts=run(q,state,n).numpy().astype(np.int64)
            expected_counts=np.stack([np.stack([((x>>a)&1)@((w>>b)&1) for b in range(4)],-1) for a in range(8)])
            np.testing.assert_array_equal(counts,expected_counts)
            ca=np.array([1,2,4,8,16,32,64,-128]);cb=np.array([1,2,4,-8])
            result=np.einsum('a,anb,b->n',ca,counts,cb)
            np.testing.assert_array_equal(result,x@w);checks+=n
    # Primitive throughput only: excludes state loading, query encoding, sum and scale.
    q,state=setup(rng.integers(-128,128,32),rng.integers(-8,8,(32,31)))
    for _ in range(5):run(q,state,31)
    start=time.perf_counter()
    for _ in range(100):run(q,state,31)
    seconds=time.perf_counter()-start
    report=dict(evidence='native_Qwen_GDN_primitive_plus_BF16_norm_boundary',
                signed_outputs_checked=checks,shape=[32,31],parallel_heads=8,
                logical_macs_per_read=992,primitive_calls_per_second=100/seconds,
                state_bytes=state.numel()*state.element_size(),
                exclusions=['state_loading','query_encoding','signed_reconstruction',
                            'group_scales','persistent_sum','full_Qwen_layers'],
                warning='primitive throughput is NOT end_to_end MAC throughput')
    Path('parallel_gdn_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
