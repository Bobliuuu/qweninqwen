"""Architecture reference tests, NOT compiled checkpoint or speed comparisons."""
import sys
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'cpu_reference'),str(ROOT/'gpu_reference')]
from chipformer.rv import asm,ref
from qgpu import ptx
from qgpu.sim import Memory,GLOBAL_BASE,launch


def cpu_dot(x,w):
    source='''
.text
_start:
 la t0, X
 la t1, W
 li t2, LENGTH
 li s0, 0
loop:
 lw t3, 0(t0)
 lw t4, 0(t1)
 mul t5, t3, t4
 add s0, s0, t5
 addi t0, t0, 4
 addi t1, t1, 4
 addi t2, t2, -1
 bne t2, zero, loop
 li a0, 0
 li a7, 93
 ecall
.data
X: .word XVALUES
W: .word WVALUES
'''.replace('LENGTH',str(len(x))).replace('XVALUES',','.join(map(str,x))).replace('WVALUES',','.join(map(str,w)))
    p=asm.assemble(source)
    m=ref.Machine.from_program(p)
    toks=m.run(100000)
    assert m.halted and m.fault is None,(m.halted,m.fault)
    got=m.x[8]
    if got>>31:got-=1<<32
    assert got==int(x@w),(got,int(x@w))
    return dict(length=len(x),result=got,instructions=m.steps,
                trace_tokens=len(toks),trace_stats=ref.trace_stats(toks))


def gpu_dot(x,w):
    mem=Memory(4096,GLOBAL_BASE)
    def buf(vs):
        a=mem.alloc(4*len(vs),align=4)
        mem.write_i32(a,vs)
        return a
    px,pw,po=buf(x),buf(w),buf([0])
    k=ptx.load(ROOT/'gpu_reference/kernels/dot.ptx')['dot']
    stats=launch(k,mem,((len(x)+3)//4,),(4,),[px,pw,po,len(x)])
    got=mem.read_i32(po,1)[0]
    assert got==int(x@w),(got,int(x@w))
    return dict(length=len(x),result=got,warp_instructions=stats.warp_instrs,
                lane_instructions=stats.lane_instrs,by_op=stats.by_op,
                logical_global_words=2*len(x)+1,
                scope='PTX_reference_only_not_lowered_4_lane_circuit')


def main():
    rng=np.random.default_rng(113)
    cpu=[];gpu=[]
    for n in (4,8,16,32):
        for extreme in (False,True):
            x=rng.integers(-128,128,n)
            w=rng.integers(-8,8,n)
            if extreme:x[:]=-128;w[:]=-8
            cpu.append(cpu_dot(x,w));gpu.append(gpu_dot(x,w))
    report=dict(evidence='upstream_CPU_and_PTX_reference_interpreters_only',
                cpu=cpu,gpu=gpu,passed_cases=len(cpu)+len(gpu))
    (ROOT/'reference_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
