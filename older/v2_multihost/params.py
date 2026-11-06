"""Analytic parameter / scaling table for merge2 layouts (no model built)."""
M06=dict(n='0.6B',hi=1024,nq=16,hd=128,I=3072,V=151936,L=28,useful=0.596e9)
MB=dict(M06,n='0.6B-Base')
M17=dict(n='1.7B',hi=2048,nq=16,hd=128,I=6144,V=151936,L=28,useful=1.72e9)
M4=dict(n='4B',hi=2560,nq=32,hd=128,I=9728,V=151936,L=36,useful=4.02e9)
def count(ms, outer_useful=0.75e9):
    N=len(ms); H=(1024+sum(m['hi'] for m in ms)+1+N+31)//32*32
    P=32+sum(m['hd']//2 for m in ms); HD=2*P+192
    NQ=8+sum(m['nq'] for m in ms); NKV=NQ//2
    I=max(3584,sum(m['I'] for m in ms)); napp=max(m['L'] for m in ms)
    V=248320+sum(m['V'] for m in ms)
    attn=2*NQ*HD*H+2*NKV*HD*H+NQ*HD*H
    mlp=3*H*I
    gdn=(6144+2048+32)*H+2048*H
    tot=2*V*H + 6*(attn+mlp) + 18*(gdn+mlp) + napp*(attn+mlp)
    return dict(models='+'.join(m['n'] for m in ms),H=H,HD=HD,NQ=NQ,I=I,layers=24+napp,params_B=round(tot/1e9,2),
                bf16_GB=round(tot*2/1e9,1),useful_B=round((sum(m['useful'] for m in ms)+outer_useful)/1e9,2),
                useful_frac=round((sum(m['useful'] for m in ms)+outer_useful)/tot,3))
if __name__=='__main__':
    for ms in ([M06],[M06,MB],[M06,M17],[M06,MB,M17],[M06,MB,MB],[M06,MB,MB,MB],[M17,M17],[M4],[M06,M4]):
        print(count(ms))
