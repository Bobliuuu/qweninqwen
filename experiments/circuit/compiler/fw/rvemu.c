// Fast RV32IM(+Zba, +Qwenputer VDOT/WB/SMAQA) emulator with instruction counting by function and class.
// Usage: rvemu prog.elf [-b ADDR blob.bin]... [-w wrom.bin] [-o out.bin] [-p] [-m MAXINSTR]
// ecall a7=1 putchar(a0) -> out file ; a7=93 exit ; a7=2 mark(a0): record cumulative instr count ; a7=64 write(a1,a2)
// Instruction semantics match chipformer.rv.ref (Qwenputer) for the base ISA; VDOT: rd += sum_k a_k*w_k, a = 8 int8 from
// (rs1,rs2) bytes, w_k = WROM cell (WP+k)-8 (u nibble), WP starts at WB base and advances 1 word per VDOT (see ref.py).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#define MEMSZ (1ull<<32)
static uint8_t *mem;
typedef struct { uint32_t addr, size; char name[48]; uint64_t n; } Sym;
static Sym *syms; static int nsym;
static uint8_t *wrom; static size_t wrom_n;
static inline uint32_t ld(uint32_t a, int n){uint32_t v=0;memcpy(&v,mem+a,n);return v;}
static inline void st(uint32_t a, uint32_t v, int n){memcpy(mem+a,&v,n);}
static int cmpsym(const void*a,const void*b){return (int)(((Sym*)a)->addr>((Sym*)b)->addr)-(int)(((Sym*)a)->addr<((Sym*)b)->addr);}
static void loadelf(const char*p, uint32_t *entry){
  FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t*b=malloc(n); fread(b,1,n,f); fclose(f);
  *entry=*(uint32_t*)(b+0x18); uint32_t phoff=*(uint32_t*)(b+0x1c); uint16_t phes=*(uint16_t*)(b+0x2a), phn=*(uint16_t*)(b+0x2c);
  for(int i=0;i<phn;i++){uint32_t*ph=(uint32_t*)(b+phoff+i*phes); if(ph[0]==1&&ph[5]) memcpy(mem+ph[3],b+ph[1],ph[4]);}
  uint32_t shoff=*(uint32_t*)(b+0x20); uint16_t shes=*(uint16_t*)(b+0x2e), shn=*(uint16_t*)(b+0x30);
  for(int i=0;i<shn;i++){uint32_t*sh=(uint32_t*)(b+shoff+i*shes); if(sh[1]==2){ // SYMTAB
      uint32_t*lk=(uint32_t*)(b+shoff+sh[6]*shes); uint8_t*str=b+lk[4]; int cnt=sh[5]/16; syms=calloc(cnt+1,sizeof(Sym));
      for(int j=0;j<cnt;j++){uint8_t*s=b+sh[4]+j*16; uint32_t nm=*(uint32_t*)s,val=*(uint32_t*)(s+4),sz=*(uint32_t*)(s+8); uint8_t info=s[12];
        if((info&0xf)==2&&sz){syms[nsym].addr=val;syms[nsym].size=sz;strncpy(syms[nsym].name,(char*)str+nm,47);nsym++;}}
      qsort(syms,nsym,sizeof(Sym),cmpsym);}}
  free(b);
}
static Sym* findsym(uint32_t pc){int lo=0,hi=nsym-1;while(lo<=hi){int m=(lo+hi)/2;if(syms[m].addr<=pc){if(pc<syms[m].addr+syms[m].size)return&syms[m];lo=m+1;}else hi=m-1;}return NULL;}
static inline int32_t sx(uint32_t v,int b){return (int32_t)(v<<(32-b))>>(32-b);}
int main(int argc,char**argv){
  mem=mmap(0,MEMSZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0); if(mem==MAP_FAILED){perror("mmap");return 1;}
  uint32_t entry; loadelf(argv[1],&entry); const char*outp=NULL; int prof=0; uint64_t maxi=~0ull;
  for(int i=2;i<argc;i++){
    if(!strcmp(argv[i],"-b")){uint32_t a=strtoul(argv[i+1],0,0);FILE*f=fopen(argv[i+2],"rb");if(!f){perror(argv[i+2]);return 1;}
      fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);size_t r=fread(mem+a,1,n,f);(void)r;fclose(f);i+=2;}
    else if(!strcmp(argv[i],"-w")){FILE*f=fopen(argv[i+1],"rb");fseek(f,0,SEEK_END);wrom_n=ftell(f);fseek(f,0,SEEK_SET);wrom=malloc(wrom_n);size_t r=fread(wrom,1,wrom_n,f);(void)r;fclose(f);i++;}
    else if(!strcmp(argv[i],"-o")){outp=argv[++i];} else if(!strcmp(argv[i],"-p")) prof=1; else if(!strcmp(argv[i],"-m")) maxi=strtoull(argv[++i],0,0);}
  uint32_t x[32]={0}; uint32_t pc=entry; x[2]=0xFFF00000u; uint64_t n=0; FILE*out=outp?fopen(outp,"wb"):NULL;
  uint64_t cls[8]={0}; // 0 alu,1 mul,2 div,3 load,4 store,5 branch,6 jump,7 custom
  uint64_t marks[1<<16]; int nm=0; uint32_t wbase=0; int exitc=-1; Sym*cur=NULL;
  uint64_t *symn=prof?calloc(nsym+1,8):NULL;
  while(n<maxi){
    uint32_t w=ld(pc,4); uint32_t op=w&0x7f, rd=(w>>7)&31, f3=(w>>12)&7, rs1=(w>>15)&31, rs2=(w>>20)&31, f7=w>>25;
    uint32_t a=x[rs1], b=x[rs2], nxt=pc+4, res=0; int wr=0; n++;
    if(prof){ if(!cur||pc<cur->addr||pc>=cur->addr+cur->size){cur=findsym(pc);} if(cur)cur->n++; }
    switch(op){
    case 0x37: res=w&0xfffff000; wr=1; cls[0]++; break;
    case 0x17: res=pc+(w&0xfffff000); wr=1; cls[0]++; break;
    case 0x6f:{int32_t im=(int32_t)((w>>31)<<20|((w>>12)&0xff)<<12|((w>>20)&1)<<11|((w>>21)&0x3ff)<<1); im=sx(im&0x1fffff,21); res=nxt;wr=1;nxt=pc+im;cls[6]++;break;}
    case 0x67: res=nxt;wr=1;nxt=(a+sx(w>>20,12))&~1u;cls[6]++;break;
    case 0x63:{int32_t im=sx(((w>>31)<<12)|(((w>>7)&1)<<11)|(((w>>25)&0x3f)<<5)|(((w>>8)&0xf)<<1),13);int t=0;
      switch(f3){case 0:t=a==b;break;case 1:t=a!=b;break;case 4:t=(int32_t)a<(int32_t)b;break;case 5:t=(int32_t)a>=(int32_t)b;break;case 6:t=a<b;break;case 7:t=a>=b;break;}
      if(t)nxt=pc+im;cls[5]++;break;}
    case 0x03:{uint32_t ea=a+sx(w>>20,12);cls[3]++;wr=1;switch(f3){case 0:res=sx(ld(ea,1),8);break;case 1:res=sx(ld(ea,2),16);break;case 2:res=ld(ea,4);break;case 4:res=ld(ea,1);break;case 5:res=ld(ea,2);break;}break;}
    case 0x23:{uint32_t ea=a+sx(((w>>25)<<5)|((w>>7)&31),12);cls[4]++;st(ea,b,1<<f3);break;}
    case 0x13:{uint32_t im=sx(w>>20,12);wr=1;cls[0]++;switch(f3){case 0:res=a+im;break;case 2:res=(int32_t)a<(int32_t)im;break;case 3:res=a<im;break;case 4:res=a^im;break;case 6:res=a|im;break;case 7:res=a&im;break;
      case 1:res=a<<(im&31);break;case 5:res=(w>>30)&1?(uint32_t)((int32_t)a>>(im&31)):a>>(im&31);break;}break;}
    case 0x33:{wr=1;
      if(f7==1){cls[1]++;switch(f3){case 0:res=a*b;break;case 1:res=(uint32_t)(((int64_t)(int32_t)a*(int32_t)b)>>32);break;case 2:res=(uint32_t)(((int64_t)(int32_t)a*(uint64_t)b)>>32);break;case 3:res=(uint32_t)(((uint64_t)a*b)>>32);break;
        case 4: cls[2]++;cls[1]--; res=b==0?~0u:((int32_t)a==INT32_MIN&&(int32_t)b==-1)?a:(uint32_t)((int32_t)a/(int32_t)b);break;
        case 5: cls[2]++;cls[1]--; res=b==0?~0u:a/b;break;
        case 6: cls[2]++;cls[1]--; res=b==0?a:((int32_t)a==INT32_MIN&&(int32_t)b==-1)?0:(uint32_t)((int32_t)a%(int32_t)b);break;
        case 7: cls[2]++;cls[1]--; res=b==0?a:a%b;break;}}
      else if(f7==16&&f3==2){cls[0]++;res=(a<<1)+b;} else if(f7==16&&f3==4){cls[0]++;res=(a<<2)+b;} else if(f7==16&&f3==6){cls[0]++;res=(a<<3)+b;}
      else {cls[0]++;switch(f3){case 0:res=(f7&0x20)?a-b:a+b;break;case 1:res=a<<(b&31);break;case 2:res=(int32_t)a<(int32_t)b;break;case 3:res=a<b;break;case 4:res=a^b;break;case 5:res=(f7&0x20)?(uint32_t)((int32_t)a>>(b&31)):a>>(b&31);break;case 6:res=a|b;break;case 7:res=a&b;break;}}
      break;}
    case 0x0f: cls[0]++; break;
    case 0x0b:{cls[7]++; // custom-0: f3=5 VDOT, 6 WB, 7 SMAQA
      if(f3==5){int32_t s=0;for(int k=0;k<4;k++){s+=(int8_t)(a>>(8*k))*((int)(wrom[wbase*8+k]&15)-8);s+=(int8_t)(b>>(8*k))*((int)(wrom[wbase*8+4+k]&15)-8);} wbase++; res=x[rd]+s; wr=1;}
      else if(f3==6){wbase=a;}
      else if(f3==7){int32_t s=0;for(int k=0;k<4;k++)s+=(int8_t)(a>>(8*k))*(int8_t)(b>>(8*k));res=x[rd]+s;wr=1;}
      break;}
    case 0x73: cls[0]++;
      if(x[17]==93){exitc=x[10];goto done;} else if(x[17]==1){if(out)fputc(x[10]&0xff,out);} else if(x[17]==2){if(nm<65536)marks[nm++]=n;}
      else if(x[17]==64){if(out)fwrite(mem+x[11],1,x[12],out);} else {fprintf(stderr,"bad ecall %u at %x\n",x[17],pc);goto done;}
      break;
    default: fprintf(stderr,"illegal %08x at %08x after %llu\n",w,pc,(unsigned long long)n); goto done;}
    if(wr&&rd)x[rd]=res; pc=nxt;
  }
done:
  printf("exit=%d instr=%llu alu=%llu mul=%llu div=%llu load=%llu store=%llu branch=%llu jump=%llu custom=%llu\n",exitc,(unsigned long long)n,
    (unsigned long long)cls[0],(unsigned long long)cls[1],(unsigned long long)cls[2],(unsigned long long)cls[3],(unsigned long long)cls[4],(unsigned long long)cls[5],(unsigned long long)cls[6],(unsigned long long)cls[7]);
  for(int i=0;i<nm;i++)printf("mark %d %llu\n",i,(unsigned long long)marks[i]);
  if(prof){for(int i=0;i<nsym;i++)if(syms[i].n)printf("sym %s %llu\n",syms[i].name,(unsigned long long)syms[i].n);}
  if(out)fclose(out); return exitc==0?0:2;
}
