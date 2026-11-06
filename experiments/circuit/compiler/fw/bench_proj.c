// instructions-per-MAC microbench: one int4/int8 projection R=3072 outputs x C=1024 inputs (Qwen3-0.6B MLP gate/up shape)
#include <stdint.h>
#define R 3072
#define C 1024
#define MARK() __asm__ volatile("li a7,2\n ecall" ::: "a7","memory")
#define SMAQA(acc,a,b) __asm__ volatile(".insn r 0x0B, 7, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
const int8_t *W8 = (const int8_t*)0x10000000;      // [R][C] int8
const int8_t *XQ = (const int8_t*)0x1f000000;      // [C]
static int32_t out[R], o1[R], o2[R];
static void wr(void*p,int n){register long a0 __asm__("a0")=1,a1 __asm__("a1")=(long)p,a2 __asm__("a2")=n,a7 __asm__("a7")=64;__asm__ volatile("ecall"::"r"(a0),"r"(a1),"r"(a2),"r"(a7):"memory");}
const int8_t *vk_x;
void vk_init_raw(int32_t*, int, int, const int32_t*, int32_t);
void vk_acc_raw(int32_t*, int, int, const int32_t*, int32_t);
void vk_acc_scaled(int32_t*, int, int, const int32_t*, int32_t);
void vk_init_scaled(int32_t*, int, int, const int32_t*, int32_t);
static int32_t S[R];
int main(void) {
  MARK();
  for (int r = 0; r < R; r++) { int32_t a = 0; const int8_t *w = W8 + r * C;                      // plain RV32IM int8
    for (int j = 0; j < C; j++) a += w[j] * XQ[j]; out[r] = a; }
  MARK();
  for (int r = 0; r < R; r++) { int32_t a = 0; const uint32_t *w = (const uint32_t*)(W8 + r * C), *x = (const uint32_t*)XQ;   // SMAQA int8
#pragma GCC unroll 8
    for (int j = 0; j < C / 4; j++) SMAQA(a, w[j], x[j]); o1[r] = a; }
  MARK();
  for (int c = 0; c < C / 64; c++) {                                                                // VDOT int4 (WROM, chunk-major)
    vk_x = XQ + 64 * c;
    if (c == 0) vk_init_raw(out, 0, R, 0, 0);
    else if (c < C / 64 - 1) vk_acc_raw(out, c * R * 8, R, 0, 0);
    else vk_acc_raw(out, c * R * 8, R, 0, 0);
  }
  MARK();
  wr(out,4*R); wr(o1,4*R);
  return 0;
}
void exit(int c){ register long a0 __asm__("a0")=c; register long a7 __asm__("a7")=93; __asm__ volatile("ecall"::"r"(a0),"r"(a7)); for(;;); }
