/* Matvec kernels y[R] = W[R x K] . x[K] (int32 out) for the ISS.  Layout: data region at 0x100000: R,K words; x int8 at 0x100100; W at 0x110000.
 * VARIANT: 0 int8 scalar C; 1 int8 SMAQA; 2 int8 SMAQA 4 rows share act loads; 3 int4 packed scalar C (nibble unpack);
 *          4 int4 VDOT 1 row; 5 int4 VDOT RB rows share act loads (acts held in regs), unrolled; (RB = -DRB=4/8, UNR = -DUNR)       */
typedef signed char i8; typedef int i32; typedef unsigned u32; typedef unsigned char u8;
#define SMAQA(acc, a, b) __asm__ volatile(".insn r 0x0b, 1, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
#define VDOT(acc, a, b) __asm__ volatile(".insn r 0x0b, 2, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
#define WBSET(p) __asm__ volatile(".insn r 0x0b, 3, 0, x0, %0, x0" :: "r"(p))
#ifndef RB
#define RB 4
#endif
#ifndef UNR
#define UNR 2
#endif
__attribute__((naked, section(".text.start"))) void _start(void) { __asm__ volatile("call main\n ecall"); }
int main(void) {
  volatile u32 *hdr = (u32 *)0x100000; u32 R = hdr[0], K = hdr[1];
  const i8 *x = (const i8 *)0x100100; const u8 *W = (const u8 *)0x110000; i32 *y = (i32 *)0x400000;
#if VARIANT == 0
  for (u32 r = 0; r < R; r++) { i32 a = 0; const i8 *w = (const i8 *)W + r * K; for (u32 k = 0; k < K; k++) a += w[k] * x[k]; y[r] = a; }
#elif VARIANT == 1
  for (u32 r = 0; r < R; r++) { i32 a = 0; const u32 *w = (const u32 *)(W + r * K); const u32 *xx = (const u32 *)x;
    for (u32 k = 0; k < K / 4; k++) SMAQA(a, w[k], xx[k]); y[r] = a; }
#elif VARIANT == 2
  for (u32 r = 0; r < R; r += 4) { i32 a0 = 0, a1 = 0, a2 = 0, a3 = 0; const u32 *w0 = (const u32 *)(W + r * K), *w1 = w0 + K / 4, *w2 = w1 + K / 4, *w3 = w2 + K / 4; const u32 *xx = (const u32 *)x;
    for (u32 k = 0; k < K / 4; k++) { u32 xv = xx[k]; SMAQA(a0, w0[k], xv); SMAQA(a1, w1[k], xv); SMAQA(a2, w2[k], xv); SMAQA(a3, w3[k], xv); }
    y[r] = a0; y[r + 1] = a1; y[r + 2] = a2; y[r + 3] = a3; }
#elif VARIANT == 3
  for (u32 r = 0; r < R; r++) { i32 a = 0; const u8 *w = W + r * (K / 2); for (u32 k = 0; k < K / 2; k++) { u32 b = w[k]; i32 lo = (i32)(b << 28) >> 28, hi = (i32)(b << 24) >> 28; a += lo * x[2 * k] + hi * x[2 * k + 1]; } y[r] = a; }
#elif VARIANT == 4
  WBSET(W); const u32 *xx = (const u32 *)x;
  for (u32 r = 0; r < R; r++) { i32 a = 0; for (u32 k = 0; k < K / 8; k++) VDOT(a, xx[2 * k], xx[2 * k + 1]); y[r] = a; }
#elif VARIANT == 5
  WBSET(W); const u32 *xx = (const u32 *)x;
  for (u32 r = 0; r < R; r += RB) { i32 a[RB]; for (int j = 0; j < RB; j++) a[j] = 0;
    for (u32 k = 0; k < K / 8; k += UNR) {
      for (int u = 0; u < UNR; u++) { u32 x0 = xx[2 * (k + u)], x1 = xx[2 * (k + u) + 1]; for (int j = 0; j < RB; j++) VDOT(a[j], x0, x1); } }
    for (int j = 0; j < RB; j++) y[r + j] = a[j]; }
#endif
  return R * 4;
}
