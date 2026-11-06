/* Minimal RV32IM instruction-set simulator with two custom instructions modelled on Qwenputer's NPU branch (README "In-weights NPU"):
 *   SMAQA rd,rs1,rs2 : rd += sum_{i<4} int8(rs1.byte[i]) * int8(rs2.byte[i])          (custom-0, funct3=1)
 *   VDOT  rd,rs1,rs2 : rd += sum_{i<8} int8(act[i]) * int4(w[i]); act = rs1 (acts 0..3) and rs2 (acts 4..7);
 *                      the 8 int4 weights are the next 4 bytes of the weight stream at WB, then WB += 4   (custom-0, funct3=2)
 *   WBSET rs1        : WB = rs1                                                          (custom-0, funct3=3)
 * One instruction == one retired instruction (the real circuit emits ~1.2-1.55 trace tokens per instruction).
 * usage: rvsim prog.bin data.bin out.bin  ; data loaded at 0x100000, out region dumped from 0x400000 (len from a0 at ecall).
 * Prints: INSTR total, per class.                                                                                          */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define MEM (64u<<20)
static uint8_t *m; static uint32_t x[32], pc, WB;
static uint64_t n_tot, n_alu, n_mul, n_ld, n_st, n_br, n_cus;
static inline uint32_t rd32(uint32_t a){ uint32_t v; memcpy(&v, m+a, 4); return v; }
static inline void wr32(uint32_t a, uint32_t v){ memcpy(m+a, &v, 4); }
int main(int argc, char **argv){
  m = calloc(MEM, 1); FILE *f = fopen(argv[1], "rb"); fread(m, 1, 1<<20, f); fclose(f);
  f = fopen(argv[2], "rb"); fread(m + 0x100000, 1, 0x2f0000, f); fclose(f);
  x[2] = MEM - 16; pc = 0;
  for (;;) {
    uint32_t i = rd32(pc), op = i & 0x7f, rd = (i>>7)&31, f3 = (i>>12)&7, r1 = (i>>15)&31, r2 = (i>>20)&31, f7 = i>>25, npc = pc + 4, v = 0;
    int32_t imi = (int32_t)i >> 20, ims = ((int32_t)(i & 0xfe000000) >> 20) | ((i>>7)&31);
    int32_t imb = ((int32_t)(i & 0x80000000) >> 19) | ((i&0x80)<<4) | ((i>>20)&0x7e0) | ((i>>7)&0x1e);
    int32_t imj = ((int32_t)(i & 0x80000000) >> 11) | (i & 0xff000) | ((i>>9)&0x800) | ((i>>20)&0x7fe);
    uint32_t a = x[r1], b = x[r2]; if (++n_tot > 2000000000ull) { fprintf(stderr, "instruction cap\n"); return 2; }
    switch (op) {
    case 0x37: v = i & 0xfffff000; x[rd] = v; n_alu++; break;
    case 0x17: x[rd] = pc + (i & 0xfffff000); n_alu++; break;
    case 0x6f: x[rd] = pc + 4; npc = pc + imj; n_br++; break;
    case 0x67: x[rd] = pc + 4; npc = (a + imi) & ~1u; n_br++; break;
    case 0x63: { int t = 0; switch (f3) { case 0: t = a == b; break; case 1: t = a != b; break; case 4: t = (int32_t)a < (int32_t)b; break;
                 case 5: t = (int32_t)a >= (int32_t)b; break; case 6: t = a < b; break; case 7: t = a >= b; break; }
                 if (t) npc = pc + imb; n_br++; break; }
    case 0x03: { uint32_t ad = a + imi; n_ld++;
                 switch (f3) { case 0: v = (int8_t)m[ad]; break; case 1: { int16_t h; memcpy(&h, m+ad, 2); v = h; break; } case 2: v = rd32(ad); break;
                               case 4: v = m[ad]; break; case 5: { uint16_t h; memcpy(&h, m+ad, 2); v = h; break; } }
                 if (rd) x[rd] = v; break; }
    case 0x23: { uint32_t ad = a + ims; n_st++; if (f3 == 0) m[ad] = b; else if (f3 == 1) memcpy(m+ad, &b, 2); else wr32(ad, b); break; }
    case 0x13: { uint32_t sh = r2; n_alu++;
                 switch (f3) { case 0: v = a + imi; break; case 2: v = (int32_t)a < imi; break; case 3: v = a < (uint32_t)imi; break; case 4: v = a ^ imi; break;
                               case 6: v = a | imi; break; case 7: v = a & imi; break; case 1: v = a << sh; break; case 5: v = (f7 & 0x20) ? (uint32_t)((int32_t)a >> sh) : a >> sh; break; }
                 if (rd) x[rd] = v; break; }
    case 0x33: {
      if (f7 == 1) { n_mul++; int64_t sa = (int32_t)a, sb = (int32_t)b; uint64_t ua = a, ub = b;
        switch (f3) { case 0: v = a * b; break; case 1: v = (uint32_t)((sa * sb) >> 32); break; case 2: v = (uint32_t)((sa * (int64_t)ub) >> 32); break; case 3: v = (uint32_t)((ua * ub) >> 32); break;
          case 4: v = b == 0 ? -1 : ((int32_t)a == INT32_MIN && (int32_t)b == -1) ? a : (int32_t)a / (int32_t)b; break; case 5: v = b == 0 ? -1 : a / b; break;
          case 6: v = b == 0 ? a : ((int32_t)a == INT32_MIN && (int32_t)b == -1) ? 0 : (int32_t)a % (int32_t)b; break; case 7: v = b == 0 ? a : a % b; break; }
      } else { n_alu++;
        switch (f3) { case 0: v = (f7 & 0x20) ? a - b : a + b; break; case 1: v = a << (b&31); break; case 2: v = (int32_t)a < (int32_t)b; break; case 3: v = a < b; break;
          case 4: v = a ^ b; break; case 5: v = (f7 & 0x20) ? (uint32_t)((int32_t)a >> (b&31)) : a >> (b&31); break; case 6: v = a | b; break; case 7: v = a & b; break; } }
      if (rd) x[rd] = v; break; }
    case 0x0b: { n_cus++;
      if (f3 == 1) { int32_t s = 0; for (int k = 0; k < 4; k++) s += (int8_t)(a >> (8*k)) * (int8_t)(b >> (8*k)); x[rd] += s; }
      else if (f3 == 2) { int32_t s = 0; uint32_t w = rd32(WB); WB += 4;
        for (int k = 0; k < 8; k++) { int8_t act = (int8_t)((k < 4 ? a : b) >> (8*(k&3))); int nib = (w >> (4*k)) & 15; nib = nib >= 8 ? nib - 16 : nib; s += act * nib; } x[rd] += s; }
      else if (f3 == 3) WB = a;
      break; }
    case 0x0f: break;
    case 0x73: { FILE *o = fopen(argv[3], "wb"); fwrite(m + 0x400000, 1, x[10], o); fclose(o);
      printf("INSTR %llu ALU %llu MUL %llu LD %llu ST %llu BR %llu CUSTOM %llu\n", (unsigned long long)n_tot, (unsigned long long)n_alu, (unsigned long long)n_mul,
             (unsigned long long)n_ld, (unsigned long long)n_st, (unsigned long long)n_br, (unsigned long long)n_cus); return 0; }
    default: fprintf(stderr, "bad op %x at %x\n", i, pc); return 1; }
    x[0] = 0; pc = npc;
  }
}
