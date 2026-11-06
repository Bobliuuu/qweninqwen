// CF unit test (reference machine only): fetch EVERY constant region through cf_fetch_raw (WROW + VDOT path) and compare with a direct read of the blob; prints the mismatch count (must be 0) and the word count.
#include <stdint.h>
#include "cfg.h"
#include "cf_cfg.h"
#define BI(off) ((const uint32_t *)(0x10000000u + (off)))
void cf_fetch_raw(int32_t *dst, int nwin, int win0, int row0);
static inline void cf_get(int32_t *dst, int row, int off, int n) { cf_fetch_raw(dst, n >> 3, (off % CF_RL) >> 3, row + off / CF_RL); }
static void wr(void *p, int n) { const uint8_t *b = (const uint8_t *)p; for (int i = 0; i < n; i++) { register long a0 __asm__("a0") = b[i], a7 __asm__("a7") = 1; __asm__ volatile("ecall" :: "r"(a0), "r"(a7) : "memory"); } }
void exit(int c) { register long a0 __asm__("a0") = c, a7 __asm__("a7") = 93; __asm__ volatile("ecall" :: "r"(a0), "r"(a7)); for (;;); }
static int32_t buf[4096 + 64]; static int bad = 0, total = 0;
static void chk(int row, int off, uint32_t boff, int n) {
    for (int r0 = 0; r0 < n; r0 += 4096) { int nn = n - r0 < 4096 ? n - r0 : 4096; cf_get(buf, row, off + r0, nn);
        const uint32_t *b = BI(boff) + r0; for (int i = 0; i < nn; i++) { total++; if ((uint32_t)buf[i] != b[i]) bad++; } }
}
int main(void) {
    chk(CF_HEAD_ROW, CFO_EMB, EMB_OFF + CF_TOK * DIM * 4, DIM); chk(CF_HEAD_ROW, CFO_ROPE, ROPE_OFF, HD / 2);
    for (int l = 0; l < NL; l++) {
        int b = CF_LBASE(l);
        chk(b, CFO_N0, NORM_OFF[l][0], DIM); chk(b, CFO_S0, SOFF[l][0], NH * HD); chk(b, CFO_S1, SOFF[l][1], NKV * HD); chk(b, CFO_S2, SOFF[l][2], NKV * HD);
        chk(b, CFO_QN, NORM_OFF[l][1], HD); chk(b, CFO_KN, NORM_OFF[l][2], HD); chk(b, CFO_S3, SOFF[l][3], DIM); chk(b, CFO_N3, NORM_OFF[l][3], DIM);
        chk(b, CFO_S4, SOFF[l][4], HID); chk(b, CFO_S5, SOFF[l][5], HID); chk(b, CFO_S6, SOFF[l][6], DIM);
    }
    chk(CF_FN_ROW, 0, FINAL_NORM_OFF, DIM); chk(CF_LS_ROW, 0, LM_SOFF, VOCAB);
    wr(&bad, 4); wr(&total, 4); exit(0); return 0;
}
