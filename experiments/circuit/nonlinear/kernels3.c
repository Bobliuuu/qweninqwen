/* attention dot-product kernels (hd = 128, int8 q/k/v): scores q.K^T and probability-weighted V sums.  Plain RV32IM vs SMAQA (Qwenputer npu ext). */
#include <stdint.h>
#define UNR _Pragma("clang loop unroll_count(4)")
static inline int32_t smaqa(int32_t acc, uint32_t a, uint32_t b) { __asm__ volatile(".insn r 0x0b, 7, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b)); return acc; }
/* scores of one q (32 words) against T keys (rows of 32 words), plain mul */
void k_score_mul(const int8_t *q, const int8_t *K, int32_t *out, int T) {
    for (int t = 0; t < T; t++) { const int8_t *k = K + 128 * t; int32_t s = 0; UNR for (int i = 0; i < 128; i++) s += q[i] * k[i]; out[t] = s; } }
/* SMAQA, 4 keys at a time sharing each q word load */
void k_score_smaqa4(const uint32_t *q, const uint32_t *K, int32_t *out, int T) {
    for (int t = 0; t < T; t += 4) { const uint32_t *k0 = K + 32 * t, *k1 = k0 + 32, *k2 = k1 + 32, *k3 = k2 + 32; int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        UNR for (int i = 0; i < 32; i++) { uint32_t w = q[i]; s0 = smaqa(s0, w, k0[i]); s1 = smaqa(s1, w, k1[i]); s2 = smaqa(s2, w, k2[i]); s3 = smaqa(s3, w, k3[i]); }
        out[t] = s0; out[t + 1] = s1; out[t + 2] = s2; out[t + 3] = s3; } }
/* PV: out[d] = sum_t p[t] * V[t][d]; p int16 Q15, V int8 row-major (t major) , plain mul, p in a register for 128 d */
void k_pv_mul(const int16_t *p, const int8_t *V, int32_t *out, int T) {
    for (int d = 0; d < 128; d++) out[d] = 0;
    for (int t = 0; t < T; t++) { int32_t pt = p[t]; const int8_t *v = V + 128 * t; UNR for (int d = 0; d < 128; d++) out[d] += pt * v[d]; } }
/* PV with int8 probabilities and V stored transposed (d major, t contiguous): SMAQA over groups of 4 positions, 4 output dims per p-word load */
void k_pv_smaqa(const uint32_t *p8, const uint32_t *VT, int32_t *out, int T) {
    int T4 = T >> 2;
    for (int d = 0; d < 128; d += 4) { const uint32_t *v0 = VT + d * T4, *v1 = v0 + T4, *v2 = v1 + T4, *v3 = v2 + T4; int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        UNR for (int i = 0; i < T4; i++) { uint32_t w = p8[i]; s0 = smaqa(s0, w, v0[i]); s1 = smaqa(s1, w, v1[i]); s2 = smaqa(s2, w, v2[i]); s3 = smaqa(s3, w, v3[i]); }
        out[d] = s0; out[d + 1] = s1; out[d + 2] = s2; out[d + 3] = s3; } }
