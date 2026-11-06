#define UNR _Pragma("clang loop unroll_count(8)")
/* Fixed-point RV32IM kernels for the Qwen3-0.6B nonlinear ops, baseline (plain RV32IM) and with the proposed custom-1 extensions.
   Compiled with: clang --target=riscv32 -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -fno-unroll-loops -c
   All pointers/constants come in as arguments (no relocations).  Formats: x Q12 (int32) activations, scores Q16 in the base-2 domain
   (log2(e)/sqrt(hd) folded into the q weights: free), logits int32. */
typedef int int32_t; typedef unsigned uint32_t; typedef long long int64_t; typedef unsigned long long uint64_t;
typedef short int16_t; typedef signed char int8_t;
#define MULH(a,b) ((int32_t)(((int64_t)(a) * (int64_t)(b)) >> 32))
#define MULHU(a,b) ((uint32_t)(((uint64_t)(a) * (uint64_t)(b)) >> 32))
#define EXT_R(name, f3) static inline uint32_t name(uint32_t a, uint32_t b) { uint32_t r; __asm__ volatile(".insn r 0x2b, " #f3 ", 0, %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); return r; }
EXT_R(x_exp2, 0) EXT_R(x_rsq, 1) EXT_R(x_sigm, 2) EXT_R(x_rot, 5)
static inline void x_amx(uint32_t v, uint32_t i) { __asm__ volatile(".insn r 0x2b, 3, 0, x0, %0, %1" :: "r"(v), "r"(i)); }
static inline uint32_t x_amxr(uint32_t rst) { uint32_t r; __asm__ volatile(".insn r 0x2b, 4, 0, %0, %1, x0" : "=r"(r) : "r"(rst)); return r; }
static inline int32_t x_smaqa(int32_t acc, uint32_t a, uint32_t b) { __asm__ volatile(".insn r 0x0b, 7, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b)); return acc; }
static inline void x_wb(uint32_t p) { __asm__ volatile(".insn r 0x0b, 6, 0, x0, %0, x0" :: "r"(p)); }
static inline int32_t x_vdot(int32_t acc, uint32_t a, uint32_t b) { __asm__ volatile(".insn r 0x0b, 5, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b)); return acc; }

/* ---------------- RMSNorm ---------------- */
#define INL static inline __attribute__((always_inline))
INL uint32_t i_sumsq(const int32_t *x, int n) {              /* sum (x>>4)^2, x Q12 -> Q16 */
    uint32_t s = 0; UNR for (int i = 0; i < n; i++) { int32_t t = x[i] >> 4; s += (uint32_t)(t * t); } return s; }
static inline int32_t mq28(int32_t a, int32_t b) { return MULH(a, b) << 4; }       /* Q28 x Q28 -> Q28 */
INL uint32_t i_rsqrt_base(uint32_t m) {                    /* m Q16 > 0  ->  m^-1/2 in Q24 (same contract as RSQ) */
    int e = 0; uint32_t t = m;
    if (t >> 16) { t >>= 16; e += 16; } if (t >> 8) { t >>= 8; e += 8; } if (t >> 4) { t >>= 4; e += 4; } if (t >> 2) { t >>= 2; e += 2; } if (t >> 1) { e += 1; }
    int E = e - 16, par = E & 1, Ee = E - par;
    int32_t Mp = (e <= 28 ? (int32_t)(m << (28 - e)) : (int32_t)(m >> (e - 28))) << par;     /* M' in [1,4) Q28 */
    int32_t y = 0x12AAAAAB - MULH(Mp, 0x2AAAAAAB);                                       /* chord y0 = 7/6 - M'/6 (Q28) */
    for (int it = 0; it < 3; it++) { int32_t y2 = mq28(y, y); int32_t t3 = 0x30000000 - mq28(Mp, y2); y = mq28(y, t3) >> 1; }
    int sh = 4 + Ee / 2;                                   /* Q28 -> Q24, times 2^(-Ee/2) */
    return sh >= 0 ? (uint32_t)y >> sh : (uint32_t)y << -sh; }
void k_rms_scale(const int32_t *x, const int32_t *g, int32_t *o, int n, uint32_t y) {   /* o = x*y*g, y Q24, x,g,o Q12 */
    UNR for (int i = 0; i < n; i++) { int32_t t = MULH(x[i] << 8, (int32_t)y); o[i] = (t * g[i]) >> 12; } }
uint32_t k_rms_inv_ext(const int32_t *x, int n, int shift) {      /* whole scalar part with RSQ: sumsq (shared) + mean + eps + RSQ */
    uint32_t s = i_sumsq(x, n); return x_rsq((s >> shift) + 1, 0); }
uint32_t k_rms_inv_base(const int32_t *x, int n, int shift) { uint32_t s = i_sumsq(x, n); return i_rsqrt_base((s >> shift) + 1); }

/* ---------------- exp2 / softmax / silu ---------------- */
INL uint32_t i_exp2_base(int32_t x) {                      /* 2^(x/2^16) in Q24, x<0.. */
    int32_t xi = x >> 16; uint32_t f = (uint32_t)x & 0xFFFF;
    if (xi >= 8) return 0xFFFFFFFF;                      /* saturate like EXP2 */
    int32_t f30 = (int32_t)(f << 14);
    int32_t p = 0x00dff96f;                            /* deg-4 near-minimax coefficients, Q30 (c4..c0) */
    p = MULH(p, f30) * 4 + 0x034fca40; p = MULH(p, f30) * 4 + 0x0f76721f; p = MULH(p, f30) * 4 + 0x2c59ab19; p = MULH(p, f30) * 4 + 0x40000ea6;
    int sh = 24 - 30 + xi;                              /* result = p(Q30) * 2^xi -> Q24 */
    uint32_t m = (uint32_t)p; if (p < 0) m = 0;
    if (sh >= 0) return m << sh; if (-sh >= 32) return 0; return m >> -sh; }
void k_softmax_base(const int32_t *s, int32_t *o, int T) {         /* s Q16 base-2 domain; o Q16 */
    int32_t mx = s[0]; UNR for (int i = 1; i < T; i++) if (s[i] > mx) mx = s[i];
    uint32_t sum = 0;
    UNR for (int i = 0; i < T; i++) { uint32_t p = i_exp2_base(s[i] - mx) >> 4; o[i] = (int32_t)p; sum += p; }
    uint32_t r = (1u << 30) / (sum >> 10);
    UNR for (int i = 0; i < T; i++) o[i] = MULH(o[i] << 8, (int32_t)r); }
void k_softmax_ext(const int32_t *s, int32_t *o, int T) {
    int32_t mx = s[0]; UNR for (int i = 1; i < T; i++) if (s[i] > mx) mx = s[i];
    uint32_t sum = 0;
    UNR for (int i = 0; i < T; i++) { uint32_t p = x_exp2(s[i] - mx, 0) >> 4; o[i] = (int32_t)p; sum += p; }
    uint32_t r = (1u << 30) / (sum >> 10);
    UNR for (int i = 0; i < T; i++) o[i] = MULH(o[i] << 8, (int32_t)r); }
void k_silu_base(const int32_t *g, const int32_t *u, int32_t *o, int n) {   /* g,u Q16 -> silu(g)*u Q16 */
    UNR for (int i = 0; i < n; i++) {
        int32_t x = g[i]; int32_t m = x >> 31; int32_t ax = (x ^ m) - m;              /* |x| */
        int32_t t = MULH(-ax << 2, 0x5C551D95);                                      /* -|x|*log2e Q16 */
        uint32_t e = i_exp2_base(t);                                                 /* exp(-|x|) Q24, <= 2^24 */
        uint32_t sp = (1u << 31) / (((1u << 24) + e) >> 9);                          /* sigmoid(|x|) Q16 */
        uint32_t sg = x >= 0 ? sp : MULHU(e << 4, sp << 4);                          /* sigmoid(x) */
        int32_t sx = MULH(x << 4, (int32_t)(sg << 12));
        o[i] = MULH(sx << 8, u[i] << 8); } }
void k_silu_ext(const int32_t *g, const int32_t *u, int32_t *o, int n) {
    UNR for (int i = 0; i < n; i++) { int32_t x = g[i]; uint32_t sg = x_sigm(x, 0); int32_t sx = MULH(x << 4, (int32_t)(sg << 12)); o[i] = MULH(sx << 8, u[i] << 8); } }

/* ---------------- RoPE ---------------- */
void k_rope_base(const int16_t *x1, const int16_t *x2, const int16_t *c, const int16_t *s, int16_t *y1, int16_t *y2, int n) {
    UNR for (int i = 0; i < n; i++) { int32_t a = x1[i], b = x2[i], cc = c[i], ss = s[i];
        int32_t u = (a * cc - b * ss + 0x4000) >> 15, v = (a * ss + b * cc + 0x4000) >> 15;
        y1[i] = u; y2[i] = v; } }
void k_rope_ext(const uint32_t *xp, const uint32_t *cs, uint32_t *yp, int n) { UNR for (int i = 0; i < n; i++) yp[i] = x_rot(xp[i], cs[i]); }
uint32_t k_sincos_base(uint32_t turns, const int16_t *tab) {      /* turns Q32 (fraction of a full turn) ; tab = 257-entry sin table over a quarter turn? -> use full 256+1 sine table */
    uint32_t i = turns >> 24, fr = (turns >> 8) & 0xFFFF;
    int32_t s0 = tab[i], s1 = tab[i + 1]; int32_t s = s0 + (((s1 - s0) * (int32_t)fr) >> 16);
    uint32_t t2 = turns + 0x40000000; uint32_t j = t2 >> 24, fr2 = (t2 >> 8) & 0xFFFF;
    int32_t c0 = tab[j], c1 = tab[j + 1]; int32_t c = c0 + (((c1 - c0) * (int32_t)fr2) >> 16);
    return ((uint32_t)s << 16) | ((uint32_t)c & 0xFFFF); }

/* ---------------- residual, requant, argmax ---------------- */
void k_add(const int32_t *a, const int32_t *b, int32_t *o, int n) { UNR for (int i = 0; i < n; i++) o[i] = a[i] + b[i]; }
void k_requant(const int32_t *acc, int8_t *o, int n, int32_t scale) {      /* acc * scale(Q31) -> int8 round clamp */
    UNR for (int i = 0; i < n; i++) { int32_t v = (MULH(acc[i], scale) + 1) >> 1; v = v > 127 ? 127 : v; v = v < -127 ? -127 : v; o[i] = v; } }
int k_argmax_base(const int32_t *a, int n) { int32_t b = a[0]; int bi = 0; UNR for (int i = 1; i < n; i++) if (a[i] > b) { b = a[i]; bi = i; } return bi; }
int k_argmax_ext(const int32_t *a, int n) { x_amxr(1); UNR for (int i = 0; i < n; i++) x_amx(a[i], i); return x_amxr(1); }
/* fused-with-matvec epilogue: value already in a register (the accumulator), index in a register */
int k_argmax_fused_base(const int32_t *a, int n) { int32_t b = -2147483647 - 1; int bi = 0; int i = 0;
    for (; i < n; i += 4) { int32_t v0 = a[i], v1 = a[i+1], v2 = a[i+2], v3 = a[i+3];
        if (v0 > b) { b = v0; bi = i; } if (v1 > b) { b = v1; bi = i+1; } if (v2 > b) { b = v2; bi = i+2; } if (v3 > b) { b = v3; bi = i+3; } } return bi; }

/* ---------------- dot products (one output row of length n) ---------------- */
int32_t k_dot_i8_base(const int8_t *w, const int8_t *a, int n) { int32_t s = 0; UNR for (int i = 0; i < n; i++) s += w[i] * a[i]; return s; }
int32_t k_dot_i4_base(const uint32_t *wp, const int8_t *a, int n) {          /* 8 nibbles per word, w = nib-8 */
    int32_t s = 0; UNR for (int i = 0; i < n; i += 8) { uint32_t w = wp[i >> 3];
        for (int k = 0; k < 8; k++) s += (int32_t)(((w >> (4 * k)) & 15) - 8) * a[i + k]; } return s; }
int32_t k_dot_smaqa(const uint32_t *wa, const uint32_t *aa, int n4) { int32_t s = 0; UNR for (int i = 0; i < n4; i++) s = x_smaqa(s, wa[i], aa[i]); return s; }
int32_t k_dot_vdot1(const uint32_t *aa, int n8) {          /* one row, WROM pointer already set (WB) */
    int32_t s = 0; UNR for (int i = 0; i < n8; i++) s = x_vdot(s, aa[2 * i], aa[2 * i + 1]); return s; }
void k_dot_vdot8(const uint32_t *aa, int n8, int32_t *out) {      /* 8 rows share the activation registers; accumulators stay in registers */
    int32_t s0 = out[0], s1 = out[1], s2 = out[2], s3 = out[3], s4 = out[4], s5 = out[5], s6 = out[6], s7 = out[7];
    UNR for (int i = 0; i < n8; i++) { uint32_t a0 = aa[2*i], a1 = aa[2*i+1];
        s0 = x_vdot(s0, a0, a1); s1 = x_vdot(s1, a0, a1); s2 = x_vdot(s2, a0, a1); s3 = x_vdot(s3, a0, a1);
        s4 = x_vdot(s4, a0, a1); s5 = x_vdot(s5, a0, a1); s6 = x_vdot(s6, a0, a1); s7 = x_vdot(s7, a0, a1); }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3; out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7; }
uint32_t k_rsqrt_ext(uint32_t m) { return x_rsq(m, 0); }
uint32_t k_exp2_ext(int32_t x) { return x_exp2(x, 0); }
uint32_t k_sigm_ext(int32_t x) { return x_sigm(x, 0); }
uint32_t k_sumsq(const int32_t *x, int n) { return i_sumsq(x, n); }
uint32_t k_rsqrt_base(uint32_t m) { return i_rsqrt_base(m); }
uint32_t k_exp2_base(int32_t x) { return i_exp2_base(x); }
