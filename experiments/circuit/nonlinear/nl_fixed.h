/* nl_fixed.h : drop-in integer-only nonlinear kernels for the Qwen3 RV32IM firmware (no FPU, no soft-float).
   Build modes (pick one):  default = plain RV32IM;  -DNL_EXT = use the custom-1 instructions (gcc/clang inline `.insn`);
                            -DNL_EXT_EMU = same semantics in C (nl_ext_ref.h) for host tests / emulators without the new opcodes.
   FORMATS  Qn = signed/unsigned integer v representing v / 2^n.
     activations x, gains g, norm output : int32 Q12     (any int32; 64-bit accumulation)
     mean-square / eps argument m         : uint32 Q16   (m = (sumsq of Q8 values)/n + 1 lsb; RSQ valid for value in [2^-8, 2^15))
     rsqrt result                         : uint32 Q24
     softmax scores s                     : int32 Q16, BASE-2 DOMAIN (fold log2(e)/sqrt(hd) into the q weights); output probs int32 Q16
     SwiGLU gate g, up u, result          : int32 Q16 (|g| < 2^19 raw ~ 8 saturates sigmoid cleanly up to |g| < 12)
     RoPE x pairs                         : packed int16 pairs (x2<<16 | x1&0xFFFF) ; cos/sin : packed Q15 (sin<<16 | cos&0xFFFF) ; output packed int16
     argmax                               : int32 logits, first maximum wins (== torch.argmax tie rule on CPU)
   ERROR BOUNDS (vs fp64 reference, measured in test_kernels.py / test_real.py):
     nl_rsqrt_q24   rel 5.6e-6 (EXT) / 3.7e-5 (soft)       nl_exp2_q24 rel 5.1e-6 (EXT) / 8.3e-6 (soft), abs <= 14 / 32 LSB of Q24
     nl_sigmoid_q16 abs 2.6e-5 (EXT)                       nl_softmax prob abs 1.5e-5            nl_rot16 abs 1.3 LSB int16 (cos/sin Q15 rounding)
   All functions are `static inline` (no relocations, no libc, no calls out). */
#ifndef NL_FIXED_H
#define NL_FIXED_H
#include <stdint.h>
#define NL_UNR _Pragma("GCC unroll 8")
#define NL_MULH(a, b) ((int32_t)(((int64_t)(a) * (int64_t)(b)) >> 32))
#define NL_MULHU(a, b) ((uint32_t)(((uint64_t)(a) * (uint64_t)(b)) >> 32))
#define NL_INL static inline __attribute__((always_inline))

#if defined(NL_EXT)
#define NL_R(name, f3) NL_INL uint32_t name(uint32_t a, uint32_t b) { uint32_t r; __asm__ volatile(".insn r 0x2b, " #f3 ", 0, %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); return r; }
NL_R(nl_exp2_op, 0) NL_R(nl_rsq_op, 1) NL_R(nl_sigm_op, 2) NL_R(nl_rot_op, 5)
NL_INL void nl_amx(uint32_t v, uint32_t i) { __asm__ volatile(".insn r 0x2b, 3, 0, x0, %0, %1" :: "r"(v), "r"(i)); }
NL_INL uint32_t nl_amxr(uint32_t rst) { uint32_t r; __asm__ volatile(".insn r 0x2b, 4, 0, %0, %1, x0" : "=r"(r) : "r"(rst)); return r; }
#define NL_HAVE_EXT 1
#elif defined(NL_EXT_EMU)
#include "nl_ext_ref.h"
static nl_amx_state nl_amx_st = { INT32_MIN, 0 };
NL_INL uint32_t nl_exp2_op(uint32_t a, uint32_t b) { (void)b; return nl_ref_exp2(a); }
NL_INL uint32_t nl_rsq_op(uint32_t a, uint32_t b) { (void)b; return nl_ref_rsq(a); }
NL_INL uint32_t nl_sigm_op(uint32_t a, uint32_t b) { (void)b; return nl_ref_sigm(a); }
NL_INL uint32_t nl_rot_op(uint32_t a, uint32_t b) { return nl_ref_rot16(a, b); }
NL_INL void nl_amx(uint32_t v, uint32_t i) { nl_ref_amx(&nl_amx_st, v, i); }
NL_INL uint32_t nl_amxr(uint32_t rst) { return nl_ref_amxr(&nl_amx_st, rst); }
#define NL_HAVE_EXT 1
#endif

/* (a*b)>>s for int32 a,b, 0<s<32, exact via mul+mulh (5 instr) */
NL_INL int32_t nl_mulshr(int32_t a, int32_t b, int s) { uint32_t lo = (uint32_t)a * (uint32_t)b; int32_t hi = NL_MULH(a, b); return (int32_t)((lo >> s) | ((uint32_t)hi << (32 - s))); }

NL_INL uint32_t nl_umulshr(uint32_t a, uint32_t b, int s) { uint32_t lo = a * b, hi = NL_MULHU(a, b); return (lo >> s) | (hi << (32 - s)); }

/* ---- RMSNorm ---------------------------------------------------------------------------------------------------------------- */
/* mean square of n = 2^log2n values x (Q12) -> uint32 Q16 (saturating): exact 64-bit accumulation (mul/mulh), no overflow for any int32 x.
   Add the RMSNorm eps as +1 (= 2^-16, larger than HF's 1e-6: negligible unless mean-square < 1e-2) before nl_rsqrt_q24. */
NL_INL uint32_t nl_meansq_q16(const int32_t *x, int n, int log2n) {
    uint32_t lo = 0, hi = 0;
    NL_UNR for (int i = 0; i < n; i++) { int32_t v = x[i]; uint32_t l = (uint32_t)v * (uint32_t)v; uint32_t h = (uint32_t)NL_MULH(v, v); lo += l; hi += h + (lo < l); }
    int sh = 8 + log2n;                                               /* Q24 sum -> Q16 mean */
    if (hi >> sh) return 0xFFFFFFFFu;
    return (hi << (32 - sh)) | (lo >> sh); }
NL_INL int32_t nl_mq28(int32_t a, int32_t b) { return NL_MULH(a, b) << 4; }
NL_INL uint32_t nl_rsqrt_q24(uint32_t m) {                           /* m Q16 > 0 -> m^-1/2 Q24 */
#ifdef NL_HAVE_EXT
    return nl_rsq_op(m, 0);
#else
    int e = 0; uint32_t t = m;
    if (t >> 16) { t >>= 16; e += 16; } if (t >> 8) { t >>= 8; e += 8; } if (t >> 4) { t >>= 4; e += 4; } if (t >> 2) { t >>= 2; e += 2; } if (t >> 1) { e += 1; }
    int E = e - 16, par = E & 1, Ee = E - par;
    int32_t Mp = (e <= 28 ? (int32_t)(m << (28 - e)) : (int32_t)(m >> (e - 28))) << par;     /* M' in [1,4) Q28 */
    int32_t y = 0x12AAAAAB - NL_MULH(Mp, 0x2AAAAAAB);                                       /* chord 7/6 - M'/6 */
    for (int it = 0; it < 3; it++) { int32_t y2 = nl_mq28(y, y); int32_t t3 = 0x30000000 - nl_mq28(Mp, y2); y = nl_mq28(y, t3) >> 1; }
    int sh = 4 + Ee / 2; return sh >= 0 ? (uint32_t)y >> sh : (uint32_t)y << -sh;
#endif
}
NL_INL uint32_t nl_rms_inv_q24(const int32_t *x, int n, int log2n) { uint32_t m = nl_meansq_q16(x, n, log2n); m += (m != 0xFFFFFFFFu); return nl_rsqrt_q24(m); }
/* o[i] = x[i] * rsqrt * g[i];  y = nl_rsqrt_q24(mean_square_q16 + eps).  x,g,o Q12 */
NL_INL void nl_rms_scale(const int32_t *x, const int32_t *g, int32_t *o, int n, uint32_t y) {
    NL_UNR for (int i = 0; i < n; i++) { int32_t t = nl_mulshr(x[i], (int32_t)y, 24); o[i] = nl_mulshr(t, g[i], 12); } }

/* ---- exp2 / softmax -------------------------------------------------------------------------------------------------------- */
NL_INL uint32_t nl_exp2_q24(int32_t x) {                             /* 2^(x/2^16) Q24; saturates to 0xFFFFFFFF for x >= 8.0 */
#ifdef NL_HAVE_EXT
    return nl_exp2_op((uint32_t)x, 0);
#else
    int32_t xi = x >> 16; uint32_t f = (uint32_t)x & 0xFFFF;
    if (xi >= 8) return 0xFFFFFFFFu;
    int32_t f30 = (int32_t)(f << 14);
    int32_t p = 0x00dff96f;
    p = NL_MULH(p, f30) * 4 + 0x034fca40; p = NL_MULH(p, f30) * 4 + 0x0f76721f; p = NL_MULH(p, f30) * 4 + 0x2c59ab19; p = NL_MULH(p, f30) * 4 + 0x40000ea6;
    int sh = 24 - 30 + xi; uint32_t m = (uint32_t)p; if (p < 0) m = 0;
    if (sh >= 0) return m << sh; if (-sh >= 32) return 0; return m >> -sh;
#endif
}
/* softmax over T scores s (Q16, base-2 domain) -> o (Q16); T <= 2048 (sum in uint32) */
NL_INL void nl_softmax_q16(const int32_t *s, int32_t *o, int T) {
    int32_t mx = s[0]; NL_UNR for (int i = 1; i < T; i++) if (s[i] > mx) mx = s[i];
    uint32_t sum = 0;
    NL_UNR for (int i = 0; i < T; i++) { uint32_t p = nl_exp2_q24(s[i] - mx) >> 4; o[i] = (int32_t)p; sum += p; }
    uint32_t r = (1u << 30) / (sum >> 10);                                /* ~1/sum, Q20, error ~1e-3 */
    uint32_t t = nl_umulshr(sum, r, 20); r = nl_umulshr(r, (2u << 20) - t, 20);   /* one Newton step: error ~1e-6 */
    NL_UNR for (int i = 0; i < T; i++) o[i] = NL_MULH(o[i] << 8, (int32_t)r); }

/* ---- SiLU / SwiGLU -------------------------------------------------------------------------------------------------------- */
NL_INL uint32_t nl_sigmoid_q16(int32_t x) {                          /* sigmoid(x/2^16) * 2^16 */
#ifdef NL_HAVE_EXT
    return nl_sigm_op((uint32_t)x, 0);
#else
    int32_t m = x >> 31; int32_t ax = (x ^ m) - m;
    int32_t t = NL_MULH(-ax << 2, 0x5C551D95);
    uint32_t e = nl_exp2_q24(t); uint32_t sp = (1u << 31) / (((1u << 24) + e) >> 9);
    return x >= 0 ? sp : NL_MULHU(e << 4, sp << 4);
#endif
}
NL_INL void nl_swiglu_q16(const int32_t *g, const int32_t *u, int32_t *o, int n) {      /* o = silu(g) * u, all Q16 */
    NL_UNR for (int i = 0; i < n; i++) { int32_t x = g[i]; uint32_t sg = nl_sigmoid_q16(x); int32_t sx = nl_mulshr(x, (int32_t)sg, 16); o[i] = nl_mulshr(sx, u[i], 16); } }

/* ---- RoPE ---------------------------------------------------------------------------------------------------------------- */
/* pairs stored interleaved (x1 = q[i], x2 = q[i+hd/2]) by permuting the q/k projection rows: free at weight-compile time */
NL_INL void nl_rope_pairs(const uint32_t *xp, const uint32_t *cs, uint32_t *yp, int n) {
#ifdef NL_HAVE_EXT
    NL_UNR for (int i = 0; i < n; i++) yp[i] = nl_rot_op(xp[i], cs[i]);
#else
    NL_UNR for (int i = 0; i < n; i++) { uint32_t a = xp[i], b = cs[i];
        int32_t x1 = (int16_t)a, x2 = (int16_t)(a >> 16), c = (int16_t)b, s = (int16_t)(b >> 16);
        int32_t u = (x1 * c - x2 * s + 0x4000) >> 15, v = (x1 * s + x2 * c + 0x4000) >> 15;
        yp[i] = ((uint32_t)(v & 0xFFFF) << 16) | (uint32_t)(u & 0xFFFF); }       /* (no int16 saturation in soft path: |x|<2^15/1.42 assumed) */
#endif
}
/* cos/sin of an angle given in turns (Q32 fraction of a full turn); tab = int16 sin table of 256+1+64 entries: tab[i] = round(32767 sin(2 pi i/256)) */
NL_INL uint32_t nl_sincos_q15(uint32_t turns, const int16_t *tab) {
    uint32_t i = turns >> 24, fr = (turns >> 8) & 0xFFFF; int32_t s0 = tab[i], s1 = tab[i + 1]; int32_t s = s0 + (((s1 - s0) * (int32_t)fr) >> 16);
    uint32_t t2 = turns + 0x40000000u; uint32_t j = t2 >> 24, fr2 = (t2 >> 8) & 0xFFFF; int32_t c0 = tab[j], c1 = tab[j + 1]; int32_t c = c0 + (((c1 - c0) * (int32_t)fr2) >> 16);
    return ((uint32_t)s << 16) | ((uint32_t)c & 0xFFFF); }

/* ---- glue --------------------------------------------------------------------------------------------------------------- */
NL_INL void nl_add(const int32_t *a, const int32_t *b, int32_t *o, int n) { NL_UNR for (int i = 0; i < n; i++) o[i] = a[i] + b[i]; }
NL_INL void nl_requant_i8(const int32_t *acc, int8_t *o, int n, int32_t scale_q31) {
    NL_UNR for (int i = 0; i < n; i++) { int32_t v = (NL_MULH(acc[i], scale_q31) + 1) >> 1; v = v > 127 ? 127 : v; v = v < -127 ? -127 : v; o[i] = (int8_t)v; } }
/* argmax: call nl_argmax_begin(), then nl_argmax_feed(logit, index) for every logit (inside the lm_head row epilogue), then nl_argmax_end() */
#ifdef NL_HAVE_EXT
NL_INL void nl_argmax_begin(void) { (void)nl_amxr(1); }
NL_INL void nl_argmax_feed(int32_t v, uint32_t i) { nl_amx((uint32_t)v, i); }
NL_INL uint32_t nl_argmax_end(void) { return nl_amxr(1); }
#else
typedef struct { int32_t v; uint32_t i; } nl_argmax_t;
#define nl_argmax_begin() nl_argmax_t nl_am = { INT32_MIN, 0 }
#define nl_argmax_feed(val, idx) do { if ((int32_t)(val) > nl_am.v) { nl_am.v = (int32_t)(val); nl_am.i = (idx); } } while (0)
#define nl_argmax_end() (nl_am.i)
#endif
#endif
