// Qwen3 inference, integer-only fixed point, bare-metal RV32IM (+optional VDOT/WB/SMAQA, +optional custom-1 EXP2/RSQ/SIGM/AMX/ROT16).
// Matvecs: int4 (1 plane) or int8 (2 int4 planes) weights in WROM via VDOT, dynamic int8 activations; attention: int8 K/V cache with SMAQA.
// Formats: residual x Q16 int32; norm outputs Q12; q/k/v matvec outputs Q12; gate/up/o/down/logits Q16.  See compiler/ISA.md.
#include <stdint.h>
#include "cfg.h"
typedef int64_t i64; typedef uint64_t u64;
#define WBASE 0x10000000u
#define BI(off) ((const int32_t *)(WBASE + (off)))
#ifdef NOMARK
#define MARK() ((void)0)
#else
#define MARK() __asm__ volatile("li a7,2\n ecall" ::: "a7", "memory")
#endif
#define NOINL __attribute__((noinline)) static
#ifdef PUTCHAR
// Qwenputer ABI: ecall a7=1 putchar(a0), one byte per ecall
static void wr(void *p, int n) { const uint8_t *b = (const uint8_t *)p; for (int i = 0; i < n; i++) { register long a0 __asm__("a0") = b[i], a7 __asm__("a7") = 1; __asm__ volatile("ecall" :: "r"(a0), "r"(a7) : "memory"); } }
#else
static void wr(void *p, int n) { register long a0 __asm__("a0") = 1, a1 __asm__("a1") = (long)p, a2 __asm__("a2") = n, a7 __asm__("a7") = 64; __asm__ volatile("ecall" :: "r"(a0), "r"(a1), "r"(a2), "r"(a7) : "memory"); }
#endif
void exit(int c) { register long a0 __asm__("a0") = c, a7 __asm__("a7") = 93; __asm__ volatile("ecall" :: "r"(a0), "r"(a7)); for (;;); }
#define SMAQA(acc, a, b) __asm__ volatile(".insn r 0x0B, 7, 0, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b))
#ifdef EXT
#define EXT_R(name, f3) static inline uint32_t name(uint32_t a) { uint32_t r; __asm__ volatile(".insn r 0x2b, " #f3 ", 0, %0, %1, x0" : "=r"(r) : "r"(a)); return r; }
EXT_R(x_exp2, 0) EXT_R(x_rsq, 1) EXT_R(x_sigm, 2)
static inline uint32_t x_rot(uint32_t a, uint32_t b) { uint32_t r; __asm__ volatile(".insn r 0x2b, 5, 0, %0, %1, %2" : "=r"(r) : "r"(a), "r"(b)); return r; }
static inline void x_amx(uint32_t v, uint32_t i) { __asm__ volatile(".insn r 0x2b, 3, 0, x0, %0, %1" :: "r"(v), "r"(i)); }
static inline uint32_t x_amxr(uint32_t rst) { uint32_t r; __asm__ volatile(".insn r 0x2b, 4, 0, %0, %1, x0" : "=r"(r) : "r"(rst)); return r; }
#endif
#define MULH(a, b) ((int32_t)(((i64)(a) * (i64)(b)) >> 32))
#define MULHU(a, b) ((uint32_t)(((u64)(a) * (u64)(b)) >> 32))
#define QD (NH * HD)
#define KVD (NKV * HD)
#define MPAD ((MAXPOS + 3) & ~3)

// ================= RAM-minimal layout (ramown): everything RW lives in a handful of buffers, tiled matvecs, fused epilogues =================
// TILE rows per matvec call (multiple of 4).  PMAX = max int4 planes of any matrix (w4:1, w8:2, w12:3).
#ifndef TILE
#define TILE 256
#endif
#ifndef PMAX
#define PMAX LM_PL
#endif
#if APL != 1
#error "ramown firmware supports APL=1 only"
#endif
#define SINGLE (MAXPOS == 1)
#if SINGLE && NTOK != 1
#error "SINGLE needs NTOK==1"
#endif
#define REP (NH / NKV)
// x: residual (persistent).  xb: norm output.  xqi: int8 input of every DIM-wide matvec.  hg: swiglu result, later reused in place as the int8 input of o_proj / down_proj.
// v32: V (SINGLE: reused in place as the attention output).  accs_/hut: one tile of accumulators / of the "up" projection.
#ifdef TWOPASS     // gate/up/swiglu computed twice (pass 1: amax of the swiglu output, pass 2: quantise straight into an int8 buffer): no 12 KB int32 hg buffer, +2/3 of the MLP VDOT work
static int32_t x[DIM], xb[DIM], v32[KVD], hg[TILE];
static int8_t xqb_[HID] __attribute__((aligned(4)));
#define XQB xqb_
#else
static int32_t x[DIM], xb[DIM], hg[HID], v32[KVD];
#define XQB ((int8_t *)hg)
#endif
static int8_t xqi[DIM + 64] __attribute__((aligned(4)));
static int32_t accs_[APL * PMAX][TILE], hut[TILE];
#if !SINGLE
static int32_t q32[QD], k32[KVD], ao[QD];
static int16_t qh[QD] __attribute__((aligned(4))), kh[KVD] __attribute__((aligned(4)));
static int8_t kq[NL][MAXPOS][KVD] __attribute__((aligned(4)));
static int32_t kmx[NL][MAXPOS][NKV], vmx[NL][MAXPOS][NKV];
static int8_t vt[NL][NKV][HD][MPAD] __attribute__((aligned(4)));
static int32_t sc[MPAD], wt[MPAD];
static int8_t w8[MPAD] __attribute__((aligned(4)));
static int8_t q8[HD] __attribute__((aligned(4)));
#endif

const int8_t *vk_x;
void vk_init_raw(int32_t *, int, int, const int32_t *, int32_t);
void vk_acc_raw(int32_t *, int, int, const int32_t *, int32_t);

// ---------------- scalar nonlinears (base = plain RV32IM fixed point; EXT = custom-1 instructions) ----------------
static inline uint32_t exp2_q(int32_t xx) {             // 2^(x/2^16) in Q24
#ifdef EXT
    return x_exp2((uint32_t)xx);
#else
    int32_t xi = xx >> 16; uint32_t f = (uint32_t)xx & 0xFFFF;
    if (xi >= 8) return 0xFFFFFFFF;
    int32_t f30 = (int32_t)(f << 14), p = 0x00dff96f;
    p = MULH(p, f30) * 4 + 0x034fca40; p = MULH(p, f30) * 4 + 0x0f76721f; p = MULH(p, f30) * 4 + 0x2c59ab19; p = MULH(p, f30) * 4 + 0x40000ea6;
    int sh = 24 - 30 + xi; uint32_t m = (uint32_t)p; if (p < 0) m = 0;
    if (sh >= 0) return m << sh; if (-sh >= 32) return 0; return m >> -sh;
#endif
}
static inline int32_t mq28(int32_t a, int32_t b) { return MULH(a, b) << 4; }
static inline uint32_t rsq_q(uint32_t m) {               // (m/2^16)^-1/2 in Q24, m>0
#ifdef EXT
    return x_rsq(m);
#else
    int e = 0; uint32_t t = m;
    if (t >> 16) { t >>= 16; e += 16; } if (t >> 8) { t >>= 8; e += 8; } if (t >> 4) { t >>= 4; e += 4; } if (t >> 2) { t >>= 2; e += 2; } if (t >> 1) { e += 1; }
    int E = e - 16, par = E & 1, Ee = E - par;
    int32_t Mp = (e <= 28 ? (int32_t)(m << (28 - e)) : (int32_t)(m >> (e - 28))) << par;
    int32_t y = 0x12AAAAAB - MULH(Mp, 0x2AAAAAAB);
    for (int it = 0; it < 3; it++) { int32_t y2 = mq28(y, y); int32_t t3 = 0x30000000 - mq28(Mp, y2); y = mq28(y, t3) >> 1; }
    int sh = 4 + Ee / 2; return sh >= 0 ? (uint32_t)y >> sh : (uint32_t)y << -sh;
#endif
}
static inline int32_t sigm_q(int32_t xx) {              // sigmoid(x/2^16)*2^16
#ifdef EXT
    return (int32_t)x_sigm((uint32_t)xx);
#else
    int32_t m = xx >> 31, ax = (xx ^ m) - m;
    int32_t t = MULH(-ax << 2, 0x5C551D95);
    uint32_t e = exp2_q(t);
    uint32_t sp = (1u << 31) / (((1u << 24) + e) >> 9);
    return xx >= 0 ? (int32_t)sp : (int32_t)MULHU(e << 4, sp << 4);
#endif
}

// ---------------- RMSNorm (Qx in, Q12 out, optional gain Q12) ----------------
NOINL void rms(int32_t *o, const int32_t *in, const int32_t *g, int n, int log2n, int Qx) {
    i64 ss = 0; for (int i = 0; i < n; i++) ss += (i64)in[i] * in[i];
    i64 mean = (ss >> log2n) + (Qx == 16 ? EPSQ16 : EPSQ12);
    int e = 63 - __builtin_clzll((u64)mean);
    int k = (2 * Qx + 1 - e) & ~1;
    i64 ms = k >= 0 ? mean << k : mean >> -k;
    uint32_t m16 = (uint32_t)(ms >> (2 * Qx - 16));
    i64 y = rsq_q(m16); y = k >= 0 ? y << (k / 2) : y >> (-k / 2);
    int sft = Qx + 24 - 12;
    if (g) for (int i = 0; i < n; i++) { i64 t = ((i64)in[i] * y) >> sft; o[i] = (int32_t)((t * g[i]) >> 12); }
    else for (int i = 0; i < n; i++) o[i] = (int32_t)(((i64)in[i] * y) >> sft);
}

// ---------------- dynamic int8 activation quantisation ----------------
NOINL int32_t quantize(const int32_t *v, int n, int8_t *dst) {      // dst may alias v (in place: byte i is written after word i>>2 <= i was read)
    int32_t amax = 1; for (int i = 0; i < n; i++) { int32_t a = v[i] < 0 ? -v[i] : v[i]; if (a > amax) amax = a; }
    uint32_t r = 0x7F000000u / (uint32_t)amax;
    for (int i = 0; i < n; i++) dst[i] = (int8_t)(((i64)v[i] * r + (1 << 23)) >> 24);
    return amax;
}

// ---------------- matvec: out[r] = W_r . x  (Qin -> OQ), W = int4 planes via VDOT ----------------
// out[0..t) (+)= rows r0..r0+t of W (R rows total, C cols) . xs ;  weights are chunk-major (64 cols) so a row tile is a contiguous run inside every chunk plane.
NOINL void mv_rows(int32_t *out, int add, const int8_t *xs, uint32_t soff, int wb, int P, int R, int r0, int t, int C, int32_t amax, int Qin, int OQ) {
    for (int p = 0; p < P; p++) {
        int wbp = wb + p * (C / 64) * R * 8 + r0 * 8; int32_t *ac = accs_[p];
        for (int c = 0; c < C / 64; c++) {
            vk_x = xs + 64 * c;
            if (c == 0) vk_init_raw(ac, wbp, t, 0, 0); else vk_acc_raw(ac, wbp + c * R * 8, t, 0, 0);
        }
    }
    int sh = 4 * P + 8 + 10 - 31; if (sh < 0) sh = 0;
    const int FB = 16, dv = 127;     // F carries FB extra fractional bits
    i64 F = OQ + FB >= Qin ? ((i64)amax << (OQ + FB - Qin)) / dv : ((i64)amax >> (Qin - OQ - FB)) / dv;
    F <<= sh;
    int kk = 0; while ((F >> kk) >= (1 << 30)) kk++;
    int32_t Fm = (int32_t)(F >> kk);
    const int32_t *s = BI(soff) + r0;
    for (int r = 0; r < t; r++) {
        i64 a64 = 0;
        for (int p = 0; p < P; p++) a64 += (i64)accs_[p][r] << (4 * (P - 1 - p));
        int32_t a = (int32_t)(a64 >> sh);
        int32_t sv = s[r], m = sv & ~31, e = sv & 31;
        int32_t tt = (int32_t)(((i64)a * m) >> 31);
        i64 v = (i64)tt * Fm; int sf = e + FB - kk;
        int32_t o = (int32_t)(sf >= 0 ? v >> sf : v << -sf);
        if (add) out[r] = (int32_t)((uint32_t)out[r] + (uint32_t)o); else out[r] = o;
    }
}
// whole matrix in tiles: out[0..R) (+)= W . xs
NOINL void mv_full(int32_t *out, int add, const int8_t *xs, uint32_t soff, int wb, int P, int R, int C, int32_t amax, int Qin, int OQ) {
    for (int r0 = 0; r0 < R; r0 += TILE) mv_rows(out + r0, add, xs, soff, wb, P, R, r0, R - r0 < TILE ? R - r0 : TILE, C, amax, Qin, OQ);
}
#define MVF(out, add, xs, l, m, R, C, am, Qin, OQ) mv_full(out, add, xs, SOFF[l][m], WBW[l][m], PLN[l][m], R, C, am, Qin, OQ)

// ---------------- RoPE on int16 Q10 pairs (i, i+HD/2) ----------------
NOINL void rope(int16_t *dst, const int32_t *src, int heads, int pos) {      // src Q12 int32 -> dst Q10 int16 rotated
    const uint32_t *cs = (const uint32_t *)(WBASE + ROPE_OFF) + pos * (HD / 2);
    for (int h = 0; h < heads; h++) {
        const int32_t *s = src + h * HD; int16_t *d = dst + h * HD;
        for (int i = 0; i < HD / 2; i++) {
            int32_t a = (s[i] + 2) >> 2, b = (s[i + HD / 2] + 2) >> 2;
            a = a > 32767 ? 32767 : a < -32768 ? -32768 : a; b = b > 32767 ? 32767 : b < -32768 ? -32768 : b;
#ifdef EXT
            uint32_t r = x_rot(((uint32_t)b << 16) | ((uint32_t)a & 0xFFFF), cs[i]);
            d[i] = (int16_t)r; d[i + HD / 2] = (int16_t)(r >> 16);
#else
            int32_t c = (int16_t)cs[i], sn = (int16_t)(cs[i] >> 16);
            int32_t u = (a * c - b * sn + 0x4000) >> 15, v = (a * sn + b * c + 0x4000) >> 15;
            u = u > 32767 ? 32767 : u < -32768 ? -32768 : u; v = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
            d[i] = (int16_t)u; d[i + HD / 2] = (int16_t)v;
#endif
        }
    }
}

#if !SINGLE
// ---------------- attention for one layer at position pos ----------------
NOINL void kv_store(int l, int pos) {
    for (int g = 0; g < NKV; g++) {
        const int16_t *k = kh + g * HD; const int32_t *v = v32 + g * HD;
        int32_t ka = 1, va = 1;
        for (int j = 0; j < HD; j++) { int32_t a = k[j] < 0 ? -k[j] : k[j]; if (a > ka) ka = a; a = v[j] < 0 ? -v[j] : v[j]; if (a > va) va = a; }
        uint32_t rk = 0x7F000000u / (uint32_t)ka, rv = 0x7F000000u / (uint32_t)va;
        for (int j = 0; j < HD; j++) {
            kq[l][pos][g * HD + j] = (int8_t)(((i64)k[j] * rk + (1 << 23)) >> 24);
            vt[l][g][j][pos] = (int8_t)(((i64)v[j] * rv + (1 << 23)) >> 24);
        }
        kmx[l][pos][g] = ka; vmx[l][pos][g] = va;
    }
}
NOINL void attn(int l, int pos) {
    int T = pos + 1, nw = (T + 3) >> 2;
    for (int h = 0; h < NH; h++) {
        int g = h / (NH / NKV);
        const int16_t *q = qh + h * HD;
        int32_t qa = 1; for (int j = 0; j < HD; j++) { int32_t a = q[j] < 0 ? -q[j] : q[j]; if (a > qa) qa = a; }
        uint32_t rq = 0x7F000000u / (uint32_t)qa;
        for (int j = 0; j < HD; j++) q8[j] = (int8_t)(((i64)q[j] * rq + (1 << 23)) >> 24);
        i64 Fq = ((i64)qa * CQ) >> 20;
        int32_t mx = -2147483647 - 1;
        for (int t = 0; t < T; t++) {
            const uint32_t *kw = (const uint32_t *)(kq[l][t] + g * HD), *qw = (const uint32_t *)q8;
            int32_t a = 0;
#pragma GCC unroll 8
            for (int j = 0; j < HD / 4; j++) SMAQA(a, qw[j], kw[j]);
            int32_t s = (int32_t)(((i64)((i64)a * kmx[l][t][g]) * Fq) >> 30);
            sc[t] = s; if (s > mx) mx = s;
        }
        uint32_t sum = 0; int32_t wm = 1;
        for (int t = 0; t < T; t++) {
            uint32_t p = exp2_q(sc[t] - mx) >> 4; sum += p;
            int32_t w = (int32_t)(((i64)p * vmx[l][t][g]) >> 10); wt[t] = w; if (w > wm) wm = w;
        }
        i64 rw = ((i64)127 << 40) / wm;
        for (int t = 0; t < T; t++) w8[t] = (int8_t)(((i64)wt[t] * rw + ((i64)1 << 39)) >> 40);
        for (int t = T; t < 4 * nw; t++) w8[t] = 0;
        i64 G = ((u64)wm << 34) / ((u64)16129 * sum);
        int32_t *o = ao + h * HD;
        for (int j = 0; j < HD; j++) {
            const uint32_t *vw = (const uint32_t *)vt[l][g][j], *pw = (const uint32_t *)w8;
            int32_t a = 0;
            for (int w = 0; w < nw; w++) SMAQA(a, pw[w], vw[w]);
            o[j] = (int32_t)(((i64)a * G) >> 24);
        }
    }
}

#else
// ctx == 1: softmax over one key is exactly 1, so q and k never influence the output and no K/V cache is needed.  The integer pipeline of the generic path
// (kv_store -> attn with T=1) reduces to the following per KV group, bit-identical:  o[j] = ((w8 * vq[j]) * G) >> 24, same for all NH/NKV heads of the group.
NOINL void attn_single(void) {
    const uint32_t p = exp2_q(0) >> 4;                  // sc[0]-mx == 0
    for (int g = 0; g < NKV; g++) {
        int32_t *v = v32 + g * HD; int32_t va = 1;
        for (int j = 0; j < HD; j++) { int32_t a = v[j] < 0 ? -v[j] : v[j]; if (a > va) va = a; }
        uint32_t rv = 0x7F000000u / (uint32_t)va;
        int32_t w = (int32_t)(((i64)p * va) >> 10), wm = 1; if (w > wm) wm = w;
        i64 rw = ((i64)127 << 40) / wm; int8_t w8s = (int8_t)(((i64)w * rw + ((i64)1 << 39)) >> 40);
        i64 G = ((u64)wm << 34) / ((u64)16129 * p);
        for (int j = 0; j < HD; j++) {
            int8_t vq = (int8_t)(((i64)v[j] * rv + (1 << 23)) >> 24); int32_t a = 0;
            SMAQA(a, (uint32_t)(uint8_t)w8s, (uint32_t)(uint8_t)vq);
            v[j] = (int32_t)(((i64)a * G) >> 24);
        }
    }
}
// = quantize(ao, QD) where ao[h*HD+j] = v32[(h/REP)*HD+j]
NOINL int32_t quantize_gqa(int8_t *dst) {
    int32_t amax = 1; for (int i = 0; i < KVD; i++) { int32_t a = v32[i] < 0 ? -v32[i] : v32[i]; if (a > amax) amax = a; }
    uint32_t r = 0x7F000000u / (uint32_t)amax;
    for (int h = 0; h < NH; h++) for (int j = 0; j < HD; j++) dst[h * HD + j] = (int8_t)(((i64)v32[(h / REP) * HD + j] * r + (1 << 23)) >> 24);
    return amax;
}
#endif

NOINL void swiglu(int32_t *g, const int32_t *u, int n) {      // g,u Q16 -> g = silu(g)*u Q12
    for (int i = 0; i < n; i++) { int32_t sg = sigm_q(g[i]); i64 sx = ((i64)g[i] * sg) >> 16; g[i] = (int32_t)((sx * u[i]) >> 20); }
}

static int forward(int tok, int pos) {
    const int32_t *emb = BI(EMB_OFF) + (uint32_t)tok * DIM;
    for (int i = 0; i < DIM; i++) x[i] = emb[i];
    for (int l = 0; l < NL; l++) {
        rms(xb, x, BI(NORM_OFF[l][0]), DIM, DLOG, 16);
        int32_t am = quantize(xb, DIM, xqi);
#if SINGLE
        MVF(v32, 0, xqi, l, 2, KVD, DIM, am, 12, 12);
        attn_single();
        am = quantize_gqa(XQB);
#else
        MVF(q32, 0, xqi, l, 0, QD, DIM, am, 12, 12); MVF(k32, 0, xqi, l, 1, KVD, DIM, am, 12, 12); MVF(v32, 0, xqi, l, 2, KVD, DIM, am, 12, 12);
        for (int h = 0; h < NH; h++) rms(q32 + h * HD, q32 + h * HD, BI(NORM_OFF[l][1]), HD, HLOG, 12);
        for (int h = 0; h < NKV; h++) rms(k32 + h * HD, k32 + h * HD, BI(NORM_OFF[l][2]), HD, HLOG, 12);
        rope(qh, q32, NH, pos); rope(kh, k32, NKV, pos);
        kv_store(l, pos);
        attn(l, pos);
        am = quantize(ao, QD, XQB);
#endif
        MVF(x, 1, XQB, l, 3, DIM, QD, am, 12, 16);                       // x += o_proj(attn)
        rms(xb, x, BI(NORM_OFF[l][3]), DIM, DLOG, 16);
        am = quantize(xb, DIM, xqi);
#ifdef TWOPASS
        int32_t mx = 1;
        for (int pass = 0; pass < 2; pass++) {
            uint32_t rq = pass ? 0x7F000000u / (uint32_t)mx : 0;
            for (int r0 = 0; r0 < HID; r0 += TILE) {
                int t = HID - r0 < TILE ? HID - r0 : TILE;
                mv_rows(hg, 0, xqi, SOFF[l][4], WBW[l][4], PLN[l][4], HID, r0, t, DIM, am, 12, 16);
                mv_rows(hut, 0, xqi, SOFF[l][5], WBW[l][5], PLN[l][5], HID, r0, t, DIM, am, 12, 16);
                swiglu(hg, hut, t);
                if (!pass) { for (int i = 0; i < t; i++) { int32_t a = hg[i] < 0 ? -hg[i] : hg[i]; if (a > mx) mx = a; } }
                else for (int i = 0; i < t; i++) XQB[r0 + i] = (int8_t)(((i64)hg[i] * rq + (1 << 23)) >> 24);
            }
        }
        am = mx;
#else
        for (int r0 = 0; r0 < HID; r0 += TILE) {                         // gate/up/swiglu per tile, result into hg
            int t = HID - r0 < TILE ? HID - r0 : TILE;
            mv_rows(hg + r0, 0, xqi, SOFF[l][4], WBW[l][4], PLN[l][4], HID, r0, t, DIM, am, 12, 16);
            mv_rows(hut, 0, xqi, SOFF[l][5], WBW[l][5], PLN[l][5], HID, r0, t, DIM, am, 12, 16);
            swiglu(hg + r0, hut, t);
        }
        am = quantize(hg, HID, XQB);                                     // in place
#endif
        MVF(x, 1, XQB, l, 6, DIM, HID, am, 12, 16);                      // x += down_proj
    }
    rms(xb, x, BI(FINAL_NORM_OFF), DIM, DLOG, 16);
    int32_t am = quantize(xb, DIM, xqi);
    // LM head in tiles with a running argmax (first maximum wins, as before); no VOCAB-sized buffer
#ifdef EXT
    x_amxr(1);
#else
    int32_t best = -2147483647 - 1; int bi = 0;
#endif
    for (int r0 = 0; r0 < VOCAB; r0 += TILE) {
        int t = VOCAB - r0 < TILE ? VOCAB - r0 : TILE;
        mv_rows(hut, 0, xqi, LM_SOFF, LM_WB, LM_PL, VOCAB, r0, t, DIM, am, 12, 16);
#ifdef DUMP
        wr(hut, 4 * t);
#endif
#ifdef EXT
        for (int i = 0; i < t; i++) x_amx(hut[i], r0 + i);
#else
        for (int i = 0; i < t; i++) if (hut[i] > best) { best = hut[i]; bi = r0 + i; }
#endif
    }
#ifdef EXT
    return x_amxr(1);
#else
    return bi;
#endif
}
int main(void) {
    static const int prompt[] = PROMPT;
    for (int pos = 0; pos < NTOK; pos++) {
        MARK(); int nxt = forward(prompt[pos], pos);
        int32_t t = nxt; wr(&t, 4);
    }
    MARK();
    return 0;
}
