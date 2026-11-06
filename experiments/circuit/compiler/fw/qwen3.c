// Qwen3 (GQA, q/k-norm, RoPE rotate_half, RMSNorm, SwiGLU, tied head, greedy argmax) inference as a bare-metal RV32IM program.
// llama2.c-style, one token per forward(); KV cache in .bss.  Generated header cfg.h supplies dims, weight offsets and the prompt.
//   MODE_F32 : fp32 weights, soft-float (libgcc) -- the literal "llama2.c on RV32IM" T0 build
//   MODE_Q4V : big projections + lm_head use int4 WROM weights and the optional VDOT/WB extension (int8 activations, per-row
//              float scale); norms/softmax/rope/silu stay soft-float.
// Weights/tables live in a blob loaded at WBASE (on the real machine: DROM/WROM contents); see compiler/ISA.md for the map.
#include <stdint.h>
#include <math.h>
#include "cfg.h"
#define WBASE 0x10000000u
#define BF(off) ((const float *)(WBASE + (off)))
#define MARK() __asm__ volatile("li a7,2\n ecall" ::: "a7", "memory")
static void wr(void *p, int n) { register long a0 __asm__("a0") = 1, a1 __asm__("a1") = (long)p, a2 __asm__("a2") = n, a7 __asm__("a7") = 64; __asm__ volatile("ecall" :: "r"(a0), "r"(a1), "r"(a2), "r"(a7) : "memory"); }
void exit(int c) { register long a0 __asm__("a0") = c, a7 __asm__("a7") = 93; __asm__ volatile("ecall" :: "r"(a0), "r"(a7)); for (;;); }

#define KVD (NKV * HD)
#define QD (NH * HD)
static float x[DIM], xb[DIM], q[QD], kk[KVD], vv[KVD], ao[QD], hg[HID], hu[HID], sc[MAXPOS], logits[VOCAB];
static float kc[NL][MAXPOS][KVD], vc[NL][MAXPOS][KVD];

#ifdef MODE_Q4V
const int8_t *vk_x;
void vk_init_raw(int32_t *, int, int, const int32_t *, int32_t);
void vk_acc_raw(int32_t *, int, int, const int32_t *, int32_t);
static int8_t xq[((HID > DIM ? HID : DIM) > QD ? (HID > DIM ? HID : DIM) : QD) + 64] __attribute__((aligned(4)));
static int32_t acc[VOCAB > HID ? VOCAB : HID];
// out[r] = scale[r] * sx * (Wq_r . xq)
__attribute__((noinline)) static void matvec(float *out, uint32_t scale_off, uint32_t wb, const float *in, int R, int C) {
    float amax = 1e-30f;
    for (int j = 0; j < C; j++) { float a = in[j] < 0 ? -in[j] : in[j]; if (a > amax) amax = a; }
    float sx = amax / 127.f, inv = 127.f / amax;
    for (int j = 0; j < C; j++) { float t = in[j] * inv; xq[j] = (int8_t)(int)(t < 0 ? t - 0.5f : t + 0.5f); }
    for (int c = 0; c < C / 64; c++) {
        vk_x = xq + 64 * c;
        if (c == 0) vk_init_raw(acc, wb, R, 0, 0); else vk_acc_raw(acc, wb + c * R * 8, R, 0, 0);
    }
    const float *s = BF(scale_off);
    for (int r = 0; r < R; r++) out[r] = (float)acc[r] * s[r] * sx;
}
#define MV(out, l, m, in, R, C) matvec(out, SOFF[l][m], WBW[l][m], in, R, C)
#else
__attribute__((noinline)) static void matvec(float *out, uint32_t off, const float *in, int R, int C) {
    const float *w = BF(off);
    for (int r = 0; r < R; r++) { float s = 0; const float *wr_ = w + (uint32_t)r * C; for (int j = 0; j < C; j++) s += wr_[j] * in[j]; out[r] = s; }
}
#define MV(out, l, m, in, R, C) matvec(out, SOFF[l][m], in, R, C)
#endif

__attribute__((noinline)) static void rmsnorm(float *o, const float *in, const float *w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += in[i] * in[i];
    float r = 1.f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) o[i] = in[i] * r * w[i];
}
__attribute__((noinline)) static void rope(float *v, int pos) {
    const float *cs = BF(ROPE_OFF) + pos * (HD / 2), *sn = BF(ROPE_OFF + 4u * MAXPOS * (HD / 2)) + pos * (HD / 2);
    for (int i = 0; i < HD / 2; i++) { float a = v[i], b = v[i + HD / 2]; v[i] = a * cs[i] - b * sn[i]; v[i + HD / 2] = b * cs[i] + a * sn[i]; }
}

__attribute__((noinline)) static void attn(int l, int pos) {
        for (int h = 0; h < NH; h++) {
            const float *qh = q + h * HD; int g = h / (NH / NKV); float mx = -1e30f;
            for (int t = 0; t <= pos; t++) { float s = 0; const float *k = kc[l][t] + g * HD; for (int j = 0; j < HD; j++) s += qh[j] * k[j]; s *= ATT_SCALE; sc[t] = s; if (s > mx) mx = s; }
            float sum = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
            float inv = 1.f / sum; float *o = ao + h * HD; for (int j = 0; j < HD; j++) o[j] = 0;
            for (int t = 0; t <= pos; t++) { float p = sc[t] * inv; const float *v = vc[l][t] + g * HD; for (int j = 0; j < HD; j++) o[j] += p * v[j]; }
        }
}
__attribute__((noinline)) static void swiglu(void) { for (int i = 0; i < HID; i++) hg[i] = hg[i] / (1.f + expf(-hg[i])) * hu[i]; }
__attribute__((noinline)) static int argmax(void) { int best = 0; for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i; return best; }

static int forward(int tok, int pos) {
    const float *emb = BF(EMB_OFF) + (uint32_t)tok * DIM;
    for (int i = 0; i < DIM; i++) x[i] = emb[i];
    for (int l = 0; l < NL; l++) {
        rmsnorm(xb, x, BF(NORM_OFF[l][0]), DIM);
        MV(q, l, 0, xb, QD, DIM); MV(kk, l, 1, xb, KVD, DIM); MV(vv, l, 2, xb, KVD, DIM);
        for (int h = 0; h < NH; h++) { rmsnorm(q + h * HD, q + h * HD, BF(NORM_OFF[l][1]), HD); rope(q + h * HD, pos); }
        for (int h = 0; h < NKV; h++) { rmsnorm(kk + h * HD, kk + h * HD, BF(NORM_OFF[l][2]), HD); rope(kk + h * HD, pos); }
        for (int i = 0; i < KVD; i++) { kc[l][pos][i] = kk[i]; vc[l][pos][i] = vv[i]; }
        attn(l, pos);
        MV(xb, l, 3, ao, DIM, QD);
        for (int i = 0; i < DIM; i++) x[i] += xb[i];
        rmsnorm(xb, x, BF(NORM_OFF[l][3]), DIM);
        MV(hg, l, 4, xb, HID, DIM); MV(hu, l, 5, xb, HID, DIM);
        swiglu();
        MV(xb, l, 6, hg, DIM, HID);
        for (int i = 0; i < DIM; i++) x[i] += xb[i];
    }
    rmsnorm(xb, x, BF(FINAL_NORM_OFF), DIM);
#ifdef MODE_Q4V
    matvec(logits, LM_SOFF, LM_WB, xb, VOCAB, DIM);
#else
    matvec(logits, EMB_OFF, xb, VOCAB, DIM);
#endif
    return argmax();
}
int main(void) {
    static const int prompt[] = PROMPT;
    int nxt = 0;
    for (int pos = 0; pos < NTOK; pos++) {
        MARK(); nxt = forward(prompt[pos], pos);
        int32_t t = nxt; wr(&t, 4);
#ifdef DUMP
        wr(logits, 4 * VOCAB);
#endif
    }
    MARK();
    return 0;
}
