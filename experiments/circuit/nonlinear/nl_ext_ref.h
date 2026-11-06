/* Bit-exact C reference semantics of the custom-1 instructions (mirror of ext.py).  Include this in an emulator (rvemu.c) to implement
   opcode 0x2b: f3=0 EXP2, 1 RSQ, 2 SIGM, 3 AMX (no rd), 4 AMXR, 5 ROT16.  Self-test: test_ext_c.c vs ext.py (gen_ext_vectors.py). */
#pragma once
#include <stdint.h>
#include "nl_tables.h"
static inline uint32_t nl_ref_exp2(uint32_t xu) {
    int32_t x = (int32_t)xu, xi = x >> 16; uint32_t f = xu & 0xFFFF, i = f >> 8, fr = f & 255;
    uint64_t m = ((uint64_t)NL_EXP_T[i] * (256 - fr) + (uint64_t)NL_EXP_T[i + 1] * fr) >> 8;
    int sh = 24 - 30 + xi; uint64_t v;
    if (sh >= 0) v = sh >= 33 ? 0xFFFFFFFFull : (m << sh); else v = (-sh < 40) ? (m >> -sh) : 0;
    return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)v; }
static inline uint32_t nl_ref_sigm(uint32_t xu) {
    int64_t x = (int32_t)xu; if (x < -(12 << 16)) x = -(12 << 16); if (x > (12 << 16) - 1) x = (12 << 16) - 1;
    uint32_t u = (uint32_t)(x + (12 << 16)), i = u >> 11, fr = u & 2047;
    int64_t d = (int64_t)NL_SIG_T[i + 1] - (int64_t)NL_SIG_T[i];
    return (uint32_t)((int64_t)NL_SIG_T[i] + ((d * (int64_t)fr) >> 11)); }
static inline uint32_t nl_ref_rsq(uint32_t m) {
    if (m == 0) return 0xFFFFFFFFu;
    int e = 31 - __builtin_clz(m), E = e - 16, par = E & 1, Ee = E - par;
    uint64_t M = ((uint64_t)m << 30) >> e;
    uint32_t u = (uint32_t)(M - (1ull << 30)), idx = u >> 21, fr = u & ((1u << 21) - 1);
    int64_t t0 = NL_RSQ_T[par * 513 + idx], t1 = NL_RSQ_T[par * 513 + idx + 1];
    int64_t y = t0 + (((t1 - t0) * (int64_t)fr) >> 21);          /* arithmetic shift */
    int sh = 24 - 30 - Ee / 2; uint64_t v = sh >= 0 ? ((uint64_t)y << sh) : ((uint64_t)y >> -sh);
    return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)v; }
static inline uint32_t nl_ref_rot16(uint32_t a, uint32_t b) {
    int32_t x1 = (int16_t)(a & 0xFFFF), x2 = (int16_t)(a >> 16), c = (int16_t)(b & 0xFFFF), s = (int16_t)(b >> 16);
    int32_t y1 = (x1 * c - x2 * s + (1 << 14)) >> 15, y2 = (x1 * s + x2 * c + (1 << 14)) >> 15;
    y1 = y1 > 32767 ? 32767 : (y1 < -32768 ? -32768 : y1); y2 = y2 > 32767 ? 32767 : (y2 < -32768 ? -32768 : y2);
    return ((uint32_t)(y2 & 0xFFFF) << 16) | (uint32_t)(y1 & 0xFFFF); }
typedef struct { int32_t maxv; uint32_t maxi; } nl_amx_state;      /* reset value: maxv = INT32_MIN, maxi = 0 */
static inline void nl_ref_amx(nl_amx_state *s, uint32_t v, uint32_t i) { if ((int32_t)v > s->maxv) { s->maxv = (int32_t)v; s->maxi = i; } }
static inline uint32_t nl_ref_amxr(nl_amx_state *s, uint32_t rst) { uint32_t r = s->maxi; if (rst) { s->maxv = INT32_MIN; s->maxi = 0; } return r; }
