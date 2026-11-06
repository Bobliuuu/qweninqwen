// Native (host) emulation of the VDOT/WB/SMAQA kernels, for fast firmware debugging before the RV32 build.
#pragma once
#include <stdio.h>
#include <stdint.h>
extern const uint8_t WROM_DATA[];
#define rv_putchar(c) putchar((unsigned char)(c))
#define SMAQA(acc, a, b) do { uint32_t a_ = (a), b_ = (b); for (int i_ = 0; i_ < 4; i_++) \
    (acc) += (int32_t)(int8_t)(a_ >> (8 * i_)) * (int32_t)(int8_t)(b_ >> (8 * i_)); } while (0)
extern const int8_t *vk_x;
static inline int32_t vk_mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }
static void vk_gen(int np, int init, int scaled, int32_t *out, int wb, int rows, const int32_t *S, int32_t F) {
    for (int r = 0; r < rows; r++) {
        int32_t acc = init ? 0 : out[r];
        const uint8_t *w = WROM_DATA + (size_t)(wb + r * np) * 8;
        for (int k = 0; k < 8 * np; k++) acc += (int32_t)vk_x[k] * ((int32_t)w[k] - 8);
        if (scaled) acc = vk_mulh((int32_t)((uint32_t)acc * (uint32_t)S[r]), F);
        out[r] = acc;
    }
}
#define KDEF(np, i, s, nm) void nm(int32_t *o, int wb, int rows, const int32_t *S, int32_t F) { vk_gen(np, i, s, o, wb, rows, S, F); }
// kernel definitions are provided at the end of q3.c under NATIVE
