/* thin wrappers around nl_fixed.h, compiled twice (-DNL_EXT and plain) to verify the drop-in header in the interpreter */
#include "nl_fixed.h"
uint32_t t_rsqrt(uint32_t m) { return nl_rsqrt_q24(m); }
uint32_t t_exp2(int32_t x) { return nl_exp2_q24(x); }
uint32_t t_sigmoid(int32_t x) { return nl_sigmoid_q16(x); }
void t_softmax(const int32_t *s, int32_t *o, int T) { nl_softmax_q16(s, o, T); }
void t_swiglu(const int32_t *g, const int32_t *u, int32_t *o, int n) { nl_swiglu_q16(g, u, o, n); }
void t_rope(const uint32_t *xp, const uint32_t *cs, uint32_t *yp, int n) { nl_rope_pairs(xp, cs, yp, n); }
uint32_t t_meansq(const int32_t *x, int n, int l2) { return nl_meansq_q16(x, n, l2); }
void t_rms_scale(const int32_t *x, const int32_t *g, int32_t *o, int n, uint32_t y) { nl_rms_scale(x, g, o, n, y); }
uint32_t t_rms_inv(const int32_t *x, int n, int l2) { return nl_rms_inv_q24(x, n, l2); }
int t_argmax(const int32_t *a, int n) { nl_argmax_begin(); for (int i = 0; i < n; i++) nl_argmax_feed(a[i], i); return nl_argmax_end(); }
