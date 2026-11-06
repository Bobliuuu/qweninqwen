#include <stdio.h>
#include "nl_ext_ref.h"
int main(void) { FILE *f = fopen("ext_vectors.txt", "r"); unsigned long long x, e, s, m, r, a, b, o; int bad = 0, n = 0;
    while (fscanf(f, "%llu %llu %llu %llu %llu %llu %llu %llu", &x, &e, &s, &m, &r, &a, &b, &o) == 8) { n++;
        if (nl_ref_exp2(x) != e) bad++; if (nl_ref_sigm(x) != s) bad++; if (nl_ref_rsq(m) != r) bad++; if (nl_ref_rot16(a, b) != o) bad++; }
    printf("%d vectors, %d mismatches\n", n, bad); return bad != 0; }
