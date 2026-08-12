/* quant.c — the arithmetic behind quant.h. See there for the scheme. */
#include "quant.h"

#include <math.h>

#define QMAX 127.0f

void quantize(const QTensor *out, const float *x, int n, int gs)
{
    int groups = n / gs;
    for (int g = 0; g < groups; g++) {
        const float *xg = x + g * gs;

        float wmax = 0.0f;
        for (int i = 0; i < gs; i++) {
            float a = fabsf(xg[i]);
            if (a > wmax) wmax = a;
        }

        float scale = wmax / QMAX;
        out->s[g] = scale;

        int8_t *qg = out->q + g * gs;
        for (int i = 0; i < gs; i++) {
            float v = scale > 0.0f ? xg[i] / scale : 0.0f;
            /* roundf then clamp; the clamp only bites on floating-point slop */
            int q = (int)lroundf(v);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qg[i] = (int8_t)q;
        }
    }
}

void dequantize(const QTensor *in, float *x, int n, int gs)
{
    for (int i = 0; i < n; i++)
        x[i] = in->q[i] * in->s[i / gs];
}

void matmul_q8(float *out, const QTensor *x, const QTensor *w,
               long woff, int n, int d, int gs)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; i++) {
        long in0 = woff + (long)i * n;          /* row i of the weight tensor */
        float val = 0.0f;
        for (int j = 0; j < n; j += gs) {
            int32_t acc = 0;
            for (int k = 0; k < gs; k++)
                acc += (int32_t)x->q[j + k] * (int32_t)w->q[in0 + j + k];
            val += (float)acc * x->s[j / gs] * w->s[(in0 + j) / gs];
        }
        out[i] = val;
    }
}
