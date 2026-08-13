/* quant.c — the arithmetic behind quant.h. See there for the scheme. */
#include "quant.h"
#include "parallel.h"

#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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

/* int32 dot product of two int8 groups. Uses the NEON dot-product instruction
 * where it exists (every recent Apple chip), otherwise a plain loop. The result
 * is bit-identical either way, since it is integer arithmetic. */
static inline int32_t dot_group_i8(const int8_t *a, const int8_t *b, int gs)
{
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int k = 0;
    for (; k + 16 <= gs; k += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + k), vld1q_s8(b + k));
    int32_t sum = vaddvq_s32(acc);
    for (; k < gs; k++)                       /* tail when gs is not a mult of 16 */
        sum += (int32_t)a[k] * (int32_t)b[k];
    return sum;
#else
    int32_t sum = 0;
    for (int k = 0; k < gs; k++)
        sum += (int32_t)a[k] * (int32_t)b[k];
    return sum;
#endif
}

typedef struct {
    float *out;
    const QTensor *x, *w;
    long woff;
    int n, gs;
} MatmulQ8;

static void matmul_q8_range(int begin, int end, void *vctx)
{
    const MatmulQ8 *c = (const MatmulQ8 *)vctx;
    for (int i = begin; i < end; i++) {
        long in0 = c->woff + (long)i * c->n;      /* row i of the weight tensor */
        float val = 0.0f;
        for (int j = 0; j < c->n; j += c->gs) {
            int32_t sum = dot_group_i8(c->x->q + j, c->w->q + in0 + j, c->gs);
            val += (float)sum * c->x->s[j / c->gs] * c->w->s[(in0 + j) / c->gs];
        }
        c->out[i] = val;
    }
}

/* Below this much work, even chunked thread dispatch costs more than it saves;
 * run the whole thing serial (still vectorized). */
#define PARALLEL_WORK_MIN (1 << 18)

void matmul_q8(float *out, const QTensor *x, const QTensor *w,
               long woff, int n, int d, int gs)
{
    MatmulQ8 ctx = { out, x, w, woff, n, gs };
    if ((long)d * n >= PARALLEL_WORK_MIN)
        mote_parallel_for(d, matmul_q8_range, &ctx);
    else
        matmul_q8_range(0, d, &ctx);
}
