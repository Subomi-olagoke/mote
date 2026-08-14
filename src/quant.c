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

/* ---- int4 ---- */

void quantize_q4(const QTensor *out, const float *x, long n, int gs)
{
    int half = gs / 2;
    long groups = n / gs;
    for (long g = 0; g < groups; g++) {
        const float *xg = x + g * gs;

        /* The scale is the group's largest-magnitude weight divided by -8,
         * keeping its sign, so that weight lands exactly on -8 and everything
         * else gets the finer step |max|/8 instead of |max|/7. Measured on
         * TinyStories this one trick is worth more than any clipping scheme
         * tried (2.642 vs 2.764 perplexity against plain [-7,7] rounding).
         *
         * On top of that, try a few slightly shrunken scales and keep the one
         * with the least round-trip error. The gain is marginal (2.640) but it
         * never hurt, and this runs at conversion time, never on the hot path. */
        float amax = 0.0f, m = 0.0f;
        for (int i = 0; i < gs; i++) {
            float a = fabsf(xg[i]);
            if (a > amax) { amax = a; m = xg[i]; }
        }

        float best_scale = 0.0f;
        float best_err = -1.0f;
        int8_t q[256];               /* gs is small; 256 is far above any real one */
        int8_t best_q[256];
        for (int c = 0; c <= 4; c++) {
            float scale = (m / -8.0f) * (1.0f - 0.05f * c);
            float err = 0.0f;
            for (int i = 0; i < gs; i++) {
                int v = scale != 0.0f ? (int)lroundf(xg[i] / scale) : 0;
                if (v > 7) v = 7;
                if (v < -8) v = -8;
                q[i] = (int8_t)v;
                float d = xg[i] - v * scale;
                err += d * d;
            }
            if (best_err < 0.0f || err < best_err) {
                best_err = err;
                best_scale = scale;
                for (int i = 0; i < gs; i++) best_q[i] = q[i];
            }
        }

        out->s[g] = best_scale;
        uint8_t *b = (uint8_t *)out->q + g * half;
        for (int k = 0; k < half; k++)
            b[k] = (uint8_t)((best_q[k] & 0x0F) | (best_q[k + half] << 4));
    }
}

/* sign-extend the two nibbles of one packed byte */
static inline int nib_lo(uint8_t b) { return (int)((int8_t)(uint8_t)(b << 4)) >> 4; }
static inline int nib_hi(uint8_t b) { return (int)(int8_t)b >> 4; }

void dequantize_q4(const QTensor *in, float *x, int n, int gs)
{
    int half = gs / 2;
    for (long g = 0; g < n / gs; g++) {
        const uint8_t *b = (const uint8_t *)in->q + g * half;
        float s = in->s[g];
        float *xg = x + g * gs;
        for (int k = 0; k < half; k++) {
            xg[k]        = nib_lo(b[k]) * s;
            xg[k + half] = nib_hi(b[k]) * s;
        }
    }
}

/* int32 dot of an int8 activation group against a packed int4 weight group.
 * The split-half packing pays off here: sign-extending the low and high nibbles
 * of a register yields w[0..15] and w[half..half+15] as two contiguous int8
 * vectors, each dotted against its slice of the activation. */
static inline int32_t dot_group_q4(const int8_t *a, const uint8_t *wb, int gs)
{
    int half = gs / 2;
    int32_t sum = 0;
    int k = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    for (; k + 16 <= half; k += 16) {
        int8x16_t b  = vld1q_s8((const int8_t *)wb + k);
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(b, 4), 4);
        int8x16_t hi = vshrq_n_s8(b, 4);
        acc = vdotq_s32(acc, vld1q_s8(a + k), lo);
        acc = vdotq_s32(acc, vld1q_s8(a + half + k), hi);
    }
    sum = vaddvq_s32(acc);
#endif
    for (; k < half; k++)                     /* tail, and the whole loop when scalar */
        sum += (int32_t)a[k] * nib_lo(wb[k]) + (int32_t)a[k + half] * nib_hi(wb[k]);
    return sum;
}

static void matmul_q4_range(int begin, int end, void *vctx)
{
    const MatmulQ8 *c = (const MatmulQ8 *)vctx;   /* same fields, int4 weights */
    int half = c->gs / 2;
    for (int i = begin; i < end; i++) {
        long in0 = c->woff + (long)i * c->n;      /* element offset of row i */
        const uint8_t *row = (const uint8_t *)c->w->q + in0 / 2;
        float val = 0.0f;
        for (int j = 0; j < c->n; j += c->gs) {
            int32_t sum = dot_group_q4(c->x->q + j, row + (long)(j / c->gs) * half, c->gs);
            val += (float)sum * c->x->s[j / c->gs] * c->w->s[(in0 + j) / c->gs];
        }
        c->out[i] = val;
    }
}

void matmul_q4(float *out, const QTensor *x, const QTensor *w,
               long woff, int n, int d, int gs)
{
    MatmulQ8 ctx = { out, x, w, woff, n, gs };
    if ((long)d * n >= PARALLEL_WORK_MIN)
        mote_parallel_for(d, matmul_q4_range, &ctx);
    else
        matmul_q4_range(0, d, &ctx);
}
