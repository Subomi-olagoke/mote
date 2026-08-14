/* forward.c — one token through the network.
 *
 * This is the heart of mote: the Llama forward pass, written out by hand. Each
 * block is pre-norm: RMSNorm, attention, add to the residual; RMSNorm, a SwiGLU
 * feed-forward, add. Rotary embeddings go straight onto the query and key
 * vectors, and a KV cache means each new token only attends over history.
 *
 * Norms, attention, and the activations stay fp32 in both paths. Only the eight
 * big matmuls change: quantized checkpoints quantize the activation on the fly
 * and run integer arithmetic (int8 x int8, or int8 x int4 for the packed
 * format), which is where all the size and, on a chip, all the speed come from.
 */
#include "model.h"
#include "parallel.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++)
        out[i] = weight[i] * (ss * x[i]);
}

static void softmax(float *x, int size)
{
    float max = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

static inline float dot_f32(const float *a, const float *b, int n)
{
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(acc);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

typedef struct { float *out; const float *x, *w; int n; } MatmulF32;

static void matmul_f32_range(int begin, int end, void *vctx)
{
    const MatmulF32 *c = (const MatmulF32 *)vctx;
    for (int i = begin; i < end; i++)
        c->out[i] = dot_f32(c->x, c->w + (long)i * c->n, c->n);
}

static void matmul_f32(float *out, const float *x, const float *w, int n, int d)
{
    MatmulF32 ctx = { out, x, w, n };
    /* only thread when the work is worth the dispatch overhead (see quant.c) */
    if ((long)d * n >= (1 << 18))
        mote_parallel_for(d, matmul_f32_range, &ctx);
    else
        matmul_f32_range(0, d, &ctx);
}

/* out(d) = W(d, n) . in(n), taking the right path for this checkpoint. For fp32,
 * `wf` is the weight base and `off` selects this layer's row 0; for quantized,
 * the activation is quantized into scratch and int8 matmul runs at `off`. */
static void mm(Transformer *t, float *out, float *in, int n, int d,
               const float *wf, const QTensor *wq, long off)
{
    if (t->quantized) {
        quantize(&t->state.xq, in, n, t->gs);
        if (t->qbits == 4)
            matmul_q4(out, &t->state.xq, wq, off, n, d, t->gs);
        else
            matmul_q8(out, &t->state.xq, wq, off, n, d, t->gs);
    } else {
        matmul_f32(out, in, wf + off, n, d);
    }
}

float *forward(Transformer *t, int token, int pos)
{
    const Config *p = &t->config;
    const Weights *w = &t->weights;
    RunState *s = &t->state;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;
    float *x = s->x;

    /* seed the residual stream with this token's embedding */
    if (t->quantized) {
        long qoff = (long)token * dim;   /* element offset of this token's row */
        QTensor row = { .q = w->q_tokens.q + (t->qbits == 4 ? qoff / 2 : qoff),
                        .s = w->q_tokens.s + qoff / t->gs };
        if (t->qbits == 4)
            dequantize_q4(&row, x, dim, t->gs);
        else
            dequantize(&row, x, dim, t->gs);
    } else {
        memcpy(x, w->token_embedding + (long)token * dim, dim * sizeof(float));
    }

    for (int l = 0; l < p->n_layers; l++) {
        /* ---- attention ---- */
        rmsnorm(s->xb, x, w->rms_att + (long)l * dim, dim, t->rms_eps);

        long loff = (long)l * p->seq_len * kv_dim;
        float *k = s->key_cache + loff + (long)pos * kv_dim;
        float *v = s->value_cache + loff + (long)pos * kv_dim;

        mm(t, s->q, s->xb, dim, dim,     w->wq, &w->q_wq, (long)l * dim * dim);
        mm(t, k,    s->xb, dim, kv_dim,  w->wk, &w->q_wk, (long)l * dim * kv_dim);
        mm(t, v,    s->xb, dim, kv_dim,  w->wv, &w->q_wv, (long)l * dim * kv_dim);

        /* Qwen-style attention bias, added to the projections before RoPE. The
         * converter permutes bq/bk to match mote's interleaved rotary layout. */
        if (t->has_qkv_bias) {
            const float *bq = w->bq + (long)l * dim;
            const float *bk = w->bk + (long)l * kv_dim;
            const float *bv = w->bv + (long)l * kv_dim;
            for (int i = 0; i < dim; i++)    s->q[i] += bq[i];
            for (int i = 0; i < kv_dim; i++) { k[i] += bk[i]; v[i] += bv[i]; }
        }

        /* rotary position embedding on q and (within kv range) k */
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(t->rope_theta, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int r = 0; r < rotn; r++) {
                float *vec = r == 0 ? s->q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

#pragma omp parallel for schedule(static)
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int tt = 0; tt <= pos; tt++) {
                const float *kt = s->key_cache + loff + (long)tt * kv_dim
                                + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                    score += q[i] * kt[i];
                att[tt] = score / sqrtf((float)head_size);
            }
            softmax(att, pos + 1);

            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int tt = 0; tt <= pos; tt++) {
                const float *vt = s->value_cache + loff + (long)tt * kv_dim
                                + (h / kv_mul) * head_size;
                float a = att[tt];
                for (int i = 0; i < head_size; i++)
                    xb[i] += a * vt[i];
            }
        }

        mm(t, s->xb2, s->xb, dim, dim, w->wo, &w->q_wo, (long)l * dim * dim);
        for (int i = 0; i < dim; i++)
            x[i] += s->xb2[i];

        /* ---- feed-forward: SwiGLU, w2( silu(w1 x) * (w3 x) ) ---- */
        rmsnorm(s->xb, x, w->rms_ffn + (long)l * dim, dim, t->rms_eps);
        mm(t, s->hb,  s->xb, dim, hidden_dim, w->w1, &w->q_w1, (long)l * dim * hidden_dim);
        mm(t, s->hb2, s->xb, dim, hidden_dim, w->w3, &w->q_w3, (long)l * dim * hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));   /* SiLU */
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        mm(t, s->xb, s->hb, hidden_dim, dim, w->w2, &w->q_w2, (long)l * hidden_dim * dim);
        for (int i = 0; i < dim; i++)
            x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final, dim, t->rms_eps);
    mm(t, s->logits, x, dim, p->vocab_size, w->wcls, &w->q_wcls, 0);
    return s->logits;
}
