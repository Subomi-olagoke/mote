/* forward.c — one token through the network.
 *
 * This is the heart of mote and the whole reason the repo exists: the Llama
 * forward pass, written out by hand with no library doing the heavy lifting.
 * Each block is pre-norm: RMSNorm, then attention, add back to the residual;
 * RMSNorm again, then a SwiGLU feed-forward, add back. Positions are encoded
 * with rotary embeddings applied straight onto the query and key vectors, and a
 * KV cache means each new token only attends over history instead of redoing it.
 *
 * There is exactly one loop that matters for speed, matmul, so it is the only
 * thing marked for optional parallelism.
 */
#include "model.h"

#include <math.h>
#include <string.h>

/* Root-mean-square norm: scale x to unit RMS, then apply the learned gain. */
static void rmsnorm(float *out, const float *x, const float *weight, int size)
{
    float ss = 0.0f;
    for (int i = 0; i < size; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int i = 0; i < size; i++)
        out[i] = weight[i] * (ss * x[i]);
}

/* In-place softmax over the first `size` entries. */
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

/* out(d) = W(d, n) . x(n). The one hot loop; everything else is cheap. */
static void matmul(float *out, const float *x, const float *w, int n, int d)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float *row = w + (long)i * n;
        for (int j = 0; j < n; j++)
            val += row[j] * x[j];
        out[i] = val;
    }
}

float *forward(Transformer *t, int token, int pos)
{
    const Config *p = &t->config;
    const Weights *w = &t->weights;
    RunState *s = &t->state;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;   /* query heads per kv head */
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;
    float *x = s->x;

    /* seed the residual stream with this token's embedding */
    memcpy(x, w->token_embedding + (long)token * dim, dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        /* ---- attention ---- */
        rmsnorm(s->xb, x, w->rms_att + (long)l * dim, dim);

        long loff = (long)l * p->seq_len * kv_dim;   /* this layer's cache */
        float *k = s->key_cache + loff + (long)pos * kv_dim;
        float *v = s->value_cache + loff + (long)pos * kv_dim;

        matmul(s->q, s->xb, w->wq + (long)l * dim * dim, dim, dim);
        matmul(k, s->xb, w->wk + (long)l * dim * kv_dim, dim, kv_dim);
        matmul(v, s->xb, w->wv + (long)l * dim * kv_dim, dim, kv_dim);

        /* rotary position embedding, applied to q and (within kv range) k */
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
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

        /* per-head scaled dot-product attention over the cache up to `pos` */
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

        matmul(s->xb2, s->xb, w->wo + (long)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++)
            x[i] += s->xb2[i];

        /* ---- feed-forward: SwiGLU, w2( silu(w1 x) * (w3 x) ) ---- */
        rmsnorm(s->xb, x, w->rms_ffn + (long)l * dim, dim);
        matmul(s->hb, s->xb, w->w1 + (long)l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (long)l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));   /* SiLU */
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + (long)l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++)
            x[i] += s->xb[i];
    }

    rmsnorm(x, x, w->rms_final, dim);
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}
