/* model.c — load a checkpoint and stand up the transformer.
 *
 * Two formats, one loader. The first int32 of the file decides: mote's magic
 * means a quantized .mq checkpoint, anything else is a legacy fp32 .bin (whose
 * first int is `dim`). Both are memory-mapped; the weight pointers, float or
 * QTensor, are just offsets into that one mapping.
 */
#include "model.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* ---- fp32 layout ---- */
static void map_weights_fp32(Weights *w, const Config *p, float *ptr, int shared)
{
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding = ptr; ptr += (unsigned long long)p->vocab_size * p->dim;
    w->rms_att = ptr;         ptr += n_layers * p->dim;
    w->wq = ptr;              ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;              ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;              ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;              ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn = ptr;         ptr += n_layers * p->dim;
    w->w1 = ptr;              ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;              ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;              ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final = ptr;       ptr += p->dim;
    ptr += p->seq_len * head_size / 2;   /* skip old RoPE tables */
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared ? w->token_embedding : ptr;
}

/* ---- quantized layout ---- */
static char *map_qtensor(QTensor *qt, char *p, long size, int gs, int bits)
{
    qt->q = (int8_t *)p;
    p += bits == 4 ? size / 2 : size;   /* int4 packs two weights per byte */
    qt->s = (float *)p;
    p += (size / gs) * (long)sizeof(float);
    return p;
}

static void map_weights_quant(Weights *w, const Config *p, int gs, int bits,
                              int shared, int has_qkv_bias, char *base)
{
    long L = p->n_layers, D = p->dim, HD = p->hidden_dim;
    int head_size = p->dim / p->n_heads;
    long kv = (long)(p->n_kv_heads * head_size);

    /* norms stay fp32, first in the payload */
    w->rms_att = (float *)base;   base += L * D * sizeof(float);
    w->rms_ffn = (float *)base;   base += L * D * sizeof(float);
    w->rms_final = (float *)base; base += D * sizeof(float);

    /* optional fp32 attention biases, right after the norms */
    if (has_qkv_bias) {
        w->bq = (float *)base; base += L * D * sizeof(float);
        w->bk = (float *)base; base += L * kv * sizeof(float);
        w->bv = (float *)base; base += L * kv * sizeof(float);
    } else {
        w->bq = w->bk = w->bv = NULL;
    }

    base = map_qtensor(&w->q_tokens, base, (long)p->vocab_size * D, gs, bits);
    base = map_qtensor(&w->q_wq, base, L * D * D, gs, bits);
    base = map_qtensor(&w->q_wk, base, L * D * kv, gs, bits);
    base = map_qtensor(&w->q_wv, base, L * D * kv, gs, bits);
    base = map_qtensor(&w->q_wo, base, L * D * D, gs, bits);
    base = map_qtensor(&w->q_w1, base, L * D * HD, gs, bits);
    base = map_qtensor(&w->q_w2, base, L * HD * D, gs, bits);
    base = map_qtensor(&w->q_w3, base, L * D * HD, gs, bits);
    if (shared)
        w->q_wcls = w->q_tokens;
    else
        map_qtensor(&w->q_wcls, base, (long)p->vocab_size * D, gs, bits);
}

static void alloc_state(RunState *s, const Config *p, int quantized, int gs)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x   = calloc(p->dim, sizeof(float));
    s->xb  = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb  = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q   = calloc(p->dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache   = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));

    /* scratch to hold whichever activation feeds a matmul, quantized. Sized to
     * the widest activation (hidden_dim >= dim). */
    if (quantized) {
        int w = p->hidden_dim > p->dim ? p->hidden_dim : p->dim;
        s->xq.q = calloc(w, sizeof(int8_t));
        s->xq.s = calloc(w / gs, sizeof(float));
    } else {
        s->xq.q = NULL;
        s->xq.s = NULL;
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->att ||
        !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "mote: out of memory allocating run state\n");
        exit(1);
    }
}

static void free_state(RunState *s)
{
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2); free(s->q);
    free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->xq.q); free(s->xq.s);
}

static void map_file(Transformer *t, const char *path)
{
    t->fd = open(path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "mote: open() failed on %s\n", path); exit(1); }
    off_t sz = lseek(t->fd, 0, SEEK_END);
    t->file_size = sz;
    t->data = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mote: mmap failed\n"); exit(1); }
}

/* Portable core: parse a model blob already in memory and wire up the
 * transformer. No files, no mmap, no platform calls. This is exactly what runs
 * on a microcontroller (blob in flash) or inside an iOS app (blob in the
 * bundle); the desktop path just mmaps a file and calls in here. */
static void init_from_blob(Transformer *t, const uint8_t *base)
{
    int32_t magic;
    memcpy(&magic, base, sizeof(magic));

    if (magic == MQ_MAGIC) {
        t->quantized = 1;
        const int32_t *h = (const int32_t *)base;
        /* h[0]=magic, h[1]=version, h[2..8]=Config, h[9]=gs, h[10]=shared,
         * h[11]=has_qkv_bias, h[12]=rope_theta bits, h[13]=rms_eps bits (v2),
         * h[14]=weight bits (v3; 0 means int8). */
        if (h[1] < 1 || h[1] > MQ_VERSION) {
            fprintf(stderr, "mote: unsupported .mq version %d\n", h[1]); exit(1);
        }
        t->config = *(const Config *)(h + 2);
        t->gs = h[9];
        int shared = h[10];
        t->has_qkv_bias = (h[1] >= 2) ? h[11] : 0;
        t->qbits = (h[1] >= 3 && h[14]) ? h[14] : 8;
        if (t->qbits != 8 && t->qbits != 4) {
            fprintf(stderr, "mote: unsupported weight width %d bits\n", t->qbits); exit(1);
        }
        /* rope_theta / eps live in the header as float bits; 0 means "unset",
         * so a v1 file (or a writer that left them blank) gets Llama defaults. */
        float rope_theta, rms_eps;
        memcpy(&rope_theta, &h[12], sizeof(float));
        memcpy(&rms_eps, &h[13], sizeof(float));
        t->rope_theta = (rope_theta > 0.0f) ? rope_theta : 10000.0f;
        t->rms_eps    = (rms_eps > 0.0f)    ? rms_eps    : 1e-5f;
        if (t->gs <= 0 || (t->config.dim % t->gs) || (t->config.hidden_dim % t->gs)
            || (t->qbits == 4 && t->gs % 2)) {
            fprintf(stderr, "mote: bad group size %d for this model\n", t->gs); exit(1);
        }
        map_weights_quant(&t->weights, &t->config, t->gs, t->qbits, shared,
                          t->has_qkv_bias, (char *)base + MQ_HEADER);
    } else {
        t->quantized = 0;
        t->gs = 0;
        t->qbits = 0;
        t->config = *(const Config *)base;
        int shared = t->config.vocab_size > 0;
        t->config.vocab_size = abs(t->config.vocab_size);
        t->has_qkv_bias = 0;
        t->rope_theta = 10000.0f;
        t->rms_eps = 1e-5f;
        float *weights_ptr = (float *)base + sizeof(Config) / sizeof(float);
        map_weights_fp32(&t->weights, &t->config, weights_ptr, shared);
    }

    alloc_state(&t->state, &t->config, t->quantized, t->gs);
}

void build_transformer_from_blob(Transformer *t, const void *blob, size_t size)
{
    /* the caller owns the blob; we neither map nor free it */
    t->data = NULL;
    t->fd = -1;
    t->file_size = (ssize_t)size;
    init_from_blob(t, (const uint8_t *)blob);
}

void build_transformer(Transformer *t, const char *path)
{
    /* desktop convenience: mmap the checkpoint, then hand the bytes to the
     * portable core. free_transformer unmaps because we own t->data and t->fd. */
    map_file(t, path);
    init_from_blob(t, (const uint8_t *)t->data);
}

void free_transformer(Transformer *t)
{
    if (t->data != MAP_FAILED && t->data != NULL)
        munmap(t->data, t->file_size);
    if (t->fd != -1)
        close(t->fd);
    free_state(&t->state);
}
