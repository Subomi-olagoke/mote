/* model.c — load a checkpoint and stand up the transformer.
 *
 * The checkpoint is memory-mapped, not read into a buffer: the weights are the
 * overwhelming majority of the file, they never change, and mmap lets the OS
 * page them in on demand and share them across processes for free. All the
 * Weights pointers are just offsets into that one mapping.
 *
 * The on-disk layout is the compact format the TinyStories models ship in: a
 * seven-int Config header, then every weight tensor back to back as float32 in
 * a fixed order. A negative vocab_size in the header is the one flag it carries,
 * it means the output classifier is stored separately rather than tied to the
 * token embedding.
 */
#include "model.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

static void map_weights(Weights *w, const Config *p, float *ptr, int shared)
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
    /* Older exports parked the RoPE frequency tables here. We recompute those
     * on the fly, so skip the space to land on the classifier. */
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared ? w->token_embedding : ptr;
}

static void alloc_state(RunState *s, const Config *p)
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
}

void build_transformer(Transformer *t, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mote: couldn't open checkpoint %s\n", path); exit(1); }
    if (fread(&t->config, sizeof(Config), 1, f) != 1) {
        fprintf(stderr, "mote: couldn't read config header from %s\n", path);
        exit(1);
    }
    int shared = t->config.vocab_size > 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    fseek(f, 0, SEEK_END);
    t->file_size = ftell(f);
    fclose(f);

    t->fd = open(path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "mote: open() failed on %s\n", path); exit(1); }
    t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mote: mmap failed\n"); exit(1); }

    /* Config is 7 int32s == 7 float32s wide; weights begin right after it. */
    float *weights_ptr = t->data + sizeof(Config) / sizeof(float);
    map_weights(&t->weights, &t->config, weights_ptr, shared);

    alloc_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
    if (t->data != MAP_FAILED && t->data != NULL)
        munmap(t->data, t->file_size);
    if (t->fd != -1)
        close(t->fd);
    free_state(&t->state);
}
