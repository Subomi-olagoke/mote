/* model.h — the transformer: its weights, its scratch memory, and its forward
 * pass. This is the whole engine's public surface.
 */
#ifndef MOTE_MODEL_H
#define MOTE_MODEL_H

#include <sys/types.h>
#include "config.h"

/* Pointers into the memory-mapped checkpoint. Nothing here is owned or copied;
 * every field just names a region of the mapped file. */
typedef struct {
    float *token_embedding;   /* (vocab, dim)                          */
    float *rms_att;           /* (layer, dim)   pre-attention norm gain */
    float *rms_ffn;           /* (layer, dim)   pre-ffn norm gain       */
    float *wq;                /* (layer, dim, n_heads*head_size)        */
    float *wk;                /* (layer, dim, n_kv_heads*head_size)     */
    float *wv;                /* (layer, dim, n_kv_heads*head_size)     */
    float *wo;                /* (layer, n_heads*head_size, dim)        */
    float *w1;                /* (layer, hidden_dim, dim)  gate         */
    float *w2;                /* (layer, dim, hidden_dim)  down         */
    float *w3;                /* (layer, hidden_dim, dim)  up           */
    float *rms_final;         /* (dim)   final norm gain                */
    float *wcls;              /* (vocab, dim)  classifier; may alias emb */
} Weights;

/* Per-step working memory. Allocated once, reused every token. */
typedef struct {
    float *x;      /* (dim)        residual stream                 */
    float *xb;     /* (dim)        a normed/temp copy of x         */
    float *xb2;    /* (dim)        second temp                     */
    float *hb;     /* (hidden_dim) ffn buffer                      */
    float *hb2;    /* (hidden_dim) ffn buffer                      */
    float *q;      /* (dim)        query                           */
    float *att;    /* (n_heads, seq_len) attention scores          */
    float *logits; /* (vocab)      output distribution             */
    float *key_cache;   /* (layer, seq_len, kv_dim)                */
    float *value_cache; /* (layer, seq_len, kv_dim)                */
} RunState;

typedef struct {
    Config  config;
    Weights weights;
    RunState state;
    int    fd;         /* file descriptor of the mmap'd checkpoint */
    float *data;       /* mmap base                                */
    ssize_t file_size;
} Transformer;

/* Map the checkpoint, wire up the weight pointers, allocate scratch memory. */
void build_transformer(Transformer *t, const char *checkpoint_path);

/* Release scratch memory and unmap the checkpoint. */
void free_transformer(Transformer *t);

/* Run one token at position `pos`. Returns a pointer to the logits over the
 * vocabulary (owned by the transformer's run state, valid until the next call). */
float *forward(Transformer *t, int token, int pos);

#endif /* MOTE_MODEL_H */
