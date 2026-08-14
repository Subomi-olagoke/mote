/* model.h — the transformer: its weights, its scratch memory, and its forward
 * pass. This is the whole engine's public surface.
 *
 * One binary runs two kinds of checkpoint. The legacy fp32 format (the
 * TinyStories .bin files) keeps every weight as a float. mote's own quantized
 * format (.mq) stores the big matrices as int8 with per-group scales; it starts
 * with a magic number so the loader can tell the two apart.
 */
#ifndef MOTE_MODEL_H
#define MOTE_MODEL_H

#include <stddef.h>
#include <sys/types.h>
#include "config.h"
#include "quant.h"

/* mote quantized format. The magic is the ASCII "mote" read as a little-endian
 * int32, which cannot collide with a legacy header (whose first int is `dim`).
 * v2 added optional QKV bias + rope_theta/eps in the header; v3 adds int4
 * weights (header slot 14 = bits). Writers stamp the lowest version that can
 * express the file, so int8 checkpoints stay readable by older runtimes. */
#define MQ_MAGIC   0x65746F6D
#define MQ_VERSION 3
#define MQ_HEADER  256          /* header padded to this, keeps weights aligned */

/* Weight tensors. In the fp32 path the `float *` fields are live and the
 * QTensors are empty; in the quantized path it is the other way around, except
 * the norm gains, which stay fp32 in both because they are small and sensitive. */
typedef struct {
    /* always fp32 */
    float *rms_att;           /* (layer, dim)  pre-attention norm gain  */
    float *rms_ffn;           /* (layer, dim)  pre-ffn norm gain        */
    float *rms_final;         /* (dim)         final norm gain          */

    /* optional fp32 attention biases (Qwen-style q/k/v have them; Llama does not).
     * NULL when the checkpoint carries no bias. bq is (layer, dim); bk and bv are
     * (layer, kv_dim). Applied right after the q/k/v projections, before RoPE. */
    float *bq, *bk, *bv;

    /* fp32 path */
    float *token_embedding;   /* (vocab, dim)                           */
    float *wq, *wk, *wv, *wo; /* attention                              */
    float *w1, *w2, *w3;      /* ffn                                    */
    float *wcls;              /* (vocab, dim)  classifier               */

    /* quantized path (same tensors, int8 + scales) */
    QTensor q_tokens;
    QTensor q_wq, q_wk, q_wv, q_wo;
    QTensor q_w1, q_w2, q_w3;
    QTensor q_wcls;
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
    QTensor xq;    /* scratch: the current activation, quantized   */
} RunState;

typedef struct {
    Config  config;
    Weights weights;
    RunState state;
    int    quantized;  /* 0 = fp32 path, 1 = quantized path        */
    int    gs;         /* quantization group size (quantized only) */
    int    qbits;      /* weight width, 8 or 4 (quantized only)    */
    /* architecture knobs that used to be constants. Defaulted for legacy
     * checkpoints (Llama/TinyStories), read from the header for .mq v2. */
    float  rope_theta; /* RoPE base frequency (Llama 10000, Qwen 1000000) */
    float  rms_eps;    /* RMSNorm epsilon (Llama 1e-5, Qwen 1e-6)         */
    int    has_qkv_bias; /* 1 if bq/bk/bv are present                     */
    int    fd;         /* file descriptor of the mmap'd checkpoint */
    void  *data;       /* mmap base                                */
    ssize_t file_size;
} Transformer;

/* Map the checkpoint (either format, auto-detected), wire up the weight
 * pointers, allocate scratch memory. Desktop convenience wrapper over the
 * portable loader below. */
void build_transformer(Transformer *t, const char *checkpoint_path);

/* Portable loader: bring the transformer up from a model blob already in
 * memory, a flash address on a microcontroller, the app bundle on iOS, or an
 * mmap on desktop. No files and no platform calls. The caller owns the blob and
 * must keep it alive for the transformer's lifetime. */
void build_transformer_from_blob(Transformer *t, const void *blob, size_t size);

/* Release scratch memory and unmap the checkpoint. */
void free_transformer(Transformer *t);

/* Run one token at position `pos`. Returns a pointer to the logits over the
 * vocabulary (owned by the transformer's run state, valid until the next call). */
float *forward(Transformer *t, int token, int pos);

#endif /* MOTE_MODEL_H */
