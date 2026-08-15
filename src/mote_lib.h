/* mote_lib.h — mote as a library.
 *
 * The CLI is one way to drive the engine; this is the other. It bundles the
 * transformer, tokenizer, and sampler behind one opaque handle, loads them from
 * memory (so it works the same on a laptop, a phone, or a chip), and streams
 * generated text back through a callback instead of printing to a terminal.
 *
 * This is the surface an iOS app or an MCU firmware links against.
 */
#ifndef MOTE_LIB_H
#define MOTE_LIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mote mote;

typedef struct {
    float temperature;          /* 0 = greedy / deterministic          */
    float topp;                 /* nucleus sampling; <=0 or >=1 = off  */
    float repeat_penalty;       /* >1 discourages recently seen tokens
                                   (1.1 is gentle); <=1 = off. Small
                                   models loop verbatim without this,
                                   especially when greedy.             */
    unsigned long long seed;    /* 0 = leave the current seed          */
    int   max_tokens;           /* cap on NEW tokens generated after the
                                   prompt; 0 = up to the context limit  */
} mote_params;

/* A model's shape, for showing what is actually running. */
typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
    int quantized;              /* 1 if the model is weight-quantized  */
    int qbits;                  /* weight width when quantized: 8 or 4 */
} mote_info_t;

void mote_get_info(const mote *m, mote_info_t *out);

/* Create an engine from a model blob and a tokenizer blob already in memory.
 * The caller may free both blobs after this returns... except the model, which
 * must stay alive for the engine's lifetime (its weights are used in place).
 * Returns NULL on failure. */
mote *mote_create(const void *model, size_t model_len,
                  const void *tokenizer, size_t tokenizer_len);

/* Convenience: create from files on disk. On iOS the bundled model and
 * tokenizer have real paths, and this uses mmap, so there is no large copy and
 * no pointer-lifetime concern. (The from-blob path above is for flash on an
 * MCU, where there is no filesystem.) */
mote *mote_create_from_files(const char *model_path, const char *tokenizer_path);

/* Generate a continuation of `prompt`. For each decoded piece of text, calls
 * on_token(piece, user); return 0 from it to stop early, non-zero to continue.
 * Returns the number of tokens generated. */
int mote_generate(mote *m, const char *prompt, const mote_params *params,
                  int (*on_token)(const char *piece, void *user), void *user);

void mote_free(mote *m);

#ifdef __cplusplus
}
#endif

#endif /* MOTE_LIB_H */
