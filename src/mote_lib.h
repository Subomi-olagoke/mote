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
    unsigned long long seed;    /* 0 = leave the current seed          */
    int   max_tokens;           /* 0 = the model's max sequence length */
} mote_params;

/* Create an engine from a model blob and a tokenizer blob already in memory.
 * The caller may free both blobs after this returns... except the model, which
 * must stay alive for the engine's lifetime (its weights are used in place).
 * Returns NULL on failure. */
mote *mote_create(const void *model, size_t model_len,
                  const void *tokenizer, size_t tokenizer_len);

/* Generate a continuation of `prompt`. For each decoded piece of text, calls
 * on_token(piece, user). Returns the number of tokens generated. */
int mote_generate(mote *m, const char *prompt, const mote_params *params,
                  void (*on_token)(const char *piece, void *user), void *user);

void mote_free(mote *m);

#ifdef __cplusplus
}
#endif

#endif /* MOTE_LIB_H */
