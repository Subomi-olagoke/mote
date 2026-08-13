/* mote_lib.c — the library API. See mote_lib.h.
 *
 * This is the same generation loop the CLI runs, wrapped so a host application
 * drives it: replay the prompt tokens, then sample the rest, handing each
 * decoded piece to the caller's callback.
 */
#include "mote_lib.h"

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

#include <stdlib.h>
#include <string.h>

struct mote {
    Transformer t;
    Tokenizer   tk;
    Sampler     s;
};

mote *mote_create(const void *model, size_t model_len,
                  const void *tok, size_t tok_len)
{
    if (!model || !tok)
        return NULL;

    mote *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;

    build_transformer_from_blob(&m->t, model, model_len);
    build_tokenizer_from_blob(&m->tk, tok, tok_len, m->t.config.vocab_size);
    /* defaults; overridden per call by mote_generate */
    build_sampler(&m->s, m->t.config.vocab_size, 1.0f, 0.9f, 1);
    return m;
}

int mote_generate(mote *m, const char *prompt, const mote_params *p,
                  void (*on_token)(const char *piece, void *user), void *user)
{
    if (!m)
        return 0;
    if (!prompt)
        prompt = "";

    /* apply this call's sampling settings */
    m->s.temperature = p ? p->temperature : 1.0f;
    m->s.topp        = p ? p->topp : 0.9f;
    if (p && p->seed)
        m->s.rng = p->seed;

    int steps = (p && p->max_tokens > 0) ? p->max_tokens : m->t.config.seq_len;
    if (steps > m->t.config.seq_len)
        steps = m->t.config.seq_len;

    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    if (!prompt_tokens)
        return 0;
    int n_prompt = 0;
    encode(&m->tk, prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt);
    if (n_prompt < 1) {
        free(prompt_tokens);
        return 0;
    }

    int token = prompt_tokens[0];
    int pos = 0, generated = 0;
    while (pos < steps) {
        float *logits = forward(&m->t, token, pos);

        int next;
        if (pos < n_prompt - 1)
            next = prompt_tokens[pos + 1];
        else
            next = sample(&m->s, logits);
        pos++;

        if (next == 1)   /* BOS marks the end for these models */
            break;

        const char *piece = decode(&m->tk, token, next);
        if (piece && on_token)
            on_token(piece, user);
        token = next;
        generated++;
    }

    free(prompt_tokens);
    return generated;
}

void mote_free(mote *m)
{
    if (!m)
        return;
    free_sampler(&m->s);
    free_tokenizer(&m->tk);
    free_transformer(&m->t);
    free(m);
}
