/* mote_lib.c — the library API. See mote_lib.h.
 *
 * This is the same generation loop the CLI runs, wrapped so a host application
 * drives it: replay the prompt tokens, then sample the rest, handing each
 * decoded piece to the caller's callback.
 */
#include "mote_lib.h"

#include "model.h"
#include "tokenizer.h"
#include "bpe.h"
#include "sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Two tokenizer schemes coexist: the original SentencePiece-style one (TinyStories
 * / Llama) and byte-level BPE (Qwen). A tokenizer file/blob starting with "MTOK" is
 * the BPE format; anything else is the legacy format. */
struct mote {
    Transformer t;
    int         is_bpe;
    Tokenizer   tk;      /* legacy path */
    BPETokenizer *bpe;   /* Qwen path   */
    Sampler     s;
};

static int blob_is_mtok(const void *b, size_t n)
{
    return n >= 4 && memcmp(b, "MTOK", 4) == 0;
}
static int file_is_mtok(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char m[4] = {0};
    size_t r = fread(m, 1, 4, f);
    fclose(f);
    return r == 4 && memcmp(m, "MTOK", 4) == 0;
}

mote *mote_create(const void *model, size_t model_len,
                  const void *tok, size_t tok_len)
{
    if (!model || !tok)
        return NULL;

    mote *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;

    build_transformer_from_blob(&m->t, model, model_len);
    if (blob_is_mtok(tok, tok_len)) {
        m->is_bpe = 1;
        m->bpe = bpe_load_from_blob(tok, tok_len, /*own=*/0);
    } else {
        build_tokenizer_from_blob(&m->tk, tok, tok_len, m->t.config.vocab_size);
    }
    /* defaults; overridden per call by mote_generate */
    build_sampler(&m->s, m->t.config.vocab_size, 1.0f, 0.9f, 1);
    return m;
}

mote *mote_create_from_files(const char *model_path, const char *tok_path)
{
    if (!model_path || !tok_path)
        return NULL;

    mote *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;

    build_transformer(&m->t, model_path);            /* mmaps the checkpoint */
    if (file_is_mtok(tok_path)) {
        m->is_bpe = 1;
        m->bpe = bpe_load(tok_path);
    } else {
        build_tokenizer(&m->tk, tok_path, m->t.config.vocab_size);
    }
    build_sampler(&m->s, m->t.config.vocab_size, 1.0f, 0.9f, 1);
    return m;
}

void mote_get_info(const mote *m, mote_info_t *out)
{
    if (!m || !out)
        return;
    const Config *c = &m->t.config;
    out->dim = c->dim;
    out->hidden_dim = c->hidden_dim;
    out->n_layers = c->n_layers;
    out->n_heads = c->n_heads;
    out->n_kv_heads = c->n_kv_heads;
    out->vocab_size = c->vocab_size;
    out->seq_len = c->seq_len;
    out->quantized = m->t.quantized;
}

int mote_generate(mote *m, const char *prompt, const mote_params *p,
                  int (*on_token)(const char *piece, void *user), void *user)
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

    /* ---- byte-level BPE path (Qwen): encode the prompt, replay it, then sample.
     * Only generated tokens are emitted (the prompt is not echoed), and generation
     * stops at the end-of-turn token. ---- */
    if (m->is_bpe) {
        int cap = (int)strlen(prompt) + 8;
        int *toks = malloc((size_t)cap * sizeof(int));
        if (!toks) return 0;
        int n_prompt = bpe_encode(m->bpe, prompt, toks, cap);
        if (n_prompt < 1) { free(toks); return 0; }
        int eos = bpe_eos(m->bpe);

        int token = toks[0], pos = 0, generated = 0;
        char piecebuf[64];
        while (pos < steps) {
            float *logits = forward(&m->t, token, pos);
            pos++;
            int next;
            if (pos < n_prompt) {
                next = toks[pos];            /* still replaying the prompt */
            } else {
                next = sample(&m->s, logits);
                if (next == eos) break;
                int len; const char *piece = bpe_decode(m->bpe, next, &len);
                if (len >= (int)sizeof(piecebuf)) len = sizeof(piecebuf) - 1;
                memcpy(piecebuf, piece, len); piecebuf[len] = '\0';
                generated++;
                if (on_token && !on_token(piecebuf, user)) break;
            }
            token = next;
        }
        free(toks);
        return generated;
    }

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
        token = next;
        generated++;
        if (piece && on_token && !on_token(piece, user))
            break;   /* caller asked to stop */
    }

    free(prompt_tokens);
    return generated;
}

void mote_free(mote *m)
{
    if (!m)
        return;
    free_sampler(&m->s);
    if (m->is_bpe)
        bpe_free(m->bpe);
    else
        free_tokenizer(&m->tk);
    free_transformer(&m->t);
    free(m);
}
