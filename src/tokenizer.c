/* tokenizer.c — the byte-pair encoder/decoder.
 *
 * Load: the .bin holds a max-token-length header, then for each id a score, a
 * length, and that many bytes of the token string. Encode: seed with single
 * bytes (falling back to raw <0xNN> byte tokens for anything not in the vocab),
 * then merge the best adjacent pair repeatedly. Decode: look up the string,
 * strip the sentinel leading space after a begin-of-sequence, and expand raw
 * byte tokens back to their byte.
 */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_tokens(const void *a, const void *b)
{
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *tk, const char *path, int vocab_size)
{
    tk->vocab_size = vocab_size;
    tk->vocab  = malloc(vocab_size * sizeof(char *));
    tk->scores = malloc(vocab_size * sizeof(float));
    tk->sorted = NULL;
    for (int i = 0; i < 256; i++) {
        tk->byte_pieces[i * 2] = (unsigned char)i;
        tk->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mote: couldn't open tokenizer %s\n", path); exit(1); }
    if (fread(&tk->max_token_length, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "mote: bad tokenizer header\n"); exit(1);
    }
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (fread(tk->scores + i, sizeof(float), 1, f) != 1 ||
            fread(&len, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "mote: truncated tokenizer\n"); exit(1);
        }
        tk->vocab[i] = malloc(len + 1);
        if (fread(tk->vocab[i], 1, len, f) != (size_t)len) {
            fprintf(stderr, "mote: truncated tokenizer string\n"); exit(1);
        }
        tk->vocab[i][len] = '\0';
    }
    fclose(f);
}

void free_tokenizer(Tokenizer *tk)
{
    for (int i = 0; i < tk->vocab_size; i++)
        free(tk->vocab[i]);
    free(tk->vocab);
    free(tk->scores);
    free(tk->sorted);
}

const char *decode(Tokenizer *tk, int prev_token, int token)
{
    const char *piece = tk->vocab[token];
    /* the first real token after BOS carries a sentinel leading space */
    if (prev_token == 1 && piece[0] == ' ')
        piece++;
    /* raw byte tokens look like "<0x0A>"; map them back to the byte */
    unsigned char byte;
    if (sscanf(piece, "<0x%02hhX>", &byte) == 1)
        piece = (const char *)tk->byte_pieces + byte * 2;
    return piece;
}

static int str_lookup(const char *str, TokenIndex *sorted, int n)
{
    TokenIndex key = { .str = str };
    TokenIndex *hit = bsearch(&key, sorted, n, sizeof(TokenIndex), compare_tokens);
    return hit ? hit->id : -1;
}

void encode(Tokenizer *tk, const char *text, int bos, int eos,
            int *tokens, int *n_tokens)
{
    if (!text) { fprintf(stderr, "mote: null text to encode\n"); exit(1); }

    if (!tk->sorted) {
        tk->sorted = malloc(tk->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < tk->vocab_size; i++)
            tk->sorted[i] = (TokenIndex){ .str = tk->vocab[i], .id = i };
        qsort(tk->sorted, tk->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    /* scratch big enough for a merged pair plus a UTF-8 run */
    char *buf = malloc(tk->max_token_length * 2 + 3);
    size_t len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    /* Llama prepends a space to non-empty input. */
    if (text[0] != '\0') {
        int space = str_lookup(" ", tk->sorted, tk->vocab_size);
        if (space != -1) tokens[(*n_tokens)++] = space;
    }

    /* first pass: one token per UTF-8 codepoint, byte-fallback if unknown */
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80)   /* not a continuation byte: start fresh */
            len = 0;
        buf[len++] = *c;
        buf[len] = '\0';
        if ((c[1] & 0xC0) == 0x80 && len < 4)
            continue;              /* more bytes of this codepoint to come */

        int id = str_lookup(buf, tk->sorted, tk->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < len; i++)   /* +3: byte tokens live after */
                tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3;
        }
        len = 0;
    }

    /* merge pass: repeatedly fuse the best-scoring adjacent pair */
    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            snprintf(buf, tk->max_token_length * 2 + 3, "%s%s",
                     tk->vocab[tokens[i]], tk->vocab[tokens[i + 1]]);
            int id = str_lookup(buf, tk->sorted, tk->vocab_size);
            if (id != -1 && tk->scores[id] > best_score) {
                best_score = tk->scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++)
            tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}
