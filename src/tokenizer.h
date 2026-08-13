/* tokenizer.h — text <-> token ids, byte-pair style.
 *
 * The vocabulary and merge scores come from a companion .bin file. Encoding
 * starts from raw bytes and greedily merges the highest-scoring adjacent pair
 * that exists in the vocabulary, over and over, which reconstructs the same
 * subword tokens the model was trained on.
 */
#ifndef MOTE_TOKENIZER_H
#define MOTE_TOKENIZER_H

#include <stddef.h>

typedef struct {
    const char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;              /* id -> string                         */
    float *scores;             /* id -> merge score                    */
    TokenIndex *sorted;        /* string-sorted view, for lookup       */
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];   /* raw byte -> 1-char string      */
} Tokenizer;

void build_tokenizer(Tokenizer *tk, const char *path, int vocab_size);

/* Load the tokenizer from a blob already in memory (the app bundle on iOS,
 * flash on an MCU). The caller may free the blob afterwards; strings are copied. */
void build_tokenizer_from_blob(Tokenizer *tk, const void *blob, size_t size,
                               int vocab_size);

void free_tokenizer(Tokenizer *tk);

/* Decode one token to a printable piece, given the previous token for the
 * leading-space rule. Returns a pointer into the tokenizer, do not free. */
const char *decode(Tokenizer *tk, int prev_token, int token);

/* Encode text into token ids. `bos`/`eos` add the begin/end markers.
 * `tokens` must have room for at least strlen(text)+3 ids. */
void encode(Tokenizer *tk, const char *text, int bos, int eos,
            int *tokens, int *n_tokens);

#endif /* MOTE_TOKENIZER_H */
