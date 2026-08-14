/* bpe.h — a byte-level BPE tokenizer (GPT-2 / Qwen family).
 *
 * mote's original tokenizer is SentencePiece-style (merge-by-score over a small
 * vocab). Modern instruct models like Qwen use a different scheme: byte-level BPE
 * over a large vocab, with a regex pre-tokenizer and explicit special tokens. This
 * is that tokenizer, loaded from the compact .mtok file tools/convert_tokenizer.py
 * produces. It is a separate module so the two schemes can coexist.
 *
 * Input is assumed NFC-normalized (what iOS text fields and normal typing produce);
 * the reference tokenizer NFC-normalizes first, which we do not, so decomposed input
 * would tokenize differently. This matches the reference exactly on NFC text.
 */
#ifndef MOTE_BPE_H
#define MOTE_BPE_H

#include <stddef.h>

typedef struct BPETokenizer BPETokenizer;

BPETokenizer *bpe_load(const char *path);
BPETokenizer *bpe_load_from_blob(const void *blob, size_t size, int own);

/* Encode UTF-8 text into token ids (special tokens like <|im_start|> are matched
 * as whole tokens). Writes up to `max` ids, returns the count. */
int bpe_encode(const BPETokenizer *t, const char *text, int *out, int max);

/* Raw bytes for one token id (not NUL-terminated; length in *out_len). Bytes may
 * be a partial UTF-8 sequence — the caller concatenates pieces. */
const char *bpe_decode(const BPETokenizer *t, int id, int *out_len);

int bpe_eos(const BPETokenizer *t);
int bpe_vocab_size(const BPETokenizer *t);
void bpe_free(BPETokenizer *t);

#endif /* MOTE_BPE_H */
