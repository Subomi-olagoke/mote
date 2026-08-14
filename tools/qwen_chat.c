/* tools/qwen_chat.c — a full offline chat turn through pure C: text in, BPE encode,
 * mote forward, BPE decode, text out. No Python, no reference tokenizer. This proves
 * the Qwen engine and the byte-level BPE tokenizer work together end to end, which
 * is exactly the path the iOS app runs.
 *
 *   qwen_chat <model.mq> <tokenizer.mtok> "<user message>" [max_new]
 */
#include "model.h"
#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int argmax(const float *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.mq> <tokenizer.mtok> <message> [max_new]\n", argv[0]);
        return 1;
    }
    int max_new = argc >= 5 ? atoi(argv[4]) : 200;

    Transformer t;
    build_transformer(&t, argv[1]);
    BPETokenizer *tk = bpe_load(argv[2]);
    if (!tk) return 1;

    /* Qwen ChatML template */
    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", argv[3]);

    int cap = 4096, *ids = malloc(cap * sizeof(int));
    int n_prompt = bpe_encode(tk, prompt, ids, cap);
    int eos = bpe_eos(tk);

    /* run the prompt through, then greedily decode until eos */
    int pos = 0;
    float *logits = NULL;
    for (; pos < n_prompt; pos++)
        logits = forward(&t, ids[pos], pos);

    for (int i = 0; i < max_new && pos < t.config.seq_len; i++) {
        int next = argmax(logits, t.config.vocab_size);
        if (next == eos) break;
        int len; const char *piece = bpe_decode(tk, next, &len);
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
        logits = forward(&t, next, pos);
        pos++;
    }
    printf("\n");

    free(ids);
    bpe_free(tk);
    free_transformer(&t);
    return 0;
}
