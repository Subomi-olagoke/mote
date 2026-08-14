/* tools/lib_chat.c — drive a chat turn through the mote_lib public API, the exact
 * surface the iOS app links against. Proves the library (not just the internals)
 * runs Qwen with the BPE tokenizer.
 *
 *   lib_chat <model.mq> <tokenizer.mtok> "<message>" [max_new]
 */
#include "mote_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int on_tok(const char *piece, void *user)
{
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return 1;   /* keep going */
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <model.mq> <tok.mtok> <message> [max]\n", argv[0]); return 1; }
    mote *m = mote_create_from_files(argv[1], argv[2]);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    char prompt[8192];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", argv[3]);

    mote_params p = { .temperature = 0.0f, .topp = 0.9f, .seed = 42,
                      .max_tokens = argc >= 5 ? atoi(argv[4]) : 120 };
    mote_generate(m, prompt, &p, on_tok, NULL);
    printf("\n");
    mote_free(m);
    return 0;
}
