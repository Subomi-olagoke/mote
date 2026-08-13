/* tools/run_ids.c — drive the engine with raw token IDs, no tokenizer.
 *
 * A model's tokenizer and its transformer are separate concerns. This harness
 * exercises just the transformer: feed it a prompt already encoded as token IDs,
 * greedily decode a few more, and print the output IDs. A caller that owns the
 * real tokenizer (e.g. a Python script with the HF tokenizer) can encode in and
 * decode out. It is how we prove a newly converted architecture runs correctly
 * before its tokenizer is ported to C.
 *
 *   run_ids <model.mq> <n_new> <id,id,id,...>
 */
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.mq> <n_new> <id,id,...> [eos_id]\n", argv[0]);
        return 1;
    }
    int n_new = atoi(argv[2]);
    int eos = (argc >= 5) ? atoi(argv[4]) : -1;   /* optional stop token */

    /* parse the comma-separated prompt IDs */
    int cap = 16, n_prompt = 0;
    int *ids = malloc(cap * sizeof(int));
    for (char *tok = strtok(argv[3], ","); tok; tok = strtok(NULL, ",")) {
        if (n_prompt == cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
        ids[n_prompt++] = atoi(tok);
    }

    Transformer t;
    build_transformer(&t, argv[1]);

    int pos = 0, token = ids[0];
    /* run the prompt through to fill the KV cache; keep the last logits */
    float *logits = NULL;
    for (; pos < n_prompt; pos++) {
        logits = forward(&t, token, pos);
        if (pos + 1 < n_prompt) token = ids[pos + 1];
    }

    /* greedy decode n_new continuation tokens */
    for (int i = 0; i < n_new && pos < t.config.seq_len; i++) {
        int next = argmax(logits, t.config.vocab_size);
        if (next == eos) break;
        printf("%d ", next);
        logits = forward(&t, next, pos);
        pos++;
    }
    printf("\n");

    free(ids);
    free_transformer(&t);
    return 0;
}
