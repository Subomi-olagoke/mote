/* mote.c — the command line.
 *
 * Wires the checkpoint, tokenizer, and sampler together and runs the generation
 * loop: encode the prompt, feed the model one token at a time, and once the
 * prompt is consumed, sample each next token and print it.
 *
 *   mote <checkpoint> [-z tokenizer] [-t temp] [-p topp] [-n steps]
 *                     [-i prompt] [-s seed]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

static long now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000;
}

/* Don't let raw control bytes from byte-fallback tokens corrupt the terminal. */
static void safe_print(const char *piece)
{
    if (piece == NULL || piece[0] == '\0')
        return;
    if (piece[1] == '\0') {
        unsigned char c = piece[0];
        int printable = (c >= 32 && c < 127) || c == '\n' || c == '\t';
        if (!printable) return;
    }
    fputs(piece, stdout);
}

static void generate(Transformer *t, Tokenizer *tk, Sampler *sampler,
                     const char *prompt, int steps)
{
    if (!prompt) prompt = "";

    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int n_prompt = 0;
    encode(tk, prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt);
    if (n_prompt < 1) {
        fprintf(stderr, "mote: prompt produced no tokens\n");
        exit(1);
    }

    long start = 0;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float *logits = forward(t, token, pos);

        int next;
        if (pos < n_prompt - 1)
            next = prompt_tokens[pos + 1];   /* still replaying the prompt */
        else
            next = sample(sampler, logits);
        pos++;

        if (next == 1)   /* BOS marks a hard stop for these models */
            break;

        safe_print(decode(tk, token, next));
        fflush(stdout);
        token = next;

        if (start == 0) start = now_ms();   /* skip the warm first token */
    }
    printf("\n");

    if (pos > 1) {
        long end = now_ms();
        fprintf(stderr, "\n[%d tokens, %.1f tok/s]\n",
                pos, (pos - 1) / (double)(end - start) * 1000.0);
    }
    free(prompt_tokens);
}

int main(int argc, char **argv)
{
    const char *checkpoint = NULL;
    const char *tokenizer_path = "models/tokenizer.bin";
    const char *prompt = "";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    unsigned long long seed = 0;

    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <checkpoint> [options]\n"
            "  -z <path>   tokenizer file (default models/tokenizer.bin)\n"
            "  -t <float>  temperature, 0 = greedy (default 1.0)\n"
            "  -p <float>  top-p / nucleus (default 0.9, 0 disables)\n"
            "  -n <int>    number of steps to run (default 256)\n"
            "  -i <string> prompt\n"
            "  -s <int>    rng seed (default: time)\n", argv[0]);
        return 1;
    }
    checkpoint = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        const char *flag = argv[i], *val = argv[i + 1];
        if (flag[0] != '-' || strlen(flag) != 2) {
            fprintf(stderr, "mote: bad flag %s\n", flag); return 1;
        }
        switch (flag[1]) {
            case 'z': tokenizer_path = val; break;
            case 't': temperature = strtof(val, NULL); break;
            case 'p': topp = strtof(val, NULL); break;
            case 'n': steps = atoi(val); break;
            case 'i': prompt = val; break;
            case 's': seed = strtoull(val, NULL, 10); break;
            default: fprintf(stderr, "mote: unknown flag %s\n", flag); return 1;
        }
    }
    if (temperature < 0.0f) temperature = 0.0f;
    if (seed == 0) seed = (unsigned long long)now_ms();

    Transformer transformer;
    build_transformer(&transformer, checkpoint);
    if (steps <= 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
