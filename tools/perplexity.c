/* tools/perplexity.c — measure a model's quality as a single number.
 *
 *   perplexity <checkpoint> <text_file> [-z tokenizer]
 *
 * Perplexity is exp(average negative log-likelihood per token): feed the model
 * real text, and at each position ask how much probability it put on the token
 * that actually came next. Lower is better. Its whole use here is comparison:
 * run it on the fp32 model and the quantized one and the gap is exactly what
 * quantization cost, no hand-waving.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/model.h"
#include "../src/tokenizer.h"

static char *read_file(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "can't open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); exit(1); }
    b[n] = '\0';
    fclose(f);
    *len = n;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <checkpoint> <text_file> [-z tokenizer]\n", argv[0]);
        return 1;
    }
    const char *checkpoint = argv[1];
    const char *text_path = argv[2];
    const char *tok_path = "models/tokenizer.bin";
    for (int i = 3; i + 1 < argc; i += 2)
        if (argv[i][0] == '-' && argv[i][1] == 'z') tok_path = argv[i + 1];

    Transformer t;
    build_transformer(&t, checkpoint);
    Tokenizer tk;
    build_tokenizer(&tk, tok_path, t.config.vocab_size);

    long len;
    char *text = read_file(text_path, &len);

    int *tokens = malloc((len + 3) * sizeof(int));
    int n = 0;
    encode(&tk, text, /*bos=*/1, /*eos=*/0, tokens, &n);
    if (n > t.config.seq_len) n = t.config.seq_len;   /* stay within context */
    if (n < 2) { fprintf(stderr, "need at least 2 tokens\n"); return 1; }

    double nll = 0.0;
    int counted = 0;
    for (int pos = 0; pos < n - 1; pos++) {
        float *logits = forward(&t, tokens[pos], pos);
        int target = tokens[pos + 1];

        /* log-softmax at the target, computed stably via log-sum-exp */
        float maxl = logits[0];
        for (int i = 1; i < t.config.vocab_size; i++)
            if (logits[i] > maxl) maxl = logits[i];
        double sum = 0.0;
        for (int i = 0; i < t.config.vocab_size; i++)
            sum += exp((double)logits[i] - maxl);
        double logprob = (double)logits[target] - (maxl + log(sum));

        nll += -logprob;
        counted++;
    }

    double avg_nll = nll / counted;
    double ppl = exp(avg_nll);
    printf("%-28s  tokens %4d   avg nll %.4f   perplexity %.3f\n",
           checkpoint, counted, avg_nll, ppl);

    free(text);
    free(tokens);
    free_tokenizer(&tk);
    free_transformer(&t);
    return 0;
}
