/* tools/perplexity.c — measure a model's quality as a single number.
 *
 *   perplexity <checkpoint> <text_file> [-z tokenizer]
 *
 * Works with either tokenizer: the legacy SentencePiece-style .bin (default)
 * or a byte-level BPE .mtok (pass it with -z; detected by its magic), so
 * Qwen-class models get the same measurement as the TinyStories ones.
 *
 * Perplexity is exp(average negative log-likelihood per token): feed the model
 * real text, and at each position ask how much probability it put on the token
 * that actually came next. Lower is better. Its whole use here is comparison:
 * run it on the fp32 model and the quantized one and the gap is exactly what
 * quantization cost, no hand-waving.
 *
 * Blank lines split the text into documents; each document is encoded and
 * scored on its own from position 0 (a story corpus is many short documents,
 * and the tokenizer's merge loop is quadratic, so one giant encode would cost
 * minutes for no accuracy). A document longer than the context is scored in
 * back-to-back windows of seq_len tokens, each starting cold.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/bpe.h"
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

    /* either tokenizer scheme: a file starting with "MTOK" is byte-level BPE
     * (Qwen-class models), anything else is the legacy SentencePiece-style one */
    BPETokenizer *bpe = NULL;
    Tokenizer tk;
    FILE *tf = fopen(tok_path, "rb");
    if (!tf) { fprintf(stderr, "can't open %s\n", tok_path); return 1; }
    char m4[4] = { 0 };
    if (fread(m4, 1, 4, tf) != 4) { fprintf(stderr, "can't read %s\n", tok_path); return 1; }
    fclose(tf);
    if (memcmp(m4, "MTOK", 4) == 0) {
        bpe = bpe_load(tok_path);
        if (!bpe) return 1;
    } else {
        build_tokenizer(&tk, tok_path, t.config.vocab_size);
    }

    long len;
    char *text = read_file(text_path, &len);

    int *tokens = malloc((len + 3) * sizeof(int));
    double nll = 0.0;
    long counted = 0;
    int T = t.config.seq_len;

    char *doc = text;
    while (doc && *doc) {
        /* a blank line ends the current document */
        char *next = strstr(doc, "\n\n");
        if (next) {
            *next = '\0';
            next += 2;
            while (*next == '\n') next++;
        }

        int n = 0;
        if (bpe)
            n = bpe_encode(bpe, doc, tokens, (int)len + 3);
        else
            encode(&tk, doc, /*bos=*/1, /*eos=*/0, tokens, &n);
        for (int start = 0; start + 1 < n; start += T) {
            int wlen = n - start < T ? n - start : T;   /* this window's tokens */
            for (int pos = 0; pos + 1 < wlen; pos++) {
                float *logits = forward(&t, tokens[start + pos], pos);
                int target = tokens[start + pos + 1];

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
        }
        doc = next;
    }
    if (counted == 0) { fprintf(stderr, "need at least 2 tokens\n"); return 1; }

    double avg_nll = nll / counted;
    double ppl = exp(avg_nll);
    printf("%-28s  tokens %6ld   avg nll %.4f   perplexity %.3f\n",
           checkpoint, counted, avg_nll, ppl);

    free(text);
    free(tokens);
    if (bpe)
        bpe_free(bpe);
    else
        free_tokenizer(&tk);
    free_transformer(&t);
    return 0;
}
