/* tools/tok_test.c — encode a line of text with a .mtok tokenizer and print the
 * token ids, so the C encoder can be diffed against the reference tokenizer.
 *
 *   tok_test <model.mtok> "some text"        # prints: id id id ...
 *   tok_test <model.mtok> --decode 9707 11   # prints the detokenized bytes
 */
#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <model.mtok> <text> | --decode ids...\n", argv[0]); return 1; }
    BPETokenizer *t = bpe_load(argv[1]);
    if (!t) return 1;

    if (strcmp(argv[2], "--decode") == 0) {
        for (int i = 3; i < argc; i++) {
            int len; const char *b = bpe_decode(t, atoi(argv[i]), &len);
            fwrite(b, 1, len, stdout);
        }
        printf("\n");
    } else {
        int max = 4096, *ids = malloc(max * sizeof(int));
        int n = bpe_encode(t, argv[2], ids, max);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n");
        free(ids);
    }
    bpe_free(t);
    return 0;
}
