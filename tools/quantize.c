/* tools/quantize.c — turn a legacy fp32 checkpoint into mote's .mq format.
 *
 *   quantize <in.bin> <out.mq> [group_size]
 *
 * Reads the fp32 weights, quantizes the big matrices to symmetric int8 with one
 * fp32 scale per group, and writes them back in mote's self-describing layout.
 * The norm gains stay fp32. Group size defaults to the largest of 64/32/16/8
 * that divides both dim and hidden_dim, and is recorded in the header so the
 * runtime reads it back.
 */
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../src/config.h"
#include "../src/model.h"   /* MQ_MAGIC / MQ_VERSION / MQ_HEADER */
#include "../src/quant.h"

static int pick_group_size(const Config *c)
{
    int candidates[] = { 64, 32, 16, 8 };
    for (int i = 0; i < 4; i++) {
        int gs = candidates[i];
        if (c->dim % gs == 0 && c->hidden_dim % gs == 0)
            return gs;
    }
    return 0;
}

/* quantize a flat fp32 tensor and append it: int8 block, then fp32 scales */
static double write_qtensor(FILE *out, const float *w, long size, int gs)
{
    int8_t *q = malloc(size * sizeof(int8_t));
    float *s = malloc((size / gs) * sizeof(float));
    if (!q || !s) { fprintf(stderr, "quantize: out of memory\n"); exit(1); }

    QTensor qt = { .q = q, .s = s };
    quantize(&qt, w, (int)size, gs);

    /* round-trip error, for the report */
    double err = 0.0;
    for (long i = 0; i < size; i++) {
        double d = (double)w[i] - (double)(q[i] * s[i / gs]);
        err += d * d;
    }

    fwrite(q, sizeof(int8_t), size, out);
    fwrite(s, sizeof(float), size / gs, out);
    free(q);
    free(s);
    return err;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.bin> <out.mq> [group_size]\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];

    /* map the fp32 checkpoint */
    int fd = open(in_path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "quantize: can't open %s\n", in_path); return 1; }
    off_t fsize = lseek(fd, 0, SEEK_END);
    float *data = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "quantize: mmap failed\n"); return 1; }

    Config c = *(Config *)data;
    int shared = c.vocab_size > 0;
    c.vocab_size = abs(c.vocab_size);

    int gs = (argc >= 4) ? atoi(argv[3]) : pick_group_size(&c);
    if (gs <= 0 || c.dim % gs || c.hidden_dim % gs) {
        fprintf(stderr, "quantize: no group size divides dim=%d and hidden=%d\n",
                c.dim, c.hidden_dim);
        return 1;
    }

    long L = c.n_layers, D = c.dim, HD = c.hidden_dim;
    int head_size = c.dim / c.n_heads;
    long kv = (long)c.n_kv_heads * head_size;

    /* walk the fp32 layout */
    float *ptr = data + sizeof(Config) / sizeof(float);
    float *tok = ptr;      ptr += (long)c.vocab_size * D;
    float *rms_att = ptr;  ptr += L * D;
    float *wq = ptr;       ptr += L * D * D;
    float *wk = ptr;       ptr += L * D * kv;
    float *wv = ptr;       ptr += L * D * kv;
    float *wo = ptr;       ptr += L * D * D;
    float *rms_ffn = ptr;  ptr += L * D;
    float *w1 = ptr;       ptr += L * D * HD;
    float *w2 = ptr;       ptr += L * HD * D;
    float *w3 = ptr;       ptr += L * D * HD;
    float *rms_final = ptr; ptr += D;
    ptr += c.seq_len * head_size / 2;
    ptr += c.seq_len * head_size / 2;
    float *wcls = shared ? tok : ptr;

    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "quantize: can't write %s\n", out_path); return 1; }

    /* header: 256 bytes, ints then zero padding */
    int32_t header[MQ_HEADER / 4] = { 0 };
    header[0] = MQ_MAGIC;
    header[1] = MQ_VERSION;
    header[2] = c.dim;
    header[3] = c.hidden_dim;
    header[4] = c.n_layers;
    header[5] = c.n_heads;
    header[6] = c.n_kv_heads;
    header[7] = c.vocab_size;
    header[8] = c.seq_len;
    header[9] = gs;
    header[10] = shared;
    /* v2 fields: this legacy path carries no attention bias and uses the Llama
     * RoPE base and epsilon. Stored as float bits so the runtime reads them back. */
    header[11] = 0;                 /* has_qkv_bias */
    float rope_theta = 10000.0f, rms_eps = 1e-5f;
    memcpy(&header[12], &rope_theta, sizeof(float));
    memcpy(&header[13], &rms_eps, sizeof(float));
    fwrite(header, 1, MQ_HEADER, out);

    /* norms, fp32 */
    fwrite(rms_att, sizeof(float), L * D, out);
    fwrite(rms_ffn, sizeof(float), L * D, out);
    fwrite(rms_final, sizeof(float), D, out);

    /* quantized tensors */
    double err = 0.0;
    long nq = 0;
    err += write_qtensor(out, tok, (long)c.vocab_size * D, gs); nq += (long)c.vocab_size * D;
    err += write_qtensor(out, wq, L * D * D, gs);   nq += L * D * D;
    err += write_qtensor(out, wk, L * D * kv, gs);  nq += L * D * kv;
    err += write_qtensor(out, wv, L * D * kv, gs);  nq += L * D * kv;
    err += write_qtensor(out, wo, L * D * D, gs);   nq += L * D * D;
    err += write_qtensor(out, w1, L * D * HD, gs);  nq += L * D * HD;
    err += write_qtensor(out, w2, L * HD * D, gs);  nq += L * HD * D;
    err += write_qtensor(out, w3, L * D * HD, gs);  nq += L * D * HD;
    if (!shared) { err += write_qtensor(out, wcls, (long)c.vocab_size * D, gs); nq += (long)c.vocab_size * D; }

    long out_size = ftell(out);
    fclose(out);
    munmap(data, fsize);
    close(fd);

    fprintf(stderr,
            "quantized %s -> %s\n"
            "  group size:  %d\n"
            "  size:        %.1f MB -> %.1f MB  (%.2fx smaller)\n"
            "  rms error:   %.6f  (per weight, averaged)\n",
            in_path, out_path, gs,
            fsize / 1e6, out_size / 1e6, (double)fsize / out_size,
            nq ? sqrt(err / nq) : 0.0);
    return 0;
}
