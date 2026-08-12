/* quant.h — 8-bit quantization, the part that makes mote small.
 *
 * The scheme is symmetric int8 with per-group scales (the "Q8_0" idea). A run of
 * `gs` weights shares one fp32 scale; each weight is stored as an int8 in
 * [-127, 127], and the real value is int8 * scale. Groups are small enough that
 * one outlier only inflates its own group, so quality holds up while the model
 * shrinks about 4x.
 *
 * The matmul multiplies int8 by int8 into an int32 accumulator per group, then
 * folds in the two scales once per group. Activations are quantized on the fly
 * the same way right before each matmul, so the hot loop is integer work.
 */
#ifndef MOTE_QUANT_H
#define MOTE_QUANT_H

#include <stdint.h>

typedef struct {
    int8_t *q;   /* quantized values                          */
    float  *s;   /* one scale per group of `gs` values        */
} QTensor;

/* Quantize x[n] into out (out->q has n slots, out->s has n/gs). Dynamic,
 * used for activations. Requires gs | n. */
void quantize(const QTensor *out, const float *x, int n, int gs);

/* Reconstruct x[n] from a quantized tensor (used for the embedding lookup). */
void dequantize(const QTensor *in, float *x, int n, int gs);

/* out(d) = W(d, n) . x(n), all int8. The weight tensor `w` may hold many rows
 * back to back; `woff` is the element offset of row 0 for this call. */
void matmul_q8(float *out, const QTensor *x, const QTensor *w,
               long woff, int n, int d, int gs);

#endif /* MOTE_QUANT_H */
