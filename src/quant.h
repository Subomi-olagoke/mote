/* quant.h — the quantization schemes, the part that makes mote small.
 *
 * Both schemes are symmetric with per-group scales. A run of `gs` weights
 * shares one fp32 scale; each weight is stored as a small integer, and the real
 * value is int * scale. Groups are small enough that one outlier only inflates
 * its own group, so quality holds up while the model shrinks.
 *
 * int8 (the "Q8_0" idea): each weight is an int8 in [-127, 127]. About 4x
 * smaller than fp32 at negligible quality cost.
 *
 * int4 (the "Q4_0" idea): each weight is a signed nibble in [-8, 7], two per
 * byte. About 8x smaller, which is what makes half-gigabyte models shippable.
 * The scale carries the sign of the group's largest weight so that weight maps
 * exactly to -8, buying every other weight a finer step (see quantize_q4 for
 * the measurements behind this and the small scale search on top).
 *
 * The matmuls multiply integers into an int32 accumulator per group, then fold
 * in the two scales once per group. Activations are quantized to int8 on the
 * fly right before each matmul in both schemes, so the hot loop is integer work.
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

/* ---- int4 ----
 * Same QTensor shape, but `q` holds packed nibbles: within each group of `gs`
 * weights, byte k carries w[k] in its low nibble and w[k + gs/2] in its high
 * nibble. That split-half layout lets the vector path sign-extend a whole
 * register of nibbles into two contiguous int8 runs. `gs` must be even. */

/* Quantize x[n] into out (out->q has n/2 bytes, out->s has n/gs scales), with a
 * per-group clipped-scale search. Write-time only; never on the hot path. */
void quantize_q4(const QTensor *out, const float *x, long n, int gs);

/* Reconstruct x[n] from packed int4 (used for the embedding lookup). */
void dequantize_q4(const QTensor *in, float *x, int n, int gs);

/* out(d) = W(d, n) . x(n): int8 activation against int4 weights. `woff` is the
 * element (not byte) offset of row 0, and must be group-aligned. */
void matmul_q4(float *out, const QTensor *x, const QTensor *w,
               long woff, int n, int d, int gs);

#endif /* MOTE_QUANT_H */
