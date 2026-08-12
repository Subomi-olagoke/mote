/* sampler.h — turn a row of logits into the next token.
 *
 * Temperature 0 is greedy (argmax). Otherwise the logits are softmaxed and a
 * token is drawn, either from the full distribution or, with top-p, from the
 * smallest set of most-likely tokens whose mass clears p.
 */
#ifndef MOTE_SAMPLER_H
#define MOTE_SAMPLER_H

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;      /* scratch for top-p                    */
    float temperature;
    float topp;
    unsigned long long rng;    /* xorshift state                       */
} Sampler;

void build_sampler(Sampler *s, int vocab_size, float temperature,
                   float topp, unsigned long long seed);
void free_sampler(Sampler *s);

/* Consumes (and may reorder) the logits buffer; returns the chosen token id. */
int sample(Sampler *s, float *logits);

#endif /* MOTE_SAMPLER_H */
