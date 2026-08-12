/* sampler.c — sampling strategies over the output distribution.
 *
 * The RNG is a small xorshift so runs are reproducible from a seed with no
 * dependency on the C library's rand(). Top-p (nucleus) sampling sorts the
 * distribution once per step, which is cheap next to the forward pass.
 */
#include "sampler.h"

#include <math.h>
#include <stdlib.h>

static void softmax(float *x, int size)
{
    float max = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* draw from the full distribution given a uniform coin in [0,1) */
static int sample_mult(const float *probs, int n, float coin)
{
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) return i;
    }
    return n - 1;   /* rounding slack */
}

static int compare_prob(const void *a, const void *b)
{
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/* nucleus sampling: restrict to the top tokens whose mass reaches topp */
static int sample_topp(const float *probs, int n, float topp,
                       ProbIndex *order, float coin)
{
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);   /* prune tiny tails fast */
    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            order[n0].index = i;
            order[n0].prob = probs[i];
            n0++;
        }
    }
    qsort(order, n0, sizeof(ProbIndex), compare_prob);

    float cumulative = 0.0f;
    int last = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += order[i].prob;
        if (cumulative > topp) { last = i; break; }
    }

    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last; i++) {
        cdf += order[i].prob;
        if (r < cdf) return order[i].index;
    }
    return order[last].index;
}

static unsigned int random_u32(unsigned long long *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1DULL) >> 32;
}

static float random_f32(unsigned long long *state)
{
    return (random_u32(state) >> 8) / 16777216.0f;   /* 24-bit -> [0,1) */
}

void build_sampler(Sampler *s, int vocab_size, float temperature,
                   float topp, unsigned long long seed)
{
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng = seed;
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *s)
{
    free(s->probindex);
}

int sample(Sampler *s, float *logits)
{
    if (s->temperature == 0.0f)
        return argmax(logits, s->vocab_size);

    for (int i = 0; i < s->vocab_size; i++)
        logits[i] /= s->temperature;
    softmax(logits, s->vocab_size);

    float coin = random_f32(&s->rng);
    if (s->topp <= 0.0f || s->topp >= 1.0f)
        return sample_mult(logits, s->vocab_size, coin);
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}
