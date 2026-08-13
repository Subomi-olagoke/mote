/* parallel.h — one portable parallel-for, split into a few big chunks.
 *
 * The matmul is embarrassingly parallel over output rows. This splits [0,count)
 * into roughly one range per core and runs the ranges concurrently: Grand
 * Central Dispatch on Apple (so it threads on iOS, not only the desktop),
 * OpenMP if the build enabled it, and a plain call everywhere else (a
 * one-core microcontroller just runs the whole range).
 *
 * It is range-based on purpose. Dispatching per row on a wide matmul fires the
 * work block thousands of times and the overhead swamps the win; one range per
 * core fires it a handful of times.
 */
#ifndef MOTE_PARALLEL_H
#define MOTE_PARALLEL_H

/* Calls fn(begin, end, ctx) for contiguous sub-ranges that cover [0, count). */
void mote_parallel_for(int count, void (*fn)(int begin, int end, void *ctx),
                       void *ctx);

#endif /* MOTE_PARALLEL_H */
