/* parallel.c — platform backends for mote_parallel_for. See parallel.h. */
#include "parallel.h"

#if defined(__APPLE__)

#include <dispatch/dispatch.h>
#include <unistd.h>

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int chunks = ncpu > 1 ? (int)ncpu : 1;
    if (chunks > count)
        chunks = count;
    if (chunks <= 1) {
        fn(0, count, ctx);
        return;
    }

    dispatch_apply((size_t)chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
        int begin = (int)((long)c * count / chunks);
        int end   = (int)(((long)c + 1) * count / chunks);
        fn(begin, end, ctx);
    });
}

#elif defined(_OPENMP)

#include <omp.h>

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int t = omp_get_thread_num();
        int begin = (int)((long)t * count / nt);
        int end   = (int)(((long)t + 1) * count / nt);
        if (begin < end)
            fn(begin, end, ctx);
    }
}

#else

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count > 0)
        fn(0, count, ctx);
}

#endif
