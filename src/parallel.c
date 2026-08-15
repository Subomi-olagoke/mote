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

#elif defined(__EMSCRIPTEN_PTHREADS__)

/* WebAssembly with threads: pthreads map onto web workers, but spawning one
 * per matmul would be absurd, so a small persistent pool is built on first
 * use and then fed jobs. The calling thread works too (chunk 0), and waits on
 * a condvar for the rest; the engine already lives in a worker, where
 * blocking is allowed. Link with -sPTHREAD_POOL_SIZE so the workers exist
 * up front — creating one lazily needs the browser's main thread, which may
 * be busy, and that way lies deadlock. */
#include <pthread.h>
#include <unistd.h>

#define POOL_MAX 8

typedef struct {
    void (*fn)(int, int, void *);
    void *ctx;
    int count, chunks;
} Job;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;
static Job job;
static int generation, pending, pool_n;

static void *worker(void *arg)
{
    int self = (int)(long)arg;   /* worker i handles chunk i+1 */
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (generation == seen)
            pthread_cond_wait(&cv_work, &mu);
        seen = generation;
        Job j = job;
        pthread_mutex_unlock(&mu);

        int c = self + 1;
        if (c < j.chunks) {
            int begin = (int)((long)c * j.count / j.chunks);
            int end   = (int)((long)(c + 1) * j.count / j.chunks);
            j.fn(begin, end, j.ctx);
        }
        pthread_mutex_lock(&mu);
        if (--pending == 0)
            pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count <= 0)
        return;

    if (pool_n == 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        pool_n = ncpu > POOL_MAX ? POOL_MAX : (ncpu > 1 ? (int)ncpu : 1);
        for (int i = 0; i + 1 < pool_n; i++) {
            pthread_t t;
            if (pthread_create(&t, NULL, worker, (void *)(long)i) != 0) {
                pool_n = i + 1;   /* run with what we got */
                break;
            }
        }
    }

    int chunks = pool_n < count ? pool_n : count;
    if (chunks <= 1) {
        fn(0, count, ctx);
        return;
    }

    pthread_mutex_lock(&mu);
    job.fn = fn; job.ctx = ctx; job.count = count; job.chunks = chunks;
    pending = pool_n - 1;
    generation++;
    pthread_cond_broadcast(&cv_work);
    pthread_mutex_unlock(&mu);

    fn(0, (int)((long)count / chunks), ctx);   /* chunk 0, on this thread */

    pthread_mutex_lock(&mu);
    while (pending > 0)
        pthread_cond_wait(&cv_done, &mu);
    pthread_mutex_unlock(&mu);
}

#else

void mote_parallel_for(int count, void (*fn)(int, int, void *), void *ctx)
{
    if (count > 0)
        fn(0, count, ctx);
}

#endif
