/* Branch predictability: identical work, identical data, only the ORDER
   differs. Sorting makes the branch outcome highly predictable; the same
   values shuffled make it near-random. Any difference is the cost of
   mispredicted branches, since instruction count and cache traffic match. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 21)
#define REPS 24

/* An opaque call the compiler cannot vectorize or if-convert, so the
   conditional below must compile to a real data-dependent branch. Without
   this, clang turns the loop into predicated NEON and there is no branch
   left to mispredict -- the effect under study disappears. */
__attribute__((noinline)) static long long acc(long long s, int v) {
    __asm__ volatile("" : "+r"(s) :: );
    return s + v;
}

static int cmp(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static long long run(const int *v, double *best) {
    long long sum = 0;
    *best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        long long s = 0;
        for (int i = 0; i < N; i++)
            if (v[i] >= 128) s = acc(s, v[i]);   /* the branch under study */
        double dt = now() - t0;
        if (dt < *best) *best = dt;
        sum += s;
    }
    return sum;
}

int main(void) {
    int *a = malloc(N * sizeof(int));
    int *b = malloc(N * sizeof(int));
    srand(1);
    for (int i = 0; i < N; i++) { a[i] = rand() % 256; b[i] = a[i]; }
    qsort(b, N, sizeof(int), cmp);       /* same multiset, predictable order */

    double t_rand, t_sorted;
    long long s1 = run(a, &t_rand);
    long long s2 = run(b, &t_sorted);
    if (s1 != s2) { fprintf(stderr, "sums differ: %lld vs %lld\n", s1, s2); return 1; }

    printf("%.6f %.6f\n", t_rand, t_sorted);
    return 0;
}
