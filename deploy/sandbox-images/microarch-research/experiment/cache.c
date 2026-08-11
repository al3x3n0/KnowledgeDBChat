/* Cache hierarchy: dependent pointer chase over a working set that grows.
   Each load's address depends on the previous one, so the loop cannot be
   overlapped and the per-access time is the load-to-use latency of whichever
   level currently holds the data. Latency steps up as the set outgrows a
   level. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPS 5

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Chase with a stride wider than a cache line so each step is a new line. */
#define STRIDE 128

int main(void) {
    size_t sizes_kb[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int n = sizeof(sizes_kb) / sizeof(sizes_kb[0]);
    printf("{\n  \"points\": [\n");
    for (int k = 0; k < n; k++) {
        size_t bytes = sizes_kb[k] * 1024;
        size_t slots = bytes / sizeof(size_t);
        size_t step = STRIDE / sizeof(size_t);
        size_t count = slots / step;
        if (count < 2) continue;
        size_t *buf = calloc(slots, sizeof(size_t));
        if (!buf) continue;
        /* Ring of dependent pointers. */
        for (size_t i = 0; i < count; i++)
            buf[i * step] = ((i + 1) % count) * step;

        size_t iters = 4000000;
        double best = 1e30;
        for (int r = 0; r < REPS; r++) {
            size_t p = 0;
            double t0 = now();
            for (size_t i = 0; i < iters; i++) p = buf[p];
            double dt = now() - t0;
            if (buf[p] == (size_t)-1) printf("x");  /* keep p live */
            if (dt < best) best = dt;
        }
        printf("    {\"kb\": %zu, \"ns_per_access\": %.3f}%s\n",
               sizes_kb[k], best * 1e9 / iters, (k == n - 1) ? "" : ",");
        fflush(stdout);
        free(buf);
    }
    printf("  ]\n}\n");
    return 0;
}
