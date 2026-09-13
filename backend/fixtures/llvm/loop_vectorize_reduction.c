#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ELEMENT_COUNT = 1 << 18 };

static volatile double observed_result;

static unsigned parse_iterations(int argc, char **argv) {
    const char prefix[] = "--iters=";
    unsigned long value;
    char *end = NULL;

    if (argc != 2 || strncmp(argv[1], prefix, sizeof(prefix) - 1) != 0) {
        return 50;
    }

    errno = 0;
    value = strtoul(argv[1] + sizeof(prefix) - 1, &end, 10);
    if (errno != 0 || end == NULL || *end != '\0' || value == 0 || value > 10000) {
        return 50;
    }
    return (unsigned)value;
}

static double reduce(const float *values, size_t count) {
    double total = 0.0;

    for (size_t index = 0; index < count; ++index) {
        total += values[index];
    }
    return total;
}

int main(int argc, char **argv) {
    const unsigned iterations = parse_iterations(argc, argv);
    float *values = malloc(sizeof(*values) * ELEMENT_COUNT);
    clock_t started;

    if (values == NULL) {
        fputs("allocation failed\n", stderr);
        return 2;
    }
    for (size_t index = 0; index < ELEMENT_COUNT; ++index) {
        values[index] = (float)((index % 1024) + 1) / 1024.0f;
    }

    started = clock();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        observed_result = reduce(values, ELEMENT_COUNT);
    }

    printf(
        "{\"iterations\":%u,\"elements\":%u,\"runtime_ms\":%.3f,"
        "\"checksum\":%.6f}\n",
        iterations,
        ELEMENT_COUNT,
        1000.0 * (double)(clock() - started) / CLOCKS_PER_SEC,
        observed_result
    );
    free(values);
    return 0;
}
