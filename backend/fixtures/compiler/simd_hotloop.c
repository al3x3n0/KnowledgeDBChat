#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ELEMENT_COUNT = 1 << 17 };

static volatile float observed_result;

static unsigned parse_repeats(int argc, char **argv) {
    const char prefix[] = "--repeat=";
    unsigned long value;
    char *end = NULL;

    if (argc != 2 || strncmp(argv[1], prefix, sizeof(prefix) - 1) != 0) {
        return 100;
    }
    errno = 0;
    value = strtoul(argv[1] + sizeof(prefix) - 1, &end, 10);
    if (errno != 0 || end == NULL || *end != '\0' || value == 0 || value > 10000) {
        return 100;
    }
    return (unsigned)value;
}

static float fused_hotloop(
    const float *left,
    const float *right,
    float *output,
    size_t count
) {
    float checksum = 0.0f;

    for (size_t index = 0; index < count; ++index) {
        const float value = left[index] * 1.25f + right[index] * 0.75f;
        output[index] = value;
        checksum += value;
    }
    return checksum;
}

int main(int argc, char **argv) {
    const unsigned repeats = parse_repeats(argc, argv);
    float *left = malloc(sizeof(*left) * ELEMENT_COUNT);
    float *right = malloc(sizeof(*right) * ELEMENT_COUNT);
    float *output = malloc(sizeof(*output) * ELEMENT_COUNT);
    clock_t started;

    if (left == NULL || right == NULL || output == NULL) {
        fputs("allocation failed\n", stderr);
        free(left);
        free(right);
        free(output);
        return 2;
    }
    for (size_t index = 0; index < ELEMENT_COUNT; ++index) {
        left[index] = (float)(index % 251) / 251.0f;
        right[index] = (float)(index % 127) / 127.0f;
    }

    started = clock();
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        observed_result = fused_hotloop(left, right, output, ELEMENT_COUNT);
    }
    printf(
        "{\"repeats\":%u,\"elements\":%u,\"runtime_ms\":%.3f,"
        "\"checksum\":%.6f}\n",
        repeats,
        ELEMENT_COUNT,
        1000.0 * (double)(clock() - started) / CLOCKS_PER_SEC,
        observed_result
    );

    free(left);
    free(right);
    free(output);
    return 0;
}
