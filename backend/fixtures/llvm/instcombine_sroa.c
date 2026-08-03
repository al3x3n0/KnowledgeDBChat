#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t low;
    uint32_t high;
} split_word;

static split_word unpack(uint64_t value) {
    split_word result = {
        .low = (uint32_t)value,
        .high = (uint32_t)(value >> 32),
    };
    return result;
}

uint64_t combine_words(const uint64_t *values, size_t count) {
    split_word accumulator = {0, 0};

    for (size_t index = 0; index < count; ++index) {
        split_word current = unpack(values[index]);
        accumulator.low ^= current.low + (uint32_t)index;
        accumulator.high += current.high ^ accumulator.low;
    }

    return ((uint64_t)accumulator.high << 32) | accumulator.low;
}
