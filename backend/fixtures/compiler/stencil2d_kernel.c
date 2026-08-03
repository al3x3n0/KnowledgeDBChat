#include <stddef.h>

void stencil2d(
    const float *restrict input,
    float *restrict output,
    size_t rows,
    size_t columns
) {
    for (size_t row = 1; row + 1 < rows; ++row) {
        for (size_t column = 1; column + 1 < columns; ++column) {
            const size_t offset = row * columns + column;
            output[offset] = (
                input[offset] * 0.5f
                + input[offset - 1] * 0.125f
                + input[offset + 1] * 0.125f
                + input[offset - columns] * 0.125f
                + input[offset + columns] * 0.125f
            );
        }
    }
}
