#include <stddef.h>

enum { TILE_SIZE = 16 };

void tiled_matmul(
    const float *restrict left,
    const float *restrict right,
    float *restrict output,
    size_t size
) {
    for (size_t row_block = 0; row_block < size; row_block += TILE_SIZE) {
        for (size_t inner_block = 0; inner_block < size; inner_block += TILE_SIZE) {
            for (
                size_t column_block = 0;
                column_block < size;
                column_block += TILE_SIZE
            ) {
                for (
                    size_t row = row_block;
                    row < row_block + TILE_SIZE && row < size;
                    ++row
                ) {
                    for (
                        size_t inner = inner_block;
                        inner < inner_block + TILE_SIZE && inner < size;
                        ++inner
                    ) {
                        const float factor = left[row * size + inner];
                        for (
                            size_t column = column_block;
                            column < column_block + TILE_SIZE && column < size;
                            ++column
                        ) {
                            output[row * size + column] +=
                                factor * right[inner * size + column];
                        }
                    }
                }
            }
        }
    }
}
