/* Injected: take one counter sample. gem5 m5 DUMP_RESET_STATS. */
#define M5_SAMPLE() __asm__ __volatile__("mov x0, #0\n\tmov x1, #0\n\t.inst 0xff420110" ::: "x0", "x1", "memory")
#include <stdint.h>
#include <stdlib.h>

#define CACHE_ELEMS (64*1024 / sizeof(int))
#define MEM_SIZE (256UL*1024*1024)
#define MEM_LINE_WORDS (MEM_SIZE / 64)
#define MEM_TOUCH_LINES 8192

volatile uint64_t sink = 0;
volatile int cache_array[CACHE_ELEMS];

static void cache_phase(void) {
  for (int r = 0; r < 16; r++) {
    for (int i = 0; i < CACHE_ELEMS; i++) {
      cache_array[i] = cache_array[i] * 3 + (i & 0xffff);
    }
  }
  sink += cache_array[1];
}

static void mem_phase(void) {
  static volatile uint8_t *buf = 0;
  if (!buf) {
    buf = (volatile uint8_t *)malloc(MEM_SIZE);
    if (!buf) __builtin_trap();
  }
  uint32_t rng = 0x12345678u;
  for (int k = 0; k < MEM_TOUCH_LINES; k++) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    size_t line = (size_t)(rng & (MEM_LINE_WORDS - 1));
    size_t off = line * 64;
    sink += buf[off];
    buf[off] = (uint8_t)(off >> 8);
  }
}

int main(void) {
  M5_SAMPLE();
  for (int i = 0; i < 100; i++) { cache_phase(); M5_SAMPLE(); }
  for (int i = 0; i < 100; i++) { mem_phase(); M5_SAMPLE(); }
  return (int)(sink & 0xff);
}