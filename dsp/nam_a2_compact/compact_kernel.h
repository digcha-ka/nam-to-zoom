#ifndef N2Z_COMPACT_KERNEL_H
#define N2Z_COMPACT_KERNEL_H

#include <stdint.h>

#ifndef COMPACT_CHANNELS
#define COMPACT_CHANNELS 2
#endif

#if COMPACT_CHANNELS != 2 && COMPACT_CHANNELS != 3
#error COMPACT_CHANNELS must be 2 or 3
#endif

#define COMPACT_LAYERS 14
#define COMPACT_HEAD_KERNEL 8
#define COMPACT_RECEPTIVE_FIELD 1644
#define COMPACT_WEIGHTS (COMPACT_CHANNELS + COMPACT_LAYERS * (4 * COMPACT_CHANNELS * COMPACT_CHANNELS + 3 * COMPACT_CHANNELS) + COMPACT_HEAD_KERNEL * COMPACT_CHANNELS + 2)
#define COMPACT_MIRROR_HISTORY_FLOATS (2 * 1658 * COMPACT_CHANNELS)

typedef struct {
    uint16_t layer_pos[COMPACT_LAYERS];
    uint16_t head_pos;
} CompactState;

float compact_process(const float *weights, float *history, CompactState *state, float x);

#endif
