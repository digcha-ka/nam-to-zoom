#ifndef N2Z_COMPACT_PAIR_H
#define N2Z_COMPACT_PAIR_H

#include <stdint.h>

#define PAIR_LAYERS 14
#define PAIR_WEIGHTS 659
#define PAIR_HISTORY_FLOATS 5019

typedef struct {
    uint16_t layer_pos[PAIR_LAYERS];
    uint16_t head_pos;
} CompactPairState;

/* Two consecutive samples, unchanged compact weights, padded history layout. */
void compact_process_pair(const float *weights, float *history,
                          CompactPairState *state, float input0, float input1,
                          float *output0, float *output1);

#endif
