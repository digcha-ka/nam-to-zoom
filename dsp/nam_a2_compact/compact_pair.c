#include "compact_pair.h"

static const unsigned char pair_dilations[PAIR_LAYERS] = {1, 3, 7, 17, 41, 101, 239,
                                                          1, 3, 7, 17, 41, 101, 239};

typedef struct {
    float current[2][3];
    float head[2][3];
} PairWork;

/* Keep outer-loop state out of the convolution's register allocation. */
#ifdef __TI_COMPILER_VERSION__
#pragma FUNC_CANNOT_INLINE(pair_layer)
#endif
static void pair_layer(const float *w, float *history, uint16_t *position, PairWork *work,
                       int dilation, float input0, float input1) {
    int capacity = 2 * dilation + 2;
    int pos0 = *position, pos1 = pos0 + 1;
    int read0 = pos0 + 2, read1;
    const float *conv = w, *bias = w + 27, *mixin = bias + 3;
    const float *residual = mixin + 3, *rbias = residual + 9;
    float activation[2][3];
    int tap0[3], tap1[3], tap, channel;
    if (pos1 == capacity)
        pos1 = 0;
    if (read0 >= capacity)
        read0 -= capacity;
    read1 = read0 + 1;
    if (read1 == capacity)
        read1 = 0;

    /* The extra slot keeps sample 0's oldest tap safe when writing sample 1. */
    history[3 * pos0] = work->current[0][0];
    history[3 * pos0 + 1] = work->current[0][1];
    history[3 * pos0 + 2] = work->current[0][2];
    history[3 * pos1] = work->current[1][0];
    history[3 * pos1 + 1] = work->current[1][1];
    history[3 * pos1 + 2] = work->current[1][2];

    for (tap = 0; tap < 3; ++tap) {
        tap0[tap] = 3 * read0;
        tap1[tap] = 3 * read1;
        read0 += dilation;
        read1 += dilation;
        if (read0 >= capacity)
            read0 -= capacity;
        if (read1 >= capacity)
            read1 -= capacity;
    }
    /* Finish one output channel at a time: four live accumulators, not twelve. */
#define PAIR_ACC(LANE, IN, TAP)                                                                    \
    do {                                                                                           \
        float v00 = history[tap0[TAP] + (IN)];                                                     \
        float v10 = history[tap1[TAP] + (IN)];                                                     \
        float coefficient = row[3 * (IN) + (TAP)];                                                 \
        z0##LANE += coefficient * v00;                                                             \
        z1##LANE += coefficient * v10;                                                             \
    } while (0)
#define PAIR_CHANNEL(OUT, RESULT0, RESULT1)                                                        \
    do {                                                                                           \
        const float *row = conv + 9 * (OUT);                                                       \
        float z0a = bias[OUT], z1a = bias[OUT];                                                    \
        float z0b = 0.0f, z1b = 0.0f;                                                              \
        PAIR_ACC(a, 0, 0);                                                                         \
        PAIR_ACC(b, 0, 1);                                                                         \
        PAIR_ACC(a, 0, 2);                                                                         \
        PAIR_ACC(b, 1, 0);                                                                         \
        PAIR_ACC(a, 1, 1);                                                                         \
        PAIR_ACC(b, 1, 2);                                                                         \
        PAIR_ACC(a, 2, 0);                                                                         \
        PAIR_ACC(b, 2, 1);                                                                         \
        PAIR_ACC(a, 2, 2);                                                                         \
        RESULT0 = z0a + z0b;                                                                       \
        RESULT1 = z1a + z1b;                                                                       \
    } while (0)
#ifdef __TI_COMPILER_VERSION__
#pragma UNROLL(1)
#endif
    for (channel = 0; channel < 3; ++channel) {
        float a0, a1;
        PAIR_CHANNEL(channel, a0, a1);
        a0 += mixin[channel] * input0;
        a1 += mixin[channel] * input1;
        if (a0 < 0.0f)
            a0 *= 0.01f;
        if (a1 < 0.0f)
            a1 *= 0.01f;
        activation[0][channel] = a0;
        activation[1][channel] = a1;
        work->head[0][channel] += a0;
        work->head[1][channel] += a1;
    }
#undef PAIR_CHANNEL
#undef PAIR_ACC
    for (channel = 0; channel < 3; ++channel) {
        float r0 = rbias[channel], r1 = rbias[channel];
        r0 += residual[3 * channel] * activation[0][0];
        r1 += residual[3 * channel] * activation[1][0];
        r0 += residual[3 * channel + 1] * activation[0][1];
        r1 += residual[3 * channel + 1] * activation[1][1];
        r0 += residual[3 * channel + 2] * activation[0][2];
        r1 += residual[3 * channel + 2] * activation[1][2];
        work->current[0][channel] += r0;
        work->current[1][channel] += r1;
    }
    ++pos1;
    if (pos1 == capacity)
        pos1 = 0;
    *position = (uint16_t)pos1;
}

void compact_process_pair(const float *weights, float *history, CompactPairState *state,
                          float input0, float input1, float *output0, float *output1) {
    PairWork work;
    const float *w = weights + 3;
    int layer, channel;
    for (channel = 0; channel < 3; ++channel) {
        work.current[0][channel] = weights[channel] * input0;
        work.current[1][channel] = weights[channel] * input1;
        work.head[0][channel] = 0.0f;
        work.head[1][channel] = 0.0f;
    }
    for (layer = 0; layer < PAIR_LAYERS; ++layer) {
        int dilation = pair_dilations[layer];
        int capacity = 2 * dilation + 2;
        pair_layer(w, history, &state->layer_pos[layer], &work, dilation, input0, input1);
        history += 3 * capacity;
        w += 45;
    }
    {
        int pos0 = state->head_pos, pos1 = pos0 + 1;
        int channel, tap;
        float y0 = w[24], y1 = w[24];
        if (pos1 == 9)
            pos1 = 0;
        for (channel = 0; channel < 3; ++channel) {
            history[3 * pos0 + channel] = work.head[0][channel];
            history[3 * pos1 + channel] = work.head[1][channel];
        }
        for (channel = 0; channel < 3; ++channel) {
            int read0 = pos0 + 2, read1 = pos1 + 2;
            if (read0 >= 9)
                read0 -= 9;
            if (read1 >= 9)
                read1 -= 9;
            for (tap = 0; tap < 8; ++tap) {
                float coefficient = w[channel * 8 + tap];
                y0 += coefficient * history[read0 * 3 + channel];
                y1 += coefficient * history[read1 * 3 + channel];
                if (++read0 == 9)
                    read0 = 0;
                if (++read1 == 9)
                    read1 = 0;
            }
        }
        ++pos1;
        if (pos1 == 9)
            pos1 = 0;
        state->head_pos = (uint16_t)pos1;
        *output0 = y0 * w[25];
        *output1 = y1 * w[25];
    }
}
