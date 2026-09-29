/* Offline reduced WaveNet kernel; the caller owns and zeros history. */
#include "compact_kernel.h"

static const unsigned char dilations[COMPACT_LAYERS] = {
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239
};

#if COMPACT_CHANNELS == 3 && defined(N2Z_OPTIMIZED_KERNEL)
float compact_process(const float *weights, float *history, CompactState *state, float x)
{
    const float *w = weights + 3;
    float current0 = weights[0] * x;
    float current1 = weights[1] * x;
    float current2 = weights[2] * x;
    float head0 = 0.0f, head1 = 0.0f, head2 = 0.0f;
    int layer;

    for (layer = 0; layer < COMPACT_LAYERS; ++layer) {
        int dilation = dilations[layer];
        int length = 2 * dilation + 1;
        int pos = state->layer_pos[layer];
        const float *conv = w;
        const float *bias = conv + 27;
        const float *mixin = bias + 3;
        const float *one_by_one = mixin + 3;
        const float *one_bias = one_by_one + 9;
        float z0a = bias[0], z0b = 0.0f;
        float z1a = bias[1], z1b = 0.0f;
        float z2a = bias[2], z2b = 0.0f;
        float z0, z1, z2;
        float a0, a1, a2, residual;
        int start = (pos + 1) * 3;
        int step = dilation * 3;

        history[pos * 3] = current0;
        history[(pos + length) * 3] = current0;
        history[pos * 3 + 1] = current1;
        history[(pos + length) * 3 + 1] = current1;
        history[pos * 3 + 2] = current2;
        history[(pos + length) * 3 + 2] = current2;

#define ACCUM(LANE, IN, TAP) do { \
    float h = history[start + (IN) + (TAP) * step]; \
    z0##LANE += conv[3 * (IN) + (TAP)] * h; \
    z1##LANE += conv[9 + 3 * (IN) + (TAP)] * h; \
    z2##LANE += conv[18 + 3 * (IN) + (TAP)] * h; \
} while (0)
        ACCUM(a, 0, 0); ACCUM(b, 0, 1); ACCUM(a, 0, 2);
        ACCUM(b, 1, 0); ACCUM(a, 1, 1); ACCUM(b, 1, 2);
        ACCUM(a, 2, 0); ACCUM(b, 2, 1); ACCUM(a, 2, 2);
#undef ACCUM

        z0 = z0a + z0b;
        z1 = z1a + z1b;
        z2 = z2a + z2b;
        z0 += mixin[0] * x;
        z1 += mixin[1] * x;
        z2 += mixin[2] * x;
        a0 = z0 >= 0.0f ? z0 : z0 * 0.01f;
        a1 = z1 >= 0.0f ? z1 : z1 * 0.01f;
        a2 = z2 >= 0.0f ? z2 : z2 * 0.01f;
        head0 += a0;
        head1 += a1;
        head2 += a2;

        residual = one_bias[0];
        residual += one_by_one[0] * a0;
        residual += one_by_one[1] * a1;
        residual += one_by_one[2] * a2;
        current0 += residual;
        residual = one_bias[1];
        residual += one_by_one[3] * a0;
        residual += one_by_one[4] * a1;
        residual += one_by_one[5] * a2;
        current1 += residual;
        residual = one_bias[2];
        residual += one_by_one[6] * a0;
        residual += one_by_one[7] * a1;
        residual += one_by_one[8] * a2;
        current2 += residual;
        state->layer_pos[layer] = (uint16_t)(pos + 1 == length ? 0 : pos + 1);
        history += 6 * length;
        w = one_bias + 3;
    }

    {
        int pos = state->head_pos;
        float y = w[24];
        int tap;
        history[pos * 3] = head0;
        history[(pos + COMPACT_HEAD_KERNEL) * 3] = head0;
        history[pos * 3 + 1] = head1;
        history[(pos + COMPACT_HEAD_KERNEL) * 3 + 1] = head1;
        history[pos * 3 + 2] = head2;
        history[(pos + COMPACT_HEAD_KERNEL) * 3 + 2] = head2;
        for (tap = 0; tap < COMPACT_HEAD_KERNEL; ++tap)
            y += w[tap] * history[(pos + 1 + tap) * 3];
        for (tap = 0; tap < COMPACT_HEAD_KERNEL; ++tap)
            y += w[8 + tap] * history[(pos + 1 + tap) * 3 + 1];
        for (tap = 0; tap < COMPACT_HEAD_KERNEL; ++tap)
            y += w[16 + tap] * history[(pos + 1 + tap) * 3 + 2];
        state->head_pos = (uint16_t)(pos + 1 == COMPACT_HEAD_KERNEL ? 0 : pos + 1);
        return y * w[25];
    }
}
#else
float compact_process(const float *weights, float *history, CompactState *state, float x)
{
    const float *w = weights + COMPACT_CHANNELS;
    float current[COMPACT_CHANNELS];
    float head_sum[COMPACT_CHANNELS] = {0};
    int layer, in, out, tap;

    for (in = 0; in < COMPACT_CHANNELS; ++in)
        current[in] = weights[in] * x;

    for (layer = 0; layer < COMPACT_LAYERS; ++layer) {
        int dilation = dilations[layer];
        int length = 2 * dilation + 1;
        int pos = state->layer_pos[layer];
        const float *conv = w;
        const float *bias = conv + 3 * COMPACT_CHANNELS * COMPACT_CHANNELS;
        const float *mixin = bias + COMPACT_CHANNELS;
        const float *one_by_one = mixin + COMPACT_CHANNELS;
        const float *one_bias = one_by_one + COMPACT_CHANNELS * COMPACT_CHANNELS;
        float z[COMPACT_CHANNELS];
        float activation[COMPACT_CHANNELS];

        for (in = 0; in < COMPACT_CHANNELS; ++in) {
            history[pos * COMPACT_CHANNELS + in] = current[in];
            history[(pos + length) * COMPACT_CHANNELS + in] = current[in];
            z[in] = bias[in];
        }
        for (in = 0; in < COMPACT_CHANNELS; ++in) {
            int start = (pos + 1) * COMPACT_CHANNELS + in;
            int step = dilation * COMPACT_CHANNELS;
            for (tap = 0; tap < 3; ++tap) {
                float h = history[start + tap * step];
                for (out = 0; out < COMPACT_CHANNELS; ++out)
                    z[out] += conv[(out * COMPACT_CHANNELS + in) * 3 + tap] * h;
            }
        }
        for (out = 0; out < COMPACT_CHANNELS; ++out) {
            z[out] += mixin[out] * x;
            activation[out] = z[out] >= 0.0f ? z[out] : z[out] * 0.01f;
            head_sum[out] += activation[out];
        }
        for (out = 0; out < COMPACT_CHANNELS; ++out) {
            float residual = one_bias[out];
            for (in = 0; in < COMPACT_CHANNELS; ++in)
                residual += one_by_one[out * COMPACT_CHANNELS + in] * activation[in];
            current[out] += residual;
        }
        state->layer_pos[layer] = (uint16_t)(pos + 1 == length ? 0 : pos + 1);
        history += 2 * length * COMPACT_CHANNELS;
        w = one_bias + COMPACT_CHANNELS;
    }

    {
        int pos = state->head_pos;
        float y = w[COMPACT_HEAD_KERNEL * COMPACT_CHANNELS];
        for (in = 0; in < COMPACT_CHANNELS; ++in) {
            history[pos * COMPACT_CHANNELS + in] = head_sum[in];
            history[(pos + COMPACT_HEAD_KERNEL) * COMPACT_CHANNELS + in] = head_sum[in];
        }
        for (in = 0; in < COMPACT_CHANNELS; ++in) {
            int start = (pos + 1) * COMPACT_CHANNELS + in;
            for (tap = 0; tap < COMPACT_HEAD_KERNEL; ++tap)
                y += w[in * COMPACT_HEAD_KERNEL + tap] * history[start + tap * COMPACT_CHANNELS];
        }
        state->head_pos = (uint16_t)(pos + 1 == COMPACT_HEAD_KERNEL ? 0 : pos + 1);
        return y * w[COMPACT_HEAD_KERNEL * COMPACT_CHANNELS + 1];
    }
}
#endif
