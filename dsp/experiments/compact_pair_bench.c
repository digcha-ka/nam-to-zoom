/* Kernel-only comparison. No Zoom ABI, controls, allocation, or hardware writes. */
#ifdef BENCH_PAIR
#include "../nam_a2_compact/compact_pair.h"
#pragma FUNC_ALWAYS_INLINE(compact_process_pair)
#include "../nam_a2_compact/compact_pair.c"
typedef CompactPairState BenchState;
#else
#define COMPACT_CHANNELS 3
#define N2Z_OPTIMIZED_KERNEL 1
#include "../nam_a2_compact/compact_kernel.h"
#pragma FUNC_ALWAYS_INLINE(compact_process)
#include "../nam_a2_compact/compact_kernel.c"
typedef CompactState BenchState;
#endif

void compact_benchmark(const float *weights, float *history, BenchState *state,
                       const float *input, float *output, unsigned int count)
{
    unsigned int frame;
    for (frame = 0; frame < count; frame += 4) {
#ifdef BENCH_PAIR
        compact_process_pair(weights, history, state, input[frame], input[frame + 1],
                             output + frame, output + frame + 1);
        compact_process_pair(weights, history, state, input[frame + 2], input[frame + 3],
                             output + frame + 2, output + frame + 3);
#else
        unsigned int i;
        for (i = 0; i < 4; ++i)
            output[frame + i] = compact_process(weights, history, state, input[frame + i]);
#endif
    }
}
