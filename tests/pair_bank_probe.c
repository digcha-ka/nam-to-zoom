/* Run the actual scalar/pair bank callbacks under a 32-bit host ABI. */
#include <math.h>
#include <stdio.h>
#include <string.h>

void scalar_reset(void);
void pair_reset(void);
void scalar_run(float *bus, const float *coeff, int invalid_descriptor);
void pair_run(float *bus, const float *coeff, int invalid_descriptor);
int scalar_ready(void);
int pair_ready(void);
int scalar_guards(void);
int pair_guards(void);

static float params[16];
static unsigned int calls, compared;

static int check(float *a, float *b, int compare) {
    unsigned int i;
    scalar_run(a, params, 0);
    pair_run(b, params, 0);
    ++calls;
    if (!scalar_guards() || !pair_guards()) return 1;
    for (i = 0; i < 32; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return 2;
        if (compare && memcmp(&a[i], &b[i], sizeof(float))) return 3;
    }
    if (compare) compared += 32;
    return 0;
}

static int settle(void) {
    unsigned int i;
    for (i = 0; i < 160; ++i) {
        float a[32] = {0}, b[32] = {0};
        int result = check(a, b, 0);
        if (result) return result;
    }
    return (!scalar_ready() || !pair_ready()) ? 4 : 0;
}

int main(void) {
    unsigned int model, block, frame;
    if (sizeof(void *) != 4) return 99;
    scalar_reset();
    pair_reset();
    params[0] = params[10] = 1.0f;
    params[5] = params[6] = params[7] = params[8] = params[9] = 0.5f;
    for (model = 0; model < 10; ++model) {
        params[3] = (float)(model % 5) * 0.25f;
        if (settle()) return 10;
        for (block = 0; block < 320; ++block) {
            float a[32], b[32];
            int result;
            params[0] = block % 7 == 0 ? 0.0f : (block % 7 == 1 ? 0.25f : 1.0f);
            params[10] = block % 3 * 0.5f;
            params[5] = (block % 11) * 0.1f;
            params[6] = (block % 5) * 0.25f;
            params[7] = (block % 3) * 0.5f;
            params[8] = (block % 5) * 0.25f;
            params[9] = (block % 3) * 0.5f;
            for (frame = 0; frame < 16; ++frame) {
                a[frame] = (float)sin((block * 16 + frame) * 0.011) * 0.3f;
                a[frame + 16] = (float)cos((block * 16 + frame) * 0.017) * 0.2f;
            }
            if (block == 23) { a[0] = NAN; a[17] = INFINITY; }
            memcpy(b, a, sizeof(a));
            result = check(a, b, 1);
            if (result) { printf("failure=%d model=%u block=%u\n", result, model, block); return 20 + result; }
        }
    }
    for (block = 0; block < 3; ++block) {
        float a[32], b[32];
        for (frame = 0; frame < 32; ++frame) a[frame] = b[frame] = 0.5f;
        if (block == 0) params[10] = NAN;
        else params[10] = 1.0f;
        scalar_run(a, params, block == 1);
        pair_run(b, params, block == 1);
        if (block < 2) {
            for (frame = 0; frame < 32; ++frame)
                if (a[frame] != 0.0f || b[frame] != 0.0f) return 30;
        }
        if (!scalar_guards() || !pair_guards()) return 31;
    }
    /* Reload into fresh firmware-style state, then confirm normal processing. */
    scalar_reset(); pair_reset();
    params[3] = 0.0f;
    if (settle()) return 40;
    printf("PASS: %u callbacks; %u bit-exact compared channel samples; guards intact\n", calls, compared);
    return 0;
}
