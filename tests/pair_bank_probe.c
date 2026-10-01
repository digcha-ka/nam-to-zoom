/* Run the actual scalar/pair bank callbacks under a 32-bit host ABI. */
#include <math.h>
#include <stdio.h>
#include <string.h>

void scalar_reset(void);
void pair_reset(void);
void scalar_run(float *bus, float *coeff, int invalid_descriptor);
void pair_run(float *bus, float *coeff, int invalid_descriptor);
void scalar_set_ui_model(int model);
void pair_set_ui_model(int model);
void scalar_poison_history(void);
void pair_poison_history(void);
unsigned scalar_active_model(void);
unsigned pair_active_model(void);
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

static int stereo_input(void) {
    float expected[16][32];
    unsigned route, block, frame;
    params[0] = params[10] = 1.0f;
    params[3] = 0.0f;
    params[5] = params[6] = params[7] = params[8] = params[9] = 0.5f;
    /* Left-only, right-only, and two half-level copies must drive NAM identically. */
    for (route = 0; route < 3; ++route) {
        scalar_reset(); pair_reset();
        if (settle()) return 41;
        for (block = 0; block < 16; ++block) {
            float a[32] = {0}, b[32];
            for (frame = 0; frame < 16; ++frame) {
                float sample = (float)sin((block * 16 + frame) * 0.11) * 0.3f;
                if (route == 0) a[frame] = sample;
                else if (route == 1) a[frame + 16] = sample;
                else a[frame] = a[frame + 16] = sample * 0.5f;
            }
            memcpy(b, a, sizeof(a));
            if (check(a, b, 1)) return 42;
            if (route == 0) memcpy(expected[block], b, sizeof(b));
            else if (memcmp(expected[block], b, sizeof(b))) return 43;
            for (frame = 0; frame < 16; ++frame)
                if (b[frame] != b[frame + 16]) return 44;
        }
    }
    for (route = 0; route < 2; ++route) {
        float a[32], b[32], dry[32];
        params[0] = route == 0 ? 1.0f : 0.0f;
        params[10] = route == 0 ? 0.0f : 1.0f;
        for (frame = 0; frame < 16; ++frame) {
            a[frame] = (float)frame * 0.01f;
            a[frame + 16] = -(float)frame * 0.02f;
        }
        memcpy(b, a, sizeof(a));
        memcpy(dry, a, sizeof(a));
        if (check(a, b, 1)) return 45;
        for (frame = 0; frame < 32; ++frame)
            if (b[frame] != dry[frame]) return 45;
    }
    return 0;
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
    /* Host entry 0 is on/off; Model is entry 2. Exercise switching back to slot 0. */
    {
        const unsigned selections[] = {0, 1, 0, 4, 0};
        unsigned index;
        scalar_set_ui_model(-1); pair_set_ui_model(-1);
        for (index = 0; index < sizeof(selections) / sizeof(selections[0]); ++index) {
            params[3] = (float)selections[index] * 0.25f;
            if (settle()) return 46;
            if (scalar_active_model() != selections[index] ||
                pair_active_model() != selections[index] ||
                params[3] != (float)selections[index] * 0.25f) return 47;
        }
    }
    /* Tuner-style scratch-memory corruption must trigger an in-place recovery. */
    params[3] = 0.75f;
    if (settle()) return 34;
    scalar_poison_history(); pair_poison_history();
    scalar_set_ui_model(3); pair_set_ui_model(3);
    params[3] = 0.0f;
    {
        float a[32], b[32];
        for (frame = 0; frame < 32; ++frame) a[frame] = b[frame] = 0.25f;
        scalar_run(a, params, 0);
        if (params[3] != 0.75f) return 35;
        params[3] = 0.0f;
        pair_run(b, params, 0);
        if (params[3] != 0.75f || scalar_ready() || pair_ready()) return 36;
        for (frame = 0; frame < 32; ++frame)
            if (a[frame] != 0.0f || b[frame] != 0.0f) return 37;
    }
    scalar_set_ui_model(-1); pair_set_ui_model(-1);
    if (settle()) return 38;
    if (scalar_active_model() != 3 || pair_active_model() != 3) return 39;
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
    {
        int result = stereo_input();
        if (result) return result;
    }
    printf("PASS: %u callbacks; %u bit-exact compared channel samples; guards intact\n", calls, compared);
    return 0;
}
