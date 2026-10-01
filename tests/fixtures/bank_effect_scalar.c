/* Offline five-slot compact NAM bank. No pedal install is performed here. */
#include <stdint.h>
#include "sh_params.h"
#include "bank_config.h"

#define COMPACT_CHANNELS 3
#include "compact_kernel.h"
#pragma FUNC_ALWAYS_INLINE(compact_process)
#include "compact_kernel.c"

#define HISTORY_FLOATS COMPACT_MIRROR_HISTORY_FLOATS
#define HISTORY_BYTES (HISTORY_FLOATS * sizeof(float))
#define CLEAR_PER_BLOCK 512u
#define INITIALIZED 0x41324231u
#define RECOVERING 0x52454331u
#define CTX_GET_PARAM 40u
#define UI_PARAM_MODEL 2 /* host entries 0 and 1 precede the user parameters */

typedef int (*GetParamFn)(void *, int);

typedef struct {
    CompactState model;
    float low;
    float upper;
    uint32_t clear_count;
    uint32_t warm_count;
    uintptr_t base_seen;
    uint32_t span_seen;
    uint32_t active_model;
    uint32_t initialized;
} EffectState;

typedef char state_fits[(sizeof(EffectState) <= SH_STATE_BYTES) ? 1 : -1];
extern const uint32_t N2ZBankWeights[];

#pragma FUNC_ALWAYS_INLINE(silence)
static void silence(float *bus)
{
    unsigned int i;
    for (i = 0; i < 2u * SH_FRAMES; ++i) bus[i] = 0.0f;
}

#pragma FUNC_ALWAYS_INLINE(blend_output)
static float blend_output(float dry, float wet, float wet_gain)
{
    return dry * (1.0f - wet_gain) + wet * wet_gain;
}

static void mute_wet(float *bus, float wet_gain)
{
    unsigned int i;
    for (i = 0; i < 2u * SH_FRAMES; ++i) {
        float dry = bus[i];
        bus[i] = dry > -8.0f && dry < 8.0f ? dry * (1.0f - wet_gain) : 0.0f;
    }
}

static float tone_process(EffectState *state, float wet, float bass,
                          float mid, float treble)
{
    state->low += 0.034992074f * (wet - state->low);   /* 250 Hz */
    state->upper += 0.299660207f * (wet - state->upper); /* 2.5 kHz */
    return 2.0f * (bass * state->low +
                   mid * (state->upper - state->low) +
                   treble * (wet - state->upper));
}

static void effect_process(void **instance, void **ctx)
{
    EffectState *state = (EffectState *)instance[2];
    float *coeff = (float *)instance[1];
    uintptr_t descriptor_address = (uintptr_t)instance[3];
    const uint32_t *descriptor;
    uintptr_t base, end;
    uint32_t span, selected;
    float *history, fade, selector, threshold, step;
    float bass, mid, treble, volume, input, mix, wet_gain;
    const float *weights;
    float *bus = (float *)ctx[SH_CTX_EFF];
    unsigned int i;

    if (state == 0 || coeff == 0 || descriptor_address == 0 ||
        (descriptor_address & 3u) != 0 ||
        ((const uint32_t *)state)[SH_STATE_GUARD_WORD] != SH_STATE_GUARD) {
        silence(bus);
        return;
    }
    descriptor = (const uint32_t *)descriptor_address;
    base = (uintptr_t)descriptor[0];
    end = (uintptr_t)descriptor[1];
    span = descriptor[2];
    fade = coeff[SH_COEFF_BYPASS];
    selector = coeff[SH_PARAM_MODEL];
    bass = coeff[SH_PARAM_BASS];
    mid = coeff[SH_PARAM_MID];
    treble = coeff[SH_PARAM_TREBLE];
    volume = coeff[SH_PARAM_VOL];
    input = coeff[SH_PARAM_INPUT];
    mix = coeff[SH_PARAM_MIX];
    if (base == 0 || (base & 3u) != 0 || end <= base ||
        end - base != span || span < HISTORY_BYTES ||
        !(fade >= 0.0f && fade <= 1.0f) ||
        !(bass >= 0.0f && bass <= 1.0f) ||
        !(mid >= 0.0f && mid <= 1.0f) ||
        !(treble >= 0.0f && treble <= 1.0f) ||
        !(volume >= 0.0f && volume <= 1.0f) ||
        !(input >= 0.0f && input <= 1.0f) ||
        !(mix >= 0.0f && mix <= 1.0f) ||
        !(selector >= 0.0f && selector <= 1.0f)) {
        silence(bus);
        return;
    }
    wet_gain = fade * mix;
    selected = 0;
    step = 1.0f / (float)BANK_SELECTOR_MAX;
    threshold = 0.5f * step;
    while (selected < BANK_SELECTOR_MAX && selector >= threshold) {
        ++selected;
        threshold += step;
    }
    if ((state->initialized == INITIALIZED || state->initialized == RECOVERING) &&
        state->active_model < BANK_MODEL_COUNT && selected != state->active_model &&
        instance[0] != 0 && ctx[CTX_GET_PARAM] != 0) {
        int raw_model = ((GetParamFn)ctx[CTX_GET_PARAM])(instance[0], UI_PARAM_MODEL);
        if (raw_model >= 0 && (uint32_t)raw_model <= BANK_SELECTOR_MAX) {
            selected = (uint32_t)raw_model;
            coeff[SH_PARAM_MODEL] = (float)selected * step;
        }
    }
    if (selected >= BANK_MODEL_COUNT) {
        state->initialized = 0;
        return; /* the one-model bank's EMPTY selector is dry pass-through */
    }
    history = (float *)base;
    weights = (const float *)N2ZBankWeights + selected * BANK_WORDS_PER_MODEL;

    if (state->initialized != INITIALIZED || state->base_seen != base ||
        state->span_seen != span || state->active_model != selected ||
        state->clear_count > HISTORY_FLOATS ||
        state->warm_count > COMPACT_RECEPTIVE_FIELD) {
        for (i = 0; i < COMPACT_LAYERS; ++i) state->model.layer_pos[i] = 0;
        state->model.head_pos = 0;
        state->low = 0.0f;
        state->upper = 0.0f;
        state->clear_count = 0;
        state->warm_count = 0;
        state->base_seen = base;
        state->span_seen = span;
        state->active_model = selected;
        state->initialized = INITIALIZED;
    }
    if (state->clear_count < HISTORY_FLOATS) {
        unsigned int remaining = HISTORY_FLOATS - state->clear_count;
        unsigned int count = remaining < CLEAR_PER_BLOCK ? remaining : CLEAR_PER_BLOCK;
        for (i = 0; i < count; ++i) history[state->clear_count + i] = 0.0f;
        state->clear_count += count;
        mute_wet(bus, wet_gain);
        return;
    }
    if (state->warm_count < COMPACT_RECEPTIVE_FIELD) {
        unsigned int remaining = COMPACT_RECEPTIVE_FIELD - state->warm_count;
        unsigned int count = remaining < SH_FRAMES ? remaining : SH_FRAMES;
        for (i = 0; i < count; ++i)
            (void)compact_process(weights, history, &state->model, 0.0f);
        state->warm_count += count;
        mute_wet(bus, wet_gain);
        return;
    }
    for (i = 0; i < SH_FRAMES; ++i) {
        float dry_l = bus[i];
        float dry_r = bus[i + SH_CH_B_OFFSET];
        float wet, out_l, out_r;
        if (!(dry_l > -8.0f && dry_l < 8.0f)) dry_l = 0.0f;
        if (!(dry_r > -8.0f && dry_r < 8.0f)) dry_r = 0.0f;
        wet = compact_process(weights, history, &state->model, (dry_l + dry_r) * input);
        if (!(wet > -8.0f && wet < 8.0f)) {
            /* Poisoned recurrent history cannot recover without a full clear. */
            state->initialized = RECOVERING;
            silence(bus);
            return;
        }
        wet = tone_process(state, wet, bass, mid, treble) * (2.0f * volume);
        out_l = blend_output(dry_l, wet, wet_gain);
        out_r = blend_output(dry_r, wet, wet_gain);
        bus[i] = out_l > -8.0f && out_l < 8.0f ? out_l : 0.0f;
        bus[i + SH_CH_B_OFFSET] = out_r > -8.0f && out_r < 8.0f ? out_r : 0.0f;
    }
}

void SH_AUDIO_FN(void **instance, void **ctx)
{
    effect_process(instance, ctx);
}
