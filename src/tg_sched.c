#include "tg_sched.h"
#include "ovg_error.h"

#include <math.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static float sched_warm(int step, int warmup_steps) {
    return (step < warmup_steps) ? (float)step / (float)warmup_steps : 1.0f;
}

static float sched_progress(int step, int total_steps) {
    if (total_steps <= 0)
        ovg_fatal("tg_lr schedule: total_steps must be > 0 (got %d)", total_steps);
    float p = (float)(step - 1) / (float)total_steps;
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    return p;
}

float tg_lr_warmup_cosine(int step, int total_steps, int warmup_steps, float base, float floor_lr) {
    float progress = sched_progress(step, total_steps);
    float warm     = sched_warm(step, warmup_steps);
    float cosine   = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return (floor_lr + (base - floor_lr) * cosine) * warm;
}

float tg_lr_warmup_linear(int step, int total_steps, int warmup_steps, float base, float floor_lr) {
    float progress = sched_progress(step, total_steps);
    float warm     = sched_warm(step, warmup_steps);
    return (base + (floor_lr - base) * progress) * warm;
}
