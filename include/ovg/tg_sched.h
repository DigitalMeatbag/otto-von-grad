#ifndef TG_SCHED_H
#define TG_SCHED_H

/* Learning-rate schedules: pure functions of the 1-based step being taken.
   warmup_steps may be 0; total_steps must be > 0. Both share:
     warm     = (step < warmup_steps) ? (float)step / (float)warmup_steps : 1.0f
     progress = clamp((float)(step - 1) / (float)total_steps, 0, 1)
   warm scales the whole value, floor included, so during warmup lr rises linearly
   from base/warmup_steps toward the decay curve. A constant LR needs no function. */

/* Cosine decay from base to floor over total_steps:
   lr = (floor + (base - floor) * 0.5 * (1 + cos(pi * progress))) * warm */
float tg_lr_warmup_cosine(int step, int total_steps, int warmup_steps, float base, float floor_lr);

/* Linear decay from base to floor over total_steps:
   lr = (base + (floor - base) * progress) * warm */
float tg_lr_warmup_linear(int step, int total_steps, int warmup_steps, float base, float floor_lr);

#endif /* TG_SCHED_H */
