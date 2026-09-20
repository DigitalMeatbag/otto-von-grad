#ifndef TG_RNG_H
#define TG_RNG_H

#include <stdint.h>

/* Seed both rand() and the dropout xorshift RNG from the same value.
 * Logs the seed to stdout so a run can be reproduced via tg_seed().
 * Note: rand() is process-global; for replay to be exact, tg_seed() must be
 * the only srand() call in the process and must occur before any RNG use.
 * Ordering contract: tg_seed() overwrites the xorshift word, so a resume load
 * (tg_checkpoint_load_run) must run AFTER any tg_seed / tg_seed_from_entropy
 * call, or the seed clobbers the restored state. */
void tg_seed(uint32_t seed);

/* Seed from OS entropy (rand_s on Windows, arc4random on Apple,
 * /dev/urandom on Linux, time^pid fallback elsewhere).
 * Calls tg_seed() internally — the chosen seed is logged to stdout. */
void tg_seed_from_entropy(void);

/* xorshift32 step — called by tg_ops.c for dropout mask generation. */
uint32_t tg_rng_xorshift32(void);

/* Uniform float in [0, 1) — wraps xorshift32. */
float tg_rng_uniform(void);

/* The xorshift32 word — saved in checkpoints and restored on resume. */
uint32_t tg_rng_get_state(void);
/* Fatal if s == 0 (xorshift32 is stuck at zero). */
void     tg_rng_set_state(uint32_t s);

#endif /* TG_RNG_H */
