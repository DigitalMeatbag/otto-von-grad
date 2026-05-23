/* tg_rng.c — RNG state and seeding for otto-von-grad.
 *
 * Owns two RNG streams:
 *   rand()        — C stdlib PRNG used by tg_fill_randn / tg_fill_uniform
 *   s_dropout_rng — xorshift32 PRNG used by tg_dropout
 *
 * Both are seeded together by tg_seed() so a single logged uint32_t value
 * is sufficient to reproduce a training run.
 */

/* _CRT_RAND_S must be defined before any stdlib.h include on MSVC. */
#if defined(_WIN32)
#  define _CRT_RAND_S
#endif

#include "tg_rng.h"
#include "ovg_error.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if !defined(_WIN32)
#  include <time.h>
#  if defined(__APPLE__)
     /* arc4random() is available on all Apple platforms — no header needed. */
#  else
#    include <unistd.h>  /* getpid() */
#  endif
#endif

/* ── xorshift32 state ────────────────────────────────────────────────────── */

static uint32_t s_dropout_rng = 0xdeadbeef;

uint32_t tg_rng_xorshift32(void) {
    s_dropout_rng ^= s_dropout_rng << 13;
    s_dropout_rng ^= s_dropout_rng >> 17;
    s_dropout_rng ^= s_dropout_rng << 5;
    return s_dropout_rng;
}

float tg_rng_uniform(void) {
    return tg_rng_xorshift32() / (float)0x100000000ULL;
}

/* ── Seeding ─────────────────────────────────────────────────────────────── */

void tg_seed(uint32_t seed) {
    /* xorshift32 is broken in the all-zeros state; use a non-zero fallback. */
    s_dropout_rng = (seed == 0) ? 0xdeadbeef : seed;
    srand(seed);
    printf("[ovg] rng seed: 0x%08x\n", seed);
    fflush(stdout);
}

void tg_seed_from_entropy(void) {
    uint32_t seed = 0;

#if defined(_WIN32)
    unsigned int tmp = 0;
    if (rand_s(&tmp) != 0)
        ovg_fatal("tg_seed_from_entropy: rand_s() failed");
    seed = (uint32_t)tmp;

#elif defined(__APPLE__)
    seed = arc4random();

#else
    /* Try /dev/urandom; fall back to time ^ pid. */
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        if (fread(&seed, sizeof(seed), 1, f) != 1)
            seed = 0;
        fclose(f);
    }
    if (seed == 0)
        seed = (uint32_t)time(NULL) ^ (uint32_t)getpid();
#endif

    tg_seed(seed);
}
