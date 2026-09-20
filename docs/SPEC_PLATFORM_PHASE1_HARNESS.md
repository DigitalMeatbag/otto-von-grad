# Spec: Platform Phase 1 — Training Harness

> **Status:** Implemented 2026-09-20 (library at `55f3472`, candide `e9132a9`, lambda `4c09cb4`, vexilloscope `ad6a75c`); every Acceptance item is checked. Derived from `docs/FOUNDATION_PLATFORM.md` (Training Harness decision, closed 2026-09-20). `AGENTS.md` describes what exists today; this document describes exactly what Phase 1 adds, and is complete when every item in [Acceptance](#acceptance) is checked.

---

## Purpose

Absorb the training machinery that `examples/candide.c`, `../lambda`, and `../vexilloscope` each hand-roll — Adam state, learning-rate schedules, gradient accumulation, clipping, the step counter, the eval-mode toggle, and checkpoints that resume exactly — into `ovg_core`, so a training loop written against the library reads as a loop and a stopped run continues where it stopped.

**Completion signal:** all three consumers train through the harness; vexilloscope's `main.c` shrinks by the harness code it no longer owns; a run stopped and restarted from a v3 checkpoint produces the same parameters as an uninterrupted run.

---

## Decisions Inherited From the Foundation Document

These are settled and this spec does not reopen them:

1. The harness lives in `ovg_core`, finishing `tg_train.h`. No fourth target.
2. Exact-resume is default on. Checkpoints written by the harness carry optimizer state, step, and RNG state.
3. Load takes an explicit mode: `resume` (default) or `init-from-weights`.
4. Format v3 with a new magic. v2 parameter-only files still load; moments are zeroed and step reset to 0, and the load reports that it did so.
5. Schedules are pure functions of step. No schedule position is stored.
6. The checkpoint header carries the RNG state (the xorshift32 word), a format version, and a caller-supplied config blob.
7. Device-resident moments are synced to host on save by the harness, not the caller.
8. No data loader, no batch abstraction, no callbacks. The harness steps on a loss tensor the caller produced.

Decisions this spec makes (assigned to it by the foundation document):

| Question | Decision |
|---|---|
| Exact v3 header fields | See [Format v3](#format-v3). |
| Weights-only export: v2 or v3? | **v3 with the optimizer flag clear.** One format is written (both save functions emit v3); v2 becomes read-only legacy. |
| Existing `tg_checkpoint_save` / `tg_checkpoint_load` signatures | **Unchanged.** They become the weights-only pair. The state-carrying pair is new. `tests/test_checkpoint.c`'s existing tests must pass without edits. |
| Does the harness own the schedule? | No. Schedules are free functions; the caller computes `lr` and passes it to the update. |
| vexilloscope's weight format | **Not migrated in Phase 1.** See [vexilloscope](#vexilloscope). |

---

## Definitions

- **Step.** The number of completed optimizer updates. `TgAdam.step` is 0 after creation and increments inside `tg_adam_update`. Adam bias correction uses this counter. A loop resuming from a checkpoint runs `for (step = opt.step + 1; ...)` (candide is the one exception: its per-run counter stays `1..steps` and `opt.step` runs alongside it — see [candide](#examplescandidec)). Schedules take the 1-based number of the step being taken.
- **Exact resume.** After `tg_checkpoint_load_run(..., TG_LOAD_RESUME)`, the parameters, Adam moments, step counter, and the library RNG state are what they were at save time. Continuing the loop from `opt.step + 1` produces the same parameters as the uninterrupted run, provided the application feeds the same data in the same order. The harness restores **library** randomness only (dropout, drop-path, sampling — everything on the xorshift32 stream). Application-side `rand()` calls (vexilloscope's balanced sampling, `img_augment`) are not restorable; an application that wants its data order restored stores its own cursor in the config blob, or draws from `tg_rng_uniform`.
- **Micro-step.** One forward + backward whose gradient contributes `1/n_micro` of a step's gradient.

---

## Deliverables

### 1. `tg_optim.h` / `src/tg_optim.c` — `TgAdam`

New public header in `include/ovg/`, new source in `ovg_core`.

```c
#include "tg_tensor.h"

typedef struct {
    Tensor **params;     /* borrowed; the caller keeps the array and the tensors alive */
    int      n_params;
    float  **m, **v;     /* per-param first/second moments; host buffers, or device buffers when on_cuda */
    int      on_cuda;    /* 1 if params live on the device; fixed at create */
    int      step;       /* completed updates; 0 after create/reset */
    float    beta1, beta2, eps;
} TgAdam;

/* Allocates zeroed m/v matching each param's numel, on the device the params are on.
   Fatal if n_params <= 0, if params mix CPU and CUDA, or on allocation failure.
   Create AFTER moving params to their final device (tg_to_cuda). */
TgAdam tg_adam_create(Tensor **params, int n_params, float beta1, float beta2, float eps);
void   tg_adam_free(TgAdam *opt);       /* frees m/v (host or device); does not touch params */
void   tg_adam_reset(TgAdam *opt);      /* zeroes m/v, step = 0 */

/* One optimizer step over accumulated grads.
   step += 1, then Adam with bias correction at t = step.
   If max_grad_norm > 0, clips first via tg_clip_grad_norm(params, n, max_grad_norm, 1e-6f).
   Returns the pre-clip gradient norm on CPU; returns 0.0f on CUDA or when not clipping
   (no host sync is added — see Performance Posture). */
float  tg_adam_update(TgAdam *opt, float lr, float max_grad_norm);

/* Accumulation helpers. A step is:
     tg_adam_zero_grads(&opt);
     for each micro-step: loss = <caller forward>; tg_adam_accumulate(&opt, loss, n_micro);
     tg_adam_update(&opt, lr, max_grad_norm);                                            */
void   tg_adam_zero_grads(TgAdam *opt);                          /* tg_zero_grads over params */
void   tg_adam_accumulate(TgAdam *opt, Tensor *loss, int n_micro);
```

`tg_adam_accumulate`: if `n_micro > 1`, `loss = tg_scale(loss, 1.0f / n_micro)`; then `tg_backward_accum(loss)`; then `tg_free_graph(loss)`. `loss` must be a non-persistent root; the caller reads its value (`tg_scalar_value`) **before** calling, because the graph is freed on return. Only non-persistent nodes are freed: persistent caller inputs (vexilloscope's patchified `p`, marked persistent so the graph walk stops there) remain the caller's to free, as today. `n_micro < 1` is fatal.

`tg_adam_update` dispatches to the existing `tg_adam_step` (CPU) or `tg_adam_step_gpu` (device) with `opt->step` after increment; the raw functions are unchanged and remain public. The struct path must be numerically identical to the raw path (tested).

`tg_sgd_step` stays as is. No SGD struct — no consumer uses SGD.

### 2. `tg_sched.h` / `src/tg_sched.c` — learning-rate schedules

Pure functions. `step` is 1-based; `warmup_steps` may be 0; `total_steps > 0`.

```c
/* Both functions share:
     warm     = (step < warmup_steps) ? (float)step / (float)warmup_steps : 1.0f
     progress = clamp((float)(step - 1) / (float)total_steps, 0, 1)
   (float division throughout — never int/int). warm scales the whole value, floor included,
   so during warmup lr rises linearly from base/warmup_steps toward the decay curve. */

/* Cosine decay from base to floor over total_steps:
   lr = (floor + (base - floor) * 0.5 * (1 + cos(pi * progress))) * warm                      */
float tg_lr_warmup_cosine(int step, int total_steps, int warmup_steps, float base, float floor_lr);

/* Linear decay from base to floor over total_steps:
   lr = (base + (floor - base) * progress) * warm                                             */
float tg_lr_warmup_linear(int step, int total_steps, int warmup_steps, float base, float floor_lr);
```

With `floor_lr = 0`, `tg_lr_warmup_cosine` is vexilloscope's schedule (`main.c:1174-1176`): `base * 0.5 * (1 + cos(pi * (step-1) / total)) * (step < warmup ? step / warmup : 1)`. The two agree to float rounding (vexilloscope evaluates `M_PI * (step-1) / total` left to right; the library computes `progress` first), which `test_lr_warmup_cosine` checks at every step of the vexilloscope grid. A constant LR needs no function; the caller passes a constant.

### 3. Eval guard — additions to `tg_train.h` (implemented in `src/tg_train.c`)

```c
/* Sets tg_training = 0 and returns the previous value. Pair with tg_eval_end. */
int  tg_eval_begin(void);
void tg_eval_end(int prev_training);

/* Running mean for an eval metric. */
typedef struct { double sum; int n; } TgMeter;
static inline void  tg_meter_add(TgMeter *m, float x) { m->sum += x; m->n += 1; }
static inline float tg_meter_mean(const TgMeter *m) { return m->n ? (float)(m->sum / m->n) : 0.0f; }
```

This is the whole eval-loop skeleton. The caller writes the loop; the library owns only the mode toggle and the arithmetic. No callbacks.

### 4. RNG state — additions to `tg_rng.h` (implemented in `src/tg_rng.c`)

```c
uint32_t tg_rng_get_state(void);        /* the xorshift32 word */
void     tg_rng_set_state(uint32_t s);  /* fatal if s == 0 (xorshift32 is stuck at zero) */
```

**Ordering contract.** `tg_seed()` overwrites the xorshift word. A resume load must run **after** any `tg_seed` / `tg_seed_from_entropy` call, or the seed clobbers the restored state. Document this on both `tg_seed` and `tg_checkpoint_load_run`.

### 5. Device float buffers — additions to `src/tg_cuda_internal.h` (CUDA builds only)

Moments are raw `float*` from `tg_cuda_malloc_floats`, not tensors, so `tg_from_cuda` does not apply and the only readback today is `tg_cuda_read_float` (one value). Add:

```c
void tg_cuda_upload_floats(float *dst_dev, const float *src_host, int n);
void tg_cuda_download_floats(float *dst_host, const float *src_dev, int n);
void tg_cuda_zero_floats(float *p, int n);
```

Internal — used by `tg_optim.c` and `tg_checkpoint.c`; not added to `tg_cuda.h`.

### 6. Checkpoint format v3 — `tg_checkpoint.h` / `src/tg_checkpoint.c`

#### Format v3

Little-endian, fixed-width. The optimizer section comes **after** the parameters so the weights-only reader can stop early, and the config comes **before** the parameters so a model can be built from the file before any tensor exists.

```text
uint32   magic        = 0x00475633   ("OVG3"; v2 was 0x00475632, v1 0x00475643)
uint32   flags        bit 0: optimizer section present. All other bits must be 0 on read.
int32    step         completed updates at save (0 for weights-only)
uint32   rng_state    xorshift32 word at save (tg_rng_get_state()) when flags bit 0 is set; 0 otherwise
int32    config_len   >= 0
uint8    config[config_len]
int32    n_params
per param:
  int32  ndim
  int32  shape[TG_MAX_DIMS]       (unused trailing entries 0)
  float  data[numel]
if flags & 1:
  int32  optimizer_kind = 1        (Adam)
  float  beta1, beta2, eps
  per param:
    float m[numel]
    float v[numel]
```

Validation on read, all failing with a message on stderr and `-1`. Header checks apply to all three readers (`tg_checkpoint_info`, `tg_checkpoint_load`, `tg_checkpoint_load_run`): magic ∈ {v2, v3}; unknown flag bits; `config_len < 0`; short read. Parameter checks apply to both loaders: `n_params` mismatch; per-param `ndim`/shape mismatch. Optimizer-section checks apply only when `tg_checkpoint_load_run` is restoring the section (RESUME on a file with flags bit 0 set): `optimizer_kind != 1`; beta1/beta2/eps differing from the target `TgAdam`'s (compared as floats, exact); `rng_state == 0` (xorshift32 cannot produce it, so it can only be corruption — reject rather than let `tg_rng_set_state` fatal). `INIT_FROM_WEIGHTS` and `tg_checkpoint_load` skip the section unread. No reader allocates for the config: `tg_checkpoint_info` copies at most `config_cap` bytes, the loaders `fseek` past it, so `config_len` needs no upper bound.

**Atomic-ish write.** Both writers write to `<path>.tmp`, then remove `<path>` and rename `<path>.tmp` → `<path>`. A process killed mid-save leaves the previous checkpoint intact. On any write error the `.tmp` is removed and `-1` returned.

#### API

```c
#define TG_CHECKPOINT_MAGIC_V2 0x00475632u
#define TG_CHECKPOINT_MAGIC_V3 0x00475633u

typedef struct {
    int      version;        /* 2 or 3 */
    int      has_optimizer;
    int      step;
    uint32_t rng_state;
    int      config_len;
    int      n_params;
} TgCheckpointInfo;

typedef enum {
    TG_LOAD_RESUME = 0,             /* params + optimizer state + step + RNG state (default) */
    TG_LOAD_INIT_FROM_WEIGHTS = 1   /* params only; optimizer reset to step 0; RNG untouched */
} TgLoadMode;

/* Unchanged signatures. Weights-only. save writes v3 with flags = 0, step = 0,
   rng_state = 0, config_len = 0. load accepts v2 and v3 and reads params only. */
int tg_checkpoint_save(const char *path, Tensor **params, int n);
int tg_checkpoint_load(const char *path, Tensor **params, int n);

/* Reads the header only. Copies min(config_len, config_cap) bytes into config_out
   (config_out may be NULL when config_cap == 0). Returns 0, or -1 on error / missing file.
   v2 files report version = 2, has_optimizer = 0, step = 0, rng_state = 0, config_len = 0. */
int tg_checkpoint_info(const char *path, TgCheckpointInfo *info, void *config_out, int config_cap);

/* Full run state: opt->params, opt's m/v/step/betas/eps, the current RNG state, and config.
   Device params and moments are synced to host by this call. */
int tg_checkpoint_save_run(const char *path, const TgAdam *opt, const void *config, int config_len);

/* Loads into opt->params (uploading to the device if they are on one).
   RESUME on a file with an optimizer section: restores m/v (uploaded if on_cuda), step, RNG state.
   RESUME on a v2 file or a v3 file without an optimizer section: params loaded, tg_adam_reset(opt),
     RNG untouched (the header's rng_state is ignored), and a notice on stdout:
     "[ovg] checkpoint <path>: no optimizer state; moments zeroed, step reset to 0".
   INIT_FROM_WEIGHTS: params loaded, tg_adam_reset(opt), RNG untouched, optimizer section skipped.
   info may be NULL. Returns 0, or -1 on any error (nothing partially applied to opt on -1
   is NOT guaranteed for params — callers treat -1 as fatal for the run). */
int tg_checkpoint_load_run(const char *path, TgAdam *opt, TgLoadMode mode, TgCheckpointInfo *info);
```

The foundation document's "flag that disables optimizer state for weights-only exports" is realised as the choice of function: `tg_checkpoint_save` is the weights-only export, `tg_checkpoint_save_run` is the default for a training loop.

### 7. Build

`CMakeLists.txt`: add `src/tg_optim.c` and `src/tg_sched.c` to `OVG_CORE_SOURCES`; add `tests/test_optim.c` to the test sources. `ovg_core` must still build standalone (`--target ovg_core`).

---

## The Loop, As Written Against the Harness

For reference and for the consumer migrations. This is what "reads as a loop" means.

```c
tg_seed(seed);                                   /* before load_run — see ordering contract */
Tensor *params[N]; int n = collect(params);
#ifdef OVG_CUDA_ENABLED
for (int i = 0; i < n; i++) tg_to_cuda(params[i]);
#endif
TgAdam opt = tg_adam_create(params, n, 0.9f, 0.999f, 1e-8f);
if (tg_checkpoint_load_run(path, &opt, TG_LOAD_RESUME, NULL) == 0)
    printf("resumed at step %d\n", opt.step);

for (int step = opt.step + 1; step <= total_steps; step++) {
    float lr = tg_lr_warmup_cosine(step, total_steps, warmup_steps, base_lr, 0.0f);
    tg_adam_zero_grads(&opt);
    for (int b = 0; b < n_micro; b++) {
        Tensor *loss = forward(...);             /* caller-owned */
        /* tg_scalar_value is a host sync on CUDA: read only when logging,
           and before accumulate frees the graph. */
        if (step % log_every == 0)
            train_loss += tg_scalar_value(loss);
        tg_adam_accumulate(&opt, loss, n_micro);
    }
    tg_adam_update(&opt, lr, 1.0f);              /* opt.step == step here */

    if (step % eval_every == 0) {
        int prev = tg_eval_begin();
        TgMeter m = {0};
        /* held-out forward passes; tg_meter_add(&m, value); tg_free_graph(...) */
        tg_eval_end(prev);
    }
    if (step % save_every == 0 || step == total_steps)
        tg_checkpoint_save_run(path, &opt, &cfg, sizeof cfg);
}
tg_adam_free(&opt);
```

---

## Consumer Migrations

Each consumer is migrated in its own repo (candide in this one), after the library work lands and its tests pass. Each migration is one commit per repo.

### `examples/candide.c`

- Replace the `m_buf`/`v_buf` allocation and the raw `tg_adam_step` with `TgAdam`. Order becomes create model → collect params → `tg_adam_create` → `load_run` (today the load precedes the buffer allocation, `candide.c:100-117`).
- Replace `tg_checkpoint_load` / `tg_checkpoint_save` with `tg_checkpoint_load_run(..., TG_LOAD_RESUME, ...)` / `tg_checkpoint_save_run` (config blob: a `TgGPTConfig`).
- **Periodic saves.** `save_run` at the existing eval site (the `step % 200 == 0` half of the condition at `candide.c:137`, not the `step == 1` half) and at the end, so a Ctrl-C loses at most 200 steps. Today the only save is at the end (`candide.c:179`).
- The demo keeps its "train `steps` more steps per run" behaviour: the per-run loop counter stays `1..steps`; `opt.step` is cumulative and is logged alongside it (`step %4d/%d (total %d)`). Bias correction now uses the cumulative count, which is the correct behaviour the comment at `candide.c:106-107` ("a resumed run restarts momentum from scratch") apologises for. Exact-resume equivalence (same parameters as an uninterrupted run) is proven by `test_checkpoint_exact_resume`, not by the demo, whose data order follows the per-run counter.
- The step body `tg_backward(loss)` / `tg_adam_step(...)` / `tg_free_graph(loss)` (`candide.c:134-135`, `156`) becomes `tg_adam_zero_grads(&opt)` / `tg_adam_accumulate(&opt, loss, 1)` / `tg_adam_update(&opt, lr, 0.0f)`. The trailing `tg_free_graph(loss)` **must be deleted** — accumulate already freed the graph, and a second walk is a use-after-free, not a leak. `tg_adam_zero_grads` is now required because `tg_backward_accum` does not zero.
- `train_loss` is read from `loss` **before** `tg_adam_accumulate` (today it is read after `tg_backward`, `candide.c:138`).
- Eval sites use `tg_eval_begin` / `tg_eval_end`.

### `../lambda`

- `TgAdam` replaces the `#ifdef`-split `m_buf`/`v_buf` blocks (`main.c:332-350`, `399-401`, `470-475`); the device branch disappears because `TgAdam` follows the params.
- Training a phase: if that phase's own checkpoint exists, `load_run(..., TG_LOAD_RESUME)` and continue from `opt.step + 1`; else if `phase > 1`, `load_run(phase N-1 file, TG_LOAD_INIT_FROM_WEIGHTS)`; else from scratch. This is a behaviour change: today phase 1 reloads its own weights with fresh moments, and phase > 1 always warm-starts from N−1 (`main.c:108-113`); no phase resumes its own optimizer state.
- **Periodic saves.** `save_run` at the existing log site and at the end; today the only save is at the end (`main.c:464`), so a mid-phase Ctrl-C would still lose the phase.
- **Completed-phase guard.** If `load_run(RESUME)` leaves `opt.step >= steps` (50000), print `[lambda] phase N complete at step %d` and skip both the training loop and the final save — the checkpoint is not rewritten. Re-running a finished phase is a no-op, which is the correct reading of "already trained".
- Inference mode keeps `tg_checkpoint_load` (weights only, no optimizer).
- Order: create model → upload params → `tg_adam_create` → `load_run`. lambda's `tg_seed` call must precede `load_run`.
- Same step-body rewrite as candide: `tg_backward(loss)` / `tg_adam_step*` / `tg_free_graph(loss)` (`main.c:397-401`, `421`) → `tg_adam_zero_grads` / `tg_adam_accumulate(&opt, loss, 1)` / `tg_adam_update(&opt, lr, 0.0f)`; the trailing `tg_free_graph(loss)` is deleted; `train_loss` (`main.c:405`) is read before accumulate.
- The constant `lr = 3e-4f` stays a constant.

### `../vexilloscope`

Migrated: the harness code in `main.c` only.

- Adam buffers (`main.c:1144-1170`, the `shape[0] * shape[1]` size computation, and the two `#ifdef` update branches) → `TgAdam`.
- Schedule (`main.c:1174-1176`) → `tg_lr_warmup_cosine(step, VX_VIT_STEPS, VX_WARMUP_STEPS, 3e-4f, 0.0f)`. Equivalence to the inline formula is checked by `test_lr_warmup_cosine` on the vexilloscope grid; the retrain is not a check (its data order is `rand()`-driven).
- Accumulation (`tg_zero_grads` / `tg_scale` / `tg_backward_accum` / `tg_free_graph`, `main.c:1180-1216`) → `tg_adam_zero_grads` / `tg_adam_accumulate`. `batch_loss` is read from `loss` before accumulate, as today.
- Clip + update (`main.c:1223-1229`) → `tg_adam_update(&opt, lr, 1.0f)`.
- The five `tg_training = 0` sites → `tg_eval_begin` / `tg_eval_end`.

**Not migrated:** `vx_vit_save` / `vx_vit_load` / `vx_vit_load_warmstart` (`vit.c:90-300`). The warm-start path loads a checkpoint into a model with a *larger* `n_labels` than the file's, which strict shape validation rejects by design. Replacing that format with `tg_checkpoint_info` + a config blob + `tg_checkpoint_load` is a Phase 2 item (vexilloscope is touched again for `TgLinear` and `ovg_vision`), where head expansion is reimplemented app-side as "build the old-shaped model, load, copy the head rows into the new model". vexilloscope does not resume mid-run today, so it loses nothing by staying weights-only in Phase 1.

---

## Tests

New file `tests/test_optim.c` (registered in `test_main.c`), additions to `tests/test_checkpoint.c`. Existing tests unchanged, including the three in `test_checkpoint.c`, which now exercise the v3 weights-only path through the same calls.

### `test_optim.c`

| Test | Checks |
|---|---|
| `test_adam_parity` | Two identical models; three updates via `TgAdam` vs three via raw `tg_adam_step` with hand-managed buffers and `step = 1,2,3`; params bitwise equal. |
| `test_adam_reset` | After updates, `tg_adam_reset`: all m/v zero, `step == 0`. |
| `test_adam_accumulate_equals_batch` | `n_micro = 4` losses through `tg_adam_accumulate` vs one loss that is the mean of the four through `tg_backward`; param grads within 1e-6. |
| `test_adam_update_returns_norm` | On CPU, `tg_adam_update(opt, lr, max)` returns the value `tg_clip_grad_norm` would; with `max = 0` returns 0 and does not scale. |
| `test_adam_step_counter` | `opt.step` equals the number of `tg_adam_update` calls. |
| `test_adam_bad_args_fatal` | `n_params = 0` and `n_micro = 0` hit `ovg_fatal` (setjmp/longjmp). |
| `test_lr_warmup_cosine` | `warmup = 10, total = 100, base = 1, floor = 0.1`: step 1 → 0.1 (warm 0.1, cosine factor 1); step 10 → ≈ 0.982 (warmup done, cosine barely started); step 101 → 0.1 (clamped); monotone non-increasing after warmup. `warmup = 0` never divides by zero. vexilloscope grid (`total = 60000, warmup = 2400, base = 3e-4, floor = 0`): `|lr_lib − lr_vex| ≤ 1e-6 × base` at every step against the inline `main.c:1174-1176` formula (absolute, scaled to `base`: `1 + cos(pi * progress)` cancels catastrophically near `progress = 1`, so a relative bound is unsatisfiable there). |
| `test_lr_warmup_linear` | Same grid: step 10 → 0.919 (1 + (0.1 − 1) × 0.09), step 51 → 0.55, step 101 → 0.1. |
| `test_eval_guard` | `tg_training = 1`; `prev = tg_eval_begin()` → `tg_training == 0`, `prev == 1`; `tg_eval_end(prev)` → 1. `TgMeter` mean of {1,2,3} is 2; empty meter is 0. |
| `test_rng_state_roundtrip` | `s = tg_rng_get_state()`; draw 3; `tg_rng_set_state(s)`; draw 3 again; sequences equal. `tg_rng_set_state(0)` is fatal. |
| `test_adam_cuda_parity` *(CUDA)* | Same model on CPU and on device through `TgAdam`; after 5 updates params within 1e-5. |
| `test_adam_mixed_device_fatal` *(CUDA)* | One CPU param and one CUDA param → `tg_adam_create` fatal. |

### `test_checkpoint.c` additions

| Test | Checks |
|---|---|
| `test_checkpoint_v3_run_roundtrip` | Model + `TgAdam`, 3 updates, `tg_rng_set_state(0xC0FFEE)`, config bytes `"cfg!"`; `save_run`; corrupt params, moments, step, RNG; `load_run(RESUME, &info)`: params, m, v bitwise restored; `step == 3`; `tg_rng_get_state() == 0xC0FFEE`; `info.version == 3`, `has_optimizer == 1`, `config_len == 4`. |
| `test_checkpoint_info_before_model` | `tg_checkpoint_info` on that file with a 16-byte buffer returns the 4 config bytes, `n_params`, `has_optimizer == 1`, `step == 3`, `rng_state == 0xC0FFEE`, with no tensors allocated. Missing file → -1. |
| `test_checkpoint_v2_loads_with_reset` | Write a v2 file by hand (magic `0x00475632`, count, params); `load_run(RESUME)` returns 0; params loaded; moments zero; `step == 0`; RNG state unchanged; `info.version == 2`. |
| `test_checkpoint_init_from_weights` | Full v3 file; `load_run(INIT_FROM_WEIGHTS)`: params loaded; moments zero; `step == 0`; RNG state unchanged. |
| `test_checkpoint_weights_only_then_run` | `tg_checkpoint_save`; `load_run(RESUME)` returns 0 with `has_optimizer == 0`, `info.rng_state == 0`, moments zero, `step == 0`, live RNG state unchanged. And the reverse: `save_run`; plain `tg_checkpoint_load` reads params and ignores the optimizer section. |
| `test_checkpoint_hparam_mismatch` | `save_run` with beta1 = 0.9; `load_run` into a `TgAdam` with beta1 = 0.8 → -1. Unknown flag bit → -1 from all three readers. Optimizer section with `rng_state == 0` → -1 on RESUME, 0 on INIT_FROM_WEIGHTS. Neither file is producible by the writer: `save_run`, then patch the `flags` / `rng_state` bytes in place (offsets 4 and 12). |
| `test_checkpoint_exact_resume` | Model with dropout 0.5 in training mode, fixed data. Both runs start from the same initial parameters (copied) and the same xorshift state (`tg_rng_set_state` before each). Run A: 6 updates. Run B: 3 updates, `save_run`, fresh model + `TgAdam`, `load_run(RESUME)`, 3 more updates. Params bitwise equal on CPU. |
| `test_checkpoint_tmp_replaced` | After a successful save no `<path>.tmp` remains; after a save to an unopenable path, -1 and no partial file. |
| `test_checkpoint_v3_cuda_roundtrip` *(CUDA)* | As `v3_run_roundtrip` with params and moments on the device; compared after sync within 1e-6. |

**Expected count:** 80 + 21 = **101 (CUDA build) / 71 + 18 = 89 (CPU build)**. The implementation sets the final number; `AGENTS.md` is updated to match.

---

## Documentation Updates (same change)

`AGENTS.md`:

- Repository Structure: add `tg_optim.h`, `tg_sched.h`, `tg_optim.c`, `tg_sched.c`, `test_optim.c`.
- Training API: add the `TgAdam` block, eval guard, and schedules; note that the raw `tg_adam_step*` remain.
- Checkpoints: replace the v2 layout with the v3 layout, the info/save_run/load_run API, the load modes, and the v2 compatibility rule.
- RNG and Seeding: add `tg_rng_get_state` / `tg_rng_set_state` and the ordering contract.
- CUDA Support → Internal: add the three float-buffer helpers; the "used by `tg_ops.c` / `tg_train.c` only" sentence (also in Include Style and Repository Structure) gains `tg_optim.c` / `tg_checkpoint.c`.
- Build Commands / Verification: new test count.
- Important Guidance: "create `TgAdam` after moving params to their device; call `tg_seed` before `tg_checkpoint_load_run`."

`docs/FOUNDATION_PLATFORM.md`: on completion, the claims-table row "Exact resume from a v3 checkpoint" moves from "not yet built" to built, and the Brownfield Baseline gains a Phase 1 line.

---

## Non-Goals (restated from the foundation document)

- Any new differentiable op.
- An SGD struct, or any optimizer beyond Adam.
- A dataset, dataloader, or batch abstraction.
- Distributed or multi-GPU training.
- Mixed-precision policy beyond `tg_cast`.
- Logging or metrics sinks. `TgMeter` is arithmetic, not a sink.
- Restoring application-side `rand()` state.
- vexilloscope's weight format (Phase 2).
- Iterative `topo_sort` / growable graph (Phase 3a).

---

## Risks and Notes for the Implementer

- **`tg_checkpoint_load_run` failure is not transactional.** A shape mismatch at param *k* leaves params `0..k-1` overwritten. This matches today's `tg_checkpoint_load`. Callers treat -1 as fatal for the run. Making it transactional (read everything into scratch first) is allowed but not required.
- **Windows `rename`** fails if the target exists; remove first. The window between remove and rename is accepted.
- **`tg_clip_grad_norm` on CUDA returns 0.0f** by design (no readback); `tg_adam_update` inherits that. Do not add a sync to report the norm.
- **Betas/eps are validated exactly** on resume. If a caller wants to change hyperparameters mid-run they use `TG_LOAD_INIT_FROM_WEIGHTS` and accept fresh moments. This is deliberate: silently continuing moments under different betas is the class of bug exact-resume exists to prevent.
- **`opt->params` must be the same tensors, in the same order,** as at save. The file validates shapes, not identity; parameter collection order is the model's contract (`tg_gpt_collect_params`, `vx_vit_collect_params`).
- Keep `tg_optim.c` free of any include from `ovg_nn` or `ovg_lm`; the layer check is grep + `--target ovg_core`.

---

## Acceptance

- [x] `tg_optim.h`, `tg_sched.h`, RNG state accessors, eval guard, CUDA float helpers implemented as specified.
- [x] Checkpoint v3 writer/readers implemented; v2 files load; the three pre-existing checkpoint tests pass unchanged.
- [x] `cmake --build --preset default` clean; `otto_von_grad_tests.exe` reports all pass at the new count; `cmake --preset cpu` build passes its count.
- [x] `ovg_core`, `ovg_nn`, `ovg_lm` each build standalone.
- [x] `examples/candide.c` migrated with periodic saves; a run stopped by Ctrl-C mid-way and restarted reports the cumulative `opt.step` of the last save and continues from it.
- [x] `../lambda` migrated and builds with no CMake edits; with no phase-2 checkpoint present, phase 2 warm-starts from phase 1 with `step == 0`; re-running a completed phase prints the complete notice and does not rewrite its checkpoint.
- [x] `../vexilloscope` migrated (harness only) and builds; `main.c` line count reduced; schedule values unchanged.
- [x] `AGENTS.md` updated as listed; `FOUNDATION_PLATFORM.md` claims table and baseline updated.
