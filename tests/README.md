# otto-von-grad tests

## Running

```powershell
cmake --preset default              # or `cpu` for a CUDA-free build (CUDA tests are skipped)
cmake --build --preset default
.\build\otto_von_grad_tests.exe
```

Or via CTest:

```powershell
ctest -C Release --test-dir build
```

The test binary exits 0 (all pass) or 1 (any failure). Each test prints `pass:` or `FAIL:`, and
the run ends with `N passed, M failed`. The seed is printed first so a failure can be replayed.

## Structure

One test file per module, linked against `ovg_lm` so every layer is reachable.

| File | Contents |
|---|---|
| `ovg_test.h` | `OVG_CHECK`, `OVG_CHECK_EQ`, `OVG_CHECK_NEAR`, `OVG_CHECK_SHAPE`, `RUN_TEST` macros |
| `test_main.c` | Entry point; seeds the RNG, calls every suite's `run_*_tests`, prints the summary |
| `test_ops.c` | Forward and gradient checks for every op (arithmetic, matmul, reductions, layer_norm, softmax, activations, dropout, embed, cross-entropy dense/sparse/no-sync, reshape, expand_dim, slice, concat); shape-mismatch and out-of-bounds error paths via `setjmp`/`longjmp`; layer_norm finite-difference check; drop-path schedule; CUDA: BF16 cast round-trip and matmul, N-D ops, large causal mask |
| `test_train.c` | SGD direction, `tg_backward` grad zeroing, `tg_backward_accum` accumulation, transpose grad accumulation, grad-norm clipping (CPU and CUDA) |
| `test_optim.c` | `TgAdam` parity with raw `tg_adam_step`, reset, gradient accumulation vs. batch mean, returned grad norm, step counter, bad-argument fatals; `tg_lr_warmup_cosine` / `tg_lr_warmup_linear` values and the vexilloscope-grid equivalence; eval guard and `TgMeter`; RNG state round-trip; CUDA: device parity, mixed-device fatal |
| `test_attention.c` | Causal attention gradients (single and multi-head), encoder attention weights, batch=1 vs batched parity, lower-ndim matmul broadcast |
| `test_gpt.c` | `tg_gpt_collect_params` capacity, forward output shape at batch 1 and batch 2 |
| `test_tokenizer.c` | Vocab build, encode/decode round-trip, `tg_tokenize` values, `tg_vocab_from_chars` |
| `test_checkpoint.c` | Weights-only save/load round-trip (CPU and CUDA), bad magic rejected, param-count mismatch rejected; v3 run-state round-trip, `tg_checkpoint_info` before any model exists, v2 file loads with moments zeroed, `TG_LOAD_INIT_FROM_WEIGHTS`, weights-only ↔ run-state cross-loading, hyperparameter / flag / rng_state rejection, exact resume with dropout, `.tmp` replacement; CUDA: v3 round-trip with device params and moments |
| `test_sample.c` | Argmax, top-k determinism and range, `tg_generate` callback count and `tg_training` restore; CUDA argmax/top-k |

## Writing new tests

Each test is a `static void` function. Use the `OVG_CHECK*` macros — on failure they print
`file:line: FAIL` to stderr, set `ovg_test_failed = 1`, and `return` from the test function.
Register the test with `RUN_TEST` inside the appropriate `run_*_tests` function. A test for a new
module gets its own `test_<module>.c`, a `run_<module>_tests` prototype called from `test_main.c`,
and an entry in the `otto_von_grad_tests` source list in `CMakeLists.txt`.

### Error-path tests

Fatal errors go through `ovg_fatal`. To test that an op rejects bad inputs:

```c
#include <setjmp.h>
static jmp_buf g_test_escape;
static char    g_last_error[512];

static void capture_handler(const char *msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    longjmp(g_test_escape, 1);  /* skips exit(1) inside ovg_fatal */
}

static void test_my_error_case(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        /* code that should trigger ovg_fatal */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);  /* restore default + reset re-entrancy guard */
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "expected keyword") != NULL);
}
```

Tensors allocated before `longjmp` will leak — acceptable in test code.

## CUDA tests

Tests guarded with `#ifdef OVG_CUDA_ENABLED` are compiled and run automatically when the library
is built with `OVG_CUDA=ON` (the `default` and `debug` presets). They are skipped in the `cpu`
preset, which is why that build reports 89 tests rather than 101.
