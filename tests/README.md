# otto-von-grad tests

## Running

```powershell
cmake -B build -G Ninja          # or add -DOVG_CUDA=ON for GPU tests
cmake --build build
.\build\otto_von_grad_tests.exe
```

Or via CTest:

```powershell
ctest --test-dir build
```

The test binary exits 0 (all pass) or 1 (any failure). Each test prints `pass:` or `FAIL:`.

## Structure

| File | Contents |
|---|---|
| `ovg_test.h` | `OVG_CHECK`, `OVG_CHECK_EQ`, `OVG_CHECK_NEAR`, `OVG_CHECK_SHAPE`, `RUN_TEST` macros |
| `test_main.c` | Entry point; calls all suite `run_*_tests` functions and prints summary |
| `test_ops.c` | Arithmetic (add/sub/mul/pow/matmul), reductions, layer_norm, softmax, cross_entropy, embed, concat_rows, row_slice, rng_uniform, drop_path schedule + inference no-op; error-path tests via `setjmp`/`longjmp` |
| `test_train.c` | SGD direction, `tg_backward` grad zeroing, `tg_backward_accum` accumulation, transpose grad accumulation |
| `test_attention.c` | Causal attention (single/multi-head) grad coverage, encoder row-sum and bidirectionality |
| `test_gpt.c` | Param-count capacity check, GPT forward output shape |

## Writing new tests

Each test is a `static void` function. Use the `OVG_CHECK*` macros — on failure they print
`file:line: FAIL` to stderr, set `ovg_test_failed = 1`, and `return` from the test function.
Register your test with `RUN_TEST` inside the appropriate `run_*_tests` function.

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

Tests guarded with `#ifdef OVG_CUDA_ENABLED` are compiled and run automatically when
the library is built with `-DOVG_CUDA=ON`. They are skipped in CPU-only builds.
