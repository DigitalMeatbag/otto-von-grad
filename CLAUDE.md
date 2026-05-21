This is **otto-von-grad** — a from-scratch tensor autograd engine in C (C11).
See `AGENTS.md` for full architecture, API reference, and constraints.

## Key rules

- Every op in `tg_ops.c` has a paired `_backward` function — keep them together
- No external ML libraries
- No silent broadcasting — shape mismatches must fail loudly (`TG_MAX_PARENTS = 8`, `TG_MAX_GRAPH = 4096`)
- All tensors are 2D `[rows x cols]`; no higher-rank ops
- `persistent = 1` marks tensors that survive `tg_free_graph`
- When adding CUDA kernels, preserve the symmetric CPU fallback path
- Correctness and readability over performance

## Build

```powershell
# CPU only
cmake -B build -G Ninja
cmake --build build
.\build\otto_von_grad.exe

# With CUDA
cmake -B build -G Ninja -DOVG_CUDA=ON
cmake --build build
.\build\otto_von_grad.exe
```

## Verification

After code changes, build and run the test binary:

```powershell
cmake --build build
.\build\otto_von_grad_tests.exe   # 23 tests; exits 0 on all-pass
```

`otto_von_grad.exe` is the GPT character-level demo (trains on `candide.txt`). It no longer runs tests at startup.
