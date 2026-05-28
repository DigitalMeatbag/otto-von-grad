This is **otto-von-grad** — a from-scratch tensor autograd engine in C (C11).
See `AGENTS.md` for full architecture, API reference, and constraints.

## Key rules

- Every op in `tg_ops.c` has a paired `_backward` function — keep them together
- No external ML libraries
- No silent broadcasting — shape mismatches must fail loudly (`TG_MAX_PARENTS = 8`, `TG_MAX_GRAPH = 8192`)
- Tensors are N-D up to `TG_MAX_DIMS = 4`; shape stored in `int shape[TG_MAX_DIMS]` with `int ndim` (see `docs/ND_SPEC.md`)
- `persistent = 1` marks tensors that survive `tg_free_graph`
- When adding CUDA kernels, preserve the symmetric CPU fallback path
- Correctness and readability over performance

## Build

Default preset: VS2026, CUDA enabled, Release mode, outputs to `build\`.

```powershell
# First configure (fresh clone or after deleting build/)
cmake --preset default

# Every subsequent build
cmake --build --preset default
```

Non-default presets (for the rare case):

```powershell
cmake --preset debug   && cmake --build --preset debug   # Debug build
cmake --preset cpu     && cmake --build --preset cpu     # CPU-only, no CUDA
```

## Verification

After code changes, build and run the test binary:

```powershell
cmake --build --preset default
.\build\otto_von_grad_tests.exe   # 68 tests; exits 0 on all-pass
```

`otto_von_grad.exe` is the GPT character-level demo (trains on `candide.txt`). It no longer runs tests at startup.
