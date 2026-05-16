This is **otto-von-grad** — a from-scratch tensor autograd engine in C (C11).
See `AGENTS.md` for full architecture, API reference, and constraints.

Key rules:
- Every op in `tg_ops.c` has a paired `_backward` function — keep them together
- No external ML libraries
- No silent broadcasting — shape mismatches must fail loudly
- All tensors are 2D `[rows x cols]`; no higher-rank ops yet
- `persistent = 1` marks tensors that survive `tg_free_graph`
- When adding CUDA kernels, preserve the symmetric CPU fallback path
- Correctness and readability over performance
