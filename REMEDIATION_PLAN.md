# Remediation Plan

This plan captures the repo review follow-up work. It is intentionally split
into small passes so correctness fixes can land before API hardening and
performance work.

## Pass 1: CUDA Correctness

- Fix `causal_mask_fwd_k` and `causal_mask_bwd_k` so row indexing matches the
  16x16 launch shape. Use `blockDim.y` or a dedicated tile constant instead of
  `BLOCK`.
- Fix CUDA transpose backward so it accumulates into the parent gradient rather
  than overwriting existing gradient values.
- Add explicit mixed-placement validation for CUDA ops. If one parent is on CUDA
  and another is still CPU-only, fail loudly with a clear error rather than
  launching kernels with null device pointers.

## Pass 2: Optimizer and API Safety

- Add CUDA-aware SGD support, or make `tg_sgd_step` reject CUDA tensors with a
  clear error. Today CPU SGD updates only host data.
- Change `tg_gpt_collect_params` to accept caller capacity and fail clearly if
  the provided array is too small.
- Update call sites for the new parameter collection signature.

## Pass 3: Regression Coverage

- Add a CUDA causal-mask check for sequence lengths greater than 16.
- Add a transpose-gradient accumulation test.
- Add a parameter-collection capacity test.
- Add a mixed CPU/GPU placement test that expects a clear failure path.

## Pass 4: Low-Risk Performance Work

- Consider blocked CPU matmul for better cache reuse.
- Add an embedding gather op to avoid one-hot token embedding matmul.
- Later, reduce attention allocation churn by avoiding repeated slice,
  transpose, and concat nodes where possible.

## Recommended Sequencing

Start with Pass 1 and Pass 2 in a focused remediation branch. Keep Pass 4 as a
separate follow-up unless the goal is a broader cleanup and speed pass.
