# legacy/

These files are the original scalar autograd learning exercises that preceded the tensor engine.
They are kept here for historical reference and are **not compiled into the library**.

| File | Description |
|------|-------------|
| `value.h / value.c` | Scalar computation graph — `Value` struct with data, grad, backward fn, arena allocator |
| `mlp.h / mlp.c` | Scalar MLP built on top of `Value` — Neuron, Layer, MLP structs demonstrating XOR learning |

The tensor engine in `src/` supersedes this work entirely. See `src/tg_tensor.h` and `src/tg_ops.h`
for the active implementation.
