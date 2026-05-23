# otto-von-grad — Planned Improvements

> Improvements identified through vexilloscope (ViT flag classifier built on OVG).
> Both items are general OVG additions — neither is ViT-specific.

---

## 1 — Row concatenation op (`tg_concat_rows`)

**Motivation:** Standard ViT uses a learned CLS token prepended to the patch sequence before
the encoder. The CLS output (row 0) replaces mean pooling for classification and is more
expressive. Implementing this in vexilloscope requires concatenating a `[1 × embed_dim]` CLS
parameter with an `[n_patches × embed_dim]` patch tensor to produce `[(n_patches+1) × embed_dim]`.
OVG has no row-concatenation op.

**Proposed API:**

```c
// Concatenate A [r_a × C] and B [r_b × C] along the row axis → [(r_a + r_b) × C]
// Both must have the same number of columns.
Tensor *tg_concat_rows(Tensor *A, Tensor *B);
```

**Backward:** distribute the output gradient back to A's rows and B's rows respectively —
a slice-and-assign on `out->grad` into `A->grad` and `B->grad`.

**Vexilloscope usage:**

```c
// In vx_patch_embedding_forward:
Tensor *cls_expanded = ...; // broadcast cls_token to [1 × embed_dim]
Tensor *seq = tg_concat_rows(cls_expanded, projected_patches); // [(n_patches+1) × embed_dim]
// ... encoder forward ...
// In vx_vit_forward, replace tg_mean_rows with row-0 slice:
Tensor *pool = tg_row_slice(enc, 0, 1); // [1 × embed_dim]
```

**General utility:** useful for any task requiring dynamic sequence construction —
prefix tokens, separator tokens, multi-segment inputs.

---

## 2 — Training mode + stochastic depth (drop path) in `TgBlock`

### 2a — Training mode flag

**Motivation:** Several regularization techniques (stochastic depth, dropout) must be disabled
at inference. `TgBlock` and `TgTransformer` currently have no training/inference distinction.

**Proposed approach:** add a global toggle (simpler than per-block flags):

```c
// tg_train.h
void tg_set_training(int is_training); // 1 = training, 0 = inference (default: 1)
int  tg_is_training(void);
```

Alternatively, add `int training` to `TgBlock` and set it via `TgTransformer`. The global
approach is simpler and sufficient for current use cases.

### 2b — Stochastic depth (drop path)

**What it is:** During training, each residual branch in a transformer block is independently
zeroed with probability `p` (and the surviving branches are scaled by `1/(1-p)` to keep
expected output unchanged). At inference the full branch always runs. Strong regularizer for
networks with 6+ blocks; reduces overfitting with no inference cost.

**Proposed addition to `TgBlock`:**

```c
// tg_block.h
typedef struct {
    // ... existing fields ...
    float drop_path_rate; // 0.0 = disabled; typical range 0.05–0.20
} TgBlock;
```

**Implementation in `tg_block_forward`:**

```c
// After computing the attention branch output `attn_out`:
if (tg_is_training() && b->drop_path_rate > 0.0f) {
    float u = tg_rng_uniform(); // draw from [0, 1)
    if (u < b->drop_path_rate) {
        attn_out = tg_scale(attn_out, 0.0f);  // drop
    } else {
        attn_out = tg_scale(attn_out, 1.0f / (1.0f - b->drop_path_rate)); // scale
    }
}
// ... add residual ...
// same pattern for FFN branch
```

Because `tg_scale` is differentiable, the backward pass for dropped steps naturally
produces zero gradients for the dropped branch — no special backward handling needed.

**Initialization:** a common schedule is linear drop rate by depth —
block `i` of `N` gets `drop_path_rate = max_rate * i / (N - 1)`, so the first block has
rate 0 and the last has `max_rate`. `tg_transformer_create_encoder` can accept a
`max_drop_path_rate` argument and set rates per block.

**Vexilloscope context:** with 6 blocks and `max_drop_path_rate = 0.10`, expected accuracy
improvement is 0.5–2 pp on confusable national flags where overfitting is most visible.

---

## Dependencies

| Item | Depends on |
|---|---|
| `tg_concat_rows` | nothing — standalone op |
| Training mode flag | nothing — standalone toggle |
| Stochastic depth | training mode flag + `tg_rng_uniform` (already in `tg_rng`) |
