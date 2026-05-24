# otto-von-grad — Planned Improvements

> Improvements identified through vexilloscope (ViT flag classifier built on OVG).
> All items are general OVG additions — none is ViT-specific.

---

## 1 — Row concatenation op (`tg_concat_rows`)

**Motivation:** Standard ViT uses a learned CLS token prepended to the patch sequence before
the encoder. The CLS output (row 0) replaces mean pooling for classification and is more
expressive. Implementing this in vexilloscope requires concatenating a `[1 × embed_dim]` CLS
parameter with an `[n_patches × embed_dim]` patch tensor to produce `[(n_patches+1) × embed_dim]`.
OVG has no row-concatenation op.

**Proposed API:**

```c
// Concatenate n_parts tensors along the row axis — all must have the same number of columns.
// parts[0] occupies the top rows, parts[n_parts-1] the bottom.
Tensor *tg_concat_rows(Tensor **parts, int n_parts);
```

**Backward:** for each input tensor, accumulate the corresponding row slice of `out->grad`
into that tensor's `->grad` (same pattern as `tg_concat_cols`).

**Vexilloscope usage:**

```c
// In vx_patch_embedding_forward:
Tensor *cls_expanded = ...; // broadcast cls_token to [1 × embed_dim]
Tensor *row_parts[2] = {cls_expanded, projected_patches};
Tensor *seq = tg_concat_rows(row_parts, 2); // [(n_patches+1) × embed_dim]
// ... encoder forward ...
// In vx_vit_forward, replace tg_mean_rows with row-0 slice:
Tensor *pool = tg_row_slice(enc, 0, 1); // [1 × embed_dim]
```

**General utility:** useful for any task requiring dynamic sequence construction —
prefix tokens, separator tokens, multi-segment inputs.

---

## 1b — Row slice op (`tg_row_slice`)

**Motivation:** The ViT forward pass extracts the CLS token (row 0) from the encoder output.
OVG has `tg_slice_cols` for column slicing; there is no row-axis counterpart.

**Proposed API:**

```c
// Extract rows [row_start, row_end) from A [R × C] → [(row_end - row_start) × C]
Tensor *tg_row_slice(Tensor *a, int row_start, int row_end);
```

**Backward:** accumulate `out->grad` into `A->grad` at rows `[row_start, row_end)` — rows outside
the slice receive zero gradient contribution from this op.

**Vexilloscope usage** (see also the concat example in §1):

```c
Tensor *pool = tg_row_slice(enc, 0, 1); // [1 × embed_dim] — CLS token
```

**General utility:** row slicing is the row-axis dual of `tg_slice_cols`; useful wherever a
sub-sequence must be extracted from a `[seq × embed]` tensor.

---

## 2 — Training mode + stochastic depth (drop path) in `TgBlock`

### 2a — Training mode flag

**Motivation:** Several regularization techniques (stochastic depth, dropout) must be disabled
at inference. `TgBlock` currently has no training/inference distinction.

**Approach:** `extern int tg_training` already exists in `tg_train.h` and is used directly
by `tg_dropout`. Stochastic depth uses the same global — no new API needed:

```c
// tg_block_forward — same pattern as tg_dropout
if (tg_training && b->drop_path_rate > 0.0f) {
    ...
}
```

Callers set `tg_training = 0` before inference, `tg_training = 1` (default) during training.

### 2b — Stochastic depth (drop path)

**What it is:** During training, each residual branch in a transformer block is independently
zeroed with probability `p` (and the surviving branches are scaled by `1/(1-p)` to keep
expected output unchanged). At inference the full branch always runs. Strong regularizer for
networks with 6+ blocks; reduces overfitting with no inference cost.

**Proposed addition to `TgBlock`:**

```c
// tg_block.h
typedef struct {
    TgSelfAttention attn;
    Tensor *gamma1, *beta1;  /* LN affine params before attention  [1 x embed_dim] */
    Tensor *gamma2, *beta2;  /* LN affine params before FFN        [1 x embed_dim] */
    Tensor *W1, *B1, *W2, *B2;
    int   embed_dim, hidden_dim;
    float dropout;
    float drop_path_rate; // 0.0 = disabled; typical range 0.05–0.20
} TgBlock;
```

**Implementation in `tg_block_forward`:**

```c
// After computing the attention branch output `attn_out`:
if (tg_training && b->drop_path_rate > 0.0f) {
    float u = tg_rng_uniform(); // [0, 1) — see §3 for definition
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

**Zero-initialization:** `tg_block_create` and `tg_block_create_encoder` must explicitly
set `b.drop_path_rate = 0.0f` — C does not zero-initialize struct fields by default.

**Initialization:** a common schedule is linear drop rate by depth —
block `i` of `N` gets `drop_path_rate = max_rate * i / (N - 1)`, so the first block has
rate 0 and the last has `max_rate`.

`tg_transformer_create_encoder` gains a new `max_drop_path_rate` parameter:

```c
// tg_transformer.h  (updated signature — breaking change)
TgTransformer tg_transformer_create_encoder(
    int n_blocks, int embed_dim, int hidden_dim,
    int seq_len, int n_heads,
    float max_drop_path_rate  // 0.0 = no drop path
);
```

Implementation sets each block's rate during construction:

```c
for (int i = 0; i < n_blocks; i++)
    t.blocks[i].drop_path_rate =
        (n_blocks > 1) ? max_drop_path_rate * i / (n_blocks - 1) : 0.0f;
```

**Vexilloscope context:** with 6 blocks and `max_drop_path_rate = 0.10`, expected accuracy
improvement is 0.5–2 pp on confusable national flags where overfitting is most visible.

---

## 3 — Add `tg_rng_uniform` to `tg_rng`

**Motivation:** Stochastic depth needs a uniform float in `[0, 1)`. `tg_rng_xorshift32` returns a
`uint32_t`; a named helper keeps callers clean and is reusable for any future stochastic op.

**Proposed addition to `tg_rng.h` / `tg_rng.c`:**

```c
// tg_rng.h
float tg_rng_uniform(void);  // returns [0, 1) via xorshift32

// tg_rng.c
float tg_rng_uniform(void) {
    return tg_rng_xorshift32() / (float)0x100000000ULL;
}
```

---

## Dependencies

| Item | Depends on |
|---|---|
| `tg_concat_rows` | nothing — standalone op |
| `tg_row_slice` | nothing — standalone op |
| `tg_rng_uniform` | nothing — wraps existing `tg_rng_xorshift32` |
| Training mode flag | nothing — `tg_training` already exists in `tg_train.h` |
| Stochastic depth | training mode flag + `tg_rng_uniform`; updates `tg_transformer_create_encoder` signature |
