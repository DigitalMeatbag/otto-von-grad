# Spec: Platform Phase 2 — Vision

> **Status:** Implemented 2026-09-20 (library at `956eb0c`, vexilloscope `2d0a14b`; this document and `FOUNDATION_PLATFORM.md` in the docs commit that follows); every Acceptance item is checked except the retrain, which is sequenced after and not gated, and the `main.c` line-count clause, which is annotated. Reviewed 2026-09-20 (the one open fork closed, see [Open Questions](#open-questions)). Derived from `docs/FOUNDATION_PLATFORM.md` (Decision Log rows "`TgLinear` fate" and "Vision-block placement", both closed 2026-09-20) and from the Phase 1 hand-off in `docs/SPEC_PLATFORM_PHASE1_HARNESS.md` → Consumer Migrations → vexilloscope ("Not migrated"). `AGENTS.md` describes what exists (at spec time: library at `25a1a0a`, tests 101 CUDA / 89 CPU; after this phase: 120 / 105); this document describes exactly what Phase 2 adds, and is complete when every item in [Acceptance](#acceptance) is checked.

---

## Open Questions

None remaining. The one fork the foundation document did not close was decided at spec review:

1. **vexilloscope train mode: resume automatically, or only on an explicit flag?** Today train mode always starts from scratch and overwrites `vit_weights.bin` at the end. Under this spec, if the file at `--weights` exists and is a training checkpoint for the compiled architecture, train mode resumes it (candide and lambda both auto-resume; the owner stops and restarts long runs; a 60,000-step run is ~5.5 h at ~3 steps/s). The alternative was a `--resume` flag with the current overwrite as default. **Closed 2026-09-20 (review): auto-resume**, with a fresh run obtained by deleting the file or passing a different `--weights`. Specified in [`main.c` train mode](#srcmainc--train-mode).

Premises in the brief that the code contradicts, recorded so the reviewer does not re-derive them:

- *"vexilloscope's data order is `rand()`-driven, which the harness does not restore."* Only when `VX_BALANCED_SAMPLING = 1`; the enum sets it to **0** (`main.c:50`), and the live branch is `idx = ((step - 1) * VX_BATCH_SIZE + b) % n_images` (`main.c:1171`), a pure function of `step`, which `opt.step` restores. The unrestored randomness is `img_augment` (sixteen `rand()` sites in `img.c`) and the balanced-sampling branch if it is ever enabled. This is what lets the spec close the resume question in favour of `save_run` / `TG_LOAD_RESUME` (see [Decisions](#decisions-this-spec-makes)).
- *"five inference modes in `main.c` that all go through `vx_vit_load`."* Four call sites exist (`--identify-json` 1003, `--identify-json-batch` 1032, `--identify` 1074, `--identify-batch` 1088); the fifth reader is `vx_vit_load_warmstart` at 1136, in train mode. All five are covered below.
- The brief describes head expansion as "copy the head rows"; `Wout` is `[embed_dim, n_labels]`, so the copy is a **per-row prefix of columns** (`vit.c:338-346` today). The spec keeps that loop.

---

## Purpose

Give the platform its second modality package. `ovg_vision` holds the two things a vision transformer needs above `ovg_nn` — a patch embedding (linear projection of flattened patches, learned positional embedding, optional CLS token) and the pooling that turns encoder tokens into one vector per image — so the ViT recipe becomes "patch embed → transformer → pool → linear" and the `[1, T, C]` CLS-extract knowledge lives in the library once. Alongside it, `TgLinear` is fixed to the v2 idiom and renamed so `ovg_nn`'s named linear layer teaches the right pattern, and vexilloscope, the first vision consumer, adopts both and drops its hand-rolled weight format for the library checkpoint.

**Completion signal:** `ovg_vision` builds standalone and is exercised by its own tests; vexilloscope links `ovg_vision` instead of `ovg_nn`, deletes `patch_embedding.c` and the `VXWT` reader/writer, trains through `tg_checkpoint_save_run` with mid-run resume, and all four `--identify*` modes plus `--warmstart` read the v3 file.

---

## Decisions Inherited From the Foundation Document

These are settled and this spec does not reopen them:

1. `TgLinear` is **fixed in place**, not deleted. `tg_mlp.h` / `tg_mlp.c` are renamed `tg_linear.h` / `tg_linear.c`. Bias becomes `[1, n_out]`, expanded to the input's leading dims at forward time via `tg_reshape` + `tg_expand_dim` (no broadcasting); the `batch` field and the `xw_out` out-parameter go away; `tg_linear_params` stays.
2. **`ovg_vision` is created now**, in Phase 2, under the second-consumer rule applied ahead of time. It links `ovg_nn`, mirrors `ovg_lm`'s shape (a few modules, own tests, no data loading), and initially holds `TgPatchEmbed` and the token-pooling helpers. Its modules assume a token layout, not pixels.
3. Patch extraction from pixels (`img_patchify`), letterboxing, and augmentation stay application-side. The library block takes `[B, n_patches, patch_size]` or `[n_patches, patch_size]` and returns tokens.
4. The patch-embed block accepts 2D (batch 1) and 3D input the way `tg_block_forward` does, and documents which it returns.
5. The CLS prepend uses `tg_concat` — no bespoke op.
6. No convolutional patch embedding (Phase 3a), no image decoding, no augmentation.
7. vexilloscope becomes `ovg_vision`'s first consumer and links it instead of `ovg_nn`.
8. vexilloscope's weight format (`vx_vit_save` / `vx_vit_load` / `vx_vit_load_warmstart`, `"VXWT"`) is replaced by `tg_checkpoint_info` + a config blob + the library loaders; head expansion is reimplemented app-side because strict shape validation rejects the larger-head load by design (Phase 1 spec, vexilloscope section).
9. Layer rules are enforced by grep plus `cmake --build --preset default --target <t>` per target, not by the test binary.
10. `../lambda` must keep building with no edits to its own CMakeLists. vexilloscope's CMakeLists may be edited (it is the consumer being migrated).

### Decisions this spec makes

Assigned to it by the foundation document or the brief:

| Question | Decision |
|---|---|
| Does `ottovongrad` link `ovg_vision`? | **Yes.** The Decision Log keeps the umbrella "indefinitely as the everything-included default"; a vision consumer linking the umbrella must get the vision package. `target_link_libraries(ottovongrad INTERFACE ovg_lm ovg_vision)`. lambda links `ottovongrad` and is unaffected (unreferenced static-library objects are dropped at link). |
| Does `tg_mean_rows` generalise to a mean over a chosen axis? | **No.** `tg_pool_mean_tokens` composes existing ops (`tg_transpose` → `tg_reshape` → `tg_mean_rows` → `tg_reshape`, see [Pooling](#3-tg_poolh--srctg_poolc--token-pooling-ovg_vision)). Mean pooling has no consumer on day one (vexilloscope pools CLS); a new `tg_mean_axis` with CPU path, CUDA kernel, backward, and tests is `ovg_core` work that a vision phase should not carry. The composition costs two extra full-size copies of `[B, T, C]` per forward (the transpose and the reshape that feeds `tg_mean_rows`), mirrored in backward, accepted under the Performance Posture. Generalising later behind the same `tg_pool_mean_tokens` signature is permitted. |
| Does vexilloscope training move to `tg_checkpoint_save_run` / `TG_LOAD_RESUME`? | **Yes**, with periodic saves. The 5.5-hour run is exactly the loss the harness exists to prevent. Guarantee, stated honestly: parameters, Adam moments, step, and library RNG (dropout, drop-path) restore exactly; the sample order restores because it is a function of `step` at `VX_BALANCED_SAMPLING = 0`; augmentation draws (`rand()`) do not restore, so a resumed run is statistically, not bitwise, equivalent to an uninterrupted one. The exact-resume proof stays `test_checkpoint_exact_resume`; vexilloscope's check is "continues from the saved `opt.step`". |
| When does the vexilloscope retrain happen; is a `VXWT` → v3 converter in scope? | **After the format change; no converter.** The only existing `vit_weights.bin` is a v1 file that already cannot load (pre-tiled `[seq_len, C]` shapes), so there is no valid `VXWT` file to convert and waiting costs nothing; retraining first would manufacture one that needs a converter. The `VXWT` reader is deleted outright. |
| `src/mlp_classifier.c` (vexilloscope's only `TgLinear` consumer, documented there as unused legacy) | **Updated to the new API**, not deleted. The `TgLinear` Fate constraint is that vexilloscope keeps compiling; deleting a file from the consumer's tree is the consumer's call and out of scope here. The ripple: `VxMlpForward` loses `xw1`/`xw2` and `vx_mlp_forward_free` stops freeing them. |
| Does the patch projection carry a bias? | **No.** `x @ Proj + b + PosEmb[t]`: a bias shared by all tokens is exactly representable by the per-token `PosEmb`, so it adds parameters without capacity. `TgPatchEmbed` therefore holds a raw `Proj` weight rather than a `TgLinear`. This is also vexilloscope's model today. |
| What rank does `tg_patch_embed_forward` return? | **Always 3D `[B, n_tokens, C]`** (`B = 1` for 2D input), matching `tg_block_forward`'s promotion. The encoder receives 3D; the pooling helpers take 3D. |
| Do the pooling helpers accept 2D input? | **No — 3D `[B, T, C]` only**, fatal otherwise. Their input is always an encoder output, which is always 3D. |
| Where does the positional embedding apply? | **To patch tokens, before the CLS prepend.** `PosEmb` is `[n_patches, C]`; the CLS token gets none (it is itself learned, so a position for it would be redundant). This is the current vexilloscope model; the alternative (`[n_patches + 1, C]` after concat) silently changes it. |
| Head-expansion mechanics | **Widen `Wout` in place after a strict load.** Build the file's architecture from the blob, `tg_checkpoint_load` every parameter with strict shapes, then replace `Wout` with a wider Xavier-initialised tensor whose per-row prefix is copied from the loaded one. One model in memory, not two; the "Wout is the last collected param" guard is no longer needed because no stream is being read around it. |

---

## Definitions

- **Patch.** One flattened image region as the application produces it: a row of `patch_size` floats. The library never sees pixels as a grid.
- **Token.** One `[C]` vector in the sequence the encoder consumes. `n_tokens = n_patches + use_cls`.
- **CLS pooling.** Token 0 of the encoder output, `[B, T, C] → [B, C]`. Requires the model to have been built with `use_cls = 1`; the library does not check this (it cannot), so the recipe documents it.
- **Mean pooling.** Mean over all `T` tokens, `[B, T, C] → [B, C]`, CLS included if present. A caller that wants patch-only mean slices first.
- **Config blob.** The bytes an application stores in the checkpoint header to rebuild its model before any tensor exists. For vexilloscope: `VxViTConfig`, a fixed-width struct of seven `int32`, written with `sizeof`, matching candide's `TgGPTConfig` precedent.

---

## Deliverables

### 1. `tg_linear.h` / `src/tg_linear.c` — `TgLinear` (rename + fix; `ovg_nn`)

`git mv include/ovg/tg_mlp.h include/ovg/tg_linear.h` and `git mv src/tg_mlp.c src/tg_linear.c`; the include guard becomes `TG_LINEAR_H`. No compatibility shim `tg_mlp.h`: nothing in `src/`, `tests/`, `examples/`, or `../lambda` includes it (verified 2026-09-20), and vexilloscope's one include is rewritten in the same phase.

```c
#include "tg_ops.h"

typedef struct {
    Tensor *W;       /* [n_in, n_out] */
    Tensor *B;       /* [1, n_out]; expanded to the input's leading dims at forward */
    int n_in, n_out;
} TgLinear;

/* W ~ randn * w_scale, B = 0; both persistent. Fatal if n_in <= 0 or n_out <= 0. */
TgLinear  tg_linear_create(int n_in, int n_out, float w_scale);
void      tg_linear_free(TgLinear *l);

/* x: [..., n_in] with ndim 2..TG_MAX_DIMS. Returns x @ W + B with shape [..., n_out].
   Fatal if x->shape[ndim-1] != n_in. F32 only (BF16 callers cast around it). */
Tensor   *tg_linear_forward(TgLinear *l, Tensor *x);

/* Unchanged: malloc'd {W, B}; *n_out = 2; caller frees the array, not the tensors. */
Tensor  **tg_linear_params(TgLinear *l, int *n_out);
```

Forward: `xw = tg_matmul(x, W)` (2D, or the documented lower-ndim second operand for 3D/4D); then the bias is made explicit: when `x->ndim > 2`, `tg_reshape(B, x->ndim, {1, …, 1, n_out})` (for 2D input `B` is already `[1, n_out]` and the reshape is skipped — a same-shape reshape would be a pointless copy); followed by `tg_expand_dim(·, i, xw->shape[i])` for each leading axis `i = 0 .. ndim-2` (for 2D input this is the single `tg_expand_dim(B, 0, N)`); then `tg_add(xw, bias)`. `tg_expand_dim` with `n = 1` is a legal no-op tile, so a size-1 leading axis needs no special case. The intermediate `xw` is an ordinary non-persistent node that `tg_free_graph` frees; the out-parameter that used to expose it is gone.

`TgGPT`'s output projection and `TgBlock`'s FFN keep their inline matmul + expansion; migrating them to `TgLinear` is not in scope (it would change nothing numerically and would touch two files this phase has no other reason to touch).

### 2. `tg_patch_embed.h` / `src/tg_patch_embed.c` — `TgPatchEmbed` (`ovg_vision`)

```c
#include "tg_ops.h"

typedef struct {
    Tensor *Cls;      /* [1, C]; NULL when use_cls == 0 */
    Tensor *Proj;     /* [patch_size, C] */
    Tensor *PosEmb;   /* [n_patches, C]; applied to patch tokens only */
    int n_patches, patch_size, embed_dim, use_cls;
} TgPatchEmbed;

/* Cls and Proj: tg_fill_xavier_uniform; PosEmb: zeros. All persistent.
   Fatal if n_patches, patch_size, or embed_dim <= 0, or if use_cls is not 0 or 1. */
TgPatchEmbed tg_patch_embed_create(int n_patches, int patch_size, int embed_dim, int use_cls);
void         tg_patch_embed_free(TgPatchEmbed *p);

/* patches: [n_patches, patch_size] (treated as batch 1) or [B, n_patches, patch_size].
   Returns [B, n_patches + use_cls, C] — always 3D. Fatal on any other rank, or if
   the trailing two dims are not exactly [n_patches, patch_size]. F32 only. */
Tensor      *tg_patch_embed_forward(TgPatchEmbed *p, Tensor *patches);

/* Fills params in the order Cls (if use_cls), Proj, PosEmb; returns 3 or 2.
   Fatal if max_params is too small. This order is the checkpoint contract. */
int          tg_patch_embed_collect_params(TgPatchEmbed *p, Tensor **params, int max_params);

static inline int tg_patch_embed_n_tokens(const TgPatchEmbed *p) { return p->n_patches + p->use_cls; }
```

Forward, all in 3D once the input is promoted (`[n_patches, patch_size]` → `tg_reshape` → `[1, n_patches, patch_size]`):

```text
X    = patches @ Proj                                  [B, n_patches, C]   (3D @ 2D broadcast)
Pos  = expand_dim(reshape(PosEmb, [1, n_patches, C]), 0, B)                 [B, n_patches, C]
X    = X + Pos
if use_cls:
  Cls3 = expand_dim(reshape(Cls, [1, 1, C]), 0, B)     [B, 1, C]
  X    = concat(Cls3, X, axis = 1)                     [B, n_patches + 1, C]   (CLS at token 0)
return X
```

Initialisation mirrors vexilloscope's `vx_patch_embedding_create` (Xavier for `Cls`/`Proj`, zeros for `PosEmb`) rather than `TgGPT`'s `randn 0.1` positional init, because vexilloscope is the model being moved and its init is not being redesigned here. The collect order `Cls, Proj, PosEmb` is vexilloscope's today, so its `vx_vit_collect_params` order — the checkpoint contract — is unchanged by the move.

`tg_patch_embed.c` includes its own header, `tg_ops.h`, `tg_tensor.h`, and `ovg_error.h` only. It must not include anything from `ovg_lm`.

### 3. `tg_pool.h` / `src/tg_pool.c` — token pooling (`ovg_vision`)

```c
#include "tg_ops.h"

/* enc: [B, T, C]. Token 0 of each sequence → [B, C].  Fatal if enc->ndim != 3. */
Tensor *tg_pool_cls(Tensor *enc);

/* enc: [B, T, C]. Mean over T → [B, C].  Fatal if enc->ndim != 3. */
Tensor *tg_pool_mean_tokens(Tensor *enc);
```

Compositions — no new op, no constant tensor, every node has an existing CPU and CUDA path and a paired backward:

```text
tg_pool_cls:          slice(enc, axis 1, start 0, len 1)  [B, 1, C]  → reshape [B, C]
tg_pool_mean_tokens:  transpose(enc, 0, 1)                [T, B, C]
                      → reshape [T, B*C]
                      → tg_mean_rows                       [1, B*C]   (mean over T per (b, c))
                      → reshape [B, C]
```

The `tg_pool_cls` composition is the two lines at `vexilloscope/src/vit.c:55-56` generalised from `B = 1`. The mean composition avoids a per-call ones tensor (which would need a device upload when `enc` is on CUDA, as `tg_block_forward`'s drop-path mask does) by letting `tg_mean_rows` do the reduction over the leading axis after a transpose. Every node is a contiguous buffer: `tg_transpose` materialises its permutation and `tg_reshape` is a copy on both paths (`memcpy` on CPU, `cuda_scale_fwd(·, 1.0f, ·)` on device — `tg_ops.c:1136-1143`), so there is no strided-view case for the device path to get wrong. The cost over a hypothetical `tg_mean_axis` is two extra full-size copies per forward — the transpose and the reshape to `[T, B*C]`, each `B·T·C` floats — plus the trailing `[1, B*C] → [B, C]` reshape, which is only `B·C`; backward mirrors them.

### 4. Build — `CMakeLists.txt`

```cmake
set(OVG_NN_SOURCES
    src/tg_linear.c            # was src/tg_mlp.c
    ...)

set(OVG_VISION_SOURCES
    src/tg_patch_embed.c
    src/tg_pool.c
)

# vision: token-layout blocks for vision transformers (patch embedding, pooling)
add_library(ovg_vision STATIC ${OVG_VISION_SOURCES})
target_link_libraries(ovg_vision PUBLIC ovg_nn)
if(NOT MSVC)
    target_compile_options(ovg_vision PRIVATE -O3 -march=native -ffast-math)
endif()

# umbrella: everything
target_link_libraries(ottovongrad INTERFACE ovg_lm ovg_vision)

add_executable(otto_von_grad_tests ... tests/test_linear.c tests/test_vision.c)
target_link_libraries(otto_von_grad_tests PRIVATE ovg_lm ovg_vision)   # neither reaches the other
```

The layer comment at the top of the file gains `vision → nn`. `candide` keeps linking `ovg_lm` only.

**Layer check** (grep + per-target build, as today):

- `cmake --build --preset default --target ovg_core`, then `ovg_nn`, `ovg_lm`, `ovg_vision` — each succeeds on its own.
- `grep -l "tg_gpt.h\|tg_tokenizer.h\|tg_sample.h" src/tg_patch_embed.c src/tg_pool.c` → nothing (vision does not reach into lm).
- `grep -l "tg_patch_embed.h\|tg_pool.h" src/tg_*.c` → only `src/tg_patch_embed.c` and `src/tg_pool.c` (nothing lower reaches into vision, and lm does not either).
- `grep -rl "tg_mlp" src include tests examples CMakeLists.txt` → nothing.

The test binary links both `ovg_lm` and `ovg_vision` and therefore still cannot detect a violation in either direction; that is the grep's job.

---

## The ViT Recipe, As Written Against `ovg_vision`

For reference and for the vexilloscope migration. This is what `vx_vit_forward` becomes.

```c
TgPatchEmbed  pe  = tg_patch_embed_create(n_patches, patch_size, C, /*use_cls=*/1);
TgTransformer enc = tg_transformer_create_encoder(n_blocks, C, hidden, n_patches + 1, n_heads, drop_path);
Tensor       *Wout = /* [C, n_labels], Xavier, persistent */;

Tensor *X      = tg_patch_embed_forward(&pe, patches);   /* [B, n_patches+1, C]; patches 2D or 3D */
Tensor *H      = tg_transformer_forward(&enc, X);         /* [B, n_patches+1, C] */
Tensor *pooled = tg_pool_cls(H);                          /* [B, C]  (requires use_cls = 1) */
Tensor *logits = tg_matmul(pooled, Wout);                 /* [B, n_labels] — 2D, ready for CE */
```

Parameter collection: `tg_patch_embed_collect_params` (3), then the 12 per block in the order `vx_vit_collect_params` uses today, then `Wout`. There is still no `tg_transformer_collect_params`; adding one is a candidate follow-up, not Phase 2 work, because `TgGPT` and vexilloscope both hand-collect today and changing one without the other buys nothing.

---

## Consumer Migrations

Only vexilloscope changes. `examples/candide.c` and `../lambda` are untouched; both must still build (lambda with no CMake edit — its `ottovongrad` link now transitively carries `ovg_vision`, which is harmless). Each migration is one commit per repo, after the library work lands and its tests pass.

### `../vexilloscope`

#### `CMakeLists.txt`

- Remove `src/patch_embedding.c` from the executable's sources.
- `target_link_libraries(vexilloscope PRIVATE ovg_vision)` replaces `ovg_nn`.
- `mlp_classifier.c` stays in the source list (see Decisions).

#### `src/patch_embedding.c` / `.h` — deleted

`TgPatchEmbed` replaces `VxPatchEmbedding` field-for-field (`ClsToken → Cls`, `PatchEmb → Proj`, `PosEmb → PosEmb`; `use_cls = 1`). `VxPatchEmbeddingForward` / `vx_patch_embedding_forward_full` (the two-tensor return exposing `projected`) have no callers outside the file and are not carried over.

#### `src/mlp_classifier.c` / `.h` — `TgLinear` API update

`#include "tg_mlp.h"` → `"tg_linear.h"`. `tg_linear_create(input_dim, hidden_dim, 1, 0.01f)` → `tg_linear_create(input_dim, hidden_dim, 0.01f)` (both layers). `tg_linear_forward(&m->hidden, image, &f.xw1)` → `tg_linear_forward(&m->hidden, image)`; likewise `output`. `VxMlpForward` drops `xw1`/`xw2`. **`vx_mlp_forward_free` becomes `tg_free_graph(f->logits)`** (then NULLs the fields): the new forward builds `xw` and the reshaped/expanded bias as internal nodes that `VxMlpForward` cannot reach, so the per-tensor `tg_free` calls it makes today would leak them. `tg_free_graph` frees every non-persistent node reachable from `logits` — `z1`, `h`, `logits`, and the internals — and stops at persistent tensors, so the caller's `image` must be `persistent = 1` (the same rule vexilloscope already applies to the ViT's patches, which `tg_adam_accumulate` frees through). Document that on `vx_mlp_forward_full` and keep the `z1`/`h`/`logits` fields for inspection. The `[1 x …]` shape comments stay accurate (`B` is now `[1, n]` rather than `[batch=1, n]`, same numbers). `vx_mlp_collect_params` is unchanged. Nothing calls this module; the check is that it compiles.

#### `src/vit.h` / `src/vit.c` — model and format

```c
#include "tg_patch_embed.h"
#include "tg_pool.h"
#include "tg_transformer.h"
#include <stdint.h>

/* Bound on the block count any file may carry; loaders fatal above it. 16 is
   an arbitrary ceiling with headroom over today's 6 (196 pointers of stack);
   main.c _Static_asserts VX_N_BLOCKS <= VX_VIT_MAX_BLOCKS, so growing the model
   past it is a one-line, compiler-enforced change here. */
#define VX_VIT_MAX_BLOCKS 16
#define VX_VIT_MAX_PARAMS (3 + VX_VIT_MAX_BLOCKS * 12 + 1)

/* Fixed-width; stored verbatim as the checkpoint config blob (sizeof == 28). */
typedef struct {
    int32_t n_patches, patch_size, embed_dim, hidden_dim, n_blocks, n_heads, n_labels;
} VxViTConfig;

typedef struct {
    TgPatchEmbed  patch_emb;   /* Cls [1, C] + Proj [patch_size, C] + PosEmb [n_patches, C] */
    TgTransformer encoder;     /* non-causal */
    Tensor       *Wout;        /* [embed_dim, n_labels] */
    int           n_labels, embed_dim;
} VxViT;

VxViT       vx_vit_create(int n_patches, int patch_size, int embed_dim, int hidden_dim,
                          int n_blocks, int n_heads, int n_labels, float max_drop_path_rate);  /* unchanged */
VxViT       vx_vit_create_from_config(const VxViTConfig *cfg, float max_drop_path_rate);
VxViTConfig vx_vit_config(const VxViT *v);
void        vx_vit_free(VxViT *v);
Tensor     *vx_vit_forward(VxViT *v, Tensor *patches);           /* [n_patches, patch_size] → [1, n_labels] */
int         vx_vit_collect_params(VxViT *v, Tensor **params);    /* unchanged order: patch_emb (3), blocks (12 each), Wout */

/* Weights-only readers over the library checkpoint (v3; v2 also accepted by the library).
   Both fatal on any failure, as today. */
VxViT       vx_vit_load(const char *path);                       /* architecture from the blob; strict shapes */
VxViT       vx_vit_load_warmstart(const char *path, int new_n_labels, float max_drop_path_rate);
```

- `vx_vit_forward` (`vit.c:40-60`): `vx_patch_embedding_forward` → `tg_patch_embed_forward`; the `tg_slice` + `tg_reshape` pair at lines 55-56 → `tg_pool_cls(enc)`. The comment block updates to the recipe above. Output stays `[1, n_labels]` for the 2D input every caller passes.
- `vx_vit_collect_params`: `vx_patch_embedding_collect_params(&v->patch_emb, params + n)` → `tg_patch_embed_collect_params(&v->patch_emb, params + n, 3)`. The block loop and trailing `Wout` are unchanged, so the on-disk parameter order is unchanged.
- `vx_vit_config` reads the seven fields off the model exactly as `vx_vit_save` does today (`n_heads` from `blocks[0].attn.n_heads`).
- **`vx_vit_save` (`vit.c:90-140`) is deleted.** Training saves through `tg_checkpoint_save_run` in `main.c` (it needs the `TgAdam`, which `vit.c` does not have). No weights-only export path remains because nothing calls one; `tg_checkpoint_save` is a one-liner if one is ever wanted.
- **A `static` helper `load_strict(const char *path, float max_drop_path_rate, VxViTConfig *cfg_out)` in `vit.c`** does the shared work: `tg_checkpoint_info(path, &info, cfg_out, sizeof *cfg_out)`; fatal if it returns -1, `info.config_len != sizeof(VxViTConfig)`, or `cfg_out->n_blocks > VX_VIT_MAX_BLOCKS`; `v = vx_vit_create_from_config(cfg_out, max_drop_path_rate)`; `Tensor *params[VX_VIT_MAX_PARAMS]; n = vx_vit_collect_params(&v, params)`; `tg_checkpoint_load(path, params, n)`, fatal on -1; returns `v`. The weights-only loader leaves a v3 optimizer section unread, so the file the training loop writes is the file identify modes read.
- **`vx_vit_load` (`vit.c:142-223`) is rewritten** as `load_strict(path, 0.0f, &cfg)` (no stochastic depth at inference, as today at `vit.c:181`) plus the `loaded weights from '%s' (%d params n_labels=%d)` notice on stderr, which stays (the JSON modes depend on stdout being clean).
- **`vx_vit_load_warmstart` (`vit.c:225-362`) is rewritten**: `v = load_strict(path, max_drop_path_rate, &cfg)` (builds `cfg`'s architecture with the caller's rate, strict load of every parameter, `Wout` included at `[C, cfg.n_labels]`); fatal if `new_n_labels < cfg.n_labels` (truncation is ambiguous — same rule, same message); then widen the head in place: `Wnew = tg_new(2, {C, new_n_labels})`, `tg_fill_xavier_uniform(Wnew)`, `Wnew->persistent = 1`; for each row `r`, copy `cfg.n_labels` floats from `Wold[r * cfg.n_labels]` to `Wnew[r * new_n_labels]` (the loop at `vit.c:338-346`, reading from a tensor instead of a stream); `tg_free(Wold)`; `v.Wout = Wnew; v.n_labels = new_n_labels`. Columns `[cfg.n_labels, new_n_labels)` keep Xavier init, as today. The two `warm-start: loaded …` notices stay.
  - **Behaviour change, deliberate:** today `vx_vit_load_warmstart` builds the model with `max_drop_path_rate = 0.0f` (`vit.c:272`; `vx_vit_load` does the same at 181, which is correct for inference), so a warm-started training run has no stochastic depth while a fresh one has 0.10. The new signature takes the rate from the caller and `main.c` passes `VX_DROP_PATH_RATE_X1000 / 1000.0f` in both branches. Called out here so the reviewer sees it; it is a fix, not a side effect.
- The `Tensor *params[512]` scratch arrays (`vit.c:104, 183, 274`; `main.c:459` in `vit_params_to_cuda`; and `main.c:1144`'s `3 + VX_N_BLOCKS * 12 + 1`) are all sized `VX_VIT_MAX_PARAMS`, defined in `vit.h` from `VX_VIT_MAX_BLOCKS = 16` (above). `VX_N_BLOCKS` is a `main.c` enum (`main.c:35`, value 6), which `vit.h` cannot see, so the bound is `vit.h`'s own constant and `main.c` adds `_Static_assert(VX_N_BLOCKS <= VX_VIT_MAX_BLOCKS, "raise VX_VIT_MAX_BLOCKS")` next to the enum. `load_strict` fatals if the blob's `n_blocks` exceeds the bound. `512` is a v1 artifact and goes (`3 + 16 * 12 + 1 = 196` pointers is the new ceiling; the compiled model uses 76).

#### `src/main.c` — train mode

Under the auto-resume decision ([Open Questions](#open-questions), closed 2026-09-20). Line numbers are as of `ad6a75c`.

- New enum constant `VX_SAVE_EVERY = 500` (the existing loss-log cadence), documented like the others.
- Model construction (`1135-1139`): `--warmstart` → `vx_vit_load_warmstart(warmstart_path, dataset.count, VX_DROP_PATH_RATE_X1000 / 1000.0f)`; else `vx_vit_create(...)` as today. Then `VxViTConfig cfg = vx_vit_config(&vit);` for the saves.
- After `tg_adam_create` (`1154`), **resume** (pseudo-code; the `fatal` messages carry the arguments they name):
  ```c
  FILE *probe = fopen(weights_path, "rb");
  if (probe) {                                   /* the file exists: from here on, every failure is fatal */
      fclose(probe);
      if (warmstart_path)                        /* first: does not depend on the file parsing */
          fatal("'%s' exists; --warmstart would overwrite it. Delete it or pass another --weights.");
      TgCheckpointInfo info; VxViTConfig file_cfg;
      if (tg_checkpoint_info(weights_path, &info, &file_cfg, sizeof file_cfg) != 0)
          fatal("'%s' exists but is not a readable checkpoint. Delete it or pass another --weights.");
      if (info.config_len != sizeof file_cfg || memcmp(&file_cfg, &cfg, sizeof cfg) != 0)
          fatal("'%s' is a checkpoint for a different architecture (n_labels %d vs %d, ...). Delete it, pass another --weights, or use --warmstart.");
      if (!info.has_optimizer)
          fatal("'%s' is weights-only (no optimizer state); resume needs a training checkpoint. Use --warmstart with a fresh --weights.");
      if (tg_checkpoint_load_run(weights_path, &opt, TG_LOAD_RESUME, &info) != 0) fatal(...);
      printf("resumed '%s' at step %d\n", weights_path, opt.step);
  }
  ```
  The existence test is the plain `fopen` probe **before** `tg_checkpoint_info`: NULL means fresh run (print nothing, skip the block); non-NULL means the file exists, and from then on any `tg_checkpoint_info` / `load_run` failure is fatal — never overwrite a file that looks like a checkpoint but does not parse. Do not use `tg_checkpoint_info`'s -1 alone to mean "missing": it returns -1 for a corrupt file too. The `has_optimizer` check exists because `load_run(RESUME)` on a v2 file or a v3 file without an optimizer section does **not** fail: it loads the params, resets the optimizer to step 0, prints the library's `no optimizer state` notice, and returns 0 — which would make train mode announce `resumed ... at step 0` and train from scratch over those weights, overwriting the file at the first periodic save. Nothing in this spec writes such a file, but a weights-only v3 (or any v2) file at `--weights` is exactly the warm-start input, and the message says so. The `tg_seed_from_entropy()` calls at `903/910/916` already precede this point, satisfying the ordering contract; the logged seed no longer describes the run's dropout stream after a resume, which the resume notice makes evident.
- **Completed-run guard**, as lambda's: if `opt.step >= VX_VIT_STEPS`, print `vexilloscope: '%s' is complete at step %d; skipping training` and skip the loop and the final save. Eval, clean accuracy, and the identify demos still run — they are the reason to rerun a finished model.
- Loop header (`1157`): `for (int step = opt.step + 1; step <= VX_VIT_STEPS; step++)`. The step body is unchanged (Phase 1 already migrated it). `idx` (`1171`) is unchanged and is therefore a function of the cumulative step.
- **Periodic save** inside the loop, after `tg_adam_update`: `if (step % VX_SAVE_EVERY == 0) tg_checkpoint_save_run(weights_path, &opt, &cfg, sizeof cfg);` — on -1, print and continue (a failed periodic save should not kill a run; the final save's failure is fatal).
- **Final save before `tg_adam_free`.** Today the order is `tg_from_cuda` (1218) → `tg_adam_free` (1220) → `vx_vit_save` (1223); `save_run` needs the live optimizer, so it becomes `tg_checkpoint_save_run(weights_path, &opt, &cfg, sizeof cfg)` (fatal on -1) → `tg_adam_free`. When `VX_VIT_STEPS` is a multiple of `VX_SAVE_EVERY` (it is: 60000 / 500, and the acceptance's 20 / 10 and 30 / 10) the last periodic save and the final save write identical state; the redundancy is accepted rather than special-cased, because the final save is the one whose failure is fatal. The `tg_from_cuda` sync at 1218 stays: `save_run` syncs for itself, but the clean-accuracy pass at the end reads host data.
- `vit_params_to_cuda` (`458`) changes only its `params[512]` to `params[VX_VIT_MAX_PARAMS]`; the four identify-mode blocks are unchanged in text. They keep working because `vx_vit_load` keeps its name and contract.

#### Retrain

After the vexilloscope commit lands: delete the v1 `vit_weights.bin`, run the full 60,000-step training. Its output is the first v3 `vit_weights.bin`; identify modes and the bot pick it up with no further change. Accuracy is reported in vexilloscope's own docs and is not an acceptance item of this spec (the schedule and model are unchanged by this phase; the retrain is owed to the v1→v2 shape change, not to Phase 2).

---

## Tests

New files `tests/test_linear.c` and `tests/test_vision.c`, each with a `run_*_tests(int *passed, int *failed)` declared and called in `test_main.c` (sections `=== linear ===` after `attention`, `=== vision ===` after `gpt`). Existing tests unchanged. Tolerances are absolute on floats of order 1.

### `test_linear.c`

| Test | Checks |
|---|---|
| `test_linear_forward_2d` | `tg_linear_create(4, 3, 0.1f)`; fill `B` with `{1, 2, 3}`; `x` `[5, 4]` random; output shape `[5, 3]`; every element equals the hand loop `sum_k x[i][k] W[k][j] + B[j]` within 1e-6. |
| `test_linear_forward_3d` | Same layer; `x` `[2, 5, 4]`; output `[2, 5, 3]`; batch `b` of the 3D output equals the 2D forward of `x[b]` within 1e-6. |
| `test_linear_backward` | `x` `[2, 5, 4]`; `tg_backward(tg_sum(out))`: `B->grad[j] == 10` (= B·T) exactly; `W->grad[k][j] == sum_{b,t} x[b][t][k]` within 1e-5; `x->grad[b][t][k] == sum_j W[k][j]` within 1e-5. |
| `test_linear_params` | `tg_linear_params` returns `{W, B}`, `n == 2`; `B` is `[1, n_out]`; both `persistent`. |
| `test_linear_bad_input_fatal` | `x` `[5, 3]` into `n_in = 4` → `ovg_fatal` (setjmp/longjmp); `tg_linear_create(0, 3, 0.1f)` → fatal. |
| `test_linear_cuda_parity` *(CUDA)* | Layer copied to the device; 3D forward + backward; output and `W`/`B` grads within 1e-5 of the host run. |

### `test_vision.c`

Fixture: `n_patches = 4, patch_size = 6, C = 8`, patches filled with a deterministic ramp (`tg_seed(42)` is set by `test_main`).

| Test | Checks |
|---|---|
| `test_patch_embed_shapes` | `create(4, 6, 8, 1)`: 2D `[4, 6]` → `[1, 5, 8]`; 3D `[2, 4, 6]` → `[2, 5, 8]`. `create(4, 6, 8, 0)`: `[2, 4, 6]` → `[2, 4, 8]`. `tg_patch_embed_n_tokens` is 5 and 4. |
| `test_patch_embed_values` | `use_cls = 1`; `Cls` filled with 0.5, `PosEmb[t][c] = t + 0.01c`; for every `b`: `out[b][0][:] == 0.5`; `out[b][t+1][c] == (patches[b][t] · Proj[:, c]) + PosEmb[t][c]` within 1e-5. |
| `test_patch_embed_batch_parity` | `[2, 4, 6]` forward; row `b` equals the 2D forward of `patches[b]` (shape `[1, 5, 8]`, row 0) within 1e-6. |
| `test_patch_embed_backward` | `[2, 4, 6]`, `tg_backward(tg_sum(out))`: `Cls->grad[c] == 2` (= B); `PosEmb->grad[t][c] == 2`; `Proj->grad[k][c] == sum_{b,t} patches[b][t][k]` within 1e-5; `patches->grad[b][t][k] == sum_c Proj[k][c]` within 1e-5. Then `tg_free_graph` leaves the three parameters and the persistent `patches` allocated (no crash on the second forward). |
| `test_patch_embed_collect_params` | `use_cls = 1`: returns 3, `params = {Cls, Proj, PosEmb}` by identity; `use_cls = 0`: returns 2, `{Proj, PosEmb}`; all `persistent`; `max_params = 2` with `use_cls = 1` → fatal. |
| `test_patch_embed_bad_shape_fatal` | `[4, 7]` (wrong patch_size), `[3, 6]` (wrong n_patches), and a 4D input each → fatal. |
| `test_pool_cls` | `enc` `[2, 5, 8]` with `enc[b][t][c] = 100b + 10t + c`; `out` `[2, 8]`, `out[b][c] == 100b + c`; `tg_backward(tg_sum(out))`: `enc->grad` is 1 at `t = 0`, 0 elsewhere. |
| `test_pool_mean_tokens` | Same `enc`; `out[b][c] == mean_t enc[b][t][c] == 100b + 20 + c` within 1e-5; backward of `tg_sum`: every `enc->grad == 1/5` within 1e-7. |
| `test_pool_mean_matches_mean_rows` | `B = 1`: `tg_pool_mean_tokens(enc [1, T, C])` equals `tg_mean_rows(tg_reshape(enc, [T, C]))` bitwise. Holds because at `B = 1` the transpose `[1, T, C] → [T, 1, C]` is the identity on memory, the reshapes are copies, and both sides reach `tg_mean_rows` with the same `[T, C]` bytes and therefore the same reduction order. |
| `test_pool_bad_ndim_fatal` | 2D `[5, 8]` into either helper → fatal. |
| `test_vit_recipe_end_to_end` | `TgPatchEmbed(4, 6, 8, 1)` → `tg_transformer_create_encoder(1, 8, 16, 5, 2, 0.0f)` → `tg_pool_cls` → `@ Wout [8, 3]`; batch-2 logits `[2, 3]`; each row within 1e-5 of the batch-1 run on that sample (in eval mode); `tg_backward(tg_cross_entropy_sparse(logits, {0, 2}, 2, 0.0f))` runs and every collected parameter (3 + 12 + 1 = 16) has a finite grad; `tg_free_graph` then a second forward succeeds. |
| `test_patch_embed_cuda_parity` *(CUDA)* | Parameters and a `[2, 4, 6]` input on the device; forward output and the three parameter grads within 1e-5 of the host run. |
| `test_pool_cuda_parity` *(CUDA)* | Both helpers on a device `[2, 5, 8]`; outputs and `enc` grads within 1e-6 of host. |

**Expected count:** `test_linear.c` 6 (1 CUDA-guarded) + `test_vision.c` 13 (2 CUDA-guarded) → **101 + 19 = 120 (CUDA build) / 89 + 16 = 105 (CPU build)**. The implementation sets the final number; `AGENTS.md` is updated to match.

---

## Documentation Updates (same change)

`AGENTS.md` (this repo):

- Opening paragraph and the "three layered CMake targets" sentence: four targets, `ovg_vision` (patch embedding, token pooling) named beside `ovg_lm`.
- Repository Structure: `tg_mlp.h` → `tg_linear.h` and `tg_mlp.c` → `tg_linear.c` under `[nn]`; new `[vision]` groups in `include/ovg/` (`tg_patch_embed.h`, `tg_pool.h`) and `src/` (`[vision → ovg_vision]`: `tg_patch_embed.c`, `tg_pool.c`); `tests/test_linear.c`, `tests/test_vision.c`.
- Library Layers table: `ovg_nn` row lists `TgLinear` unchanged; new `ovg_vision` row ("token-layout blocks for vision transformers: `TgPatchEmbed`, `tg_pool_cls`, `tg_pool_mean_tokens`", links `ovg_nn`); umbrella row links `ovg_lm` + `ovg_vision`. "Consumers link the narrowest target": a ViT links `ovg_vision`. Placing new code: add "Assumes a token layout that only vision models produce (patch embedding, token pooling) → `ovg_vision`".
- New section **Linear Layer** (`tg_linear.h`): struct, `create(n_in, n_out, w_scale)`, `forward` rank rule and bias expansion, `params`.
- New section **Vision Package** (`tg_patch_embed.h`, `tg_pool.h`): struct, forward shapes (always 3D out; PosEmb before CLS; CLS at token 0), collect order, both pools (3D in, `[B, C]` out), the mean composition, and the recipe snippet.
- Build Commands / Verification: new counts; the per-layer build list gains `ovg_vision`; the sentence about the test binary linking `ovg_lm` becomes "links `ovg_lm` and `ovg_vision`".
- Important Guidance: "`tg_patch_embed_forward` always returns 3D; the pooling helpers take 3D only; `tg_pool_cls` assumes `use_cls = 1` — the library cannot check it."

`docs/FOUNDATION_PLATFORM.md`, on completion:

- Claims table, "Independently usable modality packages": basis "`ovg_lm` and `ovg_vision` exist; `ovg_diffusion` Phase 3b"; watch "Two of three today."
- Claims table, "Proven by real applications": drop "vexilloscope needs a retrain" once the retrain has run.
- Brownfield Baseline: a **Phase 2 (vision, `docs/SPEC_PLATFORM_PHASE2_VISION.md`)** line: `TgLinear` fixed and renamed; `ovg_vision` (`TgPatchEmbed`, `tg_pool_cls`, `tg_pool_mean_tokens`) created and linked by the umbrella; tests 120 / 105; vexilloscope links `ovg_vision`, dropped `patch_embedding.c` and the `VXWT` format, trains through `save_run` with resume.
- Vision Blocks → Follow-ups: the `tg_mean_rows` question was marked Closed (compose; see this spec) at spec review on 2026-09-20 — already done. `TgLinear` Fate → Follow-ups: rename done.

`../vexilloscope/AGENTS.md`:

- Repository Structure: remove `patch_embedding.c / .h`; `vit.c / .h` line mentions `VxViTConfig` and the checkpoint readers; `mlp_classifier` line unchanged.
- Model Architecture: PatchEmbedding line names `TgPatchEmbed` (`ovg_vision`); the CLS-extract lines become `tg_pool_cls(enc) → [1 × embed_dim]`.
- Key Structs: `VxPatchEmbedding` → `TgPatchEmbed` (from `tg_patch_embed.h`); `VxViT` as above; add `VxViTConfig`.
- Training Loop: startup gains "resume from `--weights` if present (`tg_checkpoint_load_run`, `TG_LOAD_RESUME`)"; the per-step block is already harness-shaped; add "every `VX_SAVE_EVERY` steps and at the end: `tg_checkpoint_save_run`"; the "after all steps" order becomes sync → save_run → free optimizer → eval. State the resume guarantee (params/moments/step/library RNG exact; sample order by step; augmentation not restored).
- Binary CLI Flags: `--weights` doubles as the resume source in train mode; `--warmstart` with an existing `--weights` file is an error; the completed-run notice.
- **Weights Serialization → Checkpoints**: the three `vx_vit_*` prototypes replaced by `vx_vit_load` / `vx_vit_load_warmstart` over the library format; "File format: otto-von-grad checkpoint v3 (see its `AGENTS.md` → Checkpoints); config blob = `VxViTConfig` (28 bytes); parameter order = `vx_vit_collect_params`." Remove the `VXWT` sentence.
- Build Commands: replace the "Retrain required (2026-09-20)" note with the outcome (retrained on `<date>`, file is v3) or, until the retrain runs, with "the v1 file cannot load; train once to produce a v3 file".
- Dependency: otto-von-grad: `ovg_nn` → `ovg_vision` ("the vision package; it carries `ovg_nn` and `ovg_core`; `ovg_lm` is not needed").

---

## Non-Goals

- Any new differentiable op. `tg_mean_rows` stays 2D; `tg_layer_norm` is untouched.
- A `TgViT` model struct in the library. The assembly (`patch embed → encoder → pool → head`) stays in the application; `ovg_vision` ships the parts, as `ovg_lm` would if `TgGPT` did not already exist.
- `tg_transformer_collect_params`, or migrating `TgGPT`'s `Wout` / `TgBlock`'s FFN to `TgLinear`.
- Convolutional patch embedding, image decoding, letterboxing, augmentation (Phase 3a / application).
- A `VXWT` → v3 converter, or any reader for the `VXWT` format.
- Restoring `rand()`-driven augmentation on resume (Phase 1 non-goal, unchanged).
- Batching vexilloscope's training (`VX_BATCH_SIZE` stays 1; the 3D path is tested in the library, not adopted by the app in this phase).
- Deleting `mlp_classifier.c` from vexilloscope.
- Namespaced includes (Phase 4).

---

## Risks and Notes for the Implementer

- **`tg_pool_cls` cannot verify `use_cls`.** Pooling token 0 of a CLS-less model silently returns the first patch. The recipe and `AGENTS.md` say so; the app-side `VxViT` always builds with `use_cls = 1`.
- **The mean composition materialises `[T, B, C]` and then `[T, B*C]`.** Two extra full-size copies per forward (transpose, reshape) and their mirrors in backward. Acceptable; if a profile ever shows it, the fix is `tg_mean_axis` in `ovg_core` behind the same helper.
- **Head widening is not a `TgAdam` operation.** `vx_vit_load_warmstart` runs before `tg_adam_create`, so the new `Wout` gets fresh moments with everything else — correct for a warm start. Do not create the optimizer before widening.
- **Resume validates the blob by `memcmp`.** Any architecture change (including `dataset.count` growing) fails loudly and points at `--warmstart`. This is intended; the alternative is silently training a mismatched head or overwriting a 5-hour file.
- **`tg_checkpoint_info` returns -1 for both a missing and a corrupt file.** The `fopen` probe in the resume block is what tells them apart; without it a corrupt checkpoint would be silently overwritten by a fresh run.
- **Saving inside the loop costs a device→host sync of every parameter and moment** (~27 MB for the default architecture). At `VX_SAVE_EVERY = 500` that is once per ~3 minutes; negligible.
- **MSVC stdout is fully buffered to pipes and under ConPTY**, so a killed run prints nothing. Acceptance items that need mid-run output are specified as natural-exit checks with a fabricated step count (below); do not spec a Ctrl-C check.
- **`git mv` before edit** for the `TgLinear` rename so history follows the file; commit the rename and the API change together (a rename-only commit would leave `tg_linear.h` with the old API).
- **Layer discipline:** `tg_patch_embed.c` / `tg_pool.c` include nothing from `ovg_lm`; nothing in `ovg_core` / `ovg_nn` / `ovg_lm` includes `tg_patch_embed.h` / `tg_pool.h`. The grep in [Build](#4-build--cmakeliststxt) is the check.
- **Behaviour change on warm start**: stochastic depth now applies to warm-started runs (see `vit.c` migration). If the owner wants the old behaviour, pass `0.0f` in `main.c` — the library does not decide this.

---

## Acceptance

Library (this repo):

- [x] `tg_linear.h` / `tg_linear.c` replace `tg_mlp.h` / `tg_mlp.c` with the specified API; `grep -r tg_mlp` over `src include tests examples CMakeLists.txt` is empty.
- [x] `ovg_vision` target exists with `tg_patch_embed.[ch]` and `tg_pool.[ch]` as specified; `ottovongrad` links `ovg_lm` and `ovg_vision`; the test binary links both.
- [x] `cmake --preset default && cmake --build --preset default` clean; `otto_von_grad_tests.exe` reports all pass at the new count (expected 120); `cmake --preset cpu && cmake --build --preset cpu` passes its count (expected 105). *(120 passed, 0 failed / 105 passed, 0 failed.)*
- [x] `ovg_core`, `ovg_nn`, `ovg_lm`, `ovg_vision` each build standalone via `--target`; the layer greps in [Build](#4-build--cmakeliststxt) return nothing unexpected.
- [x] `examples/candide.c` untouched and builds; `../lambda` configures and builds with no edits to its own CMakeLists.

vexilloscope (`../vexilloscope`, one commit):

- [x] Links `ovg_vision`; `patch_embedding.c/.h` deleted; `vit.c` has no `fopen`, no `"VXWT"`; no `Tensor *params[512]` anywhere (`grep -n "params\[512\]" src/*.c` is empty; the `char path[512]` string buffers in `main.c` are unrelated and stay); `mlp_classifier.c` compiles against the new `TgLinear` and frees through `tg_free_graph`; zero warnings as before.
- [x] **Natural-exit resume check** (fabricated constants, then restored): with `VX_VIT_STEPS = 20`, `VX_SAVE_EVERY = 10`, a fresh run exits normally and leaves a v3 `vit_weights.bin` with `step == 20` (`tg_checkpoint_info` or the next run's notice); with `VX_VIT_STEPS = 30`, the next run prints `resumed ... at step 20`, trains 10 steps, exits with `step == 30`; a third run at 30 prints the complete notice, skips training, and still runs eval and the identify demos without rewriting the file (mtime unchanged). Run these against a reduced labels file (a dozen classes) or expect the post-training eval (`VX_EVAL_AUGS × n_flags` forward passes plus three detector demos) to take minutes on the full 402 classes — that wait is the eval, not a stuck guard. The reduced-class file is also what the next item's warm-start needs as its "fewer classes" starting point.
- [x] With the file from the previous item present at `--weights`, `--warmstart <that file>` is refused with the `--warmstart would overwrite it` message; with a fresh `--weights` path and `--warmstart <that file>` plus a labels file with more classes, the run starts with `Wout` widened (notice printed) and `opt.step == 0`.
- [x] All four `--identify*` modes load that v3 file and produce output in their existing formats (`--identify-json` stdout is a single JSON line; the seed and `loaded weights` notices are on stderr).
- [ ] `main.c` shrinks or holds (the resume block is added; the save/free reorder is neutral); `vit.c` shrinks by the deleted format code (expected well under 200 lines from 362). *(Result: `vit.c` 362 → 191 lines, as expected. `main.c` 1,257 → 1,335: the resume block (~50 lines), the periodic and final `save_run` blocks, the `_Static_assert`, and a seven-line fix for a pre-existing crash in the train-mode tail — the params were left `on_cuda = 0` after the CPU clean-accuracy pass and the identify demos, which upload their patches, died with `make_op: mixed CUDA/CPU parents` — outweigh the one deleted `vx_vit_save` call. The "shrinks or holds" premise was wrong as written; the growth is the resume path this spec adds.)*
- [ ] After the commit: the v1 `vit_weights.bin` is deleted and the full retrain is started; the file it writes is the deployed model. (Not gated on accuracy.) *(The v1 file was removed from the vexilloscope root on 2026-09-20; the 60,000-step retrain has not been started.)*

Documentation:

- [x] `AGENTS.md` updated as listed (structure, layers table, Linear Layer and Vision Package sections, counts, guidance).
- [x] `docs/FOUNDATION_PLATFORM.md` claims table ("two of three"), Brownfield Baseline Phase 2 line, and the `TgLinear` Fate follow-up marked done (the Vision Blocks follow-up was closed at spec review).
- [x] `../vexilloscope/AGENTS.md` updated as listed (structure, architecture, structs, training loop, CLI, Checkpoints, Build, Dependency).
