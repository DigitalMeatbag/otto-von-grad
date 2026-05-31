# Lambda Calculus Curriculum

A staged curriculum for training a small GPT on lambda calculus,
from raw syntax through Church encodings.  Each phase fine-tunes from the
previous phase's checkpoint (`lambda_main.c` handles staged loading).

---

## Phases

| Phase | Task | Corpus format |
|-------|------|---------------|
| 1 | Syntax — generate well-formed closed lambda terms | `\a.(a a)` |
| 2 | Single-step beta reduction | `((\a.a) b) -> b` |
| 3 | Multi-step reduction to normal form | `((\a.a \b.b) \c.c) -> (\b.b \c.c) -> \c.c` |
| 4 | Alpha-renaming / capture-avoiding substitution | `=>` rename steps plus `->` beta steps |
| 5 | Church encodings — arithmetic and booleans | anonymous Church terms using the same rewrite chains |

Vocabulary is pinned across all phases:

```text
\n <space> ( ) - . = > \ a b c d e f g h
```

That is 17 characters total. Rewrite arrows have fixed meanings:

- `->` beta-reduction
- `=>` alpha-conversion

The corpus should not introduce labels, digits, named combinators, `*`, or
variables outside `a` through `h`. Keeping this alphabet fixed means
embedding and output-projection dimensions are stable across staged fine-tuning.

---

## Corpus Generators

| Script | Phase |
|--------|-------|
| `gen_phase1.py` | Phase 1 |
| `gen_phase2.py` | Phase 2 |

Shared AST utilities (pretty-printer, free-variable analysis, substitution,
`beta_step`) live in `lambda_term.py`.

The `parse()` stub in `lambda_term.py` must be implemented before structural
evaluation can run.

---

## Evaluation Plan

Loss (cross-entropy) is a proxy metric.  The target metric at each phase is
structural or semantic correctness on a **fixed held-out test set** (different
RNG seed from the training corpus, never touched during training).

### Eval principles

- **Greedy decoding (temp=0 / argmax) for eval**, not sampling.  Beta
  reduction has one correct answer; sampling obscures whether the model
  knows it.
- **Sequence-level exact match** (or alpha-equivalence), not token-level.
  A model that outputs the wrong variable name has near-perfect token
  accuracy but is wrong.
- **Cumulative eval across phases.**  Sequential fine-tuning risks
  catastrophic forgetting.  Each phase's eval suite includes probes from
  all prior phases.

### Eval modes (phases 3–5)

Multi-step phases support two evaluation modes:

- **Stepwise (primary):** Prompt the model with the current derivation prefix
  ending in the next expected arrow (`<initial> -> <step1> ->` or
  `<initial> => <renamed> ->`), then check the next single-step output
  against the oracle.  Validates each local rewrite independently and
  localizes the first error in a derivation.
- **One-shot (secondary):** Prompt with the initial term only and let the
  model generate the complete derivation chain.  Measures end-to-end
  coherence but does not pinpoint where the model first goes wrong.

Both modes use greedy decoding.

### Per-phase metrics

**Phase 1 — syntax validity**
- Metric: `% syntactically valid` and `% closed` among N sampled completions.
- Method: sample completions, run `parse()`, check `is_closed()`.

**Phase 2 — single-step beta accuracy**
- Metric: exact string match against `beta_step()` oracle.
- Safe for exact match because redex arguments are always closed, so the
  oracle reduct is deterministic and capture-free.
- Method: prompt model with `<term> ->`, complete greedily, compare to oracle.

**Phase 3 — multi-step / normal form**
- Primary metric (stepwise): per-step alpha-equivalence to the oracle reduct.
  Report step accuracy and the distribution of first-error positions.
- Secondary metric (one-shot): alpha-equivalence of the final term to the
  oracle normal form.
- Exact string match is insufficient: normal forms are unique only up to
  alpha-equivalence, so `parse()` + alpha-equiv checking is load-bearing
  by phase 3 (not phase 4).
- Also track: does the model converge (vs. loop or truncate)?

**Phase 4 — alpha-renaming / capture avoidance**
- Primary metric (stepwise): each `=>` step must be alpha-equivalent to the
  prior term; each `->` step must match the beta oracle modulo
  alpha-equivalence.  Report per-step accuracy and first-error position.
- Secondary metric (one-shot): full chain validity.
- Additional probe: deliberately constructed capture-prone terms — verify
  the model avoids capture rather than producing a wrong result.

**Phase 5 — Church encodings**
- Primary metric (stepwise): per-step alpha-equivalence to the oracle reduct,
  same as phase 4.  Report step accuracy and first-error position.
- Secondary metric (one-shot): semantic equality — decode the final output
  Church numeral to an integer and compare to the expected value, independent
  of variable names.  This is the richest signal in the curriculum: it tests
  whether the model has internalized the *meaning* of terms, not just
  reduction mechanics.

### Infrastructure requirements

| Requirement | Needed by |
|-------------|-----------|
| `parse()` implementation in `lambda_term.py` | Phase 3 eval |
| Alpha-equivalence checker | Phase 3 eval |
| Church numeral decoder | Phase 5 eval |
| Inference-only mode for `lambda_main.c` (prompt → greedy completion) | All phases |
