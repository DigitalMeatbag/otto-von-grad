#!/usr/bin/env python3
"""
Phase 2 lambda calculus corpus generator: single-step beta reduction.

Generates output in 5 progressively harder phases written sequentially
to one file.  Each line: <term> -> <one-step-reduct>

  a  root identity       (\\x.x T) -> T
  b  root constant       (\\x.B T) -> B       x not free in B
  c  root substitution   (\\x.body T) -> ...  small body/arg, x free in body
  d  root beta large     same but larger body/arg depths
  e  contextual beta     C[(\\x.body T)] -> C[reduct]

Vocabulary (32 chars): a-h, \\, ., (, ), space, -, >, newline.

Usage:
    python gen_phase2.py                          # 100k pairs (20k per phase)
    python gen_phase2.py --count 500000
    python gen_phase2.py --output path/to/out.txt
"""
import argparse
import random
import sys
from pathlib import Path

from lambda_term import Abs, App, Term, Var, beta_step, free_vars, pretty
from gen_phase1 import VARS, sample_closed


# ---------------------------------------------------------------------------
# Redex builders
# ---------------------------------------------------------------------------

def _identity_redex(arg_depth: int) -> App:
    """(\\x.x arg) — identity function applied to arg."""
    param = random.choice(VARS)
    return App(Abs(param, Var(param)), sample_closed(arg_depth))


def _constant_redex(body_depth: int, arg_depth: int) -> App:
    """(\\x.body arg) where x not free in body — arg is discarded."""
    param = random.choice(VARS)
    # sample_closed with no initial bound vars is always closed; param can't be free.
    body = sample_closed(body_depth)
    return App(Abs(param, body), sample_closed(arg_depth))


def _using_redex(body_depth: int, arg_depth: int) -> App | None:
    """(\\x.body arg) where x IS free in body — returns None if body skips x."""
    param = random.choice(VARS)
    body = sample_closed(body_depth, bound=(param,))
    if param not in free_vars(body):
        return None
    return App(Abs(param, body), sample_closed(arg_depth))


# ---------------------------------------------------------------------------
# Context embedding (phase e)
# ---------------------------------------------------------------------------

def _embed(redex: Term, context_depth: int) -> Term:
    """Wrap a closed redex in a random closed context of the given depth."""
    if context_depth == 0:
        return redex
    r = random.random()
    if r < 0.40:
        p = random.choice(VARS)
        return Abs(p, _embed(redex, context_depth - 1))
    sibling_depth = random.randint(1, max(1, context_depth - 1))
    sibling = sample_closed(sibling_depth)
    if r < 0.70:
        return App(_embed(redex, context_depth - 1), sibling)
    return App(sibling, _embed(redex, context_depth - 1))


# ---------------------------------------------------------------------------
# Per-phase sampling
# ---------------------------------------------------------------------------

def _sample(phase: str, args) -> tuple[Term, Term] | None:
    """Return (term, reduct) for phase, or None to retry."""
    if phase == 'a':
        arg_depth = random.randint(1, args.max_arg_depth)
        term = _identity_redex(arg_depth)

    elif phase == 'b':
        body_depth = random.randint(1, args.max_body_depth)
        arg_depth  = random.randint(1, args.max_arg_depth)
        term = _constant_redex(body_depth, arg_depth)

    elif phase == 'c':
        body_depth = random.randint(1, min(2, args.max_body_depth))
        arg_depth  = random.randint(1, min(2, args.max_arg_depth))
        term = _using_redex(body_depth, arg_depth)
        if term is None:
            return None

    elif phase == 'd':
        lo = min(2, args.max_body_depth)
        body_depth = random.randint(lo, args.max_body_depth)
        lo = min(2, args.max_arg_depth)
        arg_depth  = random.randint(lo, args.max_arg_depth)
        term = _using_redex(body_depth, arg_depth)
        if term is None:
            return None

    elif phase == 'e':
        body_depth = random.randint(1, args.max_body_depth)
        arg_depth  = random.randint(1, args.max_arg_depth)
        ctx_depth  = random.randint(1, args.max_context_depth)
        redex = _using_redex(body_depth, arg_depth) or _identity_redex(arg_depth)
        term = _embed(redex, ctx_depth)
        # Reject if beta_step would fire at the root — phase e is contextual only.
        # This happens when _embed chose App(sibling, ...) and sibling is an Abs.
        if isinstance(term, App) and isinstance(term.fn, Abs):
            return None

    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    reduct = beta_step(term)
    if reduct is None:
        return None
    return term, reduct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--count", type=int, default=100_000, metavar="N",
                   help="Total pairs (split evenly across 5 phases, default: 100000)")
    p.add_argument("--max-context-depth", type=int, default=3, metavar="D",
                   help="Max context depth for phase e (default: 3)")
    p.add_argument("--max-arg-depth", type=int, default=3, metavar="D",
                   help="Max arg AST depth (default: 3)")
    p.add_argument("--max-body-depth", type=int, default=3, metavar="D",
                   help="Max body AST depth (default: 3)")
    p.add_argument("--max-attempts-multiplier", type=int, default=50, metavar="M",
                   help="Per-phase attempt cap = target * M; warns and moves on if hit (default: 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent / "data" / "text" / "lambda_phase2.txt",
                   help="Output file path")
    args = p.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    per_phase = args.count // 5
    phase_counts = {
        'a': per_phase,
        'b': per_phase,
        'c': per_phase,
        'd': per_phase,
        'e': args.count - 4 * per_phase,
    }

    seen: set[str] = set()
    total_written = 0

    with open(args.output, "w", encoding="ascii", newline="\n") as f:
        for phase in ('a', 'b', 'c', 'd', 'e'):
            target = phase_counts[phase]
            cap = target * args.max_attempts_multiplier
            written = 0
            attempts = 0
            while written < target and attempts < cap:
                attempts += 1
                result = _sample(phase, args)
                if result is None:
                    continue
                term, reduct = result
                line = f"{pretty(term)} -> {pretty(reduct)}"
                if line in seen:
                    continue
                seen.add(line)
                f.write(line + "\n")
                written += 1
                if written % 5_000 == 0:
                    print(f"  phase {phase}: {written:>7,} / {target:,}  (attempts: {attempts:,})",
                          file=sys.stderr)

            if written < target:
                print(f"  phase {phase}: WARNING — space exhausted after {attempts:,} attempts; "
                      f"wrote {written:,} / {target:,}", file=sys.stderr)
            else:
                print(f"  phase {phase}: {written:,} pairs  (attempts: {attempts:,})", file=sys.stderr)

            total_written += written

    print(f"Wrote {total_written:,} unique pairs -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
