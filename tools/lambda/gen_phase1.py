#!/usr/bin/env python3
"""
Phase 1 lambda calculus syntax corpus generator.

Generates well-formed lambda expressions and writes them as plain text
(one expression per line) for character-level language model training.

Vocabulary (30 chars): a-h, \\, ., (, ), space, newline.

Usage:
    python gen_phase1.py                          # 100k unique closed terms, depth 1-5
    python gen_phase1.py --count 500000
    python gen_phase1.py --min-depth 2 --max-depth 7
    python gen_phase1.py --open                   # allow free variables
    python gen_phase1.py --output path/to/out.txt
"""
import argparse
import random
import sys
from pathlib import Path

from lambda_term import Abs, App, Term, Var, depth, is_closed, pretty

# Fixed variable pool. Keeping it small (8 symbols) means the model never
# sees an unfamiliar identifier; smaller pool also increases reuse / binding density.
VARS = list("abcdefgh")


def sample_closed(max_depth: int, d: int = 0, bound: tuple[str, ...] = ()) -> Term:
    """Sample a random closed lambda term.

    Tracks bound variables to ensure every leaf Var refers to a binder above it.
    If no bound variables are available, forces an Abs to introduce one.
    """
    if not bound or (d >= max_depth and bound):
        if d >= max_depth:
            # Leaf: must pick a bound variable.
            return Var(random.choice(bound))

    if not bound:
        # No bound vars yet — can't produce a closed term without abstracting first.
        param = random.choice(VARS)
        return Abs(param, sample_closed(max_depth, d + 1, bound + (param,)))

    r = random.random()
    if r < 0.20:
        return Var(random.choice(bound))
    elif r < 0.50:
        param = random.choice(VARS)
        return Abs(param, sample_closed(max_depth, d + 1, bound + (param,)))
    else:
        fn = sample_closed(max_depth, d + 1, bound)
        arg = sample_closed(max_depth, d + 1, bound)
        return App(fn, arg)


def sample_open(max_depth: int, d: int = 0) -> Term:
    """Sample a random lambda term (free variables allowed)."""
    if d >= max_depth:
        return Var(random.choice(VARS))

    r = random.random()
    if r < 0.25:
        return Var(random.choice(VARS))
    elif r < 0.55:
        param = random.choice(VARS)
        return Abs(param, sample_open(max_depth, d + 1))
    else:
        return App(sample_open(max_depth, d + 1), sample_open(max_depth, d + 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=100_000, metavar="N",
                   help="Number of expressions to generate (default: 100000)")
    p.add_argument("--min-depth", type=int, default=1, metavar="D",
                   help="Minimum AST depth (default: 1)")
    p.add_argument("--max-depth", type=int, default=5, metavar="D",
                   help="Maximum AST depth (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--open", dest="closed_only", action="store_false", default=True,
                   help="Allow free variables (default: closed terms only)")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent.parent.parent / "data" / "text" / "lambda_phase1.txt",
                   help="Output file path")
    args = p.parse_args()

    if args.min_depth < 1:
        p.error("--min-depth must be >= 1")
    if args.max_depth < args.min_depth:
        p.error("--max-depth must be >= --min-depth")

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sampler = sample_closed if args.closed_only else sample_open

    seen: set[str] = set()
    written = 0
    with open(args.output, "w", encoding="ascii", newline="\n") as f:
        while written < args.count:
            d = random.randint(args.min_depth, args.max_depth)
            t = sampler(d)
            s = pretty(t)
            if s in seen:
                continue
            seen.add(s)
            f.write(s + "\n")
            written += 1

            if written % 10_000 == 0:
                print(f"  {written:>8,} / {args.count:,}", file=sys.stderr)

    print(f"Wrote {written:,} unique expressions → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
