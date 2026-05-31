#!/usr/bin/env python3
"""
Phase 2 corpus validator: checks every `lhs -> rhs` line against the
beta_step() oracle.

Each line must have exactly one ` -> ` separator.  The RHS is normalised
through parse/pretty before comparison, so harmless formatting differences
do not count as failures.

Usage:
    python eval_phase2.py
    python eval_phase2.py --corpus path/to/corpus.txt --max-examples 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambda_term import beta_step, parse, pretty

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "text" / "lambda_phase2.txt"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   metavar="PATH", help="corpus file (default: data/text/lambda_phase2.txt)")
    p.add_argument("--max-examples", type=int, default=10,
                   metavar="N", help="max failure examples to print (default: 10)")
    args = p.parse_args()

    if not args.corpus.exists():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    total = 0
    n_malformed = 0
    n_parse_err = 0
    n_no_redex = 0
    n_mismatch = 0
    shown: list[tuple[int, str, str, str]] = []  # (lineno, line, reason, detail)

    def record(lineno: int, line: str, reason: str, detail: str = "") -> None:
        if len(shown) < args.max_examples:
            shown.append((lineno, line, reason, detail))

    with open(args.corpus, encoding="ascii") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line:
                continue
            total += 1

            parts = line.split(" -> ")
            if len(parts) != 2:
                n_malformed += 1
                record(lineno, line, "malformed",
                       f"{len(parts) - 1} occurrence(s) of ' -> '")
                continue

            lhs_src, rhs_src = parts

            try:
                lhs_term = parse(lhs_src)
            except ValueError as exc:
                n_parse_err += 1
                record(lineno, line, "parse error (lhs)", str(exc))
                continue

            try:
                rhs_term = parse(rhs_src)
            except ValueError as exc:
                n_parse_err += 1
                record(lineno, line, "parse error (rhs)", str(exc))
                continue

            expected = beta_step(lhs_term)
            if expected is None:
                n_no_redex += 1
                record(lineno, line, "no redex in lhs")
                continue

            if pretty(expected) != pretty(rhs_term):
                n_mismatch += 1
                record(lineno, line, "mismatch",
                       f"  oracle: {pretty(expected)}")

    n_fail = n_malformed + n_parse_err + n_no_redex + n_mismatch
    n_pass = total - n_fail
    rate = n_pass / total if total else 0.0

    print(f"corpus:        {args.corpus}")
    print(f"total:         {total:,}")
    print(f"passed:        {n_pass:,}")
    print(f"malformed:     {n_malformed:,}")
    print(f"parse errors:  {n_parse_err:,}")
    print(f"no redex:      {n_no_redex:,}")
    print(f"mismatches:    {n_mismatch:,}")
    print(f"pass rate:     {rate:.2%}")

    if shown:
        print(f"\nfailures (up to {args.max_examples}):")
        for lineno, line, reason, detail in shown:
            print(f"  line {lineno:>6}: [{reason}]  {line}")
            if detail:
                print(f"            {detail}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
