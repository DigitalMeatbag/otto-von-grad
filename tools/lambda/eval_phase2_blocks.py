#!/usr/bin/env python3
"""
Per-block model accuracy evaluation for phase 2.

Uses explicit ``#`` delimiter lines when present, otherwise splits a legacy
corpus into --blocks equal segments. It writes all LHS prompts to a temp file,
runs otto_lambda once with --prompts-file (model loads once), then checks each
completion against the beta_step oracle.

Usage:
    python eval_phase2_blocks.py
    python eval_phase2_blocks.py --max-per-block 200
    python eval_phase2_blocks.py --phase 2 --blocks 5 --corpus path/to/corpus.txt
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambda_term import beta_step, parse, pretty

DEFAULT_LABELS = ["2a identity", "2b constant", "2c substitution", "2d root larger", "2e context"]
DEFAULT_CORPUS = REPO_ROOT / "data" / "text" / "lambda_phase2.txt"
DEFAULT_EXE    = REPO_ROOT / "build" / "otto_lambda.exe"


def load_blocks(path: Path, expected_blocks: int) -> list[list[str]]:
    """Load curriculum blocks, honoring generator delimiter lines."""
    lines = path.read_text(encoding="ascii").splitlines()
    explicit_blocks: list[list[str]] = []
    current: list[str] = []
    saw_delimiter = False

    for line in lines:
        if not line.strip():
            continue
        if line.startswith("#"):
            saw_delimiter = True
            explicit_blocks.append(current)
            current = []
        else:
            current.append(line)

    if saw_delimiter:
        explicit_blocks.append(current)
        if len(explicit_blocks) != expected_blocks:
            raise ValueError(
                f"corpus contains {len(explicit_blocks)} delimited blocks; "
                f"--blocks is {expected_blocks}"
            )
        if any(not block for block in explicit_blocks):
            raise ValueError("corpus contains an empty delimited block")
        return explicit_blocks

    # Backward compatibility for corpora generated before delimiters existed.
    content = [line for line in lines if line.strip()]
    block_size, remainder = divmod(len(content), expected_blocks)
    blocks: list[list[str]] = []
    offset = 0
    for block_index in range(expected_blocks):
        size = block_size + (1 if block_index < remainder else 0)
        blocks.append(content[offset : offset + size])
        offset += size
    return blocks


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase", type=int, default=2,
                   metavar="N", help="phase checkpoint to load (default: 2)")
    p.add_argument("--blocks", type=int, default=5,
                   metavar="N", help="number of blocks (default: 5)")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   metavar="PATH")
    p.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                   metavar="PATH")
    p.add_argument("--max-per-block", type=int, default=None,
                   metavar="N", help="max examples per block (default: all)")
    p.add_argument("--max-gen", type=int, default=100,
                   metavar="N", help="max generation steps per prompt (default: 100)")
    p.add_argument("--show-failures", type=int, default=3,
                   metavar="N", help="failure examples to print per block (default: 3)")
    p.add_argument("--labels", nargs="*", default=None,
                   metavar="LABEL", help="block label names")
    args = p.parse_args()

    if not args.corpus.exists():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)
    if not args.exe.exists():
        print(f"error: executable not found: {args.exe}", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or DEFAULT_LABELS

    try:
        blocks = load_blocks(args.corpus, args.blocks)
    except ValueError as exc:
        p.error(str(exc))
    n = sum(len(block) for block in blocks)
    if args.max_per_block is not None:
        blocks = [block[:args.max_per_block] for block in blocks]

    # Collect valid examples and build prompt list (one pass, preserving block membership)
    # valid_idx[i] = (block_index, lhs_src, expected_pretty)
    prompts:    list[str]              = []
    valid_idx:  list[tuple[int,str,str]] = []
    skipped_by_block: list[int]        = [0] * args.blocks

    for b, chunk in enumerate(blocks):
        for line in chunk:
            parts = line.split(" -> ")
            if len(parts) != 2:
                skipped_by_block[b] += 1
                continue
            lhs_src = parts[0]
            try:
                expected = beta_step(parse(lhs_src))
            except ValueError:
                skipped_by_block[b] += 1
                continue
            if expected is None:
                skipped_by_block[b] += 1
                continue
            prompts.append(lhs_src + " ->")
            valid_idx.append((b, lhs_src, pretty(expected)))

    total_prompts = len(prompts)
    cap_str = str(args.max_per_block) if args.max_per_block is not None else "all"
    print(f"corpus:  {args.corpus}  ({n} examples)")
    print(f"model:   phase {args.phase}  ({args.exe.name})")
    print(f"eval:    {args.blocks} blocks x {cap_str} -> {total_prompts} valid prompts")
    print()

    # Write prompts to a temp file and run model once
    fd, tmppath = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", newline="\n") as f:
            for pr in prompts:
                f.write(pr + "\n")

        result = subprocess.run(
            [str(args.exe), str(args.phase),
             "--prompts-file", tmppath,
             "--max-gen", str(args.max_gen)],
            capture_output=True, text=True,
        )
    finally:
        os.unlink(tmppath)

    if result.returncode != 0:
        print(f"error: model process exited with status {result.returncode}",
              file=sys.stderr)
        print(f"stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    completions = result.stdout.splitlines()

    if len(completions) != total_prompts:
        print(f"error: sent {total_prompts} prompts, got {len(completions)} completions",
              file=sys.stderr)
        print(f"stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    # Score per block
    correct_by_block:  list[int]              = [0] * args.blocks
    total_by_block:    list[int]              = [0] * args.blocks
    failures_by_block: list[list[tuple]]      = [[] for _ in range(args.blocks)]

    for (b, lhs_src, expected_str), raw_completion in zip(valid_idx, completions):
        completion = raw_completion.strip()
        total_by_block[b] += 1
        try:
            got = parse(completion)
            if pretty(got) == expected_str:
                correct_by_block[b] += 1
            else:
                failures_by_block[b].append((lhs_src, expected_str, completion))
        except ValueError:
            failures_by_block[b].append((lhs_src, expected_str, f"<parse error: {completion!r}>"))

    # Report
    total_correct = total_total = 0
    for b in range(args.blocks):
        label   = labels[b] if b < len(labels) else f"block {b+1}"
        correct = correct_by_block[b]
        total   = total_by_block[b]
        skipped = skipped_by_block[b]
        rate    = correct / total if total else 0.0
        skip_str = f"  ({skipped} skipped)" if skipped else ""
        print(f"  {label:28s}  {correct:>4}/{total:<4}  {rate:6.1%}{skip_str}")
        for lhs, exp, got in failures_by_block[b][:args.show_failures]:
            print(f"    lhs:      {lhs}")
            print(f"    expected: {exp}")
            print(f"    got:      {got}")
        total_correct += correct
        total_total   += total

    overall = total_correct / total_total if total_total else 0.0
    print()
    print(f"  {'overall':28s}  {total_correct:>4}/{total_total:<4}  {overall:6.1%}")

    sys.exit(0 if total_correct == total_total else 1)


if __name__ == "__main__":
    main()
