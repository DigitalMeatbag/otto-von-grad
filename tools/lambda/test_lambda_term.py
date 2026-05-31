#!/usr/bin/env python3
"""Tests for parse() and alpha_equiv() in lambda_term.py."""
import sys
from lambda_term import Abs, App, Var, alpha_equiv, parse, pretty

failures = 0

def check(label, got, expected):
    global failures
    if got != expected:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures += 1
    else:
        print(f"ok    {label}")


# --- parse round-trips ---

def rt(src):
    return pretty(parse(src))

check("var",          rt("a"),             "a")
check("abs",          rt("\\a.a"),         "\\a.a")
check("app",          rt("(a b)"),         "(a b)")
check("abs chain",    rt("\\a.\\b.(a b)"), "\\a.\\b.(a b)")
check("app abs abs",  rt("(\\a.a \\b.b)"), "(\\a.a \\b.b)")

# --- alpha_equiv ---

check("\\a.a == \\b.b",
      alpha_equiv(Abs("a", Var("a")), Abs("b", Var("b"))), True)

check("\\a.\\a.a == \\b.\\c.c  (inner binder shadows outer)",
      alpha_equiv(Abs("a", Abs("a", Var("a"))),
                  Abs("b", Abs("c", Var("c")))), True)

check("\\a.b != \\b.b  (free b vs bound b)",
      alpha_equiv(Abs("a", Var("b")), Abs("b", Var("b"))), False)

check("\\a.\\b.a == \\c.\\d.c  (outer var referenced)",
      alpha_equiv(Abs("a", Abs("b", Var("a"))),
                  Abs("c", Abs("d", Var("c")))), True)

check("\\a.\\b.a != \\c.\\d.d  (different binder referenced)",
      alpha_equiv(Abs("a", Abs("b", Var("a"))),
                  Abs("c", Abs("d", Var("d")))), False)

check("free vars must match by name: \\a.b == \\c.b",
      alpha_equiv(Abs("a", Var("b")), Abs("c", Var("b"))), True)

check("free vars differ: \\a.b != \\a.c",
      alpha_equiv(Abs("a", Var("b")), Abs("a", Var("c"))), False)

# --- parse errors ---

def expect_error(label, src):
    global failures
    try:
        parse(src)
        print(f"FAIL  {label}: expected ValueError, got none")
        failures += 1
    except ValueError:
        print(f"ok    {label}")

expect_error("empty string",      "")
expect_error("trailing garbage",  "a b")
expect_error("unclosed app",      "(a b")
expect_error("bad var char",      "z")

sys.exit(failures)
