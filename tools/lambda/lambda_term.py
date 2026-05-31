"""
Lambda calculus AST nodes, pretty printer, and free-variable analysis.

Notation (used by pretty()):
  Variable:    x
  Abstraction: \\x.body      (backslash, right-associative body, no parens around body)
  Application: (fn arg)      (fully parenthesized)

Variable pool: single lowercase letters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Abs:
    param: str
    body: "Term"


@dataclass(frozen=True)
class App:
    fn: "Term"
    arg: "Term"


Term = Union[Var, Abs, App]


def pretty(t: Term) -> str:
    """Serialize a Term to a string in fully-parenthesized backslash notation."""
    if isinstance(t, Var):
        return t.name
    if isinstance(t, Abs):
        return f"\\{t.param}.{pretty(t.body)}"
    if isinstance(t, App):
        return f"({pretty(t.fn)} {pretty(t.arg)})"
    raise TypeError(f"Unknown term type: {type(t)}")


def free_vars(t: Term) -> frozenset[str]:
    if isinstance(t, Var):
        return frozenset({t.name})
    if isinstance(t, Abs):
        return free_vars(t.body) - {t.param}
    if isinstance(t, App):
        return free_vars(t.fn) | free_vars(t.arg)
    raise TypeError(f"Unknown term type: {type(t)}")


def is_closed(t: Term) -> bool:
    return len(free_vars(t)) == 0


def depth(t: Term) -> int:
    """Maximum nesting depth of the AST."""
    if isinstance(t, Var):
        return 0
    if isinstance(t, Abs):
        return 1 + depth(t.body)
    if isinstance(t, App):
        return 1 + max(depth(t.fn), depth(t.arg))
    raise TypeError(f"Unknown term type: {type(t)}")


# --- Substitution and beta reduction ---

def substitute(t: Term, var: str, repl: Term) -> Term:
    """Substitute repl for free occurrences of var in t.

    Assumes repl is closed (no free variables), so capture is impossible and
    alpha-renaming is never needed.
    """
    if isinstance(t, Var):
        return repl if t.name == var else t
    if isinstance(t, Abs):
        if t.param == var:
            # var is shadowed by this binder — don't descend
            return t
        return Abs(t.param, substitute(t.body, var, repl))
    if isinstance(t, App):
        return App(substitute(t.fn, var, repl), substitute(t.arg, var, repl))
    raise TypeError(f"Unknown term type: {type(t)}")


def beta_step(t: Term) -> Term | None:
    """Perform one leftmost-outermost beta reduction step.

    Returns the reduced term, or None if no redex exists.
    """
    if isinstance(t, App):
        if isinstance(t.fn, Abs):
            # Redex at the root — fire it.
            return substitute(t.fn.body, t.fn.param, t.arg)
        # Try to reduce inside fn first (leftmost-outermost).
        reduced_fn = beta_step(t.fn)
        if reduced_fn is not None:
            return App(reduced_fn, t.arg)
        reduced_arg = beta_step(t.arg)
        if reduced_arg is not None:
            return App(t.fn, reduced_arg)
    if isinstance(t, Abs):
        reduced_body = beta_step(t.body)
        if reduced_body is not None:
            return Abs(t.param, reduced_body)
    return None


# --- Parser ---

def parse(src: str) -> Term:
    """Parse a pretty()-formatted string back into a Term.

    Accepts exactly the output format of pretty():
      Variable:    single char in a-h
      Abstraction: \\x.body   (right-associative, body extends as far as possible)
      Application: (fn arg)   (fully parenthesized, one space between fn and arg)
    """
    term, pos = _parse_term(src, 0)
    if pos != len(src):
        raise ValueError(f"Unexpected trailing input at position {pos}: {src[pos:]!r}")
    return term


def _parse_term(src: str, pos: int):
    if pos >= len(src):
        raise ValueError("Unexpected end of input")
    ch = src[pos]
    if ch == '\\':
        pos += 1
        if pos >= len(src) or src[pos] not in 'abcdefgh':
            raise ValueError(f"Expected variable after '\\' at position {pos}")
        param = src[pos]
        pos += 1
        if pos >= len(src) or src[pos] != '.':
            raise ValueError(f"Expected '.' at position {pos}")
        pos += 1
        body, pos = _parse_term(src, pos)
        return Abs(param, body), pos
    if ch == '(':
        pos += 1
        fn, pos = _parse_term(src, pos)
        if pos >= len(src) or src[pos] != ' ':
            raise ValueError(f"Expected ' ' in application at position {pos}")
        pos += 1
        arg, pos = _parse_term(src, pos)
        if pos >= len(src) or src[pos] != ')':
            raise ValueError(f"Expected ')' at position {pos}")
        return App(fn, arg), pos + 1
    if ch in 'abcdefgh':
        return Var(ch), pos + 1
    raise ValueError(f"Unexpected character {ch!r} at position {pos}")


# --- Alpha-equivalence ---

def alpha_equiv(t1: Term, t2: Term) -> bool:
    """Return True if t1 and t2 are alpha-equivalent.

    Uses a pair of renaming environments that map each bound variable name to
    a unique identity object at the point where its binder was entered.  Free
    variables are compared by name.
    """
    return _alpha_equiv(t1, t2, {}, {})


def _alpha_equiv(t1: Term, t2: Term, env1: dict, env2: dict) -> bool:
    if isinstance(t1, Var) and isinstance(t2, Var):
        return env1.get(t1.name, t1.name) == env2.get(t2.name, t2.name)
    if isinstance(t1, Abs) and isinstance(t2, Abs):
        fresh = object()
        return _alpha_equiv(t1.body, t2.body,
                            {**env1, t1.param: fresh},
                            {**env2, t2.param: fresh})
    if isinstance(t1, App) and isinstance(t2, App):
        return (_alpha_equiv(t1.fn, t2.fn, env1, env2) and
                _alpha_equiv(t1.arg, t2.arg, env1, env2))
    return False
