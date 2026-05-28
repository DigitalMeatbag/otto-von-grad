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


# --- Parser stub (to be implemented in a later phase) ---

def parse(src: str) -> Term:
    """Parse a pretty()-formatted string back into a Term.
    Stub — raises NotImplementedError until phase 2 requires it.
    """
    raise NotImplementedError("Parser not yet implemented")
