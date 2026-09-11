#!/usr/bin/env python3
"""Tests for parse() and alpha_equiv() in lambda_term.py."""
import unittest

from lambda_term import Abs, Var, alpha_equiv, parse, pretty


class ParseTests(unittest.TestCase):
    def test_round_trips(self):
        cases = [
            ("a", "a"),
            ("\\a.a", "\\a.a"),
            ("(a b)", "(a b)"),
            ("\\a.\\b.(a b)", "\\a.\\b.(a b)"),
            ("(\\a.a \\b.b)", "(\\a.a \\b.b)"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(pretty(parse(source)), expected)

    def test_parse_errors(self):
        for source in ("", "a b", "(a b", "z"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    parse(source)


class AlphaEquivalenceTests(unittest.TestCase):
    def test_equivalent_terms(self):
        cases = [
            (Abs("a", Var("a")), Abs("b", Var("b"))),
            (Abs("a", Abs("a", Var("a"))),
             Abs("b", Abs("c", Var("c")))),
            (Abs("a", Abs("b", Var("a"))),
             Abs("c", Abs("d", Var("c")))),
            (Abs("a", Var("b")), Abs("c", Var("b"))),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertTrue(alpha_equiv(left, right))

    def test_non_equivalent_terms(self):
        cases = [
            (Abs("a", Var("b")), Abs("b", Var("b"))),
            (Abs("a", Abs("b", Var("a"))),
             Abs("c", Abs("d", Var("d")))),
            (Abs("a", Var("b")), Abs("a", Var("c"))),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertFalse(alpha_equiv(left, right))


if __name__ == "__main__":
    unittest.main(verbosity=2)
