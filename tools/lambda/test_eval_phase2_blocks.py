#!/usr/bin/env python3
"""Tests for Phase 2 evaluation corpus partitioning."""
import tempfile
import unittest
from pathlib import Path

from eval_phase2_blocks import load_blocks


class LoadBlocksTests(unittest.TestCase):
    def _load(self, text: str, expected_blocks: int):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text(text, encoding="ascii", newline="\n")
            return load_blocks(path, expected_blocks)

    def test_uses_explicit_uneven_blocks(self):
        blocks = self._load("a -> a\n#\nb -> b\nc -> c\n", 2)
        self.assertEqual(blocks, [["a -> a"], ["b -> b", "c -> c"]])

    def test_rejects_delimiter_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "2 delimited blocks"):
            self._load("a -> a\n#\nb -> b\n", 3)

    def test_splits_legacy_corpus_evenly(self):
        blocks = self._load("a\nb\nc\nd\ne\n", 2)
        self.assertEqual(blocks, [["a", "b", "c"], ["d", "e"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
