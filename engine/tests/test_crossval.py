"""Çapraz doğrulamadaki B-Cubed F yardımcıları."""

from __future__ import annotations

import unittest
from collections import Counter

from engine.evaluation.crossval import (
    bcubed_from_counts,
    bootstrap_bcubed_difference,
    item_columns,
)
from engine.evaluation.metrics import reconstruction_bcubed

PAIRS = [
    ("*jak", "*jak"),
    ("*tag", "*dag"),
    ("*kara", "*qara"),
    ("*tas", "*taš"),
    ("*köl", "*göl"),
    ("*bil", "*bil"),
]


class TestCrossvalBcubed(unittest.TestCase):
    def test_counts_match_reconstruction_bcubed(self):
        # Sayım tabanlı hızlı hesap, harness'in veri kümesi düzeyi ölçüsüyle aynı olmalı.
        pooled = sum((item_columns(p) for p in PAIRS), Counter())
        self.assertAlmostEqual(
            bcubed_from_counts(pooled), reconstruction_bcubed(PAIRS)["fscore"], places=4
        )

    def test_abstention_contributes_nothing(self):
        self.assertEqual(item_columns(None), Counter())
        self.assertEqual(bcubed_from_counts(Counter()), 0.0)

    def test_paired_bootstrap_identical_systems(self):
        columns = [item_columns(p) for p in PAIRS]
        result = bootstrap_bcubed_difference(columns, columns, iterations=200)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        self.assertFalse(result["significant"])

    def test_paired_bootstrap_detects_better_system(self):
        perfect = [item_columns((g, g)) for _, g in PAIRS] * 10
        noisy = [item_columns(p) for p in PAIRS] * 10
        result = bootstrap_bcubed_difference(perfect, noisy, iterations=300)
        self.assertGreater(result["difference"], 0)
        self.assertTrue(result["a_is_better"])


if __name__ == "__main__":
    unittest.main()
