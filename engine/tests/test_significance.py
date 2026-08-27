"""
Anlamlılık testleri ve negatif kontrol bataryası testleri.

"A, B'den iyi" cümlesi ancak istatistiksel olarak gösterilirse kurulabilir.
Bu testler o mekanizmanın doğru çalıştığını korur.
"""

from __future__ import annotations

import unittest

from engine.evaluation.negative_controls import (
    ALL_BATTERIES,
    LOANWORD_TRAPS,
    OBVIOUSLY_FAKE,
    PHONOTACTICALLY_VALID,
    run_battery,
)
from engine.evaluation.significance import (
    benjamini_hochberg,
    bootstrap_difference,
    chance_resemblance_test,
    compare_systems,
    mcnemar_test,
    permutation_test,
)


class TestPermutationTest(unittest.TestCase):
    def test_identical_systems_are_not_significant(self):
        flags = [True, False, True, False] * 25
        result = permutation_test(flags, flags, iterations=500)
        self.assertEqual(result.statistic, 0.0)
        self.assertFalse(result.significant)

    def test_large_consistent_difference_is_significant(self):
        good = [True] * 100
        bad = [False] * 100
        self.assertTrue(permutation_test(good, bad, iterations=500).significant)

    def test_tiny_difference_is_not_significant(self):
        """Regresyon koruması: %1'lik fark n=200'de gürültüdür."""
        a = [True] * 51 + [False] * 149
        b = [True] * 49 + [False] * 151
        self.assertFalse(permutation_test(a, b, iterations=2000).significant)

    def test_empty_input_is_handled(self):
        self.assertEqual(permutation_test([], []).p_value, 1.0)


class TestMcNemar(unittest.TestCase):
    def test_no_discordant_pairs(self):
        flags = [True, False, True]
        result = mcnemar_test(flags, flags)
        self.assertEqual(result.p_value, 1.0)
        self.assertEqual(result.n, 0)

    def test_all_discordant_one_way_is_significant(self):
        a = [True] * 20
        b = [False] * 20
        self.assertTrue(mcnemar_test(a, b).significant)

    def test_balanced_discordance_is_not_significant(self):
        a = [True] * 10 + [False] * 10
        b = [False] * 10 + [True] * 10
        self.assertFalse(mcnemar_test(a, b).significant)


class TestBootstrapDifference(unittest.TestCase):
    def test_interval_contains_zero_for_equal_systems(self):
        flags = [True, False] * 50
        diff, low, high = bootstrap_difference(flags, flags, iterations=500)
        self.assertEqual(diff, 0.0)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)

    def test_interval_excludes_zero_for_clear_difference(self):
        _, low, high = bootstrap_difference([True] * 100, [False] * 100, iterations=500)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, 0.0)


class TestBenjaminiHochberg(unittest.TestCase):
    def test_all_significant_when_all_tiny(self):
        self.assertEqual(benjamini_hochberg([0.001, 0.002, 0.003]), [True, True, True])

    def test_none_significant_when_all_large(self):
        self.assertEqual(benjamini_hochberg([0.5, 0.6, 0.9]), [False, False, False])

    def test_controls_false_discoveries_among_many_tests(self):
        """100 testin biri salt şansla p<0,05 çıkar; FDR bunu eler."""
        p_values = [0.04] + [0.5] * 99
        self.assertFalse(any(benjamini_hochberg(p_values)))

    def test_decisions_follow_input_order(self):
        decisions = benjamini_hochberg([0.9, 0.0001, 0.8])
        self.assertEqual(decisions, [False, True, False])

    def test_empty(self):
        self.assertEqual(benjamini_hochberg([]), [])


class TestChanceResemblance(unittest.TestCase):
    """Kessler 2001: ses benzerliği tek başına akrabalık kanıtı değildir."""

    def test_chance_level_matches_are_not_significant(self):
        pool = [f"k{v}l" for v in "aeiou"] + [f"t{v}r" for v in "aeiou"]
        queries = pool[:5]
        result = chance_resemblance_test(
            observed_matches=2,
            query_forms=queries,
            candidate_pool=pool,
            similarity=lambda a, b: a[0] == b[0],
            iterations=300,
        )
        self.assertFalse(result.significant)

    def test_far_above_chance_is_significant(self):
        pool = [f"{c}{v}n" for c in "bkmst" for v in "aeiou"]
        result = chance_resemblance_test(
            observed_matches=len(pool),
            query_forms=pool,
            candidate_pool=pool,
            similarity=lambda a, b: a == b,
            iterations=300,
        )
        self.assertTrue(result.significant)

    def test_empty_pool_is_handled(self):
        self.assertEqual(
            chance_resemblance_test(0, [], [], lambda a, b: True).p_value, 1.0
        )


class TestCompareSystems(unittest.TestCase):
    def test_reports_every_non_reference_system(self):
        rows = compare_systems(
            {
                "comparative": [True] * 50,
                "baseline": [False] * 50,
                "other": [True] * 25 + [False] * 25,
            },
            reference="baseline",
        )
        self.assertEqual({r["system"] for r in rows}, {"comparative", "other"})

    def test_unknown_reference_raises(self):
        with self.assertRaises(KeyError):
            compare_systems({"a": [True]}, reference="yok")


class TestNegativeControlBatteries(unittest.TestCase):
    def test_every_battery_has_items(self):
        for name, items in ALL_BATTERIES.items():
            with self.subTest(battery=name):
                self.assertGreater(len(items), 0, name)

    def test_phonotactically_valid_fakes_obey_turkic_shape(self):
        """Bu batarya ancak maddeler GERÇEKTEN Türkçe gibi görünürse anlamlı."""
        from engine.utils.phonotactics import VOWELS

        for item in PHONOTACTICALLY_VALID:
            with self.subTest(word=item.query):
                self.assertTrue(any(ch in VOWELS for ch in item.query), item.query)
                self.assertGreaterEqual(len(item.query), 4, item.query)

    def test_obvious_fakes_are_rejected(self):
        """Bariz uydurmalarda yanlış-pozitif oranı sıfır olmalı."""
        from engine.evaluation.harness import comparative_reconstructor

        result = run_battery(comparative_reconstructor(), OBVIOUSLY_FAKE, "bariz_sahte")
        self.assertEqual(result.false_positive_rate, 0.0)

    def test_no_strong_claim_on_any_negative_control(self):
        """En kritik güvenlik özelliği.

        Motor uydurma veya alıntı bir kelimeye ASLA 🟢/🟡 rozet vermemelidir.
        Düşük güvenle bir aday üretmesi kabul edilebilir; onu güçlü bir iddia
        olarak sunması kabul edilemez.
        """
        from engine.evaluation.harness import comparative_reconstructor

        engine = comparative_reconstructor()
        for name, items in ALL_BATTERIES.items():
            with self.subTest(battery=name):
                result = run_battery(engine, items, name)
                self.assertEqual(result.strong_claim_rate, 0.0, name)

    def test_loanword_traps_are_documented(self):
        """Her tuzak, NEDEN tuzak olduğunu söylemeli."""
        for item in LOANWORD_TRAPS:
            with self.subTest(word=item.query):
                self.assertTrue(item.reason.strip(), item.query)




class TestHomonymBattery(unittest.TestCase):
    """Eşadlı kelimeye KESİN karar vermek, karar vermemekten kötüdür."""

    def test_homonyms_are_a_separate_battery(self):
        from engine.evaluation.negative_controls import HOMONYM_CASES

        self.assertIn("eşadlı", ALL_BATTERIES)
        self.assertGreaterEqual(len(HOMONYM_CASES), 3)

    def test_engine_does_not_block_reconstruction_for_homonyms(self):
        """`çay` hem "tea" (Farsça alıntı) hem "brook" (miras) demektir.

        Motorun bunu "alıntı" diye kesin karara bağlayıp miras hipotezini
        hiç kurmaması yanlış olurdu.
        """
        from engine.evaluation.negative_controls import HOMONYM_CASES
        from engine.nlp.borrowing_detector import BorrowingDetector

        detector = BorrowingDetector()
        for item in HOMONYM_CASES:
            with self.subTest(word=item.query):
                entries = [{"lang_code": c, "word": w} for c, w in item.witnesses]
                verdict = detector.detect(item.query, entries)
                self.assertFalse(verdict.blocks_inherited_reconstruction, item.query)

if __name__ == "__main__":
    unittest.main()
