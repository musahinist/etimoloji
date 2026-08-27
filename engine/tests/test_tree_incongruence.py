"""
Ağaç-uyumsuz dağılım testleri (Faz C3).

⚠️ **Bu modül DOĞRULANMADI ve doğrulanamıyor.** Türki-içi alıntı etiketi
gerekiyor; elimizdeki tek kaynak WOLD/Sakha'da ``Source_languoid ==
"Turkic"`` olan **18 madde**. O örneklemde F hesaplanabilir ama güven
aralığı kullanılamayacak kadar geniş olur.

Bu yüzden modül bir sınıflandırıcı değil, **aday üretecidir** ve karar
katmanına bağlı değildir. Testler o ayrımı korur.
"""

from __future__ import annotations

import unittest

from engine.nlp.tree_incongruence import (
    MIN_LANGUAGES_FOR_DISTRIBUTION,
    USE_AS_SIGNAL,
    analyse,
)


class TestDistributionAnalysis(unittest.TestCase):
    def test_single_branch_cannot_be_incongruent(self):
        verdict = analyse(["tr", "az", "tk", "gag", "ota", "tr", "az", "tk"])
        self.assertFalse(verdict.is_scattered)
        self.assertIn("tek kolda", verdict.explanation)

    def test_too_few_witnesses_is_refused(self):
        """⚠️ İlk sürümde bu kısıt yoktu ve sonuç kullanılamazdı: 400
        kümenin **106'sı** (%27) aday işaretlendi çünkü 3 tanıklı bir küme
        her zaman düşük kol kapsamı verir. Ölçülen şey uyumsuzluk değil,
        SEYREKLİKTİ."""
        verdict = analyse(["tr", "cv", "sah"])
        self.assertFalse(verdict.is_scattered)
        self.assertIn("ayırt edilemez", verdict.explanation)

    def test_scattered_distribution_is_flagged(self):
        """Birden çok kolda var ama her kolda tek tük dilde — miras bir kök
        girdiği kolun içinde yayılır."""
        verdict = analyse(["tr", "kk", "uz", "sah", "cv", "khk", "ky", "tyv"])
        self.assertTrue(verdict.is_scattered)
        self.assertGreaterEqual(len(verdict.branches_present), 2)

    def test_dense_distribution_is_not_flagged(self):
        """Kol içinde yaygın dağılım miras dağılımıyla uyumludur."""
        dense = ["tr", "az", "tk", "gag", "ota", "kk", "ky", "tt", "ba", "kaa",
                 "nog", "kum", "krc", "crh", "uz", "ug"]
        verdict = analyse(dense)
        self.assertFalse(verdict.is_scattered)

    def test_unknown_language_codes_are_ignored(self):
        verdict = analyse(["xx", "yy", "zz"])
        self.assertEqual(verdict.branches_present, ())

    def test_empty_input(self):
        self.assertFalse(analyse([]).is_scattered)

    def test_absent_branches_are_reported(self):
        verdict = analyse(["tr", "az", "tk", "gag", "ota", "kk", "ky", "tt"])
        self.assertTrue(verdict.branches_absent)


class TestNotWiredIntoDecisions(unittest.TestCase):
    """⚠️ Doğrulanamayan bir sinyal karar katmanına bağlanamaz."""

    def test_signal_flag_is_off(self):
        self.assertFalse(USE_AS_SIGNAL)

    def test_detector_does_not_use_it(self):
        import inspect

        from engine.nlp import borrowing_detector

        self.assertNotIn("tree_incongruence", inspect.getsource(borrowing_detector))

    def test_minimum_witness_threshold_is_meaningful(self):
        self.assertGreaterEqual(MIN_LANGUAGES_FOR_DISTRIBUTION, 5)


if __name__ == "__main__":
    unittest.main()
