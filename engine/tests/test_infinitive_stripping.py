"""Mastar eki soyma: sorgu (`gülmek` → `gül`) ve akraba tanıkları (`күлүү` → `kül`)."""
from __future__ import annotations

import unittest

from engine.nlp.historical_morphology import INFINITIVE_ENDINGS, infinitive_stem, strip_infinitive
from engine.tests.data_guards import needs_index


class TestStripInfinitive(unittest.TestCase):
    def test_language_specific_endings(self):
        cases = [
            ("az", "gülmək", "gül"),
            ("tk", "gülmek", "gül"),
            ("uz", "kulmoq", "kul"),
            ("gag", "külmää", "kül"),
            ("kk", "күлу", "kül"),
            ("ky", "күлүү", "kül"),
            ("ba", "көлөү", "köl"),
            ("tt", "күлергә", "kül"),
            ("alt", "каткырар", "katkır"),
        ]
        for code, form, expected in cases:
            with self.subTest(code=code, form=form):
                self.assertEqual(strip_infinitive(code, form), expected)

    def test_chuvash_has_no_rule(self):
        # Ölçümde zararlı çıktı: -ма/-ме kök sonu da olabiliyor.
        self.assertNotIn("cv", INFINITIVE_ENDINGS)
        self.assertEqual(strip_infinitive("cv", "кулма"), "kulma")

    def test_unknown_language_untouched(self):
        self.assertEqual(strip_infinitive("xx", "gülmek"), "gülmek")

    def test_too_short_remainder_untouched(self):
        # `emek` -> `e` olmaz; kalan en az iki harf ve bir ünlü ister.
        self.assertEqual(strip_infinitive("tr", "emek"), "emek")
        self.assertEqual(strip_infinitive("ky", "суу"), "suu")

    def test_bare_stem_untouched(self):
        self.assertEqual(strip_infinitive("az", "gül"), "gül")


class TestInfinitiveStemShape(unittest.TestCase):
    def test_non_infinitive_is_none(self):
        self.assertIsNone(infinitive_stem("kitap"))
        self.assertIsNone(infinitive_stem("mak"))


@needs_index
class TestInfinitiveStemIndex(unittest.TestCase):
    def test_verbs_are_stripped(self):
        self.assertEqual(infinitive_stem("gülmek"), "gül")
        self.assertEqual(infinitive_stem("kolaylaşmak"), "kolaylaş")

    def test_nouns_ending_in_mak_are_not(self):
        for noun in ("parmak", "ırmak", "damak", "emek"):
            with self.subTest(noun=noun):
                self.assertIsNone(infinitive_stem(noun))


if __name__ == "__main__":
    unittest.main()
