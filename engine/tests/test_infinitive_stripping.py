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

    def test_homograph_follows_query_meaning(self):
        # `kaymak` hem "krema" hem "kaygan yüzeyde gitmek".
        self.assertIsNone(infinitive_stem("kaymak", "kaymak, a creamy dairy product"))
        self.assertIsNone(infinitive_stem("ekmek", "bread, a foodstuff prepared from a dough"))
        self.assertEqual(infinitive_stem("kaymak", "to slide, to slip"), "kay")
        self.assertEqual(infinitive_stem("kaymak", "Kaygan bir yüzeyde sürtünerek gitmek"), "kay")
        # Anlam yoksa fiil okuması.
        self.assertEqual(infinitive_stem("kaymak"), "kay")

    def test_meaning_does_not_block_plain_verbs(self):
        # Yalnız fiil olan kelimede anlam denetimi uygulanmaz.
        self.assertEqual(infinitive_stem("gülmek", "laughter"), "gül")

    def test_nouns_ending_in_mak_are_not(self):
        for noun in ("parmak", "ırmak", "damak", "emek"):
            with self.subTest(noun=noun):
                self.assertIsNone(infinitive_stem(noun))



class TestBorrowingWord(unittest.TestCase):
    def test_borrowing_check_uses_full_infinitive(self):
        # Çıplak gövde (`yak`) eşsesli alıntı ada takılmasın: denetim `yakmak`la.
        from unittest import mock

        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        engine = ComparativeReconstructor()
        with mock.patch.object(ComparativeReconstructor, "_borrowing_verdict", return_value=None) as verdict:
            engine.reconstruct("yak", [], borrowing_word="yakmak")
            self.assertEqual(verdict.call_args.args[0], "yakmak")
            engine.reconstruct("yak", [])
            self.assertEqual(verdict.call_args.args[0], "yak")


if __name__ == "__main__":
    unittest.main()
