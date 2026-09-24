"""
Verici dil etiketi testleri — "alıntı mı?"dan ayrı "kimden?" adımı.

Ölçüldü (``make eval-donor``, WOLD Saha, n=440): tek havuzdaki en yakın
maddenin dili Moğolca alıntıların 79/166'sını Rusça etiketliyordu. Dil başına
en yakın madde + dilin null'ı + Starling ``monget`` doğruluğu 0,652'den
0,714'e çıkardı. Testler üç şeyi korur:

1. Etiket sinyal GÜCÜNE girmez (``monget`` güce katılınca WOLD F düşüyordu).
2. Moğolca adaylar Starling ``monget``ten gelir, kaikki'den değil.
3. Seçim ham mesafeyle değil, dilin kendi null'ına göre yapılır.
"""

from __future__ import annotations

import unittest
from unittest import mock

from engine.db import starling
from engine.nlp import borrowing_detector as bd
from engine.nlp import donor_proximity as dp


class TestMongolicFormParsing(unittest.TestCase):
    def test_references_and_glosses_are_dropped(self):
        self.assertEqual(
            starling._mongolic_forms("dölü (L 272), döl 'flame' [dial.]"), ["dölü", "döl"]
        )

    def test_starling_j_is_aligned_with_sakha_y(self):
        self.assertEqual(starling._mongolic_forms("jeke"), ["yeke"])

    def test_empty_field(self):
        self.assertEqual(starling._mongolic_forms(""), [])

    def test_missing_tables_give_nothing(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        starling.load_monget.cache_clear()
        with TemporaryDirectory() as tmp:
            self.assertEqual(starling.load_monget(Path(tmp)), ())
        starling.load_monget.cache_clear()


class TestAttributionControls(unittest.TestCase):
    def test_every_length_has_enough_controls(self):
        """⚠️ Şans denetiminin kontrolleri 6 harften uzun sorguda yetersizdi;
        etiket null'ı her uzunlukta kurulabilmeli."""
        for length in range(2, 13):
            with self.subTest(length=length):
                controls = dp._attribution_controls(length)
                self.assertGreaterEqual(len(controls), 8)
                self.assertTrue(all(len(c) == length for c in controls))


class _FakeIndex:
    exists = True

    def __init__(self, rows):
        self.rows = rows

    def by_sense(self, sense, languages=None, limit=200):
        return [r for r in self.rows if languages is None or r["lang_code"] in languages]


def _row(lang, word, comparison, gloss="x"):
    return {"lang_code": lang, "word": word, "comparison": comparison, "gloss": gloss}


class TestAttributeDonor(unittest.TestCase):
    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, monget, nulls, comparison="moğoy", languages=("ru", "mn", "evn")):
        entries = tuple((c, w, "snake", frozenset({"snake"})) for c, w in monget)
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: entries), \
                mock.patch.object(dp, "_null_distance", lambda length, pool: nulls.get(pool[0][:2], 0.0)):
            return dp.attribute_donor(comparison, "snake", languages=list(languages))

    def test_monget_replaces_kaikki_mongolian(self):
        rows = [_row("ru", "змея", "zmeya"), _row("mn", "хорхой", "horhoy")]
        result = self._run(rows, [("moğay", "moɣai")], {})
        self.assertEqual(result.lang_code, "mn")
        self.assertEqual(result.word, "moɣai")
        self.assertEqual(result.source, "starling-monget")

    def test_selection_uses_the_language_null(self):
        """Ham mesafede Rusça daha yakın (0,10 < 0,20), ama rastgele Türkçe
        kelimeler de Rusça havuza 0,15'e düşüyor, Moğolca havuza ise 0,50'ye:
        Rusça eşleşme şanstan pek ayrılmıyor, etiket Moğolca olmalı."""
        rows = [_row("ru", "могай", "mogay")]
        entries = (("moğay", "moɣai", "snake", frozenset({"snake"})),)
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: entries), \
                mock.patch.object(dp, "best_sca",
                                  lambda q, c: (0.1, c[0]) if c[0] == "mogay" else (0.2, c[0])), \
                mock.patch.object(dp, "_null_distance",
                                  lambda length, pool: 0.15 if pool == ("mogay",) else 0.5):
            chosen = dp.attribute_donor("moğoy", "snake", languages=["ru", "mn"])
        self.assertEqual(chosen.lang_code, "mn")
        self.assertEqual([alt[0] for alt in chosen.alternatives], ["ru"])

    def test_far_label_is_marked_uncertain(self):
        rows = [_row("ru", "удав", "udav")]
        result = self._run(rows, [], {}, comparison="moğoy", languages=("ru",))
        self.assertTrue(result.uncertain)
        self.assertIn("verici belirsiz", result.describe())
        self.assertTrue(result.as_dict()["donor_uncertain"])

    def test_no_candidates_is_none(self):
        self.assertIsNone(self._run([], [], {}))


class TestSignalStrengthIsUntouched(unittest.TestCase):
    """⚠️ Etiket güce girerse WOLD "alıntı mı?" F'si değişir."""

    def test_attribution_changes_label_not_strength(self):
        match = dp.DonorMatch("ru", "змея", "zmeya", "snake", 0.2, True, 0.0)
        label = dp.DonorAttribution("mn", "moɣai", "moğay", "snake", 0.05, 0.6, "starling-monget")
        with mock.patch.object(bd, "nearest_donor", lambda *a, **k: match):
            with mock.patch.object(bd, "attribute_donor", lambda *a, **k: None):
                plain = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru", "mn"])
            with mock.patch.object(bd, "attribute_donor", lambda *a, **k: label):
                labelled = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru", "mn"])
        self.assertEqual(plain.strength, labelled.strength)
        self.assertEqual(plain.fired, labelled.fired)
        self.assertEqual(labelled.evidence["donor_lang"], "ru")  # en yakın madde aynı kalır
        self.assertEqual(labelled.evidence["attributed_lang"], "mn")
        self.assertIn("verici etiketi", labelled.explanation)

    def test_no_label_when_signal_does_not_fire(self):
        far = dp.DonorMatch("ru", "змея", "zmeya", "snake", 0.9, True, None)
        called = mock.Mock()
        with mock.patch.object(bd, "nearest_donor", lambda *a, **k: far), \
                mock.patch.object(bd, "attribute_donor", called):
            signal = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru"])
        self.assertFalse(signal.fired)
        called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
