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

    def by_sense(self, sense, languages=None, limit=200, per_language=False):
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


class TestArabicViaPersian(unittest.TestCase):
    """9d: Farsça kazanan aday Farsçadaki Arapça alıntıysa etiket Arapça.

    Ölçüldü (ön kayıt ``data/cache/work/donor9d/PREREG.md``): Türkçe DEV
    etiket doğruluğu 0,484 -> 0,641 (D1, McNemar 10/0, Holm p=0,004).
    """

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, comparison, rule="d1", loans=frozenset()):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "ARABIC_VIA_RULE", rule), \
                mock.patch.object(dp, "persian_arabic_loans", lambda: loans), \
                mock.patch.object(dp, "_null_distance", lambda length, pool: 0.5):
            return dp.attribute_donor(comparison, "invitation", languages=["ar", "fa", "fr"])

    def test_same_script_skeleton_relabels_to_arabic(self):
        rows = [_row("ar", "دعوة", "dava"), _row("fa", "دعوت", "davat")]
        off = self._run(rows, "davet", rule="off")
        self.assertEqual(off.lang_code, "fa")
        on = self._run(rows, "davet")
        self.assertEqual((on.lang_code, on.via, on.word), ("ar", "fa", "دعوة"))
        self.assertIn("Farsça aracılığıyla", on.describe())
        self.assertEqual(on.as_dict()["donor_via"], "fa")
        self.assertNotIn("ar", [alt[0] for alt in on.alternatives])

    def test_etymology_mark_relabels_without_skeleton_match(self):
        rows = [_row("ar", "كلمة", "kalima"), _row("fa", "دعوت", "davat", gloss="invitation")]
        on = self._run(rows, "davet", loans=frozenset({("دعوت", "invitation")}))
        self.assertEqual((on.lang_code, on.via), ("ar", "fa"))

    def test_native_persian_stays_persian(self):
        rows = [_row("ar", "رخيص", "rahis"), _row("fa", "ارزان", "arzan")]
        self.assertEqual(self._run(rows, "arzan").lang_code, "fa")

    def test_d1_leaves_non_persian_winner(self):
        rows = [_row("ar", "دعوة", "dava"), _row("fr", "davet", "davet")]
        chosen = self._run(rows, "davet")
        self.assertEqual((chosen.lang_code, chosen.via), ("fr", ""))

    def test_skeletons(self):
        self.assertEqual(dp.script_skeleton("عَظَمَة"), dp.script_skeleton("عظمت"))
        self.assertIn(dp.consonant_skeleton("gadab"), dp._query_skeletons("gazap"))
        self.assertIn(dp.consonant_skeleton("dava"), dp._query_skeletons("davet"))


class TestFrenchVia(unittest.TestCase):
    """9e F2: Farsça/Ermenice/Yunanca kazanan biçim aslında o dilin Fransızca
    alıntısıysa (pozitron, metro) etiket Fransızca, ``via`` = kazanan dil."""

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, comparison, rule="f2", loans=frozenset()):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "FRENCH_RULE", rule), \
                mock.patch.object(dp, "dump_loans", lambda lang, phrase: loans), \
                mock.patch.object(dp, "_null_distance",
                                  lambda length, pool: 0.5 if "pozitron" in pool else 0.0):
            return dp.attribute_donor(comparison, "positron", languages=["ar", "fa", "hy", "el", "fr", "it"])

    def test_near_french_form_relabels(self):
        rows = [_row("fa", "پوزیترون", "pozitron"), _row("fr", "positron", "positron")]
        self.assertEqual(self._run(rows, "pozitron", rule="f1").lang_code, "fa")
        on = self._run(rows, "pozitron")
        self.assertEqual((on.lang_code, on.via, on.word), ("fr", "fa", "positron"))
        self.assertIn("aracılığıyla", on.describe())

    def test_etymology_mark_relabels(self):
        rows = [_row("hy", "մետրո", "pozitron", gloss="positron"), _row("fr", "électron", "elektron")]
        on = self._run(rows, "pozitron", loans=frozenset({("մետրո", "positron")}))
        self.assertEqual((on.lang_code, on.via), ("fr", "hy"))

    def test_unrelated_winner_stays(self):
        rows = [_row("fa", "پوزیترون", "pozitron"), _row("fr", "gare", "gar")]
        self.assertEqual(self._run(rows, "pozitron").lang_code, "fa")

    def test_italian_winner_is_not_touched(self):
        rows = [_row("it", "positrone", "pozitron"), _row("fr", "gare", "gar")]
        on = self._run(rows, "pozitron")
        self.assertEqual((on.lang_code, on.via), ("it", ""))


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
