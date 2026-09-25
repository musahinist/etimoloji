"""
Alıntı sinyallerinin ölçülmüş hataları — her test bir düzeltmeyi korur.

1. Arama yolu anlam geçmiyordu: verici yakınlığı üretimde hiç ateşlenmiyordu.
2. ``ses_kanunu_ihlali`` Türkçe dışı dilde Türkçe refleksi bekliyordu.
3. ``fonotaktik_ihlal`` Saha'da düzenli söz başı h-'yi ihlal sayıyordu;
   birleşik sözü tek kelime sayıp ünlü uyumu ihlali buluyordu.
4. ``zincir_kanıtı`` fiil kökünün eşsesli alıntı ad kaydına takılıyordu (G6).
"""

from __future__ import annotations

import unittest
from unittest import mock

from engine.nlp import borrowing_detector as bd


class _Prediction:
    def __init__(self, form: str, confidence: float = 0.9):
        self.form = form
        self.confidence = confidence


class TestSoundLawTargetsQueryLanguage(unittest.TestCase):
    def _detector(self, predict):
        detector = bd.BorrowingDetector(index=mock.Mock(exists=False), predictor=mock.Mock())
        detector._inherited_predictor = mock.Mock(predict=predict)
        return detector

    def test_prediction_target_is_the_query_language(self):
        calls = []

        def predict(form, source, target):
            calls.append((source, target))
            return _Prediction("kar")

        detector = self._detector(predict)
        witnesses = {"kk": "қар", "tt": "кар", "ky": "кар", "sah": "χaːr"}
        detector._sound_law_signal("χaːr", witnesses, "sah")
        self.assertTrue(calls)
        self.assertTrue(all(target == "sah" for _, target in calls))
        self.assertNotIn("sah", [source for source, _ in calls])

    def test_language_without_tables_does_not_fire(self):
        detector = self._detector(lambda form, source, target: _Prediction(form, 0.0))
        signal, expected = detector._sound_law_signal(
            "χaːr", {"kk": "қар", "tt": "кар", "ky": "кар"}, "sah"
        )
        self.assertFalse(signal.fired)
        self.assertTrue(signal.evidence.get("no_data"))
        self.assertEqual(expected, "")


class TestLanguageSpecificPhonotactics(unittest.TestCase):
    def test_sakha_initial_h_is_regular(self):
        self.assertFalse(bd.BorrowingDetector._phonotactic_signal("χaːr", "sah").fired)
        self.assertFalse(bd.BorrowingDetector._phonotactic_signal("hurt", "ba").fired)

    def test_turkish_initial_h_still_fires(self):
        self.assertTrue(bd.BorrowingDetector._phonotactic_signal("hayvan", "tr").fired)

    def test_other_prohibited_initials_still_fire_in_sakha(self):
        self.assertTrue(bd.BorrowingDetector._phonotactic_signal("lampa", "sah").fired)

    def test_compound_harmony_is_checked_per_part(self):
        signal = bd.BorrowingDetector._phonotactic_signal("kün_ortoto", "sah")
        self.assertNotIn("ünlü uyumu ihlali", signal.evidence["violations"])
        signal = bd.BorrowingDetector._phonotactic_signal("temperatura", "sah")
        self.assertIn("ünlü uyumu ihlali", signal.evidence["violations"])


class _FakeIndex:
    exists = True

    def __init__(self, rows):
        self.rows = rows

    def lookup(self, form, *, languages=None, limit=50):
        from engine.utils.orthography import to_comparison_form

        key = to_comparison_form(form)
        return [
            dict(r) for r in self.rows
            if to_comparison_form(r["word"]) == key and r.get("lang_code", "tr") in (languages or ["tr"])
        ][:limit]


def _row(word, origin, donor="", form="", pos="noun"):
    return {"word": word, "origin": origin, "donor_lang": donor, "donor_form": form, "pos": pos}


class TestChainHomonyms(unittest.TestCase):
    def _chain(self, rows, word):
        detector = bd.BorrowingDetector(index=_FakeIndex(rows))
        signal, _, donor = detector._chain_signal(word, "tr")
        return signal, donor

    def test_infinitive_counts_as_inherited(self):
        """``duy`` < Fr. *douille* (ad) ama ``duymak`` < *tuy-."""
        rows = [
            _row("duy", "alıntı", "fr", "douille"),
            _row("duy", None, pos="verb"),
            _row("duymak", "miras", "trk-pro", "*tuy-", pos="verb"),
        ]
        signal, donor = self._chain(rows, "duy")
        self.assertFalse(signal.fired)
        self.assertEqual(donor, "")

    def test_repeated_loan_rows_count_once(self):
        """``sek``: sıfat + zarf aynı Fr. *sec* — 2 alıntı sayılınca eşik geçiliyordu."""
        rows = [
            _row("sek", "alıntı", "fr", "sec", pos="adj"),
            _row("sek", "alıntı", "fr", "sec", pos="adv"),
            _row("sekmek", "miras", "trk-pro", "*sēk-", pos="verb"),
        ]
        self.assertFalse(self._chain(rows, "sek")[0].fired)

    def test_suffix_and_different_spelling_are_not_evidence(self):
        """``kar`` (*kār) ≠ ``kâr`` (Pehl.) ve ``-kâr`` (Fa. eki)."""
        rows = [
            _row("kar", "miras", "trk-pro", "*kār"),
            _row("kâr", "alıntı", "pal", "kār"),
            _row("-kâr", "alıntı", "fa-cls", "کار", pos="suffix"),
        ]
        self.assertFalse(self._chain(rows, "kar")[0].fired)

    def test_plain_loan_still_fires(self):
        rows = [_row("kitap", "alıntı", "ar", "كتاب")]
        signal, donor = self._chain(rows, "kitap")
        self.assertTrue(signal.fired)
        self.assertEqual(donor, "ar")


class TestProductionPassesSense(unittest.TestCase):
    def _detect_kitap(self, flag):
        seen = {}

        def donor_signal(word, sense, donors):
            seen["sense"], seen["donors"] = sense, donors
            return bd.Signal("verici_yakınlığı", False, 0.0, "")

        detector = bd.BorrowingDetector(index=mock.Mock(exists=False), predictor=mock.Mock())
        with mock.patch.object(bd, "SEARCH_DONOR_PROXIMITY", flag), \
                mock.patch.object(bd, "own_sense", lambda word, lang: "book"), \
                mock.patch.object(bd.BorrowingDetector, "_donor_signal", staticmethod(donor_signal)), \
                mock.patch.object(bd.BorrowingDetector, "_sound_law_signal",
                                  lambda self, w, wit, lang="tr": (bd.Signal("ses_kanunu_ihlali", False, 0.0, ""), "")), \
                mock.patch.object(bd.BorrowingDetector, "_phonotactic_model_signal",
                                  staticmethod(lambda w, lang: bd.Signal("fonotaktik_model", False, 0.0, ""))):
            detector.detect("kitap")
        return seen

    def test_search_path_sense_is_off_by_default(self):
        """Arama yolunda anlam doldurulmaz: 150 kelimede uyum 110 vs 100 (c0e8dd7)."""
        seen = self._detect_kitap(False)
        self.assertEqual(seen["sense"], "")
        self.assertEqual(seen["donors"], ["ar", "fa", "el", "hy", "fr", "it"])

    def test_missing_sense_is_read_from_the_index_when_enabled(self):
        seen = self._detect_kitap(True)
        self.assertEqual(seen["sense"], "book")

    def test_explicit_sense_is_kept(self):
        seen = {}

        def donor_signal(word, sense, donors):
            seen["sense"] = sense
            return bd.Signal("verici_yakınlığı", False, 0.0, "")

        detector = bd.BorrowingDetector(index=mock.Mock(exists=False), predictor=mock.Mock())
        with mock.patch.object(bd, "own_sense", lambda word, lang: "WRONG"), \
                mock.patch.object(bd.BorrowingDetector, "_donor_signal", staticmethod(donor_signal)), \
                mock.patch.object(bd.BorrowingDetector, "_sound_law_signal",
                                  lambda self, w, wit, lang="tr": (bd.Signal("ses_kanunu_ihlali", False, 0.0, ""), "")), \
                mock.patch.object(bd.BorrowingDetector, "_phonotactic_model_signal",
                                  staticmethod(lambda w, lang: bd.Signal("fonotaktik_model", False, 0.0, ""))):
            detector.detect("χaːr", lang="sah", sense="SNOW", donors=["ru"])
        self.assertEqual(seen["sense"], "SNOW")


if __name__ == "__main__":
    unittest.main()
