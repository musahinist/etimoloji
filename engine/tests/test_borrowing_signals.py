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


class TestProductionPassesSense(unittest.TestCase):
    def test_missing_sense_is_read_from_the_index(self):
        seen = {}

        def donor_signal(word, sense, donors):
            seen["sense"], seen["donors"] = sense, donors
            return bd.Signal("verici_yakınlığı", False, 0.0, "")

        detector = bd.BorrowingDetector(index=mock.Mock(exists=False), predictor=mock.Mock())
        with mock.patch.object(bd, "own_sense", lambda word, lang: "book"), \
                mock.patch.object(bd.BorrowingDetector, "_donor_signal", staticmethod(donor_signal)), \
                mock.patch.object(bd.BorrowingDetector, "_sound_law_signal",
                                  lambda self, w, wit, lang="tr": (bd.Signal("ses_kanunu_ihlali", False, 0.0, ""), "")), \
                mock.patch.object(bd.BorrowingDetector, "_phonotactic_model_signal",
                                  staticmethod(lambda w, lang: bd.Signal("fonotaktik_model", False, 0.0, ""))):
            detector.detect("kitap")
        self.assertEqual(seen["sense"], "book")
        self.assertEqual(seen["donors"], ["ar", "fa", "el", "hy", "fr", "it"])

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
