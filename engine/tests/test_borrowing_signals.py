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
