"""
Eğitilmiş alıntı birleştiricisi testleri.

⚠️ Bu modül bir ölçümün sonucudur: el ile konmuş ağırlıklı toplam
WOLD/Sakha'da en güçlü sinyalin kararını **bozuyordu** (madde başına
doğruluk 0,7035 vs yalnız verici yakınlığı 0,7334; fark -0,030,
%95 GA [-0,049, -0,010], p=0,004). Testler öğrenilen modelin o hatayı
tekrar etmemesini ve **künyesiz** kullanılmamasını korur.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.nlp.borrowing_combiner import (
    SIGNAL_ORDER,
    BorrowingCombiner,
    fit,
    load,
    save,
)


def _samples(n: int = 200) -> list[tuple[dict[str, float], bool]]:
    """``verici_yakınlığı`` tek başına belirleyici, ötekiler gürültü."""
    out: list[tuple[dict[str, float], bool]] = []
    for i in range(n):
        borrowed = i % 2 == 0
        out.append(
            (
                {
                    "verici_yakınlığı": 1.0 if borrowed else 0.0,
                    "değişimsiz_yayılım": 1.0 if i % 3 == 0 else 0.0,
                    "fonotaktik_ihlal": 0.5,
                },
                borrowed,
            )
        )
    return out


class TestFitting(unittest.TestCase):
    def setUp(self):
        self.model = fit(_samples(), trained_on="test/tune")

    def test_informative_signal_gets_the_weight(self):
        weights = self.model.weights
        self.assertGreater(weights["verici_yakınlığı"], 1.0)

    def test_noise_signal_is_learned_away(self):
        """El ağırlıklarında ``değişimsiz_yayılım`` toplamın %5-10'unu
        alıyordu; ablasyonda katkısı **negatifti**."""
        self.assertLess(abs(self.model.weights["değişimsiz_yayılım"]), 0.5)

    def test_training_is_deterministic(self):
        """Rastgele başlangıç yok: ölçümün tekrarlanabilirliği bunu ister."""
        again = fit(_samples(), trained_on="test/tune")
        self.assertEqual(self.model.weights, again.weights)

    def test_empty_training_set_raises(self):
        with self.assertRaises(ValueError):
            fit([], trained_on="x")

    def test_untrained_model_reports_itself(self):
        blank = BorrowingCombiner()
        self.assertFalse(blank.is_trained)
        self.assertEqual(blank.probability({"verici_yakınlığı": 1.0}), 0.0)
        self.assertIn("eğitilmemiş", blank.explain())

    def test_weights_are_readable(self):
        """Bu proje kara kutu kabul etmiyor: katsayılar raporlanabilmeli."""
        self.assertIn("verici_yakınlığı", self.model.explain())


class TestThresholdObjective(unittest.TestCase):
    """⚠️ Hedef ölçü seçimi sonucu belirler ve gizlenemez.

    Ölçüldü (WOLD/Sakha, aynı model, yalnız eşik farklı):
    F hedefli 0,5982/0,7100 · doğruluk hedefli 0,2714/0,7347.
    """

    def test_objective_is_recorded(self):
        model = fit(_samples(), trained_on="t", objective="accuracy")
        self.assertEqual(model.objective, "accuracy")
        self.assertEqual(model.as_dict()["objective"], "accuracy")

    def test_default_objective_is_fscore(self):
        self.assertEqual(fit(_samples(), trained_on="t").objective, "fscore")

    def test_threshold_is_not_blindly_half(self):
        """Sınıf dengesizliğinde 0,5 varsayılanı duyarlılığı bastırır."""
        skewed = [({"verici_yakınlığı": 1.0 if i < 20 else 0.0}, i < 20) for i in range(200)]
        model = fit(skewed, trained_on="t")
        self.assertNotEqual(model.threshold, 0.5)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "model.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        model = fit(_samples(), trained_on="test/tune")
        save(model, self.path)
        loaded = load(self.path)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.bias, model.bias, places=6)
        self.assertEqual(loaded.trained_on, "test/tune")

    def test_missing_file_returns_none(self):
        self.assertIsNone(load(self.path))

    def test_corrupt_file_is_reported_not_raised(self):
        self.path.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(load(self.path))

    def test_stale_signal_order_is_refused(self):
        """⚠️ Sinyal sırası değişmişse eski katsayıları yeni sıraya uygulamak
        her sinyale BAŞKASININ ağırlığını vermek olurdu — ve hiçbir hata
        mesajı üretmezdi."""
        model = fit(_samples(), trained_on="t")
        save(model, self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["signal_order"] = list(reversed(SIGNAL_ORDER))
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIsNone(load(self.path))


class TestDetectorIntegration(unittest.TestCase):
    def test_untrained_detector_declares_itself(self):
        """Kalibre edilmemiş bir skoru kalibreymiş gibi sunmak, hiç kalibre
        etmemekten kötüdür."""
        from unittest import mock

        from engine.nlp.borrowing_detector import BorrowingDetector

        BorrowingDetector.reset_combiner_cache()
        with mock.patch("engine.nlp.borrowing_combiner.load", lambda *a, **k: None):
            verdict = BorrowingDetector().detect("kitap")
        self.assertFalse(verdict.is_trained)
        self.assertIn("EĞİTİLMEMİŞ", verdict.combiner_note)
        BorrowingDetector.reset_combiner_cache()

    def test_verdict_and_is_borrowed_agree(self):
        """⚠️ İkisi ayrı hesaplanıyordu; eğitilmiş model devreye girince
        ``is_borrowed=True`` ile ``verdict="miras adayı"`` aynı anda
        çıkabilirdi."""
        from engine.nlp.borrowing_detector import BorrowingVerdict

        for probability in (0.05, 0.2, 0.4, 0.9):
            with self.subTest(p=probability):
                verdict = BorrowingVerdict(word="x", score=0.0)
                verdict.trained_probability = probability
                verdict._trained_threshold = 0.33
                self.assertEqual(verdict.is_borrowed, verdict.verdict == "alıntı")


if __name__ == "__main__":
    unittest.main()
