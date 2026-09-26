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
    fit_nested,
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


def _noisy_samples(n: int = 160) -> list[tuple[dict[str, float], bool]]:
    """``verici_yakınlığı`` bilgilendirici; ``fonotaktik_ihlal`` saf gürültü
    ama değişken — sabit gürültü L1'in sınaması için fazla kolay."""
    out: list[tuple[dict[str, float], bool]] = []
    for i in range(n):
        borrowed = i % 3 == 0
        out.append(
            (
                {
                    "verici_yakınlığı": (0.9 if borrowed else 0.1) if i % 7 else 0.5,
                    "fonotaktik_ihlal": 1.0 if (i * 7919) % 11 < 5 else 0.0,
                },
                borrowed,
            )
        )
    return out


class TestL1(unittest.TestCase):
    """L1 katkısız sinyalin katsayısını **tam sıfıra** indirir; L2 yalnız
    küçültür. WOLD/Sakha'da ``fonotaktik_ihlal``'ı çıkarmak F'yi artırıyordu."""

    def test_l1_zeroes_the_noise_signal(self):
        model = fit(_noisy_samples(), trained_on="t", l1=0.05)
        self.assertEqual(model.weights["fonotaktik_ihlal"], 0.0)
        self.assertGreater(model.weights["verici_yakınlığı"], 0.0)

    def test_inactive_signal_stays_zero(self):
        model = fit(_samples(), trained_on="t", active=("verici_yakınlığı",))
        self.assertEqual(model.weights["değişimsiz_yayılım"], 0.0)
        self.assertEqual(model.active, ("verici_yakınlığı",))

    def test_unknown_signal_is_refused(self):
        with self.assertRaises(ValueError):
            fit(_samples(), trained_on="t", active=("yok_böyle",))

    def test_given_threshold_is_kept(self):
        """İç ÇD'de seçilen eşik eğitim verisinde yeniden aranmamalı."""
        self.assertEqual(fit(_samples(), trained_on="t", threshold=0.42).threshold, 0.42)


#: Testte küçük ızgara ve az yineleme: yöntem sınanıyor, sayı değil.
_FAST = {"trained_on": "test/tune", "iterations": 400, "l1_grid": (0.0, 0.03), "folds": 3}


class TestNestedCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_nested(_noisy_samples(), **_FAST)

    def test_noise_signal_is_eliminated(self):
        """Daha az sinyalle aynı skor bir kazançtır."""
        self.assertNotIn("fonotaktik_ihlal", self.model.active)
        self.assertIn("verici_yakınlığı", self.model.active)

    def test_never_fired_signal_is_not_a_candidate(self):
        """Türkçede zincir kapalı: hiç ateşlenmeyen sinyal seçime girmez."""
        self.assertNotIn("zincir_kanıtı", self.model.selection["trail"][0]["signals"])

    def test_selection_is_recorded(self):
        record = self.model.as_dict()
        self.assertIn("trail", record["selection"])
        self.assertIn("l1", record)
        self.assertIn("active_signals", record)

    def test_folds_are_stratified(self):
        """⚠️ ``sıra mod kat`` ataması, her üçüncü madde alıntıyken 3 katta
        bir katı yalnız alıntılarla doldurup iç ÇD skorunu 0'a düşürüyordu."""
        from engine.nlp.borrowing_combiner import _stratified_folds

        samples = _noisy_samples(30)
        assignment = _stratified_folds(samples, 3)
        for fold in range(3):
            labels = {label for (_, label), f in zip(samples, assignment, strict=True) if f == fold}
            self.assertEqual(labels, {True, False})
        self.assertGreater(self.model.selection["cv_score"], 0.5)

    def test_nested_is_deterministic(self):
        again = fit_nested(_noisy_samples(), **_FAST)
        self.assertEqual(self.model.weights, again.weights)
        self.assertEqual(self.model.threshold, again.threshold)


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

    def test_round_trip_keeps_selection(self):
        model = fit(_samples(), trained_on="t", l1=0.01, active=("verici_yakınlığı",))
        save(model, self.path)
        loaded = load(self.path)
        self.assertEqual(loaded.active, ("verici_yakınlığı",))
        self.assertEqual(loaded.l1, 0.01)

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


    def test_old_model_without_appended_signals_still_loads(self):
        """Sona eklenen sinyal eski modeli bozmaz: katsayısı 0 sayılır."""
        model = fit(_samples(), trained_on="t")
        save(model, self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["signal_order"] = list(SIGNAL_ORDER[:-1])
        data["weights"].pop(SIGNAL_ORDER[-1], None)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load(self.path)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(
            loaded.probability({SIGNAL_ORDER[-1]: 1.0}), loaded.probability({})
        )


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


class TestWoldDonorField(unittest.TestCase):
    """⚠️ Verici ``Borrowed_base``'den okunuyordu; o alan Sakha'da hep boş.
    Doğru alan ``borrowings.csv`` / ``Source_languoid``."""

    def test_borrowed_sakha_cases_carry_a_donor(self):
        from engine.evaluation.borrowing_eval import load_wold_cases

        cases = load_wold_cases(with_witnesses=False)
        if not cases:
            self.skipTest("WOLD indirilmemiş")
        borrowed = [c for c in cases if c.is_borrowed]
        with_donor = [c for c in borrowed if c.donor]
        self.assertGreater(len(with_donor), 0.9 * len(borrowed))
        self.assertIn("Russian", {c.donor for c in with_donor})
        self.assertFalse(any(c.donor for c in cases if not c.is_borrowed))


if __name__ == "__main__":
    unittest.main()


class HardNegativeTests(unittest.TestCase):
    """D8 M9: zor negatif ağırlığı — 1,0 iken eğitim ağırlıksızla BİREBİR aynı."""

    def test_rule(self) -> None:
        from engine.nlp.borrowing_combiner import is_hard_negative

        ramp = {"verici_yakınlığı": 0.4}
        self.assertTrue(is_hard_negative(ramp, False, "ramp"))
        self.assertFalse(is_hard_negative(ramp, True, "ramp"))
        self.assertFalse(is_hard_negative({"verici_yakınlığı": 1.0}, False, "ramp"))
        phon = {"fonotaktik_ihlal": 1.0}
        self.assertFalse(is_hard_negative(phon, False, "ramp"))
        self.assertTrue(is_hard_negative(phon, False, "ramp_phon"))

    def test_weight_one_is_identity(self) -> None:
        samples = _samples(60)
        plain = fit(samples, trained_on="t", hard_negative=(1.0, "ramp", False), iterations=300)
        again = fit(samples, trained_on="t", hard_negative=(1.0, "ramp_phon", True), iterations=300)
        self.assertEqual(plain.as_dict()["weights"], again.as_dict()["weights"])
        self.assertEqual(plain.threshold, again.threshold)

    def test_weight_lowers_ramp_probability(self) -> None:
        samples = []
        for i in range(120):
            borrowed = i % 3 == 0
            ramp = 0.5 if i % 2 else 0.0
            samples.append(({"verici_yakınlığı": 1.0 if borrowed else ramp}, borrowed))
        plain = fit(samples, trained_on="t", hard_negative=(1.0, "ramp", False), iterations=500)
        heavy = fit(samples, trained_on="t", hard_negative=(5.0, "ramp", False), iterations=500)
        probe = {"verici_yakınlığı": 0.5}
        self.assertLess(heavy.probability(probe), plain.probability(probe))
        self.assertEqual(heavy.selection["hard_negative"]["weight"], 5.0)
