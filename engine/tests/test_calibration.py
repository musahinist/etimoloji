"""
Kalibrasyon ve seçici tahmin testleri.

Kalibre edilmemiş bir güven skoru kullanıcıyı yanıltır: ölçüldü, motor
ortalama 0,646 güven verirken gerçek doğruluğu 0,239'du. Bu testler o
düzeltmenin bozulmamasını korur.
"""

from __future__ import annotations

import json
import unittest

from engine.evaluation.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    auc_score,
    bootstrap_ci,
    brier_score,
    cross_validated_calibration,
    evaluate,
    expected_calibration_error,
    risk_coverage,
)
from engine.nlp.confidence import (
    BADGE_THRESHOLDS,
    CalibrationModel,
    apply_calibration,
    badge_for,
)


def _overconfident(n: int = 200) -> tuple[list[float], list[bool]]:
    """%90 güven veren ama %30 haklı çıkan bir sistem."""
    scores = [0.9] * n
    correct = [i % 10 < 3 for i in range(n)]
    return scores, correct


def _well_calibrated(n: int = 200) -> tuple[list[float], list[bool]]:
    scores: list[float] = []
    correct: list[bool] = []
    for i in range(n):
        p = (i % 10) / 10 + 0.05
        scores.append(p)
        correct.append((i // 10) % 10 < p * 10)
    return scores, correct


class TestCalibrationMetrics(unittest.TestCase):
    def test_perfect_calibration_has_zero_ece(self):
        scores = [1.0] * 50 + [0.0] * 50
        correct = [True] * 50 + [False] * 50
        ece, _ = expected_calibration_error(scores, correct)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_overconfidence_is_detected(self):
        scores, correct = _overconfident()
        report = evaluate(scores, correct, with_ci=False)
        self.assertGreater(report.overconfidence, 0.5)
        self.assertGreater(report.ece, 0.5)

    def test_brier_rewards_confident_correctness(self):
        good = brier_score([0.9, 0.9], [True, True])
        bad = brier_score([0.9, 0.9], [False, False])
        self.assertLess(good, bad)

    def test_auc_is_half_for_constant_score(self):
        """Sabit skor hiçbir şey ayırt etmez."""
        self.assertEqual(auc_score([0.5] * 10, [True] * 5 + [False] * 5), 0.5)

    def test_auc_is_one_for_perfect_ranking(self):
        self.assertEqual(auc_score([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)

    def test_auc_handles_single_class(self):
        self.assertEqual(auc_score([0.9, 0.8], [True, True]), 0.5)

    def test_bootstrap_ci_brackets_the_estimate(self):
        scores, correct = _overconfident()
        point, _ = expected_calibration_error(scores, correct)
        low, high = bootstrap_ci(
            scores, correct, lambda c, y: expected_calibration_error(c, y)[0], iterations=200
        )
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_ece_is_reported_with_interval_when_sample_is_large_enough(self):
        """n≈100'de ECE kutu tahminleri oynaktır; aralıksız rapor edilmez."""
        scores, correct = _overconfident(60)
        self.assertIsNotNone(evaluate(scores, correct).ece_ci)
        self.assertIsNone(evaluate(scores[:10], correct[:10]).ece_ci)


class TestCalibrators(unittest.TestCase):
    def test_platt_reduces_ece_on_overconfident_scores(self):
        scores, correct = _overconfident()
        before, _ = expected_calibration_error(scores, correct)
        after, _ = expected_calibration_error(
            cross_validated_calibration(scores, correct, method="platt"), correct
        )
        self.assertLess(after, before / 2)

    def test_isotonic_reduces_ece_too(self):
        scores, correct = _overconfident()
        before, _ = expected_calibration_error(scores, correct)
        after, _ = expected_calibration_error(
            cross_validated_calibration(scores, correct, method="isotonic"), correct
        )
        self.assertLess(after, before / 2)

    def test_isotonic_is_monotone(self):
        cal = IsotonicCalibrator().fit(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [False, False, True, False, True, True]
        )
        outputs = [cal.predict(x / 10) for x in range(1, 7)]
        self.assertEqual(outputs, sorted(outputs))

    def test_platt_output_is_bounded(self):
        cal = PlattCalibrator()
        for x in (-100.0, 0.0, 0.5, 100.0):
            with self.subTest(x=x):
                self.assertGreaterEqual(cal.predict(x), 0.0)
                self.assertLessEqual(cal.predict(x), 1.0)

    def test_cross_validation_does_not_train_on_the_item_it_scores(self):
        """Aynı veride hem kalibre edip hem ölçmek ECE'yi yapay sıfırlar."""
        scores = [0.9] * 100
        correct = [i < 50 for i in range(100)]
        calibrated = cross_validated_calibration(scores, correct)
        self.assertEqual(len(calibrated), 100)
        # Hepsi aynı ham skordan geldiği için kalibre skorlar da yakın olmalı
        self.assertLess(max(calibrated) - min(calibrated), 0.35)

    def test_tiny_sample_falls_back_to_raw(self):
        raw = [0.5, 0.6]
        self.assertEqual(cross_validated_calibration(raw, [True, False]), raw)


class TestRiskCoverage(unittest.TestCase):
    def test_risk_falls_as_coverage_shrinks_for_a_good_score(self):
        scores = [i / 100 for i in range(100)]
        correct = [i >= 50 for i in range(100)]
        curve = risk_coverage(scores, correct, steps=4)
        self.assertLess(curve[0]["risk"], curve[-1]["risk"])

    def test_coverage_reaches_one(self):
        curve = risk_coverage([0.5] * 10, [True] * 10, steps=5)
        self.assertEqual(curve[-1]["coverage"], 1.0)


class TestBadgesAndAbstention(unittest.TestCase):
    def test_badge_thresholds_are_descending(self):
        values = [t for t, _ in BADGE_THRESHOLDS]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_badge_for_none_is_insufficient(self):
        self.assertIn("YETERSİZ", badge_for(None))

    def test_high_score_gets_strong_badge(self):
        self.assertIn("GÜÇLÜ", badge_for(0.9))

    def test_low_calibrated_score_triggers_abstention(self):
        result = {"confidence": 0.05, "is_reconstructible": True, "reconstructed_root": "*x"}
        out = apply_calibration(result, abstention_threshold=0.15)
        self.assertTrue(out["abstained"])
        self.assertFalse(out["is_reconstructible"])
        self.assertEqual(out["reconstructed_root"], "")
        self.assertEqual(out["withheld_reconstruction"], "*x")

    def test_withheld_form_is_preserved_for_inspection(self):
        """Çekimserlik cevabı SİLMEZ; sonuç olarak sunmaz."""
        out = apply_calibration(
            {"confidence": 0.01, "is_reconstructible": True, "reconstructed_root": "*abc"},
            abstention_threshold=0.5,
        )
        self.assertEqual(out["withheld_reconstruction"], "*abc")

    def test_missing_model_is_declared_not_faked(self):
        """Kalibre edilmemiş skoru kalibreymiş gibi sunmak, hiç kalibre
        etmemekten kötüdür."""
        out = apply_calibration({"confidence": None, "is_reconstructible": False})
        self.assertFalse(out["calibrated"])
        self.assertIsNone(out["calibrated_confidence"])


class TestCalibrationModelFile(unittest.TestCase):
    def test_model_applies_sigmoid(self):
        model = CalibrationModel(
            a=1.0, b=0.0, method="platt", trained_on="t", n=1, ece_before=0.4, ece_after=0.02,
            trained_at="",
        )
        self.assertAlmostEqual(model.apply(0.0), 0.5, places=3)
        self.assertGreater(model.apply(5.0), 0.9)
        self.assertLess(model.apply(-5.0), 0.1)

    def test_shipped_model_is_trained_on_train_split_only(self):
        """dev veya test ile eğitilmiş bir kalibratör ölçümü geçersiz kılar."""
        from engine.nlp.confidence import MODEL_PATH

        if not MODEL_PATH.exists():
            self.skipTest("kalibrasyon modeli henüz eğitilmemiş")
        data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(str(data["trained_on"]).endswith("/train"), data["trained_on"])
        self.assertLess(data["ece_after"], data["ece_before"])


if __name__ == "__main__":
    unittest.main()
