"""
İleri akraba tahmini testleri.

Bu katman zincirin kopuk ilk halkasıydı: etimolojisi yapılmamış kelimenin
akrabasını bulmak için önce nasıl görüneceğini tahmin etmek gerekir.
"""

from __future__ import annotations

import unittest

from engine.config import CLDF_DIR
from engine.nlp.cognate_prediction import (
    MIN_SUPPORT,
    CognatePredictor,
    CorrespondenceTable,
    load_tables,
    position_of,
)

HAS_TABLES = bool(load_tables())
HAS_CLDF = (CLDF_DIR / "savelyevturkic" / "forms.csv").exists()


class TestPositionOf(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(position_of(0, 3), "initial")
        self.assertEqual(position_of(1, 3), "medial")
        self.assertEqual(position_of(2, 3), "final")

    def test_single_character_is_initial(self):
        self.assertEqual(position_of(0, 1), "initial")


class TestCorrespondenceTable(unittest.TestCase):
    def test_single_observation_is_noise_not_a_rule(self):
        """Tek gözlemli denklik kural değildir; MIN_SUPPORT bunu korur."""
        table = CorrespondenceTable("tr", "kk")
        table.observe("final", "ş", "s")
        predicted, probability = table.predict("final", "ş")
        self.assertEqual(predicted, "ş")
        self.assertEqual(probability, 0.0)

    def test_repeated_observation_becomes_a_rule(self):
        table = CorrespondenceTable("tr", "kk")
        for _ in range(MIN_SUPPORT):
            table.observe("final", "ş", "s")
        predicted, probability = table.predict("final", "ş")
        self.assertEqual(predicted, "s")
        self.assertEqual(probability, 1.0)

    def test_unseen_sound_is_left_untouched(self):
        """Bilinmeyen ses uydurulmaz — en muhafazakâr tahmin kaynak sestir."""
        table = CorrespondenceTable("tr", "kk")
        self.assertEqual(table.predict("initial", "q"), ("q", 0.0))

    def test_position_specific_beats_pooled(self):
        table = CorrespondenceTable("tr", "kk")
        for _ in range(5):
            table.observe("initial", "k", "q")
        for _ in range(20):
            table.observe("final", "k", "k")
        self.assertEqual(table.predict("initial", "k")[0], "q")
        self.assertEqual(table.predict("final", "k")[0], "k")

    def test_round_trip_serialisation(self):
        table = CorrespondenceTable("tr", "kk")
        for _ in range(3):
            table.observe("final", "z", "s")
        restored = CorrespondenceTable.from_dict(table.as_dict())
        self.assertEqual(restored.predict("final", "z"), table.predict("final", "z"))

    def test_low_support_rules_are_not_serialised(self):
        table = CorrespondenceTable("tr", "kk")
        table.observe("final", "x", "y")
        self.assertEqual(table.as_dict()["rules"], {})


class TestPredictorWithoutTables(unittest.TestCase):
    def test_missing_table_does_not_invent(self):
        predictor = CognatePredictor(tables={})
        prediction = predictor.predict("göz", "tr", "kk")
        self.assertEqual(prediction.confidence, 0.0)
        self.assertEqual(prediction.form, "göz")

    def test_empty_word(self):
        predictor = CognatePredictor(tables={})
        self.assertEqual(predictor.predict("", "tr", "kk").form, "")


@unittest.skipUnless(HAS_TABLES, "denklik tabloları öğrenilmemiş (make correspondences)")
class TestLearnedPredictions(unittest.TestCase):
    """Ölçülmüş hataların regresyon koruması."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = CognatePredictor()

    def test_sibilant_correspondence_was_learned(self):
        """Elle yazılmış kurallarda ``ş ~ s`` denkliği YOKTU.

        Bu yüzden ``baş ~ бас`` ve ``taş ~ тас`` bulunamıyordu.
        """
        self.assertEqual(self.predictor.predict("baş", "tr", "kk").form, "bas")
        self.assertEqual(self.predictor.predict("taş", "tr", "kk").form, "tas")

    def test_front_vowel_correspondence(self):
        self.assertEqual(self.predictor.predict("göz", "tr", "kk").form, "köz")

    def test_prediction_is_deterministic(self):
        first = self.predictor.predict("kar", "tr", "kk").form
        second = self.predictor.predict("kar", "tr", "kk").form
        self.assertEqual(first, second)

    def test_predict_all_returns_sorted_languages(self):
        predictions = self.predictor.predict_all("göz", "tr")
        self.assertGreater(len(predictions), 5)
        confidences = [p.confidence for p in predictions]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_steps_explain_each_change(self):
        """Tahmin açıklanabilir olmalı: hangi ses neye, hangi olasılıkla."""
        prediction = self.predictor.predict("baş", "tr", "kk")
        self.assertTrue(prediction.steps)
        for step in prediction.steps:
            self.assertIn("from", step)
            self.assertIn("to", step)
            self.assertIn("probability", step)


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş (make data)")
class TestLearningIsolation(unittest.TestCase):
    def test_tables_are_learned_from_train_only(self):
        """dev/test kavramlarını görmek ileri tahmin ölçümünü geçersiz kılar."""
        import json

        from engine.nlp.cognate_prediction import CORRESPONDENCE_PATH

        if not CORRESPONDENCE_PATH.exists():
            self.skipTest("tablolar henüz öğrenilmemiş")
        data = json.loads(CORRESPONDENCE_PATH.read_text(encoding="utf-8"))
        self.assertTrue(str(data["trained_on"]).endswith("/train"), data["trained_on"])


if __name__ == "__main__":
    unittest.main()
