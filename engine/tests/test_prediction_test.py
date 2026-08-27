"""
Öngörü testi (üret-kilitle-doğrula) testleri.

Protokol Bodt & List 2022 ve Blum ve ark. 2024'ten alınmıştır. Bu testler
protokolün bozulmamasını korur: kilit tutmazsa, ön-kayıt iddiası yerelde
kalırsa veya sahte eşleşmeler bulgu sayılırsa buradan kırmızı yanar.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from engine.evaluation import prediction_test as pt


def make(hypothesis="*köŕ", target="kk", form="köz"):
    return pt.Prediction(
        hypothesis=hypothesis,
        source_word="göz",
        source_languages=("tr", "cv"),
        target_language=target,
        predicted_form=form,
        confidence=0.4,
        rationale="test",
    )


class TestPredictionDigest(unittest.TestCase):
    def test_same_content_same_digest(self):
        self.assertEqual(make().digest(), make().digest())

    def test_changed_form_changes_digest(self):
        self.assertNotEqual(make(form="köz").digest(), make(form="küz").digest())

    def test_changed_target_changes_digest(self):
        self.assertNotEqual(make(target="kk").digest(), make(target="tt").digest())


class TestRegistrySealing(unittest.TestCase):
    def test_seal_detects_tampering(self):
        """Kilitlendikten sonra değiştirilen sicille doğrulama geçerli değildir."""
        with TemporaryDirectory() as tmp:
            with mock.patch.object(pt, "PREDICTIONS_DIR", Path(tmp)):
                registry = pt.PredictionRegistry(name="t", predictions=[make()])
                path = registry.save()

                data = json.loads(path.read_text(encoding="utf-8"))
                data["predictions"][0]["predicted_form"] = "başka"
                path.write_text(json.dumps(data), encoding="utf-8")

                with self.assertRaises(ValueError):
                    pt.PredictionRegistry.load("t")

    def test_round_trip_preserves_predictions(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.object(pt, "PREDICTIONS_DIR", Path(tmp)):
                registry = pt.PredictionRegistry(name="t", predictions=[make(), make(target="tt")])
                registry.save()
                loaded = pt.PredictionRegistry.load("t")
                self.assertEqual(len(loaded.predictions), 2)
                self.assertEqual(loaded.seal(), registry.seal())

    def test_cannot_overwrite_a_locked_registry(self):
        """Kilitli sicilin üzerine yazmak ön-kaydı geçersiz kılar."""
        with TemporaryDirectory() as tmp:
            with mock.patch.object(pt, "PREDICTIONS_DIR", Path(tmp)):
                pt.PredictionRegistry(name="t", predictions=[make()]).save()
                with self.assertRaises(FileExistsError):
                    pt.PredictionRegistry(name="t", predictions=[make()]).save()

    def test_local_timestamp_is_not_preregistration(self):
        """Kendi repondaki zaman damgası ön-kayıt yerine geçmez."""
        registry = pt.PredictionRegistry(name="t", predictions=[make()])
        self.assertFalse(registry.is_preregistered)

    def test_external_doi_marks_preregistration(self):
        registry = pt.PredictionRegistry(
            name="t", predictions=[make()], external_doi="10.17605/OSF.IO/XXXXX"
        )
        self.assertTrue(registry.is_preregistered)


class TestSpuriousMatches(unittest.TestCase):
    """Sözlükte olan her şey sözcük değildir."""

    def test_single_letter_is_spurious(self):
        self.assertTrue(pt.is_spurious_match("a", "The first letter of the alphabet"))

    def test_letter_name_gloss_is_spurious(self):
        self.assertTrue(pt.is_spurious_match("be", "The second letter of the alphabet"))

    def test_alternative_spelling_is_spurious(self):
        self.assertTrue(pt.is_spurious_match("qan", "Cyrillic spelling of qan"))

    def test_real_word_is_not_spurious(self):
        self.assertFalse(pt.is_spurious_match("köz", "eye"))

    def test_turkish_letter_gloss_is_spurious(self):
        self.assertTrue(pt.is_spurious_match("ke", "alfabenin harfi"))


class TestScoring(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(pt.score_verifications([])["n"], 0)

    def test_hit_rate_excludes_untestable(self):
        verifications = [
            pt.Verification("a", pt.HIT),
            pt.Verification("b", pt.MISS),
            pt.Verification("c", pt.UNTESTABLE),
        ]
        summary = pt.score_verifications(verifications)
        self.assertEqual(summary["testable"], 2)
        self.assertEqual(summary["hit_rate"], 0.5)

    def test_near_counts_towards_combined_rate(self):
        verifications = [pt.Verification("a", pt.HIT), pt.Verification("b", pt.NEAR)]
        self.assertEqual(pt.score_verifications(verifications)["hit_or_near_rate"], 1.0)

    def test_reference_warns_against_direct_comparison(self):
        """Bodt & List'in %70'i uzman seçimli ve saha doğrulamalıydı."""
        summary = pt.score_verifications([pt.Verification("a", pt.HIT)])
        self.assertIn("Doğrudan karşılaştırılamaz", summary["reference"])


class TestVerificationWithoutIndex(unittest.TestCase):
    def test_missing_index_marks_untestable_not_miss(self):
        """İndeks yoksa "tutmadı" denemez — denetlenemedi denir."""
        registry = pt.PredictionRegistry(name="t", predictions=[make()])
        with mock.patch("engine.db.lexicon_index.LexiconIndex.exists", False):
            results = pt.verify(registry)
        self.assertEqual(results[0].outcome, pt.UNTESTABLE)


if __name__ == "__main__":
    unittest.main()
