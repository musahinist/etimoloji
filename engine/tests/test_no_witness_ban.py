"""Tanıksız kök yasağı ve üretilmiş sahte kök bataryası."""

from __future__ import annotations

import unittest
from unittest import mock

from engine.evaluation.negative_controls import (
    GENERATED_FAKES,
    GENERATED_PHONOTACTIC_FAKES,
    PHONOTACTICALLY_VALID,
    fake_witnesses,
)
from engine.nlp.comparative_reconstruction import ComparativeReconstructor
from engine.nlp.hypothesis_ranking import HypothesisRanker


def _index_available() -> bool:
    try:
        from engine.db.lexicon_index import LexiconIndex

        return LexiconIndex().exists
    except Exception:
        return False


class _FakeAligner:
    pass


def _reconstruct(word: str, witnesses: list[tuple[str, str]], attested: int | None):
    engine = ComparativeReconstructor(aligner=_FakeAligner())
    entries = [{"lang_code": code, "word": form} for code, form in witnesses]
    with mock.patch.object(
        ComparativeReconstructor, "_attested_witness_count", return_value=attested
    ):
        return engine.reconstruct(word, entries, check_borrowing=False)


WITNESSES = [("kk", "tirbek"), ("ky", "tirbek"), ("uz", "tirbak")]


class TestUnattestedBan(unittest.TestCase):
    def test_zero_attested_produces_no_comparative_root(self):
        result = _reconstruct("tirbek", WITNESSES, attested=0)
        self.assertEqual(result["method"], "anchor_fallback")
        self.assertTrue(result["unattested_ban"])
        self.assertFalse(result["evidence_available"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["confidence_badge"], "⚪ YETERSİZ KANIT")
        # Türetilen aday gizlenmez, ama sonuç olarak sunulmaz.
        self.assertTrue(result["withheld_reconstruction"].startswith("*"))

    def test_attested_witness_keeps_comparative_root(self):
        result = _reconstruct("tirbek", WITNESSES, attested=1)
        self.assertEqual(result["method"], "comparative")
        self.assertNotIn("unattested_ban", result)

    def test_unmeasurable_attestation_is_not_banned(self):
        """İndeks yoksa (None) "sıfır tanık" sayılmaz."""
        result = _reconstruct("tirbek", WITNESSES, attested=None)
        self.assertEqual(result["method"], "comparative")

    def test_implausible_form_still_abstains(self):
        """Proto-Türkçe olamayacak biçim geri-dönüş adayı almaz."""
        result = _reconstruct("zzzqx", [("kk", "zzzqy"), ("tt", "zzzqz")], attested=0)
        self.assertFalse(result["is_reconstructible"])

    def test_hypothesis_is_insufficient_evidence(self):
        ranker = HypothesisRanker(
            reconstructor=mock.Mock(
                reconstruct=mock.Mock(
                    return_value=_reconstruct("tirbek", WITNESSES, attested=0)
                )
            ),
            borrowing_detector=mock.Mock(
                detect=mock.Mock(
                    return_value=mock.Mock(
                        donor_language="", signals=[], score=0.0, chain=[],
                        expected_if_inherited="", word="tirbek",
                    )
                )
            ),
        )
        with mock.patch(
            "engine.nlp.borrowing_chain.source_loan_step", return_value=None
        ):
            ranked = ranker.rank("tirbek", [])
        inherited = next(h for h in ranked.hypotheses if h.kind == "inherited")
        self.assertLessEqual(inherited.score, 0.05)
        self.assertIn("tanıksız", inherited.claim)
        self.assertEqual(inherited.detail["confidence_badge"], "⚪ YETERSİZ KANIT")
        self.assertEqual(ranked.selected.kind, "unknown")


class TestGeneratedFakeBattery(unittest.TestCase):
    def test_battery_has_at_least_50_generated_fakes(self):
        self.assertGreaterEqual(len(GENERATED_FAKES), 50)
        self.assertEqual(len(set(GENERATED_FAKES)), len(GENERATED_FAKES))
        self.assertGreaterEqual(len(PHONOTACTICALLY_VALID + GENERATED_PHONOTACTIC_FAKES), 58)

    def test_generated_fakes_obey_turkic_shape(self):
        from engine.utils.phonotactics import (
            VOWELS,
            has_vowel_harmony,
            initial_consonant_violation,
        )

        for word in GENERATED_FAKES:
            with self.subTest(word=word):
                self.assertTrue(4 <= len(word) <= 7)
                self.assertGreaterEqual(sum(ch in VOWELS for ch in word), 2)
                self.assertTrue(has_vowel_harmony(word))
                self.assertFalse(initial_consonant_violation(word)[0])

    def test_witnesses_are_regular_correspondences(self):
        self.assertEqual(
            fake_witnesses("yençeği"),
            [("kk", "jenşeği"), ("ky", "jençeği"), ("tt", "yänçäği")],
        )

    @unittest.skipUnless(_index_available(), "sözlük indeksi yok")
    def test_generated_fakes_absent_from_lexicon_index(self):
        engine = ComparativeReconstructor()
        for word in GENERATED_FAKES:
            with self.subTest(word=word):
                forms = [word, *(w for _, w in fake_witnesses(word))]
                self.assertEqual(engine._attested_witness_count(forms), 0)


if __name__ == "__main__":
    unittest.main()
