"""
Rakip hipotez sıralaması ve karşıtsal red gerekçesi testleri.

Reddedilen hipotezin ÇIKTIDA KALMASI ve NEDEN reddedildiğinin söylenmesi
bu modülün varlık sebebidir; testler o sözleşmeyi korur.
"""

from __future__ import annotations

import unittest

from engine.nlp.hypothesis_ranking import (
    HYPOTHESIS_KINDS,
    Hypothesis,
    HypothesisRanker,
    RankedHypotheses,
)
from engine.tests.data_guards import needs_index


def h(kind, score, **kwargs):
    return Hypothesis(kind=kind, claim=kind, score=score, **kwargs)


class TestRankedHypotheses(unittest.TestCase):
    def test_selected_is_the_highest_unrejected(self):
        ranked = RankedHypotheses(
            "x", [h("inherited", 0.2), h("borrowed", 0.6)]
        )
        HypothesisRanker._write_rejections(ranked)
        self.assertEqual(ranked.selected.kind, "borrowed")

    def test_rejected_hypotheses_are_kept_not_deleted(self):
        """Reddedilenler çıktıda KALIR — asıl özellik budur."""
        ranked = RankedHypotheses("x", [h("inherited", 0.2), h("borrowed", 0.6)])
        HypothesisRanker._write_rejections(ranked)
        self.assertEqual(len(ranked.hypotheses), 2)
        self.assertTrue(any(x.is_rejected for x in ranked.hypotheses))

    def test_every_rejection_states_a_reason(self):
        ranked = RankedHypotheses(
            "x", [h("inherited", 0.2), h("borrowed", 0.6, supporting=["kanıt"])]
        )
        HypothesisRanker._write_rejections(ranked)
        for hypothesis in ranked.hypotheses:
            if hypothesis.is_rejected:
                self.assertTrue(hypothesis.rejected_because.strip())

    def test_every_rejection_states_a_counterfactual(self):
        """"Doğru olsaydı ne beklerdik?" olmadan gerekçe yanlışlanabilir değil."""
        ranked = RankedHypotheses("x", [h("inherited", 0.2), h("borrowed", 0.6)])
        HypothesisRanker._write_rejections(ranked)
        rejected = [x for x in ranked.hypotheses if x.is_rejected]
        self.assertTrue(rejected)
        for hypothesis in rejected:
            self.assertTrue(hypothesis.counterfactual.strip(), hypothesis.kind)

    def test_rejection_reason_cites_the_winning_evidence(self):
        """"Skoru düşüktü" bir gerekçe değildir."""
        ranked = RankedHypotheses(
            "kitap",
            [
                h("inherited", 0.19),
                h("borrowed", 0.60, supporting=["Arapça alıntı olarak tanıklanmış"]),
            ],
        )
        HypothesisRanker._write_rejections(ranked)
        inherited = next(x for x in ranked.hypotheses if x.kind == "inherited")
        self.assertIn("Arapça", inherited.rejected_because)

    def test_tied_hypotheses_are_not_rejected(self):
        ranked = RankedHypotheses("x", [h("inherited", 0.5), h("borrowed", 0.5)])
        HypothesisRanker._write_rejections(ranked)
        self.assertFalse(any(x.is_rejected for x in ranked.hypotheses))

    def test_close_call_is_flagged_as_contested(self):
        """Kırılgan karar kullanıcıdan saklanmaz."""
        ranked = RankedHypotheses("x", [h("inherited", 0.50), h("borrowed", 0.45)])
        self.assertTrue(ranked.is_contested)

    def test_clear_win_is_not_contested(self):
        ranked = RankedHypotheses("x", [h("inherited", 0.10), h("borrowed", 0.80)])
        self.assertFalse(ranked.is_contested)

    def test_all_kinds_have_readable_labels(self):
        for kind in HYPOTHESIS_KINDS:
            self.assertTrue(h(kind, 0.1).label.strip())


class TestRankerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.ranker = HypothesisRanker()

    @needs_index
    def test_borrowing_wins_for_a_known_loanword(self):
        entries = [{"lang_code": c, "word": w} for c, w in
                   [("kk", "kitap"), ("tt", "kitap"), ("uz", "kitob")]]
        ranked = self.ranker.rank("kitap", entries)
        self.assertEqual(ranked.selected.kind, "borrowed")

    def test_inheritance_wins_for_a_core_word(self):
        entries = [{"lang_code": c, "word": w} for c, w in
                   [("tr", "göz"), ("kk", "көз"), ("cv", "куҫ"), ("tt", "күз")]]
        ranked = self.ranker.rank("göz", entries)
        self.assertEqual(ranked.selected.kind, "inherited")

    def test_modern_coinage_wins_for_a_republican_derivation(self):
        ranked = self.ranker.rank("bilgisayar", [])
        self.assertEqual(ranked.selected.kind, "modern_coinage")

    def test_output_always_contains_more_than_one_hypothesis(self):
        """Tek cevap vermek bu modülün reddettiği şeydir."""
        ranked = self.ranker.rank("göz", [{"lang_code": "kk", "word": "köz"}])
        self.assertGreaterEqual(len(ranked.hypotheses), 2)

    def test_unknown_when_no_evidence(self):
        ranked = self.ranker.rank("qqqq", [])
        self.assertIsNotNone(ranked.selected)

    def test_serialisable(self):
        data = self.ranker.rank("göz", [{"lang_code": "cv", "word": "kus"}]).as_dict()
        self.assertIn("hypotheses", data)
        self.assertIn("explanation", data)
        self.assertIn("margin", data)

    def test_deterministic(self):
        entries = [{"lang_code": "cv", "word": "kus"}, {"lang_code": "kk", "word": "köz"}]
        first = self.ranker.rank("göz", entries).as_dict()
        second = self.ranker.rank("göz", entries).as_dict()
        self.assertEqual(first["explanation"], second["explanation"])


if __name__ == "__main__":
    unittest.main()


class TestDonorProximityWithAttestedDonor(unittest.TestCase):
    """Tanıklı verici biçim varken verici yakınlığı ıskası karşı kanıt değildir."""

    @staticmethod
    def _borrowing(chain, donor="ar"):
        from types import SimpleNamespace

        from engine.nlp.borrowing_detector import Signal

        return SimpleNamespace(
            word="asker",
            donor_language=donor,
            score=0.6,
            chain=chain,
            expected_if_inherited="",
            signals=[
                Signal("verici_yakınlığı", False, 0.0, "verici sözlüğünde yakın karşılık yok"),
                Signal("ünlü_uyumu", False, 0.0, "ünlü uyumu korunuyor"),
            ],
        )

    def test_attested_donor_moves_miss_to_not_evaluated(self):
        hyp = HypothesisRanker._borrowed_hypothesis(self._borrowing(["Türkçe asker", "Arapça عَسْكَر"]))
        self.assertNotIn("verici sözlüğünde yakın karşılık yok", hyp.against)
        self.assertIn("ünlü uyumu korunuyor", hyp.against)
        self.assertTrue(any("Arapça عَسْكَر" in x for x in hyp.not_evaluated))

    def test_without_donor_form_miss_stays_counter_evidence(self):
        hyp = HypothesisRanker._borrowed_hypothesis(self._borrowing([], donor=None))
        self.assertIn("verici sözlüğünde yakın karşılık yok", hyp.against)

    def test_source_loan_chain_also_clears_the_miss(self):
        borrowing = self._borrowing([], donor=None)
        base = HypothesisRanker._borrowed_hypothesis(borrowing)
        entries = [{"lang_code": "donor", "lang_name": "Fransızca", "word": "accumulateur",
                    "etymology": "Alıntı", "source": "kaynak"}]
        from unittest import mock

        step = {"lang_name": "Fransızca", "word": "accumulateur", "source": "kaynak"}
        with mock.patch("engine.nlp.borrowing_chain.source_loan_step", return_value=step):
            hyp = HypothesisRanker._with_source_loan(base, borrowing, entries)
        self.assertNotIn("verici sözlüğünde yakın karşılık yok", hyp.against)
        self.assertTrue(any("accumulateur" in x for x in hyp.not_evaluated))
