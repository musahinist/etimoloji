"""
N-best üretimi ve yeniden sıralama testleri (Faz D5).

⚠️ **Yeniden sıralama KAPALIDIR ve bu ölçülmüş bir karardır.** Doğru cevap
adayların içinde (oracle 0,506 vs top-1 0,434) ama elimizdeki sıralayıcı onu
öne çıkaramıyor: P2D üretim uyumu tek başına 0,3614, hiçbir karışım
konsensüsü geçmiyor. Testler adayların üretilmeye devam etmesini (rakip
hipotez değeri) ve seçilen biçmin değişmemesini korur.
"""

from __future__ import annotations

import unittest

from engine.nlp.nbest_reranking import (
    MIN_ALTERNATIVE_SCORE,
    generate,
    generation_fit,
    rerank,
)
from engine.nlp.proto_phonology import ColumnDecision


def _decision(sound: str, alternatives: tuple[tuple[str, float], ...]) -> ColumnDecision:
    return ColumnDecision(sound, None, "arkaik_agirlik", 0.8, alternatives=alternatives)


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.decisions = [
            _decision("k", (("k", 0.7), ("g", 0.3))),
            _decision("ö", (("ö", 0.6), ("u", 0.4))),
            _decision("ŕ", (("ŕ", 1.0),)),
        ]

    def test_top_one_is_always_first(self):
        """⚠️ Yeniden sıralama başarısız olduğunda sistem eski davranışına
        dönebilmeli; top-1 kararı listenin başında durmalı."""
        self.assertEqual(generate(self.decisions)[0][0], "köŕ")

    def test_alternatives_are_combined(self):
        forms = {form for form, _ in generate(self.decisions)}
        self.assertIn("guŕ", forms)
        self.assertIn("kuŕ", forms)

    def test_single_alternative_column_does_not_branch(self):
        self.assertTrue(all(form.endswith("ŕ") for form, _ in generate(self.decisions)))

    def test_weak_alternatives_are_dropped(self):
        """Eşiksiz taşımak tek gözlemi 'aday' mertebesine çıkarır."""
        weak = [_decision("k", (("k", 0.95), ("q", MIN_ALTERNATIVE_SCORE / 2)))]
        self.assertEqual([f for f, _ in generate(weak)], ["k"])

    def test_candidate_cap_is_respected(self):
        many = [_decision("a", (("a", 0.4), ("b", 0.3), ("c", 0.3)))] * 6
        self.assertLessEqual(len(generate(many, max_candidates=10)), 10)

    def test_no_decisions(self):
        self.assertEqual(generate([]), [])

    def test_decision_without_alternatives_still_yields_its_sound(self):
        self.assertEqual(generate([_decision("t", ())]), [("t", 1.0)])


class TestGenerationFit(unittest.TestCase):
    def test_identical_forms_fit_perfectly(self):
        self.assertGreater(generation_fit("kum", {"tr": "kum", "kk": "kum"}), 0.5)

    def test_unrelated_form_fits_poorly(self):
        good = generation_fit("kum", {"tr": "kum"})
        bad = generation_fit("zzz", {"tr": "kum"})
        self.assertGreater(good, bad)

    def test_empty_input(self):
        self.assertEqual(generation_fit("", {"tr": "kum"}), 0.0)
        self.assertEqual(generation_fit("kum", {}), 0.0)


class TestRerankingIsMeasuredNotAssumed(unittest.TestCase):
    def test_rerank_returns_scored_candidates(self):
        decisions = [
            _decision("k", (("k", 0.7), ("g", 0.3))),
            _decision("u", (("u", 0.6), ("o", 0.4))),
            _decision("m", (("m", 1.0),)),
        ]
        ranked = rerank(decisions, {"tr": "kum", "kk": "kum"})
        self.assertTrue(ranked)
        self.assertTrue(all(0.0 <= c.score <= 1.0 for c in ranked))

    def test_reconstruction_does_not_use_the_reranker(self):
        """⚠️ Ölçüldü: yeniden sıralama doğruluğu 0,4337'den 0,3614'e
        DÜŞÜRÜYOR. Seçilen biçim sütun kararlarından gelmeli."""
        import inspect

        from engine.nlp import comparative_reconstruction as cr

        source = inspect.getsource(cr)
        self.assertNotIn("rerank(", source)
        self.assertIn("generate_candidates", source)

    def test_alternatives_are_offered_to_the_user(self):
        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        out = ComparativeReconstructor().reconstruct(
            "göz",
            [
                {"lang_code": "kk", "word": "köz"},
                {"lang_code": "cv", "word": "kuś"},
                {"lang_code": "tt", "word": "küz"},
            ],
        )
        self.assertTrue(out["alternative_forms"])
        self.assertNotIn(out["reconstructed_root"], out["alternative_forms"])


class TestColumnAlternatives(unittest.TestCase):
    def test_diagnostic_decision_has_no_alternatives(self):
        """⚠️ Tanısal karar bir TANIMDIR (Lir-Şaz); alternatifi yoktur."""
        from engine.nlp.multi_alignment import AlignedColumn
        from engine.nlp.proto_phonology import pick_proto_sound

        column = AlignedColumn(1, {"cv": "r", "tr": "z", "kk": "z"}, 2)
        decision = pick_proto_sound(column, "final")
        self.assertEqual(decision.method, "tanisal")
        self.assertEqual(decision.alternatives, (("ŕ", 1.0),))

    def test_weighted_vote_carries_its_runners_up(self):
        from engine.nlp.multi_alignment import AlignedColumn
        from engine.nlp.proto_phonology import pick_proto_sound

        column = AlignedColumn(0, {"tr": "g", "kk": "k", "tt": "k"}, 1)
        decision = pick_proto_sound(column, "initial")
        self.assertGreaterEqual(len(decision.alternatives), 2)


if __name__ == "__main__":
    unittest.main()
