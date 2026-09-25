import unittest

from engine.nlp import neural_reconstruction as neural
from engine.tests.data_guards import needs_torch


class CharsTests(unittest.TestCase):
    def test_length_mark_unified(self):
        # savelyev ``o:`` ile Starling ``ō`` aynı dizi olur.
        self.assertEqual(neural.chars("o:t"), neural.chars("ōt"))

    def test_strips_hyphen_and_case(self):
        self.assertEqual(neural.chars("Büt-"), ["b", "ü", "t"])

    def test_source_tokens_order_is_fixed(self):
        a = neural.source_tokens(neural.SRC_SAV, neural.TGT_PROTO, "kül", {"tt": "köl", "kk": "kül"})
        b = neural.source_tokens(neural.SRC_SAV, neural.TGT_PROTO, "kül", {"kk": "kül", "tt": "köl"})
        self.assertEqual(a, b)


class SelectionTests(unittest.TestCase):
    def _cands(self):
        C = neural.Candidate
        return [
            C("*kül", column=-1.0, neural=-0.2, rank=0, in_beam=True),
            C("*köl", column=-0.5, neural=-0.4, rank=1, in_beam=True),
            C("*kul", column=-0.1, neural=-2.0, is_column=True),
        ]

    def test_select_column_only_from_beam(self):
        self.assertEqual(neural.select_column(self._cands()), "*köl")

    def test_vote_tie_prefers_column(self):
        # sıralar: sinir kül0 köl1 kul2 · sütun kul0 köl1 kül2 -> kül ve kul eşit, sütun top-1 kazanır
        self.assertEqual(neural.select_vote(self._cands()), "*kul")

    def test_ranker_uses_weights(self):
        cands = self._cands()
        self.assertEqual(neural.select_ranker(cands, {"neural": 5.0}, 0.0, 3, "kül", []), "*kül")
        self.assertEqual(neural.select_ranker(cands, {"is_column": 5.0}, 0.0, 3, "kül", []), "*kul")


@needs_torch
class TrainSmokeTests(unittest.TestCase):
    def test_tiny_training_returns_candidates(self):
        examples = [
            neural.Example(neural.SRC_STA, "kül", {"kk": "kül", "tt": "köl", "cv": "kəl"}, "kül"),
            neural.Example(neural.SRC_SAV, "ot", {"kk": "ot", "tt": "ut"}, "ōt", 2),
        ]
        model = neural.train(examples, epochs=1)
        out = model.reconstruct("kül", [{"lang_code": "kk", "word": "kül"}])
        self.assertIn("is_reconstructible", out)
        score = model.score("kül", [{"lang_code": "kk", "word": "kül"}], "*kül")
        self.assertLess(score, 0.0)


class EngineHookTests(unittest.TestCase):
    def _fake(self, chosen):
        class FakeSelector:
            def select(self, word, entries, informative, column, table, original):
                return chosen, [neural.Candidate(chosen or original, rank=0, in_beam=True)]
        return FakeSelector()

    def _run(self, result, chosen):
        from unittest import mock

        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        with mock.patch.object(neural, "active_selector", return_value=self._fake(chosen)), \
                mock.patch("engine.nlp.column_model.active_model", return_value=object()):
            return ComparativeReconstructor._neural_select(
                dict(result), "kül", [{"lang_code": "kk", "word": "kül"}, {"lang_code": "tt", "word": "köl"}]
            )

    def test_replaces_comparative_root(self):
        out = self._run({"method": "comparative", "is_reconstructible": True, "reconstructed_root": "*köl"}, "*kül")
        self.assertEqual(out["reconstructed_root"], "*kül")
        self.assertEqual(out["neural_selection"]["column_model_root"], "*köl")
        self.assertEqual(out["alternative_forms"][0], "*köl")

    def test_fallback_untouched(self):
        result = {"method": "anchor_fallback", "is_reconstructible": True, "reconstructed_root": "*köl"}
        self.assertEqual(self._run(result, "*kül"), result)

    def test_no_selector_keeps_root(self):
        from unittest import mock

        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        result = {"method": "comparative", "is_reconstructible": True, "reconstructed_root": "*köl"}
        with mock.patch.object(neural, "active_selector", return_value=None):
            self.assertEqual(ComparativeReconstructor._neural_select(dict(result), "kül", []), result)


if __name__ == "__main__":
    unittest.main()
