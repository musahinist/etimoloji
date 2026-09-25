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


if __name__ == "__main__":
    unittest.main()
