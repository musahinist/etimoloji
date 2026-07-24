"""
Çözülmemiş Kelimeler İçin Otonom Etimoloji Rekonstrüksiyon Mimarisi Birim Testleri
(Jenerik Ek Soyma, Tahminleyici Proto-Kök Türetimi ve İnatçı A-HVP Hipotez Kurma)
Sıfır kelime bazlı hardcode içerir.
"""
import unittest
from engine.nlp.unsupervised_morpheme_segmenter import UnsupervisedMorphemeSegmenter
from engine.nlp.predictive_reconstructor import PredictiveReconstructor
from engine.nlp.iterative_hypothesis_prover import IterativeHypothesisProver

class TestUnattestedEtymologyReconstruction(unittest.TestCase):

    def setUp(self):
        self.segmenter = UnsupervisedMorphemeSegmenter()
        self.reconstructor = PredictiveReconstructor()
        self.prover = IterativeHypothesisProver()

    def test_unsupervised_morpheme_segmentation(self):
        """Kelimelerin yapım/çekim eklerini hiçbir hardcode kelime olmadan jenerik soyma testi"""
        res = self.segmenter.segment_morphemes("yağmurluk")
        self.assertTrue(res["is_segmented"])
        self.assertIn("-luk", res["affixes"])

    def test_predictive_proto_form_reconstruction(self):
        """Diyalektik ses kurallarını geriye projeksiyon ile *proto-form hesaplama testi"""
        res = self.reconstructor.reconstruct_unattested_proto_form("korak")
        self.assertTrue(res["reconstructed_proto_form"].startswith("*"))
        self.assertGreater(res["reconstruction_confidence"], 0.70)

    def test_iterative_unattested_hypothesis_proving(self):
        """Sözlükte kaydı bulunmayan çözülmemiş kelimeler için otonom A-HVP hipotez PR üretimi testi"""
        res = self.prover.prove_unattested_word("kurtarılmak")
        self.assertIn("proven_hypothesis", res)
        hypo = res["proven_hypothesis"]
        self.assertTrue(hypo.get("is_unattested_ai_pr"))
        self.assertIn("validation_report", hypo)

if __name__ == "__main__":
    unittest.main()
