"""
Yeni Hesaplamalı Dilbilim ve Bilimsel NLP Modülleri Birim Testleri
(PanPhon artikülatör vektörleri, CLDF standart aktarıcısı, Wiktextract fetcher)
"""

import unittest
from engine.nlp.phonological_feature_engine import PhonologicalFeatureEngine
from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine
from engine.db.cldf_exporter import CldfExporter
from engine.fetchers.wiktextract_local import WiktextractFetcher

class TestNewScientificModules(unittest.TestCase):

    def setUp(self):
        self.panphon = PhonologicalFeatureEngine()
        self.lingpy_aligner = CldfLingPyAligner()
        self.semantic_engine = DiachronicSemanticEngine()
        self.cldf_exporter = CldfExporter()
        self.wiktextract = WiktextractFetcher()

    def test_panphon_articulatory_feature_vector(self):
        """PanPhon 21-boyutlu IPA artikülatör fonetik vektör ve Hamming uzaklığı testi"""
        v_a = self.panphon.get_feature_vector('a')
        v_e = self.panphon.get_feature_vector('e')
        self.assertEqual(len(v_a), 21)
        self.assertEqual(len(v_e), 21)

        # Ünlü-Ünlü ve Ünlü-Ünsüz mesafesi
        dist_ae = self.panphon.articulatory_hamming_distance('a', 'e')
        dist_ab = self.panphon.articulatory_hamming_distance('a', 'b')
        self.assertLess(dist_ae, dist_ab)

        # Dizilim artikülatör mesafesi
        res = self.panphon.sequence_phonological_distance("sub", "suv")
        self.assertGreater(res["phonetic_similarity"], 0.70)

    def test_cldf_lingpy_sca_dolgopolsky_alignment(self):
        """LingPy SCA Dolgopolsky ses sınıfları hizalaması testi"""
        res = self.lingpy_aligner.align_sequences("teŋiz", "deniz")
        self.assertIn("sound_class_seq1", res)
        self.assertGreater(res["phonetic_similarity"], 0.75)

    def test_cldf_exporter(self):
        """Cross-Linguistic Data Formats (CLDF) standart CSV ve metadata testi"""
        finding = {
            "query_word": "göz",
            "root": {"proto_turkic": "*göŕ", "meaning": "eye"},
            "turkic_languages": [
                {"lang_code": "tr", "word": "göz"},
                {"lang_code": "az", "word": "göz"},
                {"lang_code": "kk", "word": "көз"}
            ]
        }
        exported = self.cldf_exporter.export_to_cldf(finding)
        self.assertIn("cldf_forms_csv", exported)
        self.assertIn("cldf_cognates_csv", exported)
        self.assertIn("form_1", exported["cldf_forms_csv"])

    def test_wiktextract_fetcher_structure(self):
        """Wiktextract / Kaikki fetcher yapısı ve dönen verinin jenerik doğrulama testi"""
        res = self.wiktextract.fetch("deniz")
        self.assertIn("root", res)
        self.assertIn("turkic_languages", res)

if __name__ == "__main__":
    unittest.main()
