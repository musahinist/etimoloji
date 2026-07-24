"""
İleri Hesaplamalı Etimoloji Modülleri Birim Testleri
(LingPy Hizalama, Diyakronik Vektör Manifoldu, Otomatik Ses Kanunu İndüksiyonu ve Neo4j Graf Şeması)
"""
import unittest
from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine
from engine.nlp.sound_law_induction import SoundLawInductionEngine
from engine.db.graph_database import GraphDatabaseManager

class TestAdvancedEtymologyModules(unittest.TestCase):

    def setUp(self):
        self.aligner = CldfLingPyAligner()
        self.semantic_engine = DiachronicSemanticEngine()
        self.induction_engine = SoundLawInductionEngine()
        self.graph_db = GraphDatabaseManager()

    def test_cldf_lingpy_sequence_alignment(self):
        """Needleman-Wunsch tabanlı LingPy fonetik dizi hizalama testi"""
        res = self.aligner.align_sequences("*sub", "su")
        self.assertIn("aligned_seq1", res)
        self.assertIn("aligned_seq2", res)
        self.assertGreater(res["phonetic_similarity"], 0.20)
        self.assertTrue(len(res["aligned_pairs"]) > 0)

    def test_diachronic_semantic_vector_trajectory(self):
        """Diyakronik semantik vektör yörüngesi ve ivme hesabı testi"""
        res = self.semantic_engine.evaluate_diachronic_trajectory("göz, görme organı", "görüş organı, göz")
        self.assertTrue(res["is_plausible"])
        self.assertLessEqual(res["semantic_acceleration"], res["theta_threshold"])

    def test_automated_sound_law_induction(self):
        """Akraba sözcük çiftlerinden otonom ses kanunu türetme testi"""
        res = self.induction_engine.induce_sound_law("köz", "göz")
        self.assertTrue(res["has_induced_rule"])
        self.assertTrue(any(r["rule_id"] == "INITIAL_K_G" for r in res["induced_rules"]))

    def test_graph_database_schema_export(self):
        """Neo4j ve Cytoscape uyumlu graf düğüm ve kenar şeması testi"""
        hypothesis = {
            "hypothesis_type": "Asli Öz Türkçe Köken Hipotezi",
            "confidence_score": 0.95,
            "donor_language": "Proto-Türkçe"
        }
        attestations = ["8. yy Orhun Yazıtları (kös)", "11. yy DLT (göz)"]
        cognates = ["görmek", "gözlem", "gözlük"]

        graph = self.graph_db.build_etymology_graph("göz", "*göŕ", hypothesis, attestations, cognates)
        self.assertGreater(graph["node_count"], 3)
        self.assertGreater(graph["edge_count"], 3)
        self.assertIn("elements", graph["cytoscape_format"])

if __name__ == "__main__":
    unittest.main()
