"""
A-HVP (AI Hypothesis Validation Protocol) ve Kurallar Motoru Birim Testleri
"""
import unittest
from engine.nlp.hypothesis_validation_protocol import (
    HypothesisValidationProtocol,
    ChronologicalTimeLock,
    PhoneticChainVerifier,
    SemanticDriftEvaluator
)

class TestHypothesisValidationProtocol(unittest.TestCase):

    def setUp(self):
        self.protocol = HypothesisValidationProtocol()
        self.time_lock = ChronologicalTimeLock()
        self.phonetic_verifier = PhoneticChainVerifier()
        self.semantic_evaluator = SemanticDriftEvaluator()

    def test_valid_proto_turkic_etymology(self):
        """Geçerli Öz Türkçe kök hipotezinin (göz -> *göŕ) yeşil doğrulama alması"""
        word = "göz"
        hypothesis = {
            "hypothesis_type": "Asli Öz Türkçe Köken Hipotezi",
            "origin_form": "*göŕ",
            "donor_language": "Proto-Türkçe M.Ö. 500",
            "proof_summary": "Söz sonu -z ~ -ŕ Lir-Şaz sızıcılaşma kanunu uyumu",
            "historical_meaning": "göz, görme organı"
        }
        attestation = {"first_attestation_record": "8. yüzyıl Orhun Yazıtları (kös)"}

        report = self.protocol.validate_hypothesis(word, hypothesis, attestation)
        
        # göz→*göŕ için skor %70+ ve fonetik/kronolojik geçerlilik bekliyoruz
        # (Cognate triangulation dar yayılım gösterdiğinden VALIDATED veya NEEDS_REVIEW olabilir)
        self.assertIn(report["status_code"], ["VALIDATED", "NEEDS_REVIEW"])
        self.assertGreaterEqual(report["final_confidence_score"], 0.70)
        self.assertTrue(report["stage_breakdown"]["stage1_phonetic_chain"]["is_valid"])
        self.assertTrue(report["stage_breakdown"]["stage2_time_lock"]["is_valid"])

    def test_anachronism_lock_rejection(self):
        """Anakronizm Kilidi (T_kaynak > T_hedef) nedeniyle hipotezin kırmızı reddedilmesi"""
        word = "su"
        # Sahte hipotez: 1950 Fransızca kelimesinden geldiği iddiası ama kelime 735 Köktürkçe kayıtlarda var
        hypothesis = {
            "hypothesis_type": "Sahte Alıntı Hipotezi",
            "origin_form": "sous",
            "donor_language": "Fransızca (1950 üretimi)",
            "proof_summary": "1950 modern Fransızca alıntısı",
            "historical_meaning": "su"
        }
        attestation = {"first_attestation_record": "735 yılı Orhun Yazıtları (sub)"}

        report = self.protocol.validate_hypothesis(word, hypothesis, attestation)

        self.assertEqual(report["status_code"], "REJECTED")
        self.assertLessEqual(report["final_confidence_score"], 0.40)
        self.assertFalse(report["stage_breakdown"]["stage2_time_lock"]["is_valid"])
        self.assertTrue(any("ANAKRONİZM ENGELİ" in rej for rej in report["rejection_reasons"]))

    def test_broken_phonetic_chain_violation(self):
        """Uydurma/kopuk ses dönüşümü durumunda kırık fonetik halka uyarısı üretilmesi"""
        result = self.phonetic_verifier.verify("xyzq", "göz")
        self.assertFalse(result["is_valid"])
        self.assertLess(result["score"], 0.50)
        self.assertTrue(len(result["violations"]) > 0)

    def test_time_lock_year_parser(self):
        """Tarih ve yüzyıl ayrıştırma testi"""
        self.assertEqual(self.time_lock.parse_year_or_century("11. yüzyıl Divanü Lugati't-Türk"), 1050)
        self.assertEqual(self.time_lock.parse_year_or_century("735 yılı Orhun"), 735)
        self.assertEqual(self.time_lock.parse_year_or_century("Cumhuriyet dönemi 1935"), 1935)

if __name__ == "__main__":
    unittest.main()
