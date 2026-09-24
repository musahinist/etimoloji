"""
A-HVP Hakem Protokolü Birim Testleri

Bu testlerin merkezinde tek bir ilke var: **KANIT YOKSA PUAN DA YOK**.
Önceki sürümde dört aşamanın dördü de kanıt yokluğunda cömert varsayılan
puanlar veriyor, uydurma bir kelime %96 "🟢 VALIDATED" alabiliyordu.
"""
import unittest

from engine.nlp.hypothesis_validation_protocol import (
    ChronologicalTimeLock,
    CrossCognateTriangulator,
    HypothesisValidationProtocol,
    PhoneticChainVerifier,
    SemanticDriftEvaluator,
)
from engine.tests.data_guards import needs_panphon


def make_entries(pairs, origin="live", source="TestSource"):
    """Test için gerçekçi dil kaydı listesi üretir."""
    return [
        {"lang_code": code, "word": form, "source": src, "origin": origin}
        for code, form, src in ((c, f, source) for c, f in pairs)
    ]


class TestHypothesisValidationProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = HypothesisValidationProtocol()
        self.time_lock = ChronologicalTimeLock()
        self.phonetic_verifier = PhoneticChainVerifier()
        self.semantic_evaluator = SemanticDriftEvaluator()
        self.triangulator = CrossCognateTriangulator()
        self.goz_entries = make_entries([
            ("tr", "göz"), ("az", "göz"), ("kk", "көз"), ("tt", "күз"),
            ("cv", "куҫ"), ("ky", "көз"), ("otk", "köz"), ("ba", "күҙ"),
        ], origin="local")

    # --- Kanıt kapsamı ---------------------------------------------------

    def test_fabricated_word_gets_insufficient_evidence(self):
        """Uydurma kelime, kanıt olmadan DOĞRULANMAMALIDIR.

        Regresyon: 'zzzqx' eskiden %96 '🟢 VALIDATED (Bilimsel Hakem Onaylı)'
        alıyordu.
        """
        report = self.protocol.validate_hypothesis(
            "zzzqx",
            {"origin_form": "*zzzqx", "donor_language": "Proto-Türkçe",
             "historical_meaning": "", "modern_meaning": ""},
            attestation_record=None,
            turkic_entries=[],
        )
        self.assertEqual(report["status_code"], "INSUFFICIENT_EVIDENCE")
        self.assertLess(report["evidence_coverage"], 0.5)
        self.assertNotIn("VALIDATED", report["badge"])

    def test_missing_evidence_is_reported_not_scored(self):
        """Kanıt üretemeyen aşamalar puana KATILMAMALI, açıkça raporlanmalıdır."""
        report = self.protocol.validate_hypothesis(
            "zzzqx",
            {"origin_form": "*zzzqx", "donor_language": "Proto-Türkçe"},
            attestation_record=None,
            turkic_entries=[],
        )
        self.assertIn("chronology", report["missing_evidence"])
        self.assertIn("triangulation", report["missing_evidence"])
        self.assertEqual(
            report["stage_breakdown"]["stage2_time_lock"]["evidence_available"], False
        )
        self.assertIsNone(report["stage_breakdown"]["stage2_time_lock"]["score"])

    @needs_panphon
    def test_well_attested_etymology_scores_high(self):
        """Gerçek, çok tanıklı bir etimoloji yüksek skor ve kapsam almalıdır."""
        report = self.protocol.validate_hypothesis(
            "göz",
            {"origin_form": "*köŕ", "donor_language": "Proto-Türkçe",
             "historical_meaning": "göz", "modern_meaning": "göz"},
            attestation_record={"verified": True, "first_attestation_year": 1074},
            turkic_entries=self.goz_entries,
        )
        self.assertIn(report["status_code"], ("VALIDATED", "NEEDS_REVIEW"))
        self.assertGreaterEqual(report["evidence_coverage"], 0.80)
        self.assertGreater(report["final_confidence_score"], 0.60)
        self.assertTrue(report["stage_breakdown"]["stage1_phonetic_chain"]["is_valid"])

    # --- Aşama 2: anakronizm ---------------------------------------------

    def test_anachronism_is_rejected(self):
        """Kaynak dil teması tanıklamadan SONRAYSA hipotez reddedilmelidir."""
        report = self.protocol.validate_hypothesis(
            "su",
            {"origin_form": "sous", "donor_language": "Fransızca",
             "historical_meaning": "su", "modern_meaning": "su"},
            attestation_record={"verified": True, "first_attestation_year": 735},
            turkic_entries=self.goz_entries,
        )
        self.assertEqual(report["status_code"], "REJECTED")
        self.assertFalse(report["stage_breakdown"]["stage2_time_lock"]["is_valid"])
        self.assertTrue(any("ANAKRONİZM" in r for r in report["rejection_reasons"]))

    def test_time_lock_without_dates_yields_no_evidence(self):
        """Tarih bilinmiyorsa aşama 1.0 DEĞİL, 'kanıt yok' döndürmelidir."""
        res = self.time_lock.verify("", None)
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["score"])
        self.assertIsNone(res["is_valid"])

    def test_year_parser_ignores_page_numbers(self):
        """Sayfa ve cilt numaraları yıl sanılmamalıdır."""
        self.assertIsNone(self.time_lock.parse_year_or_century("Clauson EDPT s. 456"))
        self.assertIsNone(self.time_lock.parse_year_or_century("cilt 2 sayfa 130"))
        self.assertEqual(self.time_lock.parse_year_or_century("735 yılı Orhun"), 735)
        self.assertEqual(self.time_lock.parse_year_or_century("11. yüzyıl"), 1050)
        self.assertEqual(self.time_lock.parse_year_or_century("Cumhuriyet dönemi 1935"), 1935)

    def test_donor_contact_periods(self):
        """Donör dillerin temas dönemleri makul sırada olmalıdır."""
        self.assertLess(
            self.time_lock.donor_contact_year("Arapça"),
            self.time_lock.donor_contact_year("Fransızca"),
        )
        self.assertIsNone(self.time_lock.donor_contact_year("Klingonca"))

    # --- Aşama 1: fonetik zincir -----------------------------------------

    def test_broken_phonetic_chain_is_invalid(self):
        """Alâkasız biçimler arasında geçerli ses zinciri bulunmamalıdır."""
        result = self.phonetic_verifier.verify("xyzq", "göz")
        self.assertTrue(result["evidence_available"])
        self.assertFalse(result["is_valid"])
        self.assertLess(result["score"], 0.50)
        self.assertTrue(result["violations"])

    def test_regular_correspondence_is_valid(self):
        """Düzenli ses denklikleri (*köŕ ~ göz) geçerli sayılmalıdır."""
        result = self.phonetic_verifier.verify("*köŕ", "göz")
        self.assertTrue(result["is_valid"])
        self.assertGreater(result["score"], 0.55)

    def test_phonetic_verifier_without_input_has_no_evidence(self):
        res = self.phonetic_verifier.verify("", "")
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["score"])

    # --- Aşama 4: triangulation ------------------------------------------

    def test_triangulation_uses_real_entries_only(self):
        """Aşama 4 GERÇEK kayıtları saymalı, üretilmiş varyantları değil.

        Regresyon: eski sürüm kendi ürettiği transkripsiyonları sayıp
        her kelimeye sabit 0.95 veriyordu.
        """
        empty = self.triangulator.verify("zzzqx", [])
        self.assertFalse(empty["evidence_available"])
        self.assertIsNone(empty["score"])

        rich = self.triangulator.verify("göz", self.goz_entries)
        self.assertTrue(rich["evidence_available"])
        self.assertEqual(rich["cognate_count"], 8)
        self.assertGreater(rich["score"], 0.5)

    def test_triangulation_ignores_pseudo_language_codes(self):
        """'donor' gibi sözde kodlar lehçe sayısına dâhil edilmemelidir."""
        entries = [
            {"lang_code": "tr", "word": "göz", "source": "TDK", "origin": "live"},
            {"lang_code": "donor", "word": "X", "source": "TDK", "origin": "live"},
            {"lang_code": "ai", "word": "Y", "source": "TDK", "origin": "live"},
        ]
        res = self.triangulator.verify("göz", entries)
        self.assertEqual(res["cognate_count"], 1)

    # --- Aşama 3: semantik -----------------------------------------------

    def test_semantic_stage_without_data_has_no_evidence(self):
        res = self.semantic_evaluator.verify("", "")
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["score"])


if __name__ == "__main__":
    unittest.main()


class TestHistoricalGlossCrossReference(unittest.TestCase):
    def test_reference_skipped_and_parenthesised_form_matched(self):
        """terlik: "bk. derlik" (mesafe 0) değil, "derlik (terlik)" tanımı seçilmeli."""
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        entries = [
            {"lang_code": "otk", "word": "ter", "meaning": "ter"},
            {"lang_code": "otk", "word": "terlik", "meaning": "bk. derlik"},
            {"lang_code": "otk", "word": "derlik (terlik)", "meaning": "Üstten giyilen ince elbise."},
        ]
        self.assertEqual(_historical_gloss(entries, "terlik"), "Üstten giyilen ince elbise")


class TestAuditConsistency(unittest.TestCase):
    """105 kelimelik tutarlılık denetiminin (seed 99) A-HVP bulguları."""

    def test_sample_cognates_exclude_donor_chain(self):
        """master: numune akraba satırında `maistre, magister` görünüyordu."""
        entries = [
            {"lang_code": "donor", "word": "maistre", "source": "a"},
            {"lang_code": "donor", "word": "magister", "source": "a"},
            {"lang_code": "kk", "word": "мастер", "source": "b"},
            {"lang_code": "ky", "word": "мастер", "source": "c"},
        ]
        sample = CrossCognateTriangulator().verify("master", entries)["sample_cognates"]
        self.assertNotIn("maistre", sample)
        self.assertNotIn("magister", sample)
        self.assertIn("мастер", sample)

    def test_attested_root_is_tested_not_engine_reconstruction(self):
        """uçmak: başlık *uč- (kaynak), A-HVP *uça'yı sınıyordu."""
        from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine

        recon = {"evidence_available": True, "reconstructed_root": "*uça"}
        hyp = IterativeHypothesisEngine._select_hypothesis(
            "uçmak", {}, None, None, recon, attested_root="*uč-"
        )
        self.assertEqual(hyp["origin_form"], "*uč-")
        self.assertEqual(hyp["engine_reconstruction"], "*uça")
        self.assertTrue(hyp["origin_form_attested"])
        # Tanıklı kök yoksa motorun rekonstrüksiyonu sınanır.
        hyp = IterativeHypothesisEngine._select_hypothesis("uçmak", {}, None, None, recon)
        self.assertEqual(hyp["origin_form"], "*uça")

    def test_attested_root_is_not_first_fetcher_root(self):
        """açıkgöz: EtimolojiTürkçe'nin ETü tabanı (*açuk) tanıklı kök değildir."""
        from unittest import mock

        from engine.nlp.iterative_hypothesis_engine import _attested_proto_root

        starling = {"root": {"proto_turkic": "*uč-", "starling_proto": "*uč-"}}
        self.assertEqual(_attested_proto_root("açıkgöz", {"proto_turkic": "*açuk"}, []), "")
        # Çağıran Starling sonucunu getirmediyse veritabanına gidilmez.
        self.assertEqual(_attested_proto_root("uçmak", {}, [], []), "")
        with mock.patch("engine.fetchers.starling.StarlingFetcher.fetch", return_value=starling):
            self.assertEqual(_attested_proto_root("uçmak", {}, [], [starling]), "*uč-")
            # Kök varyantının (`kulluk` -> `kul`) Starling kökü sorgunun değildir.
            kul = {"root": {"proto_turkic": "*Kul", "starling_proto": "*Kul"}}
            self.assertEqual(_attested_proto_root("kulluk", {}, [], [kul]), "")


class TestAttestedRootReachesTimeLock(unittest.TestCase):
    """B1: kaynağın verdiği kök, karşılaştırmalı yöntem için tanık yetmese de
    A-HVP'de sınanır; tarihli tanık 2. aşamaya (zaman kilidi) ulaşır."""

    def test_attested_root_without_reconstruction_forms_hypothesis(self):
        from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine

        recon = {"evidence_available": False, "reconstructed_root": "*bak", "witness_count": 0}
        hyp = IterativeHypothesisEngine._select_hypothesis(
            "bak", {}, None, None, recon, attested_root="*bạk-"
        )
        self.assertIsNotNone(hyp)
        self.assertEqual(hyp["origin_form"], "*bạk-")
        self.assertEqual(hyp["donor_language"], "Proto-Türkçe")
        self.assertEqual(hyp["evidence_kind"], "attested_root")
        # Kaynak kökü de yoksa hipotez UYDURULMAZ.
        self.assertIsNone(IterativeHypothesisEngine._select_hypothesis("bak", {}, None, None, recon))

    def test_starling_year_reaches_stage2(self):
        from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine

        starling = {
            "root": {"proto_turkic": "*bạk-", "starling_proto": "*bạk-"},
            "first_attestation": {"form": "baq-", "source": "KB (Starling #890)", "year": 1069},
            "turkic_languages": [],
        }
        entries = [{"lang_code": "tr", "word": "bak", "source": "Yerel", "origin": "local"}]
        from unittest import mock

        with mock.patch("engine.fetchers.starling.StarlingFetcher.fetch", return_value=starling):
            out = IterativeHypothesisEngine().prove_etymological_hypothesis(
                "bak", {"root": {}}, entries, [starling]
            )
        stage2 = out["validation_report"]["stage_breakdown"]["stage2_time_lock"]
        self.assertEqual(stage2["attestation_year"], 1069)


class TestBadgeIsNetworkIndependent(unittest.TestCase):
    """B2: canlı kaynağın o an yanıt vermesi kanıt değildir."""

    def _report(self, entries):
        return HypothesisValidationProtocol().validate_hypothesis(
            "göz",
            {"origin_form": "*köŕ", "donor_language": "Proto-Türkçe",
             "historical_meaning": "", "modern_meaning": ""},
            attestation_record={"verified": True, "first_attestation_year": 1074},
            turkic_entries=entries,
        )

    def test_same_badge_with_and_without_live_sources(self):
        local = [
            {"lang_code": c, "word": f, "source": s, "origin": "local"}
            for c, f, s in (("tr", "göz", "Vikisözlük"), ("az", "göz", "Vikisözlük"),
                            ("kk", "көз", "Apertium"), ("tt", "күз", "NorthEuraLex"))
        ]
        # TDK ve Nişanyan yanıt verdi: sorgunun kendi Türkçe maddesi.
        live = [
            {"lang_code": "tr", "word": "göz", "source": "TDK", "origin": "live"},
            {"lang_code": "tr", "word": "göz", "source": "Nişanyan", "origin": "live"},
        ]
        off, on = self._report(local), self._report(local + live)
        self.assertEqual(off["badge"], on["badge"])
        self.assertEqual(off["final_confidence_score"], on["final_confidence_score"])
        tri_off = off["stage_breakdown"]["stage4_cognate_triangulation"]
        tri_on = on["stage_breakdown"]["stage4_cognate_triangulation"]
        self.assertEqual(tri_off["score"], tri_on["score"])
