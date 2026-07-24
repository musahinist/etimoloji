"""
Yapay Zeka Hipotez Doğrulama ve Hakemlik Protokolü (AI Hypothesis Validation Protocol - A-HVP)
Kelime etimolojisi hipotezlerini 5 aşamalı akademik süzgeçten geçirir:
1. Fonetik Evrim & IPA Kural Kontrolü (No Broken Phonetic Chain & LingPy Sequence Alignment)
2. Kronolojik Zaman Kilidi (Anachronism Lock: T_source < T_target)
3. Diyakronik Semantik Vektör & Yörünge Analizi (d^2S/dt^2 < theta)
4. Çapraz Akraba Kelime Triangulation (Cognates)
5. Hakem Raporu & Güven Skoru Basımı (%85+ Validated, <50% Rejected)
"""

from typing import Dict, Any, List, Optional
import re
from engine.utils.phonetic_rules import verify_phonetic_chain
from engine.utils.sound_shifts import generate_turkic_cognate_candidates
from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine

class PhoneticChainVerifier:
    """A-HVP Aşama 1: Fonetik Evrim ve LingPy Ses Hizalama Kontrolü"""
    def __init__(self):
        self.lingpy_aligner = CldfLingPyAligner()

    def verify(self, origin_form: str, word: str) -> Dict[str, Any]:
        clean_origin = (origin_form or "").strip().lstrip("*")
        if not clean_origin:
            clean_origin = word

        base_verification = verify_phonetic_chain(clean_origin, word)
        align_res = self.lingpy_aligner.align_sequences(clean_origin, word)

        # Base verification score ile LingPy hizalama skorunun sentezi
        combined_score = round((base_verification["score"] * 0.6) + (align_res["phonetic_similarity"] * 0.4), 3)

        return {
            "is_valid": base_verification["is_valid"] and (align_res["phonetic_similarity"] > 0.15 or clean_origin == word),
            "score": combined_score,
            "violations": base_verification["violations"],
            "matched_rules": base_verification["matched_rules"],
            "alignment_details": align_res
        }

class ChronologicalTimeLock:
    """A-HVP Aşama 2: Kronolojik Zaman Kilidi Testi (Anachronism Lock)"""
    def parse_year_or_century(self, text: str) -> Optional[int]:
        if not text:
            return None
        text_lower = text.lower()
        
        is_bc = "m.ö" in text_lower or "mö" in text_lower or "bc" in text_lower
        
        year_match = re.search(r'\b(1\d{3}|20\d{2}|\d{3})\b', text)
        if year_match:
            year = int(year_match.group(1))
            return -year if is_bc else year
            
        century_match = re.search(r'(\d{1,2})\.?\s*(yy|yüzyıl|century)', text_lower)
        if century_match:
            century = int(century_match.group(1))
            approx_year = (century - 1) * 100 + 50
            return -approx_year if is_bc else approx_year

        if "orhun" in text_lower or "köktürk" in text_lower:
            return 735
        if "divan" in text_lower or "kaşgarlı" in text_lower:
            return 1074
        if "cumhuriyet" in text_lower or "tdk" in text_lower:
            return 1935
        if "fransızca" in text_lower or "latin" in text_lower or "batı" in text_lower:
            return 1850

        return None

    def verify(self, source_period: str, target_attestation: str) -> Dict[str, Any]:
        t_source = self.parse_year_or_century(source_period)
        t_target = self.parse_year_or_century(target_attestation)

        if t_source is not None and t_target is not None:
            if t_source > t_target:
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "violation": f"ANAKRONİZM ENGELİ: Kaynak dönemi ({source_period} -> ~{t_source}) hedef kelimenin ilk tanıklanma tarihinden ({target_attestation} -> ~{t_target}) daha sonradır (T_kaynak > T_hedef)."
                }
        
        return {
            "is_valid": True,
            "score": 1.0,
            "violation": None
        }

class SemanticDriftEvaluator:
    """A-HVP Aşama 3: Diyakronik Semantik Vektör ve Yörünge Kontrolü"""
    def __init__(self):
        self.semantic_engine = DiachronicSemanticEngine()

    def verify(self, source_meaning: str, target_meaning: str) -> Dict[str, Any]:
        s_m = (source_meaning or "").lower()
        t_m = (target_meaning or "").lower()

        if not s_m or not t_m:
            return {"is_valid": True, "score": 0.85, "reason": "Semantik veri tam değil, varsayılan uyum kabul edildi."}

        trajectory_res = self.semantic_engine.evaluate_diachronic_trajectory(s_m, t_m)

        return {
            "is_valid": trajectory_res["is_plausible"],
            "score": max(0.20, round(1.0 - trajectory_res["total_shift_distance"], 3)),
            "reason": trajectory_res["reason"],
            "trajectory_details": trajectory_res
        }

class CrossCognateTriangulator:
    """A-HVP Aşama 4: Çapraz Akraba Kelime Sorgusu (Triangulation)"""
    def verify(self, word: str) -> Dict[str, Any]:
        cognates = generate_turkic_cognate_candidates(word)
        has_dialectic_matches = len(cognates) > 1

        score = 0.95 if has_dialectic_matches else 0.60
        return {
            "is_valid": True,
            "score": score,
            "cognate_count": len(cognates),
            "sample_cognates": cognates[:6]
        }

class HypothesisValidationProtocol:
    """A-HVP Master Doğrulama ve Otomatik Hakemlik Mimarisi"""
    def __init__(self):
        self.phonetic_verifier = PhoneticChainVerifier()
        self.time_lock = ChronologicalTimeLock()
        self.semantic_evaluator = SemanticDriftEvaluator()
        self.cognate_triangulator = CrossCognateTriangulator()

    def validate_hypothesis(self, word: str, hypothesis: Dict[str, Any], attestation_record: Dict[str, Any]) -> Dict[str, Any]:
        raw_origin = hypothesis.get("origin_form") or word
        clean_origin = raw_origin.strip().lstrip("*")
        if not clean_origin:
            clean_origin = word
        origin_form = f"*{clean_origin}" if hypothesis.get("donor_language") == "Proto-Türkçe" else clean_origin

        donor_lang = hypothesis.get("donor_language", "")
        proof_summary = hypothesis.get("proof_summary", "")
        hist_meaning = hypothesis.get("historical_meaning", "")
        first_attestation = attestation_record.get("first_attestation_record", "")

        # 1. Fonetik Halka & LingPy Hizalama Kontrolü
        phonetic_res = self.phonetic_verifier.verify(origin_form, word)

        # 2. Kronolojik Zaman Kilidi Testi
        time_res = self.time_lock.verify(donor_lang + " " + proof_summary, first_attestation)

        # 3. Diyakronik Semantik Kontrol
        semantic_res = self.semantic_evaluator.verify(hist_meaning, proof_summary)

        # 4. Çapraz Akraba Triangulation
        cognate_res = self.cognate_triangulator.verify(word)

        # Ağırlıklı Güven Skoru Hesabı
        p_score = phonetic_res["score"]
        t_score = time_res["score"]
        s_score = semantic_res["score"]
        c_score = cognate_res["score"]

        total_score = (p_score * 0.35) + (t_score * 0.30) + (s_score * 0.15) + (c_score * 0.20)

        # Otomatik Red Şartları
        rejections = []
        if not phonetic_res["is_valid"]:
            rejections.extend(phonetic_res["violations"])
            total_score *= 0.4 # Fonetik halka kırıksa skoru düşür
        if not time_res["is_valid"]:
            rejections.append(time_res["violation"])
            total_score = min(total_score, 0.30) # Anakronizm varsa skor en fazla %30 olabilir

        # Hakem Kararı ve Rozet
        if total_score >= 0.75 and not rejections:
            status_code = "VALIDATED"
            badge = "🟢 VALIDATED (Bilimsel Hakem Onaylı)"
        elif total_score >= 0.50 and not rejections:
            status_code = "NEEDS_REVIEW"
            badge = "🟡 NEEDS REVIEW (İnceleme Gerekli Hipotez)"
        else:
            status_code = "REJECTED"
            badge = "🔴 REJECTED (Akademik Red / Anakronizm veya Fonetik İhlal)"

        referee_report = {
            "status_code": status_code,
            "badge": badge,
            "final_confidence_score": round(total_score, 3),
            "score_percentage": f"%{round(total_score * 100, 1)}",
            "stage_breakdown": {
                "stage1_phonetic_chain": {
                    "is_valid": phonetic_res["is_valid"],
                    "matched_rules": phonetic_res["matched_rules"],
                    "alignment_details": phonetic_res.get("alignment_details", {}),
                    "violations": phonetic_res.get("violations", [])
                },
                "stage2_time_lock": {
                    "is_valid": time_res["is_valid"],
                    "violation": time_res.get("violation")
                },
                "stage3_semantic_drift": {
                    "is_valid": semantic_res["is_valid"],
                    "reason": semantic_res.get("reason"),
                    "trajectory_details": semantic_res.get("trajectory_details", {})
                },
                "stage4_cognate_triangulation": {
                    "is_valid": cognate_res["is_valid"],
                    "sample_cognates": cognate_res.get("sample_cognates", [])
                }
            },
            "rejection_reasons": rejections
        }

        return referee_report
