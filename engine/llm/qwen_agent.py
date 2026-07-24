import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List

from engine.llm.agent_guideline import QWEN_AGENT_SYSTEM_GUIDELINE
from engine.llm.advanced_tools import (
    tool_wiktionary_multilingual_api,
    tool_ipa_phonetic_analyzer,
    tool_donor_pattern_analyzer,
    tool_sound_change_matrix,
    tool_historical_corpus_search
)
from engine.llm.research_tools import (
    tool_verify_attestation,
    tool_analyze_phonotactics,
    tool_extract_suffixes,
    tool_web_search
)
from engine.nlp.full_web_scraper import scrape_full_web_pages_for_results
from engine.nlp.neologism_detector import NeologismDetector
from engine.nlp.historical_attestation_verifier import HistoricalAttestationVerifier

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

def clean_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', clean).strip()

class QwenEtymologyAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.neologism_detector = NeologismDetector()
        self.attestation_verifier = HistoricalAttestationVerifier()

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = [m.get("name") for m in data.get("models", [])]
                return any(self.model_name in m for m in models)
        except Exception:
            return False

    def research_and_enrich(self, word: str, initial_finding: Dict[str, Any]) -> Dict[str, Any]:
        """Qwen2.5:14b ajanı süzülmüş metin ve otonom yedekleme (fail-safe fallback) ile kesintisiz sentez üretir."""
        ipa_res = tool_ipa_phonetic_analyzer(word)
        donor_pattern_res = tool_donor_pattern_analyzer(word)
        corpus_res = tool_historical_corpus_search(word)
        suffixes_analysis = tool_extract_suffixes(word)
        
        neologism_res = self.neologism_detector.detect(word)
        attestation_res = self.attestation_verifier.verify_attestation(word, initial_finding.get("turkic_languages"))

        raw_web_results = tool_web_search(word)
        full_page_web_results = scrape_full_web_pages_for_results(raw_web_results, max_pages=2)

        nlp_analysis = initial_finding.get("nlp_analysis", {})
        proven_hypo = nlp_analysis.get("proven_hypothesis", {}) or {}
        val_report = proven_hypo.get("validation_report", {}) or {}

        proto_r = initial_finding.get('root', {}).get('proto_turkic', word)
        root_meaning = initial_finding.get('root', {}).get('meaning', '')
        turkic_entries = initial_finding.get('turkic_languages', [])

        entries_summary = []
        for e in turkic_entries[:6]:
            lname = e.get("lang_name", "")
            w_form = e.get("word", "")
            m_text = e.get("meaning", "")
            if m_text and not m_text.startswith("Online"):
                entries_summary.append(f"{lname} ({w_form}): {m_text}")

        # Fail-safe sentez paragrafı hazırlığı
        donor_info = nlp_analysis.get("donor_matching", {}) or {}
        donor_lang = proven_hypo.get("donor_language") or donor_info.get("donor_language", "")
        origin_form = proven_hypo.get("origin_form") or donor_info.get("origin_form", "")

        # Somut Donör ve Etimoloji Açıklama Fallback Metni
        if donor_lang and donor_lang != "Proto-Türkçe":
            fallback_text = (
                f"'{word}' kelimesi etimolojik açıdan {donor_lang} kaynaklı ({origin_form}) bir alıntıdır. "
                f"Asıl anlamı ve türeyişi {proven_hypo.get('proof_summary', 'komşu dil temasları')} çerçevesinde gelişmiş "
                f"ve Türkçe ağızlarına diyalekt teması ile geçmiştir. "
                f"Tarihsel kronolojide {attestation_res.get('first_attestation_record')} kaydıyla belgelenmektedir."
            )
        else:
            fallback_text = (
                f"'{word}' kelimesi etimolojik açıdan Proto-Türkçe (*{proto_r}) köküne dayanmaktadır. "
                f"Anlamı '{root_meaning}' şeklinde tespit edilmiştir. "
                f"Tarihsel kronolojide {attestation_res.get('first_attestation_record')} kaydıyla belgelenmektedir."
            )

        if not self.is_available():
            initial_finding["ai_agent_enrichment"] = fallback_text
            initial_finding["discovered_web_sources"] = raw_web_results
            return initial_finding

        prompt = f"""
{QWEN_AGENT_SYSTEM_GUIDELINE}

[ARAŞTIRILACAK KELİME]: {word}

[DONÖR DİL VE KÖKEN KARTI (SOMUT ETIMOLOJİ KANITLARI)]:
- Donör Kaynak Dil: {donor_lang}
- Orijinal Kök / İmla: {origin_form}
- Etimolojik İnceleme & Geçiş Yörüngesi: {proven_hypo.get('proof_summary', donor_info.get('donor_meaning', 'Diyalekt Teması'))}

[SÖZLÜK VE AKADEMİK VERİ KATMANI (KESİN ANLAM KANITLARI)]:
- Tespit Edilen Ana Anlam: {root_meaning if root_meaning else 'Yerel/Ağız Anlamı'}
- Gerçek Sözlük Maddeleri ve Anlamları: {json.dumps(entries_summary, ensure_ascii=False) if entries_summary else 'Sözlük kaydı'}

[A-HVP AKADEMİK HAKEM PROTOKOLÜ ROZETİ VE DOĞRULAMA ÇIKTISI]:
- Hakem Kararı & Rozet: {val_report.get('badge', 'Bilinmiyor')}
- Hakem Skoru (% Yüzde): {val_report.get('score_percentage', '%0')}
- Hipotez Türü: {proven_hypo.get('hypothesis_type')}

[DOĞRULANMIŞ HİPOTEZ VE KRONOLOJİ]:
- GERÇEK İLK YAZILI TANIKLAMA TARİHİ: {attestation_res.get('first_attestation_record')}

TALİMAT:
1. "Etimolojik kökeni komşu dil alıntılarından kaynaklanmaktadır", "IPA ünlü uyumu gösterir" gibi JENERİK BOŞ LAFLARI KESİNLİKLE YAZMA.
2. Varsa donör dili ({donor_lang}) ve orijinal kökü ({origin_form}) açıkça söyleyerek kelimenin Türkçeye ve bölge ağızlarına nasıl geçtiğini net anlat.
3. Giriş/Gelişme/Sonuç veya Markdown başlıkları (#, ##, ###) KULLANMA. İstem talimatlarını TEKRARLAMA.
"""

        req_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 1024,
                "num_predict": 250,
                "temperature": 0.15
            }
        }

        try:
            json_payload = json.dumps(req_data).encode('utf-8')
            req = urllib.request.Request(OLLAMA_API_URL, data=json_payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                enrichment_text = result.get("response", "").strip()
                if enrichment_text:
                    initial_finding["ai_agent_enrichment"] = enrichment_text
                else:
                    initial_finding["ai_agent_enrichment"] = fallback_text

                initial_finding["discovered_web_sources"] = raw_web_results
        except Exception:
            initial_finding["ai_agent_enrichment"] = fallback_text
            initial_finding["discovered_web_sources"] = raw_web_results

        return initial_finding
