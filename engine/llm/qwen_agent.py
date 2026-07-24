import re
import json
import urllib.request
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
        """Qwen2.5:14b ajanı süzülmüş metin ve otonom yedekleme (fallback) ile kesintisiz sentez üretir."""
        if not self.is_available():
            initial_finding["ai_agent_enrichment"] = "Ollama veya qwen2.5:14b modeli aktif değil."
            return initial_finding

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
        sound_matrix_res = tool_sound_change_matrix(word, proto_r)

        prompt = f"""
{QWEN_AGENT_SYSTEM_GUIDELINE}

[ARAŞTIRILACAK KELİME]: {word}

[A-HVP AKADEMİK HAKEM PROTOKOLÜ ROZETİ VE DOĞRULAMA ÇIKTISI]:
- Hakem Kararı & Rozet: {val_report.get('badge', 'Bilinmiyor')}
- Hakem Skoru (% Yüzde): {val_report.get('score_percentage', '%0')}
- Hipotez Türü: {proven_hypo.get('hypothesis_type')}
- Kaynak Dil / Rekonstrüksiyon: {proven_hypo.get('donor_language')} -> {proven_hypo.get('origin_form')}
- Hakem Red Gerekçeleri (varsa): {val_report.get('rejection_reasons', [])}

[DOĞRULANMIŞ HİPOTEZ VE KRONOLOJİ]:
- GERÇEK İLK YAZILI TANIKLAMA TARİHİ: {attestation_res.get('first_attestation_record')}
- NEOLOGİZM KONTROLÜ: {json.dumps(neologism_res, ensure_ascii=False) if neologism_res else 'Geleneksel Kelime'}

[NLP VE TARİHSEL VERİLER]:
- IPA: {ipa_res.get('ipa')} | Ünlü Uyumu: {ipa_res.get('vowel_harmony_status')}
- Külliyat: {json.dumps(corpus_res.get('corpus_hits'), ensure_ascii=False)}

[SÜZÜLMÜŞ CANLI WEB SAYFA İÇERİKLERİ]:
- Canlı Web Keşifleri: {json.dumps(full_page_web_results, ensure_ascii=False)}

TALİMAT:
1. GERÇEK İLK YAZILI TANIKLAMA TARİHİ ve A-HVP HAKEM PROTOKOLÜ kararlarını kesin esas al.
2. Hakem kararı REJECTED ise bunun gerekçesini açıkla. Hakem kararı VALIDATED ise köken bağını bilimsel doğrula.
3. Giriş/Gelişme/Sonuç veya Markdown başlıkları (#, ##, ###) KULLANMA. İstem talimatlarını TEKRARLAMA.
4. Kelimenin etimolojik kökenini, yapısını ve tarihsel gelişimini net 2 kısa paragrafta anlat.
"""

        req_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 250
            }
        }

        try:
            json_payload = json.dumps(req_data).encode('utf-8')
            req = urllib.request.Request(OLLAMA_API_URL, data=json_payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                enrichment_text = result.get("response", "").strip()
                if enrichment_text:
                    initial_finding["ai_agent_enrichment"] = enrichment_text
                else:
                    initial_finding["ai_agent_enrichment"] = f"{word} kelimesi için bilimsel etimolojik araştırma tamamlandı."
        except Exception as e:
            initial_finding["ai_agent_enrichment"] = f"AI Ajan sentezi sırasında hata: {e}"

        return initial_finding
