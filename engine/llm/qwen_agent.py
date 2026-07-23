import json
import urllib.request
from typing import Dict, Any, List

from engine.llm.agent_guideline import QWEN_AGENT_SYSTEM_GUIDELINE
from engine.llm.research_tools import (
    tool_verify_attestation,
    tool_verify_sound_law,
    tool_verify_donor_language,
    tool_analyze_phonotactics,
    tool_extract_suffixes,
    tool_align_cognates,
    tool_donor_nearest_neighbor,
    tool_web_search,
    tool_web_scrape_url,
    tool_search_academic
)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"

class QwenEtymologyAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

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
        """Qwen2.5:14b ajanı Yönergeye (System Guideline) uyarak Doğrulama ve Keşif Protokollerini çalıştırır."""
        if not self.is_available():
            initial_finding["ai_agent_enrichment"] = "Ollama veya qwen2.5:14b modeli aktif değil."
            return initial_finding

        # 1. Bilimsel ve NLP Araçlarını Çalıştır (Tool Execution Phase)
        attestation_verify = tool_verify_attestation(word)
        donor_verify = tool_verify_donor_language(word)
        phonotactics_analysis = tool_analyze_phonotactics(word)
        suffixes_analysis = tool_extract_suffixes(word)
        cognates_alignment = tool_align_cognates(word)
        donor_neighbor = tool_donor_nearest_neighbor(word)
        web_results = tool_web_search(word)
        academic_results = tool_search_academic(word)

        scraped_content = ""
        if web_results:
            first_url = web_results[0].get("url", "")
            if first_url and first_url.startswith("http"):
                scraped_content = tool_web_scrape_url(first_url)

        # 2. Qwen2.5:14b İçin Yönerge Destekli Derinleştirme Prompt'u Hazırla
        prompt = f"""
{QWEN_AGENT_SYSTEM_GUIDELINE}

[ARAŞTIRILACAK KELİME]: {word}

[İLK STATİK BULGULAR]:
- Kök / Rekonstrüksiyon: {initial_finding.get('root', {}).get('proto_turkic')}
- Anlam: {initial_finding.get('root', {}).get('meaning')}
- Morfoloji: {initial_finding.get('morphology')}
- Bulunan Türki Dil Sayısı: {len(initial_finding.get('turkic_languages', []))}

[BİLİMSEL VE NLP ARAÇLARI ÇIKTILARI (TOOL RESULTS)]:
1. Tarihsel Tanıklama [tool_verify_attestation]: {attestation_verify}
2. Alıntı & Kaynak Dil Doğrulama [tool_verify_donor_language]: {donor_verify}
3. Fonotaktik İhlal Analizi [tool_analyze_phonotactics]: {phonotactics_analysis}
4. Ek & Kök Ayrıştırma [tool_extract_suffixes]: {suffixes_analysis}
5. Çapraz Hizalama Skoru [tool_align_cognates]: {cognates_alignment}
6. Komşu Dil Eşleşmesi [tool_donor_nearest_neighbor]: {donor_neighbor}
7. Canlı Web & Akademik Keşif [tool_web_search]: {json.dumps(web_results[:2], ensure_ascii=False)} | {academic_results}
8. Keşfedilen Metin [tool_web_scrape_url]: {scraped_content[:300]}...

Yönerge protokollerine göre kelimenin etimolojisini doğrula veya otonom keşfet. Sonucunu akademik, tutarlı ve akıcı bir Türkçe sentez paragrafı olarak yaz.
"""

        req_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            req = urllib.request.Request(
                OLLAMA_API_URL,
                data=json.dumps(req_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                analysis = res.get("response", "").strip()

                initial_finding["ai_agent_enrichment"] = analysis
                initial_finding["discovered_web_sources"] = web_results
                
                if analysis:
                    initial_finding["sources"].append(f"Qwen2.5:14b Yönergeli Otonom Araştırma Ajanı ({self.model_name})")
                    initial_finding["turkic_languages"].append({
                        "lang_code": "ai",
                        "lang_name": "Qwen2.5:14b Bilimsel Ajan Yönergesi Sentezi",
                        "word": word,
                        "meaning": analysis,
                        "script": "Latin"
                    })
        except Exception as e:
            initial_finding["ai_agent_enrichment"] = f"Qwen2.5:14b analiz hatası: {e}"

        return initial_finding
