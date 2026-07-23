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
        """Qwen2.5:14b ajanı İleri Düzey Araçları ve Soyut Dilbilimsel Eleştirel Yargılama Protokolünü çalıştırır."""
        if not self.is_available():
            initial_finding["ai_agent_enrichment"] = "Ollama veya qwen2.5:14b modeli aktif değil."
            return initial_finding

        # --- 1. İLERİ DÜZEY ARAÇLARI ÇALIŞTIR ---
        wiktionary_res = tool_wiktionary_multilingual_api(word)
        ipa_res = tool_ipa_phonetic_analyzer(word)
        donor_pattern_res = tool_donor_pattern_analyzer(word)
        corpus_res = tool_historical_corpus_search(word)
        attestation_verify = tool_verify_attestation(word)
        phonotactics_analysis = tool_analyze_phonotactics(word)
        suffixes_analysis = tool_extract_suffixes(word)
        web_results = tool_web_search(word)

        nlp_analysis = initial_finding.get("nlp_analysis", {})
        proven_hypo = nlp_analysis.get("proven_hypothesis", {})

        proto_r = initial_finding.get('root', {}).get('proto_turkic', word)
        sound_matrix_res = tool_sound_change_matrix(word, proto_r)

        # 2. Qwen2.5:14b İçin Soyut Eleştirel Yargılama Prompt'u Hazırla
        prompt = f"""
{QWEN_AGENT_SYSTEM_GUIDELINE}

[ARAŞTIRILACAK KELİME]: {word}

[BİLİMSEL KANITLANMIŞ HİPOTEZ VE KÖKEN]:
- Hipotez Tipi: {proven_hypo.get('hypothesis_type')}
- Kaynak Dil / Rekonstrüksiyon: {proven_hypo.get('donor_language')} -> {proven_hypo.get('origin_form')}
- Kanıt Özeti: {proven_hypo.get('proof_summary')}
- Anlamı: {proven_hypo.get('historical_meaning')}

[BİLİMSEL VE NLP ARAÇ ÇIKTILARI]:
1. IPA Fonetik Analiz: IPA={ipa_res.get('ipa')}, Ünlü Uyumu={ipa_res.get('vowel_harmony_status')}
2. Kaynak Dil Vezin/Yapı: {json.dumps(donor_pattern_res.get('detected_donor_patterns'), ensure_ascii=False)}
3. Tarihsel Külliyat Taraması: {json.dumps(corpus_res.get('corpus_hits'), ensure_ascii=False)}
4. Morfolojik Ek/Kök Tespiti: {suffixes_analysis}
5. Canlı Web Taraması Bulguları: {json.dumps(web_results[:3], ensure_ascii=False)}

YÖNERGE PROTOKOLÜ:
Tüm verileri Soyut Eleştirel Kanıt Protokolüne (Ses kayması mantığı, tarihsel kültür/ticaret uyumu, halk etimolojisi eleştirisi) göre değerlendir.
Yüzeysel harf benzerliği taşıyan uydurma iddiaları elerken, tarihsel ve fonetik açıdan en güçlü bilimsel kanıtı seçerek raporunu yaz.
"""

        req_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": 512,
                "temperature": 0.2
            }
        }

        try:
            req = urllib.request.Request(
                OLLAMA_API_URL,
                data=json.dumps(req_data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                analysis = res.get("response", "").strip()

                initial_finding["ai_agent_enrichment"] = analysis
                initial_finding["discovered_web_sources"] = web_results
                
                if analysis:
                    initial_finding["sources"].append(f"Qwen2.5:14b İleri Düzey Bilimsel Ajanı ({self.model_name})")
                    initial_finding["turkic_languages"].append({
                        "lang_code": "ai",
                        "lang_name": "Qwen2.5:14b İleri Düzey Bilimsel Ajan Sentezi",
                        "word": word,
                        "meaning": analysis,
                        "script": "Latin"
                    })
        except Exception as e:
            initial_finding["ai_agent_enrichment"] = f"Qwen2.5:14b analiz hatası: {e}"

        return initial_finding
