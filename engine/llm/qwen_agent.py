import json
import urllib.request
from typing import Dict, Any, List

from engine.llm.research_tools import (
    tool_web_search,
    tool_web_scrape_url,
    tool_search_academic,
    tool_search_archive_books
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
        """Qwen2.5:14b ajanı ilk bulguları inceler, otonom araçlarla web araması ve araştırması yaparak bulguyu derinleştirir."""
        if not self.is_available():
            initial_finding["ai_agent_enrichment"] = "Ollama veya qwen2.5:14b modeli aktif değil."
            return initial_finding

        # 1. Otonom Araçları Çalıştır (Web Araması + Akademik + Kitap Taraması)
        web_search_results = tool_web_search(word)
        academic_results = tool_search_academic(word)
        archive_results = tool_search_archive_books(word)

        scraped_content = ""
        if web_search_results:
            first_url = web_search_results[0].get("url", "")
            if first_url and first_url.startswith("http"):
                scraped_content = tool_web_scrape_url(first_url)

        # 2. Qwen2.5:14b İçin Derinleştirme Prompt'u Hazırla
        prompt = f"""
Sen dünya çapında uzman bir Türki Diller Etimoloji ve Dilbilim Ajanısın (Qwen2.5:14b).
Aşağıda '{word}' kelimesi için internetten ve 20 veri katmanından toplanmış ilk bulgular ile yeni keşfedilen web araştırma verileri bulunmaktadır.

[İLK BULGULAR]:
- Kök / Rekonstrüksiyon: {initial_finding.get('root', {}).get('proto_turkic')}
- Anlam: {initial_finding.get('root', {}).get('meaning')}
- Morfoloji: {initial_finding.get('morphology')}
- Bulunan Türki Dil Karşılık Sayısı: {len(initial_finding.get('turkic_languages', []))}

[YENİ KEŞFEDİLEN AKADEMİK VE WEB ARAŞTIRMA VERİLERİ]:
- Web Aramaları: {json.dumps(web_search_results, ensure_ascii=False)}
- DergiPark Makaleleri: {academic_results}
- Archive.org Kitapları: {archive_results}
- Keşfedilen Sayfa İçeriği: {scraped_content[:400]}...

Görevlerin:
1. Bu kelimenin etimolojik kökeni, tarihi tanıklığı ve diyalekt yayılımı hakkında 2-3 cümlelik yüksek akademik kalitede derinleştirilmiş bir analiz sentezi yaz.
2. Varsa kaynak dildeki (Arapça, Farsça, Çince, Grekçe, Latince) veya Proto-Türkçedeki en doğru kökü ve türeyiş biçimini açıkla.

Lütfen yanıtını doğrudan Türkçe olarak yaz.
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                analysis = res.get("response", "").strip()

                initial_finding["ai_agent_enrichment"] = analysis
                initial_finding["discovered_web_sources"] = web_search_results
                
                # Derinleştirilmiş analizi veritabanına ve sonuçlara ekle
                if analysis:
                    initial_finding["sources"].append(f"Qwen2.5:14b Otonom Araştırma Ajanı ({self.model_name})")
                    initial_finding["turkic_languages"].append({
                        "lang_code": "ai",
                        "lang_name": "Qwen2.5:14b Otonom Araştırma Sentezi",
                        "word": word,
                        "meaning": analysis,
                        "script": "Latin"
                    })
        except Exception as e:
            initial_finding["ai_agent_enrichment"] = f"Qwen2.5:14b analiz hatası: {e}"

        return initial_finding
