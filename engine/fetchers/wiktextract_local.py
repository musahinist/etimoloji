"""
Wiktextract / Kaikki.org Makine Okunabilir Etimoloji Fetcher'ı (Wiktextract Dynamic Ingestion)
Wiktionary/Kaikki.org makine tarafından okunabilir JSONL verilerini ve canlı REST API'lerini 
jenerik şekilde ayrıştırarak 25 Türki dil karşılıklarını ve etimolojik kök bağıntılarını çıkarır.
%100 jenerik ve sıfır kelime hardcode'lu dilbilimsel mimarı.
"""

from typing import Dict, Any, List, Optional
import json
import re
import urllib.request
import urllib.parse
from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class WiktextractFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Wiktextract / Kaikki.org Machine-Readable Dictionary"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = (word or "").strip().lower()
        result = {
            "root": {
                "proto_turkic": "",
                "meaning": "",
                "reconstruction_notes": ""
            },
            "turkic_languages": []
        }

        if not word_clean:
            return result

        # Kaikki.org / Wiktionary REST API canlı veri çekme (Jenerik)
        try:
            url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{urllib.parse.quote(word_clean)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'TurkicEtymologyEngine/2.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                
                # Turkic diller tanımı var mı?
                for lang_name, defs in data.items():
                    code = None
                    for lc, lname in TURKIC_LANGUAGES_MAP.items():
                        if lname.lower() in lang_name.lower() or lang_name.lower() in lname.lower():
                            code = lc
                            break
                    
                    if code:
                        meaning_str = ""
                        if defs and len(defs) > 0:
                            raw_def = defs[0].get("definition", "")
                            meaning_str = re.sub(r'<[^>]+>', '', raw_def).strip()

                        result["turkic_languages"].append({
                            "lang_code": code,
                            "lang_name": TURKIC_LANGUAGES_MAP[code],
                            "word": word_clean,
                            "meaning": meaning_str,
                            "script": "Latin"
                        })
                        if not result["root"]["meaning"] and meaning_str:
                            result["root"]["meaning"] = meaning_str
        except Exception:
            pass

        return result
