import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class GlosbeFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Glosbe Multilingual Turkic Dictionary API"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # Glosbe API query
        url = f"https://glosbe.com/gapi/translate?from=tur&dest=az&phrase={urllib.parse.quote(word_clean)}&format=json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                tuc = data.get("tuc", [])
                for item in tuc[:2]:
                    phrase = item.get("phrase", {})
                    if phrase and phrase.get("text"):
                        result["turkic_languages"].append({
                            "lang_code": "az",
                            "lang_name": "Glosbe Azerbaycan Türkçesi Karşılığı",
                            "word": phrase.get("text"),
                            "meaning": f"Glosbe online çeviri kaydı ({word_clean})",
                            "script": "Latin"
                        })
        except Exception:
            pass

        return result
