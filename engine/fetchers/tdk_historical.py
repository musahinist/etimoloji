import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class TdkTaramaFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK Tarama Sözlüğü (Tarihi Türkçe Metinler 13.-19. yy)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://sozluk.gov.tr/tarama?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    tarama_list = item.get("tarama", [])
                    if tarama_list:
                        hist_word = tarama_list[0].get("kelime", word_clean)
                        meaning = tarama_list[0].get("anlam", "")
                        result["turkic_languages"].append({
                            "lang_code": "otk",
                            "lang_name": "Tarihi Türkçe / Osmanlıca (13.-19. yy)",
                            "word": hist_word,
                            "meaning": meaning,
                            "script": "Latin"
                        })
                        result["root"]["reconstruction_notes"] = f"TDK Tarama Sözlüğü (Tarihi Türkçe): {hist_word} - {meaning}"
        except Exception:
            pass

        return result


class TdkDerlemeFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK Derleme Sözlüğü (Türk Ağızları ve Diyalektleri)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://sozluk.gov.tr/derleme?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:3]:
                        m_word = item.get("madde", word_clean)
                        meaning = item.get("anlam", "")
                        city = item.get("sehir", "")
                        if meaning:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"Türk Ağızları ({city})" if city else "Türk Ağızları",
                                "word": m_word,
                                "meaning": meaning,
                                "script": "Latin"
                            })
        except Exception:
            pass

        return result
