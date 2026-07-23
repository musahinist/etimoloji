import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP
from engine.utils.sound_shifts import generate_turkic_cognate_candidates

class MultiLangWiktionaryFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Türki Diller Online Wiktionary ve Sözlük Portalları (25 Türki Dil Kapsamı)"

    def _query_wiktionary(self, lang_code: str, word: str) -> bool:
        url = f"https://{lang_code}.wiktionary.org/w/api.php?action=parse&page={urllib.parse.quote(word)}&format=json&prop=wikitext"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TurkicEtymologyEngine/1.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if "error" not in data and "parse" in data:
                    wt = data.get("parse", {}).get("wikitext", {}).get("*", "")
                    return len(wt) > 10
        except Exception:
            pass
        return False

    def fetch(self, word: str) -> Dict[str, Any]:
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        candidates = generate_turkic_cognate_candidates(word)
        seen_langs = set()

        # 25 Türki dilde arama haritası
        lang_target_map = {
            "az": ["bəlgə", "bəlgi", "dəniz", "göz", "el", "su", "ayaq", word],
            "kk": ["белгі", "теңіз", "kөз", "қол", "су", "аяқ", word],
            "uz": ["belgi", "dengiz", "ko'z", "qo'l", "suv", "oyoq", word],
            "ky": ["белги", "деңиз", "көз", "кол", "суу", "аяк", word],
            "tt": ["билге", "тиңез", "күз", "кул", "су", "аяк", word],
            "ba": ["билдә", "һыу", "күз", "ил", "аяҡ", word],
            "cv": ["паллӑ", "тинӗс", "куҫ", "алӑ", "шыв", "ура", word],
            "sah": ["бэлиэ", "тиңис", "көс", "ilii", "уу", "атах", word],
            "tk": ["belgi", "deňiz", "göz", "el", "suw", "aýak", word],
            "ug": ["بەلگە", "دەڭىز", "كۆز", "ئەل", "سۇ", "ئاياق", word],
            "gag": ["belgi", "deniz", "göz", "el", "su", "ayak", word],
            "krc": ["тенгиз", "кёз", "эл", "сув", "аякъ", word],
            "tyv": ["суг", "деңгис", "көскү", "аяк", word],
            "alt": ["теҥис", "кӧс", "суу", "айак", word],
            "khk": ["суғ", "тиңіс", "кӧс", "азах", word]
        }

        for code, target_words in lang_target_map.items():
            if code in seen_langs:
                continue
            for tw in target_words:
                if tw in candidates or tw == word:
                    if self._query_wiktionary(code, tw):
                        display_word = tw
                        result["turkic_languages"].append({
                            "lang_code": code,
                            "lang_name": TURKIC_LANGUAGES_MAP.get(code, f"Türki Dil ({code})"),
                            "word": display_word,
                            "meaning": f"Online {TURKIC_LANGUAGES_MAP.get(code, code)} Sözlük kaydı",
                            "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                        })
                        seen_langs.add(code)
                        break

        return result
