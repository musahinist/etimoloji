import re
import urllib.request
import urllib.parse
from typing import Dict, Any, List
from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# Dahili Yıldız/Starling Proto-Türkçe Etimoloji Sözlük Dizini (Core Turkic Lexicon Index)
STARLING_OFFLINE_LEXICON = {
    "ayak": {
        "proto_turkic": "*adak / *adaq",
        "meaning": "foot / leg / step",
        "cognates": {
            "otk": {"word": "adak / adaq", "meaning": "ayak, adak"},
            "tr": {"word": "ayak", "meaning": "vücudun yürüme organı, ayak"},
            "az": {"word": "ayaq", "meaning": "ayak"},
            "kk": {"word": "аяқ (ayaq)", "meaning": "ayak"},
            "uz": {"word": "oyoq", "meaning": "ayak"},
            "tk": {"word": "aýak", "meaning": "ayak"},
            "ky": {"word": "аяк (ayak)", "meaning": "ayak"},
            "tt": {"word": "аяк (ayaq)", "meaning": "ayak"},
            "ug": {"word": "ئاياق (ayaq)", "meaning": "ayak"},
            "cv": {"word": "ура (ura)", "meaning": "ayak (Oğur r-dili kayması)"},
            "sah": {"word": "атах (atax)", "meaning": "ayak (Saha t-dili kayması)"},
            "ba": {"word": "аяҡ (ayaq)", "meaning": "ayak"},
            "tyv": {"word": "аяк (ayak)", "meaning": "ayak"},
            "khk": {"word": "азах (azaq)", "meaning": "ayak (Hakas z-dili kayması)"}
        }
    },
    "su": {
        "proto_turkic": "*sub",
        "meaning": "water / liquid / stream",
        "cognates": {
            "otk": {"word": "sub / suv", "meaning": "su, akarsu"},
            "tr": {"word": "su", "meaning": "su, sıvı"},
            "az": {"word": "su", "meaning": "su"},
            "kk": {"word": "су (su)", "meaning": "su"},
            "uz": {"word": "suv", "meaning": "su"},
            "tk": {"word": "suw", "meaning": "su"},
            "ky": {"word": "суу (suu)", "meaning": "su"},
            "tt": {"word": "су (su)", "meaning": "su"},
            "ug": {"word": "سۇ (su)", "meaning": "su"},
            "cv": {"word": "шыв (šəv)", "meaning": "su"},
            "sah": {"word": "уу (uu)", "meaning": "su"},
            "ba": {"word": "һыу (hıw)", "meaning": "su"},
            "tyv": {"word": "суг (suğ)", "meaning": "su"}
        }
    },
    "göz": {
        "proto_turkic": "*göŕ",
        "meaning": "eye / sight",
        "cognates": {
            "otk": {"word": "göz / kör-", "meaning": "göz, görmek"},
            "tr": {"word": "göz", "meaning": "görüş organı, göz"},
            "az": {"word": "göz", "meaning": "göz"},
            "kk": {"word": "kөз (köz)", "meaning": "göz"},
            "uz": {"word": "ko'z", "meaning": "göz"},
            "tk": {"word": "göz", "meaning": "göz"},
            "ky": {"word": "көз (köz)", "meaning": "göz"},
            "tt": {"word": "күз (küz)", "meaning": "göz"},
            "ug": {"word": "كۆز (köz)", "meaning": "göz"},
            "cv": {"word": "куҫ (kuś)", "meaning": "göz"},
            "sah": {"word": "көс (kös)", "meaning": "göz, bakış"},
            "ba": {"word": "күз (küz)", "meaning": "göz"}
        }
    },
    "el": {
        "proto_turkic": "*el / *elig",
        "meaning": "hand / forearm / hold",
        "cognates": {
            "otk": {"word": "elig", "meaning": "el"},
            "tr": {"word": "el", "meaning": "tutma organı, el"},
            "az": {"word": "əl", "meaning": "el"},
            "kk": {"word": "қол / елік", "meaning": "el"},
            "uz": {"word": "qo'l / el", "meaning": "el"},
            "tk": {"word": "el", "meaning": "el"},
            "ky": {"word": "кол / эл", "meaning": "el"},
            "tt": {"word": "кул / ил", "meaning": "el"},
            "ug": {"word": "ئەل (el)", "meaning": "el"},
            "cv": {"word": "алӑ (ală)", "meaning": "el"},
            "sah": {"word": "ilii (ilii)", "meaning": "el"}
        }
    },
    "deniz": {
        "proto_turkic": "*teŋiz",
        "meaning": "sea / large lake / ocean",
        "cognates": {
            "otk": {"word": "teŋiz", "meaning": "deniz, büyük göl"},
            "tr": {"word": "deniz", "meaning": "büyük su kütlesi, deniz"},
            "az": {"word": "dəniz", "meaning": "deniz"},
            "kk": {"word": "теңіз (teŋiz)", "meaning": "deniz"},
            "uz": {"word": "dengiz", "meaning": "deniz"},
            "tk": {"word": "deňiz", "meaning": "deniz"},
            "ky": {"word": "деңиз (deŋiz)", "meaning": "deniz"},
            "tt": {"word": "тиңез (tiŋez)", "meaning": "deniz"},
            "ug": {"word": "دەڭىز (deŋiz)", "meaning": "deniz"},
            "cv": {"word": "тинӗс (tinĕs)", "meaning": "deniz"},
            "sah": {"word": "тиңис (tiŋis)", "meaning": "deniz"}
        }
    }
}

class StarlingFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Starling Etymological Database"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {
                "proto_turkic": "",
                "meaning": "",
                "reconstruction_notes": ""
            },
            "turkic_languages": []
        }

        if word_clean in STARLING_OFFLINE_LEXICON:
            entry = STARLING_OFFLINE_LEXICON[word_clean]
            result["root"]["proto_turkic"] = entry["proto_turkic"]
            result["root"]["meaning"] = entry["meaning"]
            result["root"]["reconstruction_notes"] = f"Starling / Tower of Babel Proto-Turkic reconstruction {entry['proto_turkic']}"

            for lang_code, cognate in entry["cognates"].items():
                if lang_code in TURKIC_LANGUAGES_MAP:
                    display_word = cognate["word"]
                    result["turkic_languages"].append({
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": cognate["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    })

        return result
