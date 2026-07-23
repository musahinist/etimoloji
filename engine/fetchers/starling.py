import re
import urllib.request
import urllib.parse
from typing import Dict, Any, List
from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# Dahili Yıldız/Starling Proto-Türkçe Etimoloji Sözlük Dizini (Core Turkic Lexicon Index)
STARLING_OFFLINE_LEXICON = {
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
    },
    "tengri": {
        "proto_turkic": "*teŋri / *taŋrï",
        "meaning": "sky / heaven / deity",
        "cognates": {
            "otk": {"word": "teŋri", "meaning": "gök, tanrı"},
            "tr": {"word": "tanrı", "meaning": "ilahi varlık, tanrı"},
            "az": {"word": "tanrı", "meaning": "tanrı"},
            "kk": {"word": "тәңір (täŋir)", "meaning": "tanrı"},
            "uz": {"word": "tengri", "meaning": "tanrı"},
            "tk": {"word": "taňry", "meaning": "tanrı"},
            "ky": {"word": "теңир (teŋir)", "meaning": "tanrı"},
            "tt": {"word": "тәңре (täŋre)", "meaning": "tanrı"},
            "ug": {"word": "تەڭرى (teŋri)", "meaning": "tanrı"},
            "sah": {"word": "tanara (taŋara)", "meaning": "gök tanrı"}
        }
    },
    "gök": {
        "proto_turkic": "*gök",
        "meaning": "blue / sky / heaven",
        "cognates": {
            "otk": {"word": "kök", "meaning": "gök, mavi"},
            "tr": {"word": "gök", "meaning": "gökyüzü, mavi"},
            "az": {"word": "göy", "meaning": "gök, mavi"},
            "kk": {"word": "көк (kök)", "meaning": "gök, mavi"},
            "uz": {"word": "ko'k", "meaning": "gök, mavi"},
            "tk": {"word": "gök", "meaning": "gök, mavi"},
            "ky": {"word": "көк (kök)", "meaning": "gök, mavi"},
            "tt": {"word": "көк (kök)", "meaning": "gök"},
            "cv": {"word": "вак / кӑвак", "meaning": "mavi, gök"}
        }
    },
    "gün": {
        "proto_turkic": "*gün / *kün",
        "meaning": "sun / day",
        "cognates": {
            "otk": {"word": "kün", "meaning": "güneş, gün"},
            "tr": {"word": "gün", "meaning": "güneş, 24 saatlik zaman"},
            "az": {"word": "gün", "meaning": "gün, güneş"},
            "kk": {"word": "күн (kün)", "meaning": "gün, güneş"},
            "uz": {"word": "kun", "meaning": "gün"},
            "tk": {"word": "gün", "meaning": "gün"},
            "ky": {"word": "күн (kün)", "meaning": "gün"},
            "tt": {"word": "көн (kön)", "meaning": "gün"},
            "sah": {"word": "kүн (kün)", "meaning": "güneş, gün"},
            "cv": {"word": "кун (kun)", "meaning": "gün"}
        }
    },
    "ay": {
        "proto_turkic": "*āń",
        "meaning": "moon / month",
        "cognates": {
            "otk": {"word": "ay", "meaning": "ay, ay ayı"},
            "tr": {"word": "ay", "meaning": "dünyanın uydusu, ay kaydı"},
            "az": {"word": "ay", "meaning": "ay"},
            "kk": {"word": "ай (ay)", "meaning": "ay"},
            "uz": {"word": "oy", "meaning": "ay"},
            "tk": {"word": "ay", "meaning": "ay"},
            "ky": {"word": "ай (ay)", "meaning": "ay"},
            "tt": {"word": "ай (ay)", "meaning": "ay"},
            "sah": {"word": "ый (ıy)", "meaning": "ay"},
            "cv": {"word": "уйӑх (uyăh)", "meaning": "ay"}
        }
    },
    "baş": {
        "proto_turkic": "*baĺ",
        "meaning": "head / top / chief",
        "cognates": {
            "otk": {"word": "baş", "meaning": "baş, kafa, lider"},
            "tr": {"word": "baş", "meaning": "kafa, lider, başlangıç"},
            "az": {"word": "baş", "meaning": "baş"},
            "kk": {"word": "бас (bas)", "meaning": "baş, kafa"},
            "uz": {"word": "bosh", "meaning": "baş"},
            "tk": {"word": "baş", "meaning": "baş"},
            "ky": {"word": "баш (baş)", "meaning": "baş"},
            "tt": {"word": "баш (baş)", "meaning": "baş"},
            "cv": {"word": "puҫ (puś)", "meaning": "baş"},
            "sah": {"word": "bas (bas)", "meaning": "baş"}
        }
    },
    "kan": {
        "proto_turkic": "*kān",
        "meaning": "blood",
        "cognates": {
            "otk": {"word": "kan", "meaning": "kan"},
            "tr": {"word": "kan", "meaning": "vücuttaki kırmızı sıvı"},
            "az": {"word": "qan", "meaning": "kan"},
            "kk": {"word": "қан (qan)", "meaning": "kan"},
            "uz": {"word": "qon", "meaning": "kan"},
            "tk": {"word": "gan", "meaning": "kan"},
            "ky": {"word": "кан (kan)", "meaning": "kan"},
            "tt": {"word": "кан (kan)", "meaning": "kan"},
            "sah": {"word": "haan (haan)", "meaning": "kan"},
            "cv": {"word": "юн (yun)", "meaning": "kan"}
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

        # Offline etimoloji veritabanı kontrolü
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
