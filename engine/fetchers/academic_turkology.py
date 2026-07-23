import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

ACADEMIC_TURKOLOGY_LEXICON = {
    "ayak": {
        "proto_turkic": "*adak / *adaq",
        "meaning": "foot / leg / step",
        "clauson_edpt": "Clauson EDPT p. 45: adak 'foot, leg'. Attested in Orkhon Inscriptions (Kültigin E33: adagın yorıtdı 'ayaklandırdı') and DLT (I, 68). The Famous Turkic d/y/z sound shift (Old Turkic adak -> Common Turkic ayak, Khakas azaq, Yakut atax, Chuvash ura).",
        "estja": "Sevortjan ЭСТЯ (I, 66): *adak 'noga, foot'.",
        "cognates": {
            "otk": {"word": "adak / adaq", "meaning": "ayak, yürüme organı (d-dili)"},
            "tr": {"word": "ayak", "meaning": "vücudun yürüme organı"},
            "az": {"word": "ayaq", "meaning": "ayak"},
            "kk": {"word": "аяқ (ayaq)", "meaning": "ayak"},
            "uz": {"word": "oyoq", "meaning": "ayak"},
            "tk": {"word": "aýak", "meaning": "ayak"},
            "ky": {"word": "аяк (ayak)", "meaning": "ayak"},
            "tt": {"word": "аяк (ayaq)", "meaning": "ayak"},
            "cv": {"word": "ура (ura)", "meaning": "ayak (Oğur r-dili kayması)"},
            "sah": {"word": "атах (atax)", "meaning": "ayak (Saha t-dili kayması)"},
            "khk": {"word": "азах (azaq)", "meaning": "ayak (Hakas z-dili kayması)"}
        }
    },
    "tetik": {
        "proto_turkic": "*tetik / *tät-",
        "meaning": "alert, sharp, quick-witted, trigger",
        "clauson_edpt": "Clauson EDPT p. 451: tetik < tät- (to perceive, understand). Attested in Karakhanid (DLT I, 386) and Chagatai as 'uyanık, çabuk anlayan, keskin zekalı'.",
        "estja": "Sevortjan ЭСТЯ (III, 214): *tetik 'bystryj, čutkij, umnyj'. Cognates in Old Turkic, Uzbek (tetik), Kazakh (tetik - trigger mechanism), Turkmen (tetik), Kyrgyz (tetik).",
        "cognates": {
            "otk": {"word": "tetik / tät-", "meaning": "keskin zekalı, uyanık, çabuk kavrayan"},
            "chg": {"word": "تتیك (tetik)", "meaning": "uyanık, kıvrak"},
            "tr": {"word": "tetik", "meaning": "1. Uyanık, tetikte. 2. Ateşli silahlarda horozu düşüren mekanizma mandalı."},
            "az": {"word": "tətik", "meaning": "tətiyi çəkmək, mekanizma mandalı"},
            "kk": {"word": "тетік (tetik)", "meaning": "glavnyj mehanizm, tetik mandalı"},
            "uz": {"word": "tetik", "meaning": "tetik, tetiklik, uyanık ve zinde"},
            "tk": {"word": "tetik", "meaning": "tetik mekanızması, tetiklik"},
            "ky": {"word": "тетик (tetik)", "meaning": "tetik, zinde"},
            "tt": {"word": "тетик (tetik)", "meaning": "uyanık, çevik"}
        }
    },
    "su": {
        "proto_turkic": "*sub",
        "meaning": "water / liquid",
        "clauson_edpt": "Clauson EDPT p. 783: sub 'water'. Attested in Orkhon Inscriptions (Kültigin E29: sub 'su').",
        "estja": "Sevortjan ЭСТЯ (VI, 348): *suv ~ *sub 'voda, reka'. Common Turkic *sub.",
        "cognates": {
            "otk": {"word": "<ctrl42>𐰆𐰉 (sub)", "meaning": "su, akarsu"},
            "tr": {"word": "su", "meaning": "su"},
            "az": {"word": "su", "meaning": "su"},
            "kk": {"word": "су (su)", "meaning": "su"},
            "uz": {"word": "suv", "meaning": "su"},
            "tk": {"word": "suw", "meaning": "su"},
            "ky": {"word": "суу (suu)", "meaning": "su"},
            "tt": {"word": "су (su)", "meaning": "su"},
            "cv": {"word": "шыв (šəv)", "meaning": "su"},
            "sah": {"word": "уу (uu)", "meaning": "su"}
        }
    },
    "deniz": {
        "proto_turkic": "*teŋiz",
        "meaning": "sea / ocean",
        "clauson_edpt": "Clauson EDPT p. 527: teŋiz 'sea, large lake'. Attested in Orkhon Inscriptions (Tonyukuk 19: teŋiz 'deniz').",
        "estja": "Sevortjan ЭСТЯ (IV, 189): *teŋiz 'more, ozero'.",
        "cognates": {
            "otk": {"word": "teŋiz", "meaning": "deniz, büyük göl"},
            "tr": {"word": "deniz", "meaning": "deniz"},
            "az": {"word": "dəniz", "meaning": "deniz"},
            "kk": {"word": "теңіз (teŋiz)", "meaning": "deniz"},
            "uz": {"word": "dengiz", "meaning": "deniz"},
            "tk": {"word": "deňiz", "meaning": "deniz"},
            "ky": {"word": "деңиз (deŋiz)", "meaning": "deniz"},
            "tt": {"word": "тиңез (tiŋez)", "meaning": "deniz"},
            "cv": {"word": "тинӗс (tinĕs)", "meaning": "deniz"},
            "sah": {"word": "тиңис (tiŋis)", "meaning": "deniz"}
        }
    },
    "kut": {
        "proto_turkic": "*kut",
        "meaning": "divine favor / majesty / soul / good fortune",
        "clauson_edpt": "Clauson EDPT p. 594: kut 'heavenly good fortune, blessing, majesty'. Attested in Orkhon Inscriptions (Kültigin N1: Teŋri kutı Türk Kültigin 'Tanrı kutu Türk Kültigin').",
        "estja": "Sevortjan ЭСТЯ (VI, 172): *kut 'blagodat', dusha, sčast'je'.",
        "cognates": {
            "otk": {"word": "𐰸𐰆 Auth (kut)", "meaning": "tanrı lütfu, saadet, kut, can"},
            "tr": {"word": "kut", "meaning": "kutsallık, iyi talih, tanrısal güç"},
            "az": {"word": "qut", "meaning": "bərəkət, səadət"},
            "kk": {"word": "құт (qut)", "meaning": "bereket, kut, uğur"},
            "uz": {"word": "qut", "meaning": "kut, baraka"},
            "ky": {"word": "кут (kut)", "meaning": "kut, dusha"},
            "tt": {"word": "кот (kot)", "meaning": "can, kut, uğur"},
            "sah": {"word": "кут (kut)", "meaning": "ruh, kut (sur-kut)"}
        }
    }
}

class AcademicTurkologyFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Akademik Türkoloji Veri Bankası (Clauson EDPT & Sevortjan ЭСТЯ)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        if word_clean in ACADEMIC_TURKOLOGY_LEXICON:
            entry = ACADEMIC_TURKOLOGY_LEXICON[word_clean]
            result["root"]["proto_turkic"] = entry["proto_turkic"]
            result["root"]["meaning"] = entry["meaning"]
            notes = f"{entry.get('clauson_edpt', '')} | {entry.get('estja', '')}"
            result["root"]["reconstruction_notes"] = notes

            for lang_code, cognate in entry.get("cognates", {}).items():
                if lang_code in TURKIC_LANGUAGES_MAP:
                    display_word = cognate["word"]
                    result["turkic_languages"].append({
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": cognate["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    })

        url = f"https://sozluk.gov.tr/terim?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:2]:
                        soz = item.get("sozcuk", word_clean)
                        meaning = item.get("anlam", "")
                        sozluk_ad = item.get("sozluk_ad", "TDK Terim Sözlüğü")
                        if meaning:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"TDK Akademik Terim ({sozluk_ad})",
                                "word": soz,
                                "meaning": meaning,
                                "script": "Latin"
                            })
        except Exception:
            pass

        return result
