import re
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# Divanü Lugati't-Türk (1074), Kamus-ı Türkî (1901), Codex Cumanicus (1303), Mukaddimetü'l-Edeb (12. yy), Sanglax & Güncel Sözlükler
HISTORICAL_MODERN_LEXICON = {
    "tetik": {
        "dlt": "Divanü Lugati't-Türk (Kaşgarlı Mahmud, 1074): tetik (تِتِكْ) 'çabuk anlayan, uyanık, çabuk kavrayan adam'. tät- eyleminden.",
        "kamus": "Kamus-ı Türkî (Şemseddin Sami, 1901): tetik (تتیك) 'uyanık, zeki, gayretli, silah mekanizması mandalı'.",
        "codex": "Codex Cumanicus (1303 Kıpçakça): tetiq / tetik 'promptus, vigil, uyanık'.",
        "sanglax": "Sanglax (Çağatayca Nevai Sözlüğü): tetik 'kıvrak, zinde, zeki'."
    },
    "su": {
        "dlt": "Divanü Lugati't-Türk (1074): sub (سُبْ) 'su, akarsu, ırmak'.",
        "kamus": "Kamus-ı Türkî (1901): su (صو) 'mayi, sıvı, hayat kaynağı'. Eski Türkçe suv / sub.",
        "codex": "Codex Cumanicus (1303): suv / su 'aqua'.",
        "sanglax": "Sanglax (Çağatayca): suv / su."
    },
    "deniz": {
        "dlt": "Divanü Lugati't-Türk (1074): teŋiz (تِңِزْ) 'deniz, ulu göl'.",
        "kamus": "Kamus-ı Türkî (1901): deniz (دниз) 'büyük su kütlesi, bahr'. Eski Türkçe teŋiz.",
        "codex": "Codex Cumanicus (1303): deŋiz 'mare'.",
        "sanglax": "Sanglax (Çağatayca): teŋiz / deniz."
    },
    "göz": {
        "dlt": "Divanü Lugati't-Türk (1074): göz (كُؤزْ) 'göz, basar organı'.",
        "kamus": "Kamus-ı Türkî (1901): göz (كوز) 'görüş organı, ayn'.",
        "codex": "Codex Cumanicus (1303): küz / göz 'oculus'.",
        "sanglax": "Sanglax (Çağatayca): köz."
    },
    "el": {
        "dlt": "Divanü Lugati't-Türk (1074): elig (اِلِكْ) 'el, tutma organı' & el (اِلْ) 'halk, vilayet, memleket'.",
        "kamus": "Kamus-ı Türkî (1901): el (ايل) 'yed, tutma organı, memleket, yabancı'.",
        "codex": "Codex Cumanicus (1303): el / il 'manus, populus'.",
        "sanglax": "Sanglax (Çağatayca): elig / el."
    },
    "belge": {
        "dlt": "Divanü Lugati't-Türk (1074): belgü (بِلْكُو) 'nişan, işaret, alamet'.",
        "kamus": "Kamus-ı Türkî (1901): belge / belgü 'işaret, alamet, senet, vesika'.",
        "codex": "Codex Cumanicus (1303): belgi / belgü 'signum'.",
        "sanglax": "Sanglax (Çağatayca): belgi."
    },
    "bilgi": {
        "dlt": "Divanü Lugati't-Türk (1074): biliğ (بِلِكْ) 'akıl, ilim, bilgi'. bil- eyleminden.",
        "kamus": "Kamus-ı Türkî (1901): bilgi / biliğ 'ilm, malumat, vukuf'.",
        "codex": "Codex Cumanicus (1303): bilik / bilgi 'scientia'.",
        "sanglax": "Sanglax (Çağatayca): bilik."
    },
    "kut": {
        "dlt": "Divanü Lugati't-Türk (1074): kut (قُتْ) 'devlet, saadet, uğur, tanrı lütfu'.",
        "kamus": "Kamus-ı Türkî (1901): kut (قوت) 'saadet, baht, kutsallık'.",
        "codex": "Codex Cumanicus (1303): kut 'fortuna, gratia'.",
        "sanglax": "Sanglax (Çağatayca): kut."
    }
}

class HistoricalModernLexiconFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Tarihi Türk Lehçeleri Sözlükleri (DLT 1074, Kamus-ı Türkî 1901, Codex Cumanicus 1303, Sanglax)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        if word_clean in HISTORICAL_MODERN_LEXICON:
            entry = HISTORICAL_MODERN_LEXICON[word_clean]
            
            if "dlt" in entry:
                result["turkic_languages"].append({
                    "lang_code": "otk",
                    "lang_name": "Divanü Lugati't-Türk (Kaşgarlı Mahmud, 1074)",
                    "word": word_clean,
                    "meaning": entry["dlt"],
                    "script": "Latin / Arabic"
                })

            if "kamus" in entry:
                result["turkic_languages"].append({
                    "lang_code": "ota",
                    "lang_name": "Kamus-ı Türkî (Şemseddin Sami, 1901)",
                    "word": word_clean,
                    "meaning": entry["kamus"],
                    "script": "Latin / Arabic"
                })

            if "codex" in entry:
                result["turkic_languages"].append({
                    "lang_code": "krc", # Kıpçakça
                    "lang_name": "Codex Cumanicus (Kıpçakça Metinler, 1303)",
                    "word": word_clean,
                    "meaning": entry["codex"],
                    "script": "Latin"
                })

            if "sanglax" in entry:
                result["turkic_languages"].append({
                    "lang_code": "chg",
                    "lang_name": "Sanglax (Klasik Çağatayca Nevai Sözlüğü)",
                    "word": word_clean,
                    "meaning": entry["sanglax"],
                    "script": "Latin / Arabic"
                })

        return result
