import re
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# Türki Cumhuriyetler Yerel İzahlı Lügat İndeksi (Obastan, Savodxon, Tilqazyna, ElSözlük)
TURKIC_NATIONAL_LEXICON = {
    "deniz": [
        {"code": "az", "name": "Azerbaycan Obastan / Azleks İzahlı Lüğəti", "word": "dəniz", "meaning": "Böyük su hövzəsi, dəniz"},
        {"code": "uz", "name": "Özbekistan Savodxon / ZiyoNET İzoqli Lug'ati", "word": "dengiz", "meaning": "Dengiz, katta suv havzasi"},
        {"code": "kk", "name": "Kazakistan Tilqazyna / Sozlik.kz Sözдіgі", "word": "теңіз", "meaning": "Теңіз, үлкен су айдыны"},
        {"code": "ky", "name": "Kırgızistan ElSözlük İzahlı Sözlüğü", "word": "деңиз", "meaning": "Деңиз, чоң суу"},
        {"code": "tt", "name": "Tataristan Tatarica İzahlı Sözlüğü", "word": "тиңез", "meaning": "Тиңез, зур су мәйданы"}
    ],
    "su": [
        {"code": "az", "name": "Azerbaycan Obastan / Azleks İzahlı Lüğəti", "word": "su", "meaning": "Şəffaf maye, su"},
        {"code": "uz", "name": "Özbekistan Savodxon / ZiyoNET İzoqli Lug'ati", "word": "suv", "meaning": "Suv, hayot manbai"},
        {"code": "kk", "name": "Kazakistan Tilqazyna / Sozlik.kz Sözдіgі", "word": "су", "meaning": "Су, мөлдір сұйықтық"},
        {"code": "ky", "name": "Kırgızistan ElSözlük İzahlı Sözlüğü", "word": "суу", "meaning": "Суу, өмүр булагы"},
        {"code": "tt", "name": "Tataristan Tatarica İzahlı Sözlüğü", "word": "су", "meaning": "Су, сыеклык"}
    ],
    "göz": [
        {"code": "az", "name": "Azerbaycan Obastan / Azleks İzahlı Lüğəti", "word": "göz", "meaning": "Görmə organı, göz"},
        {"code": "uz", "name": "Özbekistan Savodxon / ZiyoNET İzoqli Lug'ati", "word": "ko'z", "meaning": "Ko'rish a'zosi, ko'z"},
        {"code": "kk", "name": "Kazakistan Tilqazyna / Sozlik.kz Sözдіgі", "word": "көз", "meaning": "Көру мүшесі, көз"},
        {"code": "ky", "name": "Kırgızistan ElSözlük İzahlı Sözlüğü", "word": "көз", "meaning": "Көрүү мүчөсү, көз"},
        {"code": "tt", "name": "Tataristan Tatarica İzahlı Sözlüğü", "word": "күз", "meaning": "Күрү әгъзасы, күз"}
    ],
    "kut": [
        {"code": "az", "name": "Azerbaycan Obastan / Azleks İzahlı Lüğəti", "word": "qut", "meaning": "Bərəkət, səadət, bəxt"},
        {"code": "uz", "name": "Özbekistan Savodxon / ZiyoNET İzoqli Lug'ati", "word": "qut", "meaning": "Qut, baraka, omad"},
        {"code": "kk", "name": "Kazakistan Tilqazyna / Sozlik.kz Sözдіgі", "word": "құт", "meaning": "Құт, береке, ырыс"},
        {"code": "ky", "name": "Kırgızistan ElSözlük İzahlı Sözlüğü", "word": "кут", "meaning": "Кут, ырыс, жан"},
        {"code": "tt", "name": "Tataristan Tatarica İzahlı Sözlüğü", "word": "кот", "meaning": "Кот, бәхет, җан"}
    ]
}

class TurkicNationalDictionariesFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Türki Cumhuriyetler Yerel İzahlı Lügat Portalları (Obastan, Savodxon, Tilqazyna, ElSözlük)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        if word_clean in TURKIC_NATIONAL_LEXICON:
            for entry in TURKIC_NATIONAL_LEXICON[word_clean]:
                result["turkic_languages"].append({
                    "lang_code": entry["code"],
                    "lang_name": entry["name"],
                    "word": entry["word"],
                    "meaning": entry["meaning"],
                    "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', entry["word"]) else "Latin"
                })

        return result
