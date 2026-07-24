"""
Neologizm & Dil Devrimi Türetmeleri Sözlük Modülü (Neologism & Language Reform Detector)
1932-1935 TDK Özleştirme Kılavuzları ve Cumhuriyet dönemi yeni türetmelerini (neologisms) tespit eder.
"""
from typing import Dict, Any, Optional

NEOLOGISM_CATALOG = {
    "okul": {
        "root_verb": "oku-",
        "suffix": "-l",
        "year": "1935 (Cumhuriyet Dönemi TDK Türetmesi)",
        "creator": "TDK / Nurullah Ataç",
        "phonetic_influence": "Fransızca école (ekol) kelimesine ses çağrışımı yaptırılarak türetilmiştir.",
        "historical_note": "Eski Türkçe veya Osmanlıca el yazmalarında bulunmaz (1935 öncesinde mektep / medrese kullanılmaktaydı)."
    },
    "öğretmen": {
        "root_verb": "öğret-",
        "suffix": "-men",
        "year": "1935 (TDK Özleştirme)",
        "creator": "TDK Özleştirme Komisyonu",
        "phonetic_influence": "Öz Türkçe kökten türetilmiştir.",
        "historical_note": "1935 öncesinde muallim / müderris kullanılmaktaydı."
    },
    "uçak": {
        "root_verb": "uç-",
        "suffix": "-ak",
        "year": "1935 (TDK Özleştirme)",
        "creator": "TDK",
        "phonetic_influence": "Öz Türkçe kökten türetilmiştir.",
        "historical_note": "1935 öncesinde tayyare kullanılmaktaydı."
    },
    "bilgisayar": {
        "root_verb": "bilgi + say-",
        "suffix": "-ar",
        "year": "1969",
        "creator": "Aydın Köksal",
        "phonetic_influence": "Türkçe birleşik türetmedir (Computer karşılığı).",
        "historical_note": "20. yüzyıl sonu bilişim türetmesidir."
    },
    "bağımsızlık": {
        "root_verb": "bağ + -ım + -sız",
        "suffix": "+lık",
        "year": "1935 (TDK Özleştirme)",
        "creator": "TDK",
        "phonetic_influence": "Öz Türkçe kökten türetilmiştir.",
        "historical_note": "1935 öncesinde istiklal / müstakiliyet kullanılmaktaydı."
    }
}

class NeologismDetector:
    def detect(self, word: str) -> Optional[Dict[str, Any]]:
        w = word.strip().lower()
        if w in NEOLOGISM_CATALOG:
            item = NEOLOGISM_CATALOG[w]
            return {
                "word": w,
                "is_neologism": True,
                "derivation_type": f"Cumhuriyet Dönemi Dil Devrimi Türetmesi ({item['year']})",
                "etymology_details": f"Kök: '{item['root_verb']}' + Ek: '{item['suffix']}'. {item['phonetic_influence']} {item['historical_note']}"
            }
        return None
