import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# Dahili Osmanlıca ve Klasik Türkçe Lügat Dizini (Kubbealtı, Lehçe-i Osmanî, LexiQamus)
OSMANLICA_LUGAT_INDEX = {
    "deniz": {
        "word_osmanli": "دڭز / deniz",
        "meaning": "Deniz, umman, büyük su kütlesi. Kubbealtı Lugatı: (Eski Türkçe teŋiz).",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "su": {
        "word_osmanli": "صو / su",
        "meaning": "Su, ma, ab. Kubbealtı Lugatı: (Eski Türkçe sub / suv).",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "göz": {
        "word_osmanli": "كوز / göz",
        "meaning": "Göz, basar, ayn. Kubbealtı Lugatı: (Eski Türkçe göz / köz).",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "el": {
        "word_osmanli": "يد / el / elig",
        "meaning": "1. Tutma organı (Eski Türkçe elig). 2. Devlet, memleket.",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "ayak": {
        "word_osmanli": "ایاق / ayak",
        "meaning": "Ayak, kadem. Kubbealtı Lugatı: (Eski Türkçe adak).",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "kut": {
        "word_osmanli": "قوت / kut",
        "meaning": "1. Tanrı lütfu, iyi talih, kut. 2. Can, gıda, azık.",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    },
    "tengri": {
        "word_osmanli": "تڭرى / tanrı",
        "meaning": "Tanrı, ilah, Huda, Halık. Kubbealtı Lugatı: (Eski Türkçe teŋri).",
        "source": "Kubbealtı Lugatı & Ahmet Vefik Paşa (Lehçe-i Osmanî)"
    }
}

class OsmanlicaLugatFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Osmanlıca ve Klasik Türkçe Lügat Portalları (Kubbealtı & Lehçe-i Osmanî)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # 1. Dahili Sözlük Kontrolü
        if word_clean in OSMANLICA_LUGAT_INDEX:
            entry = OSMANLICA_LUGAT_INDEX[word_clean]
            result["turkic_languages"].append({
                "lang_code": "ota",
                "lang_name": f"Osmanlıca Lügat ({entry['source']})",
                "word": entry["word_osmanli"],
                "meaning": entry["meaning"],
                "script": "Arabic"
            })
            result["root"]["reconstruction_notes"] = f"Osmanlıca Lügat Kaydı: {entry['meaning']}"

        # 2. Canlı Lugatim Web İsteği
        url = f"https://www.lugatim.com/s/{urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=4) as resp:
                html = resp.read().decode('utf-8')
                m = re.search(r'<div class=\"[^\"]*meaning[^\"]*\"[^>]*>(.*?)</div>', html, re.DOTALL)
                if m:
                    clean_m = re.sub(r'<[^>]+>', '', m.group(1)).strip()
                    if clean_m:
                        result["turkic_languages"].append({
                            "lang_code": "ota",
                            "lang_name": "Kubbealtı Lugatı (Lugatim.com Live)",
                            "word": word_clean,
                            "meaning": clean_m[:150],
                            "script": "Latin"
                        })
        except Exception:
            pass

        return result
