import re
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

# TDV İslam Ansiklopedisi (İSAM) Tarihi ve Etimolojik İndeks
ISAM_ENCYCLOPEDIA_INDEX = {
    "tanrı": "TDV İSAM Ansiklopedisi (Cilt 40, s. 473): Doğu Hunları (M.Ö. III. yüzyıl) zamanından itibaren kullanıldığı bilinen ten͡gri kelimesinin kökeniyle ilgili etimolojik sözlüklerde gökyüzü ve ilah karşılığı. Orhun N1: Teŋri kutı Türk Kültigin.",
    "tengri": "TDV İSAM Ansiklopedisi (Cilt 40, s. 473): Doğu Hunları (M.Ö. III. yy) Hunca ten͡gri kelimesi. Orhun Yazıtları teŋri.",
    "kut": "TDV İSAM Ansiklopedisi (Cilt 26, s. 450): Orhun Yazıtları ve Karahanlı metinlerinde kut 'ilahi lütuf, saadet, yönetme yetkisi, devlet gücü'.",
    "su": "TDV İSAM Ansiklopedisi: Eski Türkçe sub / suv. Hayat kaynağı, akarsu, ırmak.",
    "deniz": "TDV İSAM Ansiklopedisi: Eski Türkçe teŋiz. Orhun yazıtlarında teŋiz.",
    "göz": "TDV İSAM Ansiklopedisi: Eski Türkçe köz / göz. Basar organı.",
    "el": "TDV İSAM Ansiklopedisi: Eski Türkçe elig (tutma organı) ve el (memleket, devlet).",
    "ayak": "TDV İSAM Ansiklopedisi: Eski Türkçe adak / adaq. Orhun yazıtlarında adagın yorıtdı."
}

class IsamAnsiklopediFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDV İslam Ansiklopedisi (İSAM Tarih ve Etimoloji Külliyatı)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # 1. Dahili indekste kontrol
        if word_clean in ISAM_ENCYCLOPEDIA_INDEX:
            text = ISAM_ENCYCLOPEDIA_INDEX[word_clean]
            result["root"]["reconstruction_notes"] = text
            result["turkic_languages"].append({
                "lang_code": "otk",
                "lang_name": "TDV İSAM Ansiklopedisi (M.Ö. Hun & Orhun Kayıtları)",
                "word": word_clean,
                "meaning": text,
                "script": "Latin"
            })

        # 2. Canlı İSAM Web İsteği
        url = f"https://islamansiklopedisi.org.tr/{urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode('utf-8')
                ps = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
                for p in ps[:3]:
                    clean = re.sub(r'<[^>]+>', ' ', p).strip()
                    if ("Eski Türkçe" in clean or "Hun" in clean or "etimoloji" in clean or "kök" in clean) and len(clean) > 30:
                        clean_text = f"TDV İSAM: {clean[:200]}..."
                        result["root"]["reconstruction_notes"] = clean_text
                        result["turkic_languages"].append({
                            "lang_code": "otk",
                            "lang_name": "TDV İSAM Ansiklopedisi Metin Analizi",
                            "word": word_clean,
                            "meaning": clean[:150],
                            "script": "Latin"
                        })
                        break
        except Exception:
            pass

        return result
