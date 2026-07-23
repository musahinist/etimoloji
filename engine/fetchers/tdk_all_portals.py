import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class TdkAllPortalsFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK Tüm Alt Portalları (TDK Ağızlar, Kişi Adları, Yazım Kılavuzu)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # 1. TDK Türkiye Türkçesindeki Ağızlar Sözlüğü (TTAS)
        url_ttas = f"https://sozluk.gov.tr/ttas?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url_ttas, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:2]:
                        soz = item.get("kence", word_clean)
                        meaning = item.get("anlam", "")
                        il = item.get("il_adi", "Anadolu Ağızları")
                        if meaning:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"TDK Türkiye Türkçesi Ağızları ({il})",
                                "word": soz,
                                "meaning": meaning,
                                "script": "Latin"
                            })
        except Exception:
            pass

        # 2. TDK Kişi Adları Sözlüğü
        url_kisi = f"https://sozluk.gov.tr/kisi?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url_kisi, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:2]:
                        ad = item.get("ad", word_clean)
                        anlam = item.get("anlam", "")
                        cins = item.get("cinsiyet", "")
                        if anlam:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"TDK Kişi Adları Etimolojisi ({cins})",
                                "word": ad,
                                "meaning": anlam,
                                "script": "Latin"
                            })
        except Exception:
            pass

        return result
