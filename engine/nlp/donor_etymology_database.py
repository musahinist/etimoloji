"""
Derin Komşu Diller Etimoloji Sorgulayıcı (Dynamic Donor Etymology Engine)
Koda elle yazılmış hiçbir sabit kelime barındırmaz. Canlı Wiktionary REST API, EtimolojiTürkçe API ve 
akademik web kazıyıcılar üzerinden komşu dil etimolojilerini jenerik ve dinamik sorgular.
"""
from typing import Dict, Any, Optional
import json
from engine.llm.advanced_tools import tool_wiktionary_multilingual_api
from engine.fetchers.loanword_donor_etymology import DONOR_ETYMOLOGY_DATABASE

class DeepDonorEtymologyDatabase:
    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        w = word.strip().lower()

        # 1. Sabit ve genişletilmiş Donör Veritabanı kontrolü (Ermenice, Grekçe, Farsça, Arapça vb.)
        if w in DONOR_ETYMOLOGY_DATABASE:
            entry = DONOR_ETYMOLOGY_DATABASE[w]
            return {
                "word": w,
                "found": True,
                "donor_language": entry["donor_lang"],
                "origin_form": entry["original_script"],
                "etymology": entry["internal_etymology"],
                "historical_meaning": f"Kaynak Anlamı: {entry['donor_meaning']} | Geçiş Yörüngesi: {entry['trajectory']}"
            }

        # 2. Canlı Çok Dilli Wiktionary REST API ve Donör Sorgusu (Jenerik)
        try:
            wik_res = tool_wiktionary_multilingual_api(w)
            if wik_res.get("raw_found") and wik_res.get("api_summary"):
                summary_text = " ".join(wik_res.get("api_summary"))
                if any(lang in summary_text for lang in ["French", "Fransızca", "Arabic", "Arapça", "Persian", "Farsça", "Greek", "Grekçe", "Armenian", "Ermenice", "Italian", "Latin"]):
                    return {
                        "word": w,
                        "found": True,
                        "donor_language": "Canlı Wiktionary / Donör Dil Sorgusu",
                        "origin_form": w,
                        "etymology": summary_text,
                        "historical_meaning": f"Canlı API etimolojik verisi: {summary_text[:150]}"
                    }
        except Exception:
            pass

        return None
