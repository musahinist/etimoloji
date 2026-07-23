import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class TdkFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK (Türk Dil Kurumu)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://sozluk.gov.tr/gts?ara={urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0 and "anlamlarList" in data[0]:
                    meanings = [item["anlam"] for item in data[0]["anlamlarList"] if "anlam" in item]
                    meaning_str = "; ".join(meanings[:2])
                    
                    lisan = data[0].get("lisan", "")
                    
                    result["turkic_languages"].append({
                        "lang_code": "tr",
                        "lang_name": TURKIC_LANGUAGES_MAP["tr"],
                        "word": word_clean,
                        "meaning": meaning_str,
                        "script": "Latin"
                    })
                    result["root"]["meaning"] = meaning_str
                    if lisan:
                        result["root"]["reconstruction_notes"] = f"TDK Köken Bilgisi: {lisan}"
        except Exception:
            pass

        return result


class NisanyanFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Nişanyan Etimoloji Sözlüğü"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://www.nisanyansozluk.com/kelime/{urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode('utf-8')
                tokens = re.findall(r'text:\"([^\"]+)\"', html)
                if not tokens:
                    return result

                text_full = "".join(tokens)
                
                # Eski Türkçe veya Ana Türkçe kök tespiti
                etü_match = re.search(r'Eski\s+Türkçe\s+([a-zçğıöşüA-ZÇĞİÖŞÜ\*]+)\s+“([^”]+)”', text_full)
                if etü_match:
                    etü_word = etü_match.group(1).strip()
                    etü_meaning = etü_match.group(2).strip()
                    result["turkic_languages"].append({
                        "lang_code": "otk",
                        "lang_name": TURKIC_LANGUAGES_MAP["otk"],
                        "word": etü_word,
                        "meaning": etü_meaning,
                        "script": "Latin"
                    })
                    result["root"]["proto_turkic"] = f"*{etü_word}"
                    result["root"]["meaning"] = etü_meaning

                # Ana Türkçe / Proto-Turkic kök tespiti
                root_match = re.search(r'\*([a-zçğıöşüA-ZÇĞİÖŞÜ\-]+)\s+“([^”]+)”', text_full)
                if root_match:
                    proto_w = root_match.group(1).strip()
                    proto_m = root_match.group(2).strip()
                    result["root"]["proto_turkic"] = f"*{proto_w}"
                    if not result["root"]["meaning"]:
                        result["root"]["meaning"] = proto_m

                result["root"]["reconstruction_notes"] = f"Nişanyan Etimoloji: {text_full[:300]}..."
        except Exception:
            pass

        return result
