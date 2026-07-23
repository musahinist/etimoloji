import re
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class EtimolojiTurkceFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "EtimolojiTürkçe Portal (Tarihi Tanıklamalar & İlk Kullanımlar)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://www.etimolojiturkce.com/kelime/{urllib.parse.quote(word_clean)}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode('utf-8')
                
                # Etimoloji açıklama paragraflarını tara
                p_tags = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
                for p in p_tags:
                    if "Eski Türkçe" in p or "Ana Türkçe" in p or "Proto-Türkçe" in p or "kök" in p:
                        clean_text = re.sub(r'<[^>]+>', ' ', p).strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        
                        # Proto-Turkic root tespiti (*bel-, *sub, *teŋiz vb.)
                        root_match = re.search(r'\*([a-zçğıöşüA-ZÇĞİÖŞÜ\-]+)\s+“([^”]+)”', clean_text)
                        if root_match:
                            proto_w = root_match.group(1).strip()
                            proto_m = root_match.group(2).strip()
                            result["root"]["proto_turkic"] = f"*{proto_w}"
                            result["root"]["meaning"] = proto_m

                        # Eski Türkçe biçim ve metin tanıklamaları (Kaşgarlı Mahmud, Uygurca vb.)
                        etü_match = re.search(r'Eski\s+Türkçe\s+([a-zçğıöşüA-ZÇĞİÖŞÜ\*]+)\s+“([^”]+)”', clean_text)
                        if etü_match:
                            etü_w = etü_match.group(1).strip()
                            etü_m = etü_match.group(2).strip()
                            result["turkic_languages"].append({
                                "lang_code": "otk",
                                "lang_name": "Eski Türkçe (Uygur & DLT Tanıklamaları)",
                                "word": etü_w,
                                "meaning": etü_m,
                                "script": "Latin"
                            })

                        result["root"]["reconstruction_notes"] = f"EtimolojiTürkçe: {clean_text[:250]}"
                        break
        except Exception:
            pass

        return result
