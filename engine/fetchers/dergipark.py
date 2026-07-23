import re
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class DergiParkFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "DergiPark (TÜBİTAK ULAKBİM Akademik Dergiler & Tez Portalı)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://dergipark.org.tr/tr/search?q={urllib.parse.quote(word_clean)}+etimoloji"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
            with urllib.request.urlopen(req, timeout=4) as resp:
                html = resp.read().decode('utf-8')
                titles = re.findall(r'<a[^>]*class=\"[^\"]*card-title[^\"]*\"[^>]*>(.*?)</a>', html, re.DOTALL)
                for t in titles[:2]:
                    clean_t = re.sub(r'<[^>]+>', '', t).strip()
                    if clean_t and not "doğrulayınız" in clean_t:
                        result["turkic_languages"].append({
                            "lang_code": "tr",
                            "lang_name": "DergiPark Akademik Makale Dizin",
                            "word": word_clean,
                            "meaning": f"Makale Başlığı: {clean_t}",
                            "script": "Latin"
                        })
        except Exception:
            pass

        return result
