import json
import urllib.request
import urllib.parse
from typing import Dict, Any

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class ArchiveOrgFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Internet Archive (Archive.org Dijital Kitaplar Külliyatı)"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        query = f"{word_clean} turkic etymology"
        url = f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(query)}&fl[]=identifier,title,creator,year&rows=2&page=1&output=json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                docs = data.get("response", {}).get("docs", [])
                for doc in docs:
                    title = doc.get("title", "")
                    creator = doc.get("creator", "Bilinmeyen Yazar")
                    year = doc.get("year", "Tarihsiz")
                    if title:
                        result["turkic_languages"].append({
                            "lang_code": "otk",
                            "lang_name": f"Archive.org Kitap Külliyatı ({year})",
                            "word": word_clean,
                            "meaning": f"Kitap: {title} (Yazar: {creator})",
                            "script": "Latin"
                        })
                        result["root"]["reconstruction_notes"] = f"Archive.org Dijital Kitap: {title} ({year})"
        except Exception:
            pass

        return result
