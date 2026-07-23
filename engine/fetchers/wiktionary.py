import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP

class WiktionaryFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Wiktionary"

    def _http_get(self, url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TurkicEtymologyEngine/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode('utf-8')
        except Exception:
            return None

    def _get_page_wikitext(self, page_title: str) -> Optional[str]:
        url = f"https://en.wiktionary.org/w/api.php?action=parse&page={urllib.parse.quote(page_title)}&format=json&prop=wikitext"
        raw = self._http_get(url)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if "error" in data:
                return None
            return data.get("parse", {}).get("wikitext", {}).get("*", "")
        except Exception:
            return None

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {
                "proto_turkic": "",
                "meaning": "",
                "reconstruction_notes": ""
            },
            "turkic_languages": []
        }

        # 1. Ana kelime sayfasını çek
        wt = self._get_page_wikitext(word_clean)
        proto_page_title = None

        if wt:
            # Proto-Turkic kök bağlantısını ara (örn. {{inh|tr|trk-pro|*sub|t=water}})
            proto_match = re.search(r'\{\{(?:inh|der|cog)\|[a-z\-]+\|trk-pro\|\*?([^|\}]+)(?:\|t=([^|\}]+))?', wt)
            if proto_match:
                proto_root = proto_match.group(1).strip()
                proto_meaning = proto_match.group(2).strip() if proto_match.group(2) else ""
                result["root"]["proto_turkic"] = f"*{proto_root}"
                result["root"]["meaning"] = proto_meaning
                proto_page_title = f"Reconstruction:Proto-Turkic/{proto_root}"

        # 2. Eğer Proto-Turkic rekonstruksiyon sayfası bulunursa, akraba kelimeleri oradan çek
        if proto_page_title:
            recon_wt = self._get_page_wikitext(proto_page_title)
            if recon_wt:
                self._parse_reconstruction_page(recon_wt, result)

        # 3. Ana sayfadan da tanımları ve Türki dilleri topla
        if wt:
            self._parse_word_page(wt, word_clean, result)

        return result

    def _parse_reconstruction_page(self, wt: str, result: Dict[str, Any]) -> None:
        # Anlam çekme
        meaning_match = re.search(r'==Proto-Turkic==.*?(?:#\s*\[\[(.*?)\]\]|#\s*(.*?)\n)', wt, re.DOTALL)
        if meaning_match and not result["root"]["meaning"]:
            meaning = (meaning_match.group(1) or meaning_match.group(2) or "").strip()
            result["root"]["meaning"] = meaning

        # Türki diller türevlerini parsing
        # Template format: {{desc|code|word|...}} veya {{desctree|code|word|...}}
        desc_pattern = r'\{\{desc(?:tree)?\|([a-z0-9\-]+)\|([^|\}]+)(?:\|([^|\}]+))?(?:\|ts=([^|\}]+))?'
        
        seen_langs = {item["lang_code"]: item for item in result["turkic_languages"]}

        for match in re.finditer(desc_pattern, wt):
            lang_code = match.group(1).strip()
            entry_word = match.group(2).strip()
            extra_word = match.group(3) or ""
            ts_trans = match.group(4) or ""

            # Türki diller haritasında var mı?
            if lang_code in TURKIC_LANGUAGES_MAP:
                display_word = entry_word
                if ts_trans and not ts_trans.startswith("t="):
                    display_word = f"{entry_word} ({ts_trans})"
                elif extra_word and not extra_word.startswith("t=") and not extra_word.startswith("bor="):
                    display_word = f"{entry_word} ({extra_word})"

                if lang_code not in seen_langs:
                    item = {
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": result["root"]["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    }
                    result["turkic_languages"].append(item)
                    seen_langs[lang_code] = item

    def _parse_word_page(self, wt: str, word_clean: str, result: Dict[str, Any]) -> None:
        seen_langs = {item["lang_code"]: item for item in result["turkic_languages"]}

        # Wiktionary dil başlıklarını ara ==Turkish==, ==Azerbaijani== vs.
        lang_sections = re.split(r'==\s*([A-Za-z\s]+)\s*==', wt)
        for i in range(1, len(lang_sections) - 1, 2):
            lang_header = lang_sections[i].strip()
            section_content = lang_sections[i+1]

            # Header ile lang_code eşleştir
            code = None
            for lc, lname in TURKIC_LANGUAGES_MAP.items():
                if lname.lower() in lang_header.lower() or lang_header.lower() in lname.lower():
                    code = lc
                    break

            if code and code not in seen_langs:
                # Anlam çıkar
                m = re.search(r'#\s*\[\[(.*?)\]\]|#\s*(.*?)\n', section_content)
                meaning = ""
                if m:
                    meaning = (m.group(1) or m.group(2) or "").strip()

                item = {
                    "lang_code": code,
                    "lang_name": TURKIC_LANGUAGES_MAP[code],
                    "word": word_clean,
                    "meaning": meaning or result["root"]["meaning"],
                    "script": "Latin"
                }
                result["turkic_languages"].append(item)
                seen_langs[code] = item
