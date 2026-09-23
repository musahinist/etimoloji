import json
import re
import urllib.parse
import urllib.request
from typing import Any

from engine import config
from engine.fetchers.base import (
    TURKIC_LANGUAGES_MAP,
    BaseFetcher,
    detect_script,
    lang_code_from_wiktionary_header,
)
from engine.logging_setup import get_logger
from engine.utils.network import fetch as http_get

logger = get_logger(__name__)



#: Sayfanın ``==Turkish==`` bölümü (bir sonraki 2. düzey başlığa kadar).
_TURKISH_SECTION = re.compile(r"^==Turkish==\s*$(.*?)(?=^==[^=]|\Z)", re.M | re.S)
_TURKISH_INHERITED = re.compile(r"\{\{inh\|tr\|trk-pro\|([^}]*)\}\}")
_TURKISH_BORROWED = re.compile(r"\{\{(?:bor|bor\+|lbor|slbor)\|tr\||Borrowed from", re.I)


def _turkish_proto_link(wikitext: str) -> tuple[str, str] | None:
    """Türkçe bölümündeki Proto-Türkçe MİRAS bağlantısı; alıntıysa ``None``.

    ⚠️ Eskiden sayfanın herhangi bir yerindeki ilk ``trk-pro`` bağlantısı
    (``inh``/``der``/``cog``, herhangi bir dil bölümü) alınıyor ve o kökün
    BÜTÜN torunları tanık yapılıyordu: Farsça alıntı `parça` için 25 dilde
    *bar "var olmak" biçimleri raporlandı.
    """
    section = _TURKISH_SECTION.search(wikitext)
    if not section or _TURKISH_BORROWED.search(section.group(1)):
        return None
    link = _TURKISH_INHERITED.search(section.group(1))
    if not link:
        return None
    args = link.group(1).split("|")
    positional = [a.strip() for a in args if "=" not in a]
    named = dict(a.split("=", 1) for a in args if "=" in a)
    root = positional[0].lstrip("*") if positional else ""
    # Anlam `t=` ya da 3. konumsal argüman ({{inh|tr|trk-pro|*köŕ||eye}}).
    meaning = named.get("t", "") or (positional[2] if len(positional) > 2 else "")
    return (root, meaning.strip()) if root else None


def _desc_parts(raw_args: str) -> tuple[str, str]:
    """``{{desc|kk|бар|tr=bar}}`` argümanlarından (biçim, okunuş).

    Adlandırılmış argümanlar (``tr=``, ``der=1``, ``bor=1``) biçim DEĞİLDİR;
    eskiden "tr=bar", "- (der=1)" diye tanık listesine basılıyordu.
    """
    positional = [a.strip() for a in raw_args.split("|") if "=" not in a]
    named = dict(a.split("=", 1) for a in raw_args.split("|") if "=" in a)
    form = positional[0] if positional else ""
    return ("" if form in ("", "-") else form), named.get("tr", "").strip()


class WiktionaryFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Wiktionary"

    def _http_get(self, url: str) -> str | None:
        return http_get(url, timeout=config.HTTP_TIMEOUT_LONG)

    def _get_page_wikitext(self, page_title: str) -> str | None:
        url = f"https://en.wiktionary.org/w/api.php?action=parse&page={urllib.parse.quote(page_title)}&format=json&prop=wikitext"
        raw = self._http_get(url)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("Wiktionary wikitext JSON ayrıştırılamadı: %s", page_title, exc_info=True)
            return None
        if "error" in data:
            logger.debug("Wiktionary sayfası yok: %s", page_title)
            return None
        return data.get("parse", {}).get("wikitext", {}).get("*", "")

    def fetch(self, word: str) -> dict[str, Any]:
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
            proto_match = _turkish_proto_link(wt)
            if proto_match:
                proto_root, proto_meaning = proto_match
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

    def _parse_reconstruction_page(self, wt: str, result: dict[str, Any]) -> None:
        # Anlam çekme
        meaning_match = re.search(r'==Proto-Turkic==.*?(?:#\s*\[\[(.*?)\]\]|#\s*(.*?)\n)', wt, re.DOTALL)
        if meaning_match and not result["root"]["meaning"]:
            meaning = (meaning_match.group(1) or meaning_match.group(2) or "").strip()
            result["root"]["meaning"] = meaning

        # Türki diller türevlerini parsing
        # Template format: {{desc|code|word|...}} veya {{desctree|code|word|...}}
        desc_pattern = r'\{\{desc(?:tree)?\|([a-z0-9\-]+)\|([^}]*)\}\}'

        seen_langs = {item["lang_code"]: item for item in result["turkic_languages"]}

        for match in re.finditer(desc_pattern, wt):
            lang_code = match.group(1).strip()
            entry_word, reading = _desc_parts(match.group(2))
            if not entry_word:
                continue

            # Türki diller haritasında var mı?
            if lang_code in TURKIC_LANGUAGES_MAP:
                display_word = f"{entry_word} ({reading})" if reading and reading != entry_word else entry_word

                if lang_code not in seen_langs:
                    item = {
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": result["root"]["meaning"],
                        "script": detect_script(display_word),
                    }
                    result["turkic_languages"].append(item)
                    seen_langs[lang_code] = item

    def _parse_word_page(self, wt: str, word_clean: str, result: dict[str, Any]) -> None:
        seen_langs = {item["lang_code"]: item for item in result["turkic_languages"]}

        # Yalnızca 2. seviye dil başlıkları: ==Turkish==, ==Azerbaijani==
        # Not: eski desen `===Noun===` gibi ALT başlıkları da yakalıyor ve
        # bölüm içeriğini ikiye bölerek anlamların kaybolmasına yol açıyordu.
        lang_sections = re.split(r'^==\s*([^=\n]+?)\s*==\s*$', wt, flags=re.M)
        for i in range(1, len(lang_sections) - 1, 2):
            lang_header = lang_sections[i].strip()
            section_content = lang_sections[i+1]

            # Wiktionary başlıkları İNGİLİZCEDİR ("==Turkish=="); harita ise
            # Türkçe adlar tutar. Doğrudan karşılaştırma hiçbir zaman
            # eşleşmiyordu ve bu ayrıştırıcı fiilen ölüydü.
            code = lang_code_from_wiktionary_header(lang_header)

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
                    "script": detect_script(word_clean),
                }
                result["turkic_languages"].append(item)
                seen_langs[code] = item
