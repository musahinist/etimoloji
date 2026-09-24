"""
Proto-Türkçe kökün torunları — yerel Wiktionary dökümünden (kaikki ``trk-pro``).

Canlı ``WiktionaryFetcher``'ın yaptığını ağa çıkmadan yapar: sorgunun miras
kaydındaki Proto-Türkçe biçimi (indeks ``donor_form``) bulur ve o kökün
sayfasındaki torun ağacını tanık olarak döndürür.

⚠️ Eşsesli kök: `el` için hem *el "el (organ)" hem *ēl "halk" kaydı çıkabilir.
Torunlar kökün ANLAMINI taşır ve ``meaning_check`` ile işaretlenir; arama
motoru onu sorgunun anlamıyla karşılaştırır, yanlış kökün torunları toplu
elenir. Ağaçta ``borrowed`` etiketli dallar (Eski Yunancaya geçmiş Hazarca
biçim gibi) ve yıldızlı ara biçimler tanık değildir.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from functools import lru_cache
from typing import Any

from engine.config import LEXICON_DIR
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher, detect_script

DUMP = LEXICON_DIR / "trk-pro.jsonl.gz"


def _key(proto: str) -> str:
    """Yazım farkını kapatan anahtar: Wiktionary sayfa başlığı `körᶻ`,
    indeks `*köŕ` yazar (ŕ = rᶻ, ĺ = lˢ, e = ä); uzunluk işaretleri atılır."""
    text = proto.strip().lstrip("*").strip("-").replace("rᶻ", "ŕ").replace("lˢ", "ĺ").replace("ä", "e")
    text = unicodedata.normalize("NFD", text.lower())
    return unicodedata.normalize("NFC", "".join(c for c in text if c not in "\u0304\u0306"))


@lru_cache(maxsize=1)
def _pages() -> dict[str, dict[str, Any]]:
    if not DUMP.exists():
        return {}
    pages: dict[str, dict[str, Any]] = {}
    with gzip.open(DUMP, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("word"):
                pages.setdefault(_key(str(record["word"])), record)
    return pages


def _is_borrowed(node: dict[str, Any]) -> bool:
    """Alıntı dalı ya da ek eklenerek yeniden biçimlenmiş türev (Çuvaşça
    куҫлӗх "gözlük" `göz`ün akrabası değil, Çuvaşçada türemiş kelimedir)."""
    tags = [str(t).lower() for t in [*(node.get("tags") or []), *(node.get("raw_tags") or [])]]
    return any("borrow" in t or "reshaped" in t or "addition of morphemes" in t for t in tags)


def descendants(proto: str) -> list[tuple[str, str, str]]:
    """``(dil, biçim, okunuş)`` — alıntı dalları ve yıldızlı ara biçimler hariç."""
    page = _pages().get(_key(proto))
    out: list[tuple[str, str, str]] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes or []:
            if _is_borrowed(node):
                continue
            code, word = str(node.get("lang_code") or ""), str(node.get("word") or "").strip()
            if code in TURKIC_LANGUAGES_MAP and word and not word.startswith(("*", "-")):
                out.append((code, word, str(node.get("roman") or "")))
            walk(node.get("descendants") or [])

    if page:
        walk(page.get("descendants") or [])
    return out


def gloss(proto: str) -> str:
    page = _pages().get(_key(proto))
    for sense in (page or {}).get("senses") or []:
        if sense.get("glosses"):
            return str(sense["glosses"][0])
    return ""


class LocalProtoTurkicFetcher(BaseFetcher):
    #: Yerel veri.
    is_seed_source = True
    #: Kök varyantı almaz (bkz. `NorthEuraLexFetcher`).
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "Proto-Türkçe kök torunları (yerel Wiktionary dökümü)"

    def fetch(self, word: str) -> dict[str, Any]:
        from engine.db.lexicon_index import LexiconIndex

        result = self.empty_result()
        query = (word or "").strip().lower()
        index = LexiconIndex()
        if not query or not index.exists or not _pages():
            return result
        protos = []
        for row in index.lookup(query, languages=["tr", "ota"], limit=10):
            form = re.sub(r"<[^<>]*>", "", str(row.get("donor_form") or "")).strip()
            if row.get("origin") == "miras" and row.get("donor_lang") == "trk-pro" and form and form not in protos:
                protos.append(form)
        seen: set[tuple[str, str]] = set()
        for proto in protos:
            meaning = gloss(proto)
            for code, form, reading in descendants(proto):
                if code == "tr" or (code, form) in seen:  # sorgu kendi tanığı olmaz
                    continue
                seen.add((code, form))
                entry = self.make_entry(code, form, meaning, script=detect_script(form))
                if reading:
                    entry["latin_transliteration"] = reading
                entry["etymology"] = f"Proto-Türkçe *{proto.lstrip('*')} torunu (Wiktionary)"
                # Kökün anlamı sorgunun anlamıyla doğrulanır (eşsesli kök).
                entry["meaning_check"] = True
                result["turkic_languages"].append(entry)
        return result
