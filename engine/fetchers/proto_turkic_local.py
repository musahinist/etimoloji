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
import os
import re
import sqlite3
import threading
import unicodedata
from functools import lru_cache
from typing import Any

from engine.config import LEXICON_DIR, PROJECT_ROOT
from engine.fetchers.base import (
    TURKIC_LANGUAGES_MAP,
    WIKTIONARY_CODE_ALIASES,
    BaseFetcher,
    detect_script,
    lang_code_from_wiktionary,
)
from engine.logging_setup import get_logger

logger = get_logger(__name__)

DUMP = LEXICON_DIR / "trk-pro.jsonl.gz"
#: Dökümden türetilmiş küçük önbellek: anahtar başına hazır torun listesi ve
#: ilk anlam. Döküm değişince (boyut/mtime) yeniden kurulur.
CACHE = PROJECT_ROOT / "data" / "cache" / "trk-pro.sqlite3"
_CACHE_VERSION = "2"


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


def _page_descendants(page: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes or []:
            if _is_borrowed(node):
                continue
            # kaikki ISO kodu yazar (Hakasça `kjh`, Salarca `slr`); motor
            # koduna çevrilmeden süzülünce 511 + 598 torun atılıyordu.
            code = lang_code_from_wiktionary(str(node.get("lang_code") or ""))
            word = str(node.get("word") or "").strip()
            if code in TURKIC_LANGUAGES_MAP and word and not word.startswith(("*", "-")):
                out.append((code, word, str(node.get("roman") or "")))
            walk(node.get("descendants") or [])

    if page:
        walk(page.get("descendants") or [])
    return out


def _page_gloss(page: dict[str, Any] | None) -> str:
    for sense in (page or {}).get("senses") or []:
        if sense.get("glosses"):
            return str(sense["glosses"][0])
    return ""


def _dump_signature() -> str:
    """Döküm (boyut, mtime) ve önbelleğe gömülü süzgeç (Türk dili kodları ve
    Wiktionary kod eşlemesi) değişince önbellek geçersizleşir; ``_is_borrowed``
    değişirse sürümü artır."""
    stat = DUMP.stat()
    langs = ",".join(sorted(TURKIC_LANGUAGES_MAP))
    aliases = ",".join(f"{k}>{v}" for k, v in sorted(WIKTIONARY_CODE_ALIASES.items()))
    return f"{_CACHE_VERSION}:{stat.st_size}:{stat.st_mtime_ns}:{langs}:{aliases}"


def _build_cache(signature: str) -> None:
    """Dökümü bir kez tarar; geçici dosyaya yazıp atomik olarak yerine koyar
    (paralel süreçler yarım dosya görmez)."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        conn = sqlite3.connect(tmp)
        conn.execute("CREATE TABLE meta (signature TEXT NOT NULL)")
        conn.execute("CREATE TABLE roots (key TEXT PRIMARY KEY, descendants TEXT NOT NULL, gloss TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO roots VALUES (?, ?, ?)",
            (
                (key, json.dumps(_page_descendants(page), ensure_ascii=False), _page_gloss(page))
                for key, page in _pages().items()
            ),
        )
        conn.execute("INSERT INTO meta VALUES (?)", (signature,))
        conn.commit()
        conn.close()
        os.replace(tmp, CACHE)
    finally:
        tmp.unlink(missing_ok=True)
    _pages.cache_clear()


def _cache_signature() -> str | None:
    try:
        conn = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT signature FROM meta").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cache_ready() -> bool:
    """Önbellek dökümle uyumluysa True; değilse kurar. Döküm yoksa False."""
    if not DUMP.exists():
        return False
    with _lock:
        signature = _dump_signature()
        if _cache_signature() != signature:
            _build_cache(signature)
    return True


@lru_cache(maxsize=512)
def _entry(key: str) -> tuple[tuple[tuple[str, str, str], ...], str] | None:
    if not _cache_ready():
        return None
    conn = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT descendants, gloss FROM roots WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return tuple(tuple(item) for item in json.loads(row[0])), row[1]


def descendants(proto: str) -> list[tuple[str, str, str]]:
    """``(dil, biçim, okunuş)`` — alıntı dalları ve yıldızlı ara biçimler hariç."""
    entry = _entry(_key(proto))
    return [tuple(item) for item in entry[0]] if entry else []  # type: ignore[misc]


def gloss(proto: str) -> str:
    entry = _entry(_key(proto))
    return entry[1] if entry else ""


class LocalProtoTurkicFetcher(BaseFetcher):
    #: Yerel, indirilmiş veri (tohum değil).
    is_local = True
    #: Kök varyantı almaz (bkz. `NorthEuraLexFetcher`).
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "Proto-Türkçe kök torunları (yerel Wiktionary dökümü)"

    def fetch(self, word: str) -> dict[str, Any]:
        # Sözleşme: fetch() istisna atmaz (bozuk döküm, kilitli/bozuk SQLite
        # önbelleği ya da indeks aramayı düşürmesin).
        try:
            return self._fetch(word)
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
            return self.empty_result()

    def _fetch(self, word: str) -> dict[str, Any]:
        from engine.db.lexicon_index import LexiconIndex

        result = self.empty_result()
        query = (word or "").strip().lower()
        index = LexiconIndex()
        if not query or not index.exists or not _cache_ready():
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
