"""Türkçe kelime -> İngilizce anlam köprüsü (9l S1; yalnız verici ETİKETİ için).

Kör/tam indekste birçok Türkçe maddenin tek anlamı Türkçedir (Türkçe Vikisözlük
tanımı); verici havuzlarının (``donors.db``) anlamları İngilizcedir ve anlam
araması (``DonorIndex.by_sense``) boş döner. Köprü, kelimenin İngilizce
karşılıklarını iki sözlük ÇEVİRİ kaynağından toplar:

- ``trwiktionary`` (kaikki ham dökümü): Türkçe maddelerin çeviri bölümündeki
  İngilizce karşılıklar (``tr -> en``) ve İngilizce maddelerin tek sözcüklük
  Türkçe tanımları (ters yön, ``en -> tr``);
- ``enwiktionary`` (kaikki İngilizce sözlüğü): İngilizce maddelerin çeviri
  tablolarındaki Türkçe karşılıklar (ters yön);
- yedek: sözlük indeksinde aynı karşılaştırma biçimli Osmanlıca (``ota``,
  en-Wiktionary) maddenin İngilizce anlamı (:func:`ottoman_sense`).

⚠️ Köken bilgisi KULLANILMAZ: etimoloji alanları, kategoriler ("... kökenli
Türkçe sözcükler") ve İngilizce dışı çeviriler (İtalyanca, Fransızca ... —
çoğu zaman etimonun kendisi) okunmaz; TDK tanımı da kullanılmaz. Anahtar
kelimenin karşılaştırma biçimidir. Sonuç ``data/lexicons/sense_bridge/tr_en.db``
(git-ignored; künye ``tr_en.provenance.json`` commit edilir: iki dökümün ve
tablonun içerik SHA-256'sı).

Kurulum: ``make sense-bridge`` (``scripts/download_sense_bridge.py``; dökümleri
indirir, künyedeki SHA ile karşılaştırır ve tabloyu kurar). Elle:
python -m engine.db.sense_bridge --build --trwikt <raw.jsonl.gz> --enwikt <English.jsonl.gz>

⚠️ Tablo yoksa köprü sessizce kapanmaz, ama ETKİSİZ olur: ``is_english_sense``
her ASCII anlamı İngilizce sayar. Bu yüzden ilk sorguda uyarı loglanır.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form
from engine.utils.provenance import write_if_changed

logger = get_logger(__name__)

BRIDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "lexicons" / "sense_bridge"
BRIDGE_DB = BRIDGE_DIR / "tr_en.db"
PROVENANCE = BRIDGE_DIR / "tr_en.provenance.json"
#: Köprü anlamına en çok bu kadar İngilizce karşılık girer (sık olan önce).
MAX_TERMS = 4

_EN_WORD = re.compile(r"[A-Za-z][A-Za-z' -]{1,40}")
_TR_WORD = re.compile(r"[a-zçğıöşüâîû]{2,}")
_TR_GLOSS = re.compile(r"^\s*([a-zçğıöşüâîû]{2,})\s*[.;]?\s*$")


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def _translations(record: dict) -> list[dict]:
    out = list(record.get("translations") or [])
    for sense in record.get("senses") or []:
        out += sense.get("translations") or []
    return out


def _code(t: dict) -> str:
    return t.get("lang_code") or t.get("code") or ""


def _en_ok(word: str) -> bool:
    return bool(word) and _EN_WORD.fullmatch(word) is not None and len(word.split()) <= 2


def collect(trwikt: Path | None, enwikt: Path | None) -> tuple[dict[str, Counter], set[str]]:
    """(Türkçe karşılaştırma biçimi -> İngilizce karşılık sayacı, İngilizce başlık sözcükleri)."""
    pairs: dict[str, Counter] = defaultdict(Counter)
    vocab: set[str] = set()
    if trwikt:
        for line in _open(trwikt):
            r = json.loads(line)
            lang = r.get("lang_code")
            if lang == "tr" and r.get("pos") != "name":
                key = to_comparison_form(r.get("word") or "")
                for t in _translations(r):
                    w = (t.get("word") or "").strip()
                    if _code(t) == "en" and _en_ok(w) and key:
                        pairs[key][w.lower()] += 2
            elif lang == "en" and r.get("pos") != "name":
                en = (r.get("word") or "").strip()
                if not _en_ok(en):
                    continue
                for sense in r.get("senses") or []:
                    for g in sense.get("glosses") or []:
                        for part in re.split(r"[,;]", g):
                            m = _TR_GLOSS.match(part.lower())
                            if m:
                                pairs[to_comparison_form(m.group(1))][en.lower()] += 1
    if enwikt:
        for line in _open(enwikt):
            r = json.loads(line)
            if r.get("lang_code") not in (None, "en") or r.get("pos") == "name":
                continue
            en = (r.get("word") or "").strip()
            if not _en_ok(en):
                continue
            if " " not in en:
                vocab.add(en.lower())
            if '"tr"' not in line:
                continue
            for t in _translations(r):
                w = (t.get("word") or "").strip().lower()
                if _code(t) == "tr" and _TR_WORD.fullmatch(w):
                    pairs[to_comparison_form(w)][en.lower()] += 1
    return {k: v for k, v in pairs.items() if k and len(k) >= 2}, vocab


def _sha(path: Path | None) -> str:
    if not path:
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_digest(db: Path) -> str:
    """Tablonun İÇERİK özeti (SHA-256; iki tablonun sıralı satırları).

    SQLite dosyasının baytları sayfa düzenine bağlıdır; aynı satırları taşıyan
    iki tabloyu karşılaştırmak için içerik özeti kullanılır (künyede commit edilir)."""
    h = hashlib.sha256()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for table, query in (("bridge", "SELECT comparison, english FROM bridge ORDER BY comparison"),
                             ("en_vocab", "SELECT word FROM en_vocab ORDER BY word")):
            h.update(f"#{table}\n".encode())
            for row in con.execute(query):
                h.update(("\t".join(row) + "\n").encode("utf-8"))
    finally:
        con.close()
    return h.hexdigest()


def provenance_path(out: Path) -> Path:
    """``tr_en.db`` -> ``tr_en.provenance.json`` (aynı dizinde)."""
    return out.with_name(f"{out.stem}.provenance.json")


def build(trwikt: Path | None, enwikt: Path | None, out: Path = BRIDGE_DB) -> dict:
    pairs, vocab = collect(trwikt, enwikt)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.execute("CREATE TABLE bridge (comparison TEXT PRIMARY KEY, english TEXT NOT NULL)")
    con.executemany("INSERT INTO bridge VALUES (?, ?)",
                    [(k, json.dumps(v.most_common(12), ensure_ascii=False)) for k, v in sorted(pairs.items())])
    con.execute("CREATE TABLE en_vocab (word TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO en_vocab VALUES (?)", [(w,) for w in sorted(vocab)])
    con.commit()
    con.close()
    meta = {
        "_schema": "turkic-etymology-sense-bridge/v1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "trwiktionary": {"url": "https://kaikki.org/trwiktionary/raw-wiktextract-data.jsonl.gz",
                             "sha256": _sha(trwikt)},
            "enwiktionary": {"url": "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz",
                             "sha256": _sha(enwikt)},
        },
        "fields_used": "yalnız çeviri (tr->en, en->tr) ve İngilizce maddelerin tek sözcüklük Türkçe tanımı; "
                       "etimoloji/kategori/İngilizce dışı çeviri YOK",
        "entries": len(pairs),
        "en_vocab": len(vocab),
        "content_sha256": table_digest(out),
    }
    # Aynı dökümlerden aynı tablo: yalnız kurulum zamanı değiştiyse künyeye dokunulmaz.
    return write_if_changed(provenance_path(out), meta, json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                            volatile=("built_at",))


@lru_cache(maxsize=1)
def _connection() -> sqlite3.Connection | None:
    if not BRIDGE_DB.exists():
        # Sessizce kötüleşmesin: tablo yoksa S1 (SENSE_BRIDGE) fiilen kapalıdır.
        logger.warning("Anlam köprüsü tablosu yok (%s): Türkçe anlamlı maddelerde verici etiketi "
                       "köprüsüz kalır; kurmak için `make sense-bridge`", BRIDGE_DB)
        return None
    return sqlite3.connect(f"file:{BRIDGE_DB}?mode=ro", uri=True, check_same_thread=False)


@lru_cache(maxsize=50000)
def english_sense(comparison: str, max_terms: int = MAX_TERMS) -> str:
    """Kelimenin İngilizce köprü anlamı ("band, stern"); yoksa boş."""
    con = _connection()
    if con is None or not comparison:
        return ""
    row = con.execute("SELECT english FROM bridge WHERE comparison = ?", (comparison,)).fetchone()
    if not row:
        return ""
    return ", ".join(w for w, _ in json.loads(row[0])[:max_terms])


@lru_cache(maxsize=50000)
def ottoman_sense(comparison: str) -> str:
    """Yedek köprü: aynı karşılaştırma biçimli Osmanlıca maddenin İngilizce anlamı."""
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not comparison or not index.exists:
        return ""
    with index.connect() as con:
        rows = con.execute("SELECT gloss FROM entries WHERE lang_code = 'ota' AND comparison = ?"
                           " AND gloss IS NOT NULL AND gloss != '' ORDER BY id", (comparison,)).fetchall()
    for (gloss,) in rows:
        if is_english_sense(gloss):
            return gloss
    return ""


_EN_STOP = frozenset("the of a an to or and in on for by with from as which that is used any one who "
                     "something someone person thing".split())
_TR_STOP = frozenset("bir ve ile veya olan için gibi bu da de ya çok en olarak eden yapan durumu türlü "
                     "biçimde işi kimse şey ait".split())
_TR_CHARS = re.compile(r"[çğışöüâî]")


def is_english_sense(sense: str) -> bool:
    """Anlam metni İngilizce mi? (Türkçe Vikisözlük tanımlarını ayırmak için kaba sezgi.)"""
    tokens = re.findall(r"[^\W\d_]+", sense.lower())
    if not tokens:
        return False
    en = sum(t in _EN_STOP for t in tokens)
    tr = sum(t in _TR_STOP for t in tokens)
    if en:
        return en > tr
    if tr or _TR_CHARS.search(sense.lower()):
        return False
    # Durak sözcüğü yok, ASCII: İngilizce başlık sözlüğünde olan sözcükler çoğunluk mu?
    content = [t for t in tokens if len(t) > 2] or tokens
    known = _known_english(tuple(content))
    return known * 2 > len(content)


def _known_english(tokens: tuple[str, ...]) -> int:
    con = _connection()
    if con is None:
        return len(tokens)
    return sum(con.execute("SELECT 1 FROM en_vocab WHERE word = ?", (t,)).fetchone() is not None for t in tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--trwikt", type=Path)
    ap.add_argument("--enwikt", type=Path)
    a = ap.parse_args()
    if a.build:
        print(json.dumps(build(a.trwikt, a.enwikt), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
