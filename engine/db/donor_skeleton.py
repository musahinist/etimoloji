"""
Verici maddelerinin ÜNSÜZ İSKELETİ dizini — anlamdan bağımsız, biçim-öncelikli arama (9o).

``donors.db`` yalnız anlam (FTS) ve karşılaştırma biçimi eşitliğiyle aranabilir. 9o tanısı
(``data/cache/work/donor9o/PREREG.md``): kesin verici etiketi verilemeyen maddelerde doğru
etimon çoğu zaman sözlükte VAR ama anlam havuzuna girmiyor — İngilizce tanım ile verici anlamı
ortak sözcük taşımıyor (sultan "A monarchic title …" ~ سلطان "sultan"), Türkçe anlamın köprüsü
yok ya da sırasız ``LIMIT 200`` kesiyor. Bu dizin karşılaştırma biçiminin ünsüz iskeletinden
(:func:`engine.nlp.donor_proximity.consonant_skeleton`) madde kimliğine gider; aday kümesi
küçüktür ve anlam denetimi sonradan, geniş anlam sözcükleriyle yapılır.

``donors.db`` YENİDEN KURULMAZ: dizin ayrı dosyadır (:data:`SKELETON_DB`) ve ilk kullanımda
``donors.db``den türetilir; ``donors.db`` değişirse (boyut / değişiklik zamanı) yeniden kurulur.

    python -m engine.db.donor_skeleton --build
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.db.donor_index import DEFAULT_DB
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SKELETON_DB = DEFAULT_DB.with_name("donor_skeleton.db")
#: İskelet tanımı değişirse artırılır (dizin yeniden kurulur).
SKELETON_VERSION = "1"


def _stamp(donors: Path) -> str:
    st = donors.stat()
    return f"{SKELETON_VERSION}:{st.st_size}:{st.st_mtime_ns}"


def build(donors: Path = DEFAULT_DB, out: Path = SKELETON_DB) -> int:
    """``donors.db`` -> (iskelet, madde kimliği) tablosu. Döndürür: satır sayısı."""
    from engine.nlp.donor_proximity import consonant_skeleton

    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{donors}?mode=ro", uri=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE skel (skeleton TEXT NOT NULL, id INTEGER NOT NULL)")
    con.execute("CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT)")
    n = 0
    batch: list[tuple[str, int]] = []
    for row_id, comparison in src.execute("SELECT id, comparison FROM donor_entries WHERE from_turkic = 0"):
        sk = consonant_skeleton(comparison or "")
        if len(sk) >= 2:
            batch.append((sk, row_id))
        if len(batch) >= 50000:
            con.executemany("INSERT INTO skel VALUES (?, ?)", batch)
            n += len(batch)
            batch.clear()
    con.executemany("INSERT INTO skel VALUES (?, ?)", batch)
    n += len(batch)
    con.execute("CREATE INDEX idx_skel ON skel(skeleton)")
    con.execute("INSERT INTO info VALUES ('stamp', ?)", (_stamp(donors),))
    con.commit()
    con.close()
    src.close()
    os.replace(tmp, out)
    logger.info("verici iskelet dizini kuruldu: %s (%d satır)", out, n)
    return n


def _fresh(donors: Path, out: Path) -> bool:
    if not out.exists():
        return False
    try:
        con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
        row = con.execute("SELECT value FROM info WHERE key = 'stamp'").fetchone()
        con.close()
    except sqlite3.Error:
        return False
    return bool(row) and row[0] == _stamp(donors)


@lru_cache(maxsize=4)
def _ready(donors: Path, out: Path) -> bool:
    if not donors.exists():
        return False
    if not _fresh(donors, out):
        try:
            build(donors, out)
        except (OSError, sqlite3.Error):
            logger.warning("verici iskelet dizini kurulamadı", exc_info=True)
            return False
    return True


def reset_cache() -> None:
    _ready.cache_clear()


def by_skeleton(skeletons: set[str] | list[str], languages: list[str] | None = None,
                donors: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    """Ünsüz iskeleti verilenlerden biri olan verici maddeleri (Türkçeden alınmışlar hariç).

    Dizin ``donors`` ile aynı dizindeki ``donor_skeleton.db``dir; yoksa ya da eskiyse kurulur.
    """
    skeletons = sorted({s for s in skeletons if s})
    donors = Path(donors)
    out = donors.with_name(SKELETON_DB.name)
    if not skeletons or not _ready(donors, out):
        return []
    con = sqlite3.connect(f"file:{donors}?mode=ro", uri=True)
    try:
        con.execute("ATTACH DATABASE ? AS sk", (f"file:{out}?mode=ro",))
        query = ("SELECT e.id, e.lang_code, e.word, e.comparison, e.gloss FROM sk.skel s"
                 " JOIN donor_entries e ON e.id = s.id"
                 f" WHERE s.skeleton IN ({','.join('?' * len(skeletons))})")
        params: list[Any] = list(skeletons)
        if languages:
            query += f" AND e.lang_code IN ({','.join('?' * len(languages))})"
            params += list(languages)
        return [{"id": r[0], "lang_code": r[1], "word": r[2], "comparison": r[3], "gloss": r[4] or ""}
                for r in con.execute(query + " ORDER BY e.id", params)]
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        print(build())


if __name__ == "__main__":
    main()
