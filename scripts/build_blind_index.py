"""Köken etiketleri boşaltılmış KÖR sözlük indeksi (döngüsellik denetimi K2).

Türk dilleri arası alıntı altını (``engine/evaluation/xturkic_gold.py``)
kaikki **en** dökümünün etimoloji şablonlarından kurulur. Motorun
``zincir_kanıtı`` sinyali ve ``_index_attests_loan`` aynı etiketi indeksin
``origin``/``donor_*`` sütunlarından okur; ölçüm o sütunlar doluyken
yapılırsa sistem kendi girdisini "doğrular".

Bu betik ``data/lexicons/index.db``'nin bir KOPYASINI kurar ve bütün Türk
dillerinin satırlarında şu alanları ``NULL`` yapar:

    origin, donor_lang, donor_form, etymology, formation, cognates

Sözcük, karşılaştırma biçimi, anlam (gloss), IPA ve uzun ünlüler kalır —
tanık arama ve verici yakınlığı onlarla çalışır. FTS tablosu yeniden kurulur
(``etymology`` sütunu içerik tablosundan okunur).

⚠️ Asıl ``index.db``'ye DOKUNULMAZ, yeniden KURULMAZ; yalnız dosya kopyalanır.

Kullanım::

    python scripts/build_blind_index.py [--out data/cache/work/xtr/index_blind.db]
    ETY_LEXICON_INDEX=data/cache/work/xtr/index_blind.db python -m ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.db.lexicon_index import TURKIC_FAMILY_CODES  # noqa: E402
from engine.fetchers.base import TURKIC_LANGUAGES_MAP  # noqa: E402

#: Türk dili sayılan kodlar. ⚠️ İkisinin BİRLEŞİMİ: indeks kodları
#: ``TURKIC_LANGUAGES_MAP``ten gelir (``khk`` = Hakasça, ``oui`` = Eski
#: Uygurca), şablon kodları ``TURKIC_FAMILY_CODES``ten; yalnız ikincisi
#: kullanılınca bu iki dil boşaltılmadan kalıyordu (ölçüldü).
TURKIC = frozenset(TURKIC_FAMILY_CODES) | frozenset(TURKIC_LANGUAGES_MAP)

SOURCE = ROOT / "data" / "lexicons" / "index.db"
DEFAULT_OUT = ROOT / "data" / "cache" / "work" / "xtr" / "index_blind.db"

#: Boşaltılan sütunlar — köken beyanı taşıyan her alan.
BLINDED_COLUMNS = ("origin", "donor_lang", "donor_form", "etymology", "formation", "cognates")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, out: Path) -> dict[str, object]:
    if not source.exists():
        raise SystemExit(f"kaynak indeks yok: {source}")
    if out.resolve() == source.resolve():
        raise SystemExit("çıktı kaynak indeksin kendisi olamaz")
    source_sha = sha256_of(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    shutil.copyfile(source, tmp)

    connection = sqlite3.connect(tmp)
    try:
        langs = [r[0] for r in connection.execute("SELECT DISTINCT lang_code FROM entries")]
        turkic = sorted(lang for lang in langs if lang in TURKIC)
        other = sorted(lang for lang in langs if lang not in TURKIC)
        placeholders = ",".join("?" * len(turkic))
        before = connection.execute(
            f"SELECT COUNT(*) FROM entries WHERE origin IS NOT NULL "
            f"AND lang_code IN ({placeholders})",
            turkic,
        ).fetchone()[0]
        assignments = ", ".join(f"{col} = NULL" for col in BLINDED_COLUMNS)
        connection.execute(
            f"UPDATE entries SET {assignments} WHERE lang_code IN ({placeholders})", turkic
        )
        connection.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
        left = {
            col: connection.execute(
                f"SELECT COUNT(*) FROM entries WHERE {col} IS NOT NULL "
                f"AND lang_code IN ({placeholders})",
                turkic,
            ).fetchone()[0]
            for col in BLINDED_COLUMNS
        }
        info = {
            "blind": "1",
            "blind_source_sha256": source_sha,
            "blind_columns": json.dumps(BLINDED_COLUMNS),
            "blind_built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO build_info(key, value) VALUES (?, ?)", info.items()
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    tmp.replace(out)
    report = {
        "source": str(source),
        "source_sha256": source_sha,
        "out": str(out),
        "blinded_languages": turkic,
        "untouched_languages": other,
        "origin_rows_before": before,
        "non_null_after": left,
    }
    if any(left.values()):
        raise SystemExit(f"boşaltma eksik: {left}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    report = build(args.source, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
