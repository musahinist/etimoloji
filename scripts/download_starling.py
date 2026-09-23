#!/usr/bin/env python3
"""
Starling Altay paketinden Türk ve Moğol etimoloji tablolarını indirir — ``make starling``.

Paket (``ALTAIC.exe``) kendiliğinden açılan bir zip arşividir; Python'un
``zipfile`` modülü başındaki yürütülebilir kısmı atlayıp açabilir. Yalnız
``turcet`` (Türk) ve ``monget`` (Moğol, verici dil) tabloları çıkarılır.

Ham dosyalar repoya alınmaz; künye ve SHA-256 ``data/starling/_provenance.json``
dosyasında commit edilir.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db.starling import STARLING_DIR, load_turcet  # noqa: E402

URL = "https://starlingdb.org/download/ALTAIC.exe"
TABLES = ("turcet.dbf", "turcet.var", "monget.dbf", "monget.var")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    response = requests.get(URL, timeout=120, headers={"User-Agent": "turkic-etymology-engine/3.0 (+research)"})
    response.raise_for_status()
    archive = response.content

    STARLING_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = {Path(n).name.lower(): n for n in bundle.namelist()}
        for table in TABLES:
            if table not in names:
                print(f"⚠️ arşivde yok: {table}")
                continue
            data = bundle.read(names[table])
            (STARLING_DIR / table).write_bytes(data)
            files[table] = {"sha256": _sha256(data), "bytes": len(data)}

    roots = len(load_turcet())
    provenance = {
        "_schema": "turkic-etymology-starling-provenance/v1",
        "url": URL,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "archive_sha256": _sha256(archive),
        "citation": "Dybo, A. & Starostin, S. (2005). Turkic etymology. Starling / Tower of Babel.",
        "role": "Proto-Türkçe kök ve tarihli Eski Türkçe tanık (ikinci görüş)",
        "caveat": (
            "Altay hipotezi okulundandır: Proto-Türkçe biçimler ve Türk içi tanıklar "
            "kullanılır, Altay karşılaştırması akrabalık kanıtı sayılmaz. Tablo tüm "
            "Türk söz varlığını kapsamaz (ör. deniz, bağla- yok)."
        ),
        "turkic_roots": roots,
        "files": files,
    }
    (STARLING_DIR / "_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Starling: {roots} Türk kökü -> {STARLING_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
