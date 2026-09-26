#!/usr/bin/env python3
"""
Eren (1999) ve Gülensoy (2007) Türkiye Türkçesi etimoloji sözlükleri —
``make erengul``.

archive.org'daki tesseract OCR metinlerini (``_djvu.txt``) ``data/erengul/``
altına indirir ve ``engine.db.eren_gulensoy`` ile JSONL'e ayrıştırır.
⚠️ Metinler telifli: ``_djvu.txt`` ve JSONL repoya ALINMAZ (``.gitignore``);
yalnız künye ve SHA-256'lar ``_provenance.json`` ile commit edilir. Atlama
kararı dosyaların kendisine (SHA-256) bakar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db.eren_gulensoy import ERENGUL_DIR, JSONL_NAME, SOURCES, parse_dir, write_jsonl  # noqa: E402
from engine.utils.provenance import write_if_changed  # noqa: E402

DOWNLOAD = "https://archive.org/download/{}/{}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(directory: Path) -> dict:
    provenance_path = directory / "_provenance.json"
    if not provenance_path.exists():
        return {}
    return json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}


def is_current(directory: Path) -> bool:
    files = _files(directory)
    return bool(files) and all(
        (directory / name).is_file() and _sha256(directory / name) == info.get("sha256")
        for name, info in files.items()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Eren + Gülensoy indirici/ayrıştırıcı")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir ve ayrıştır")
    args = ap.parse_args(argv)
    if not args.force and is_current(ERENGUL_DIR):
        print(f"Eren/Gülensoy: zaten var, SHA-256 künyeyle aynı (--force ile yeniden) -> {ERENGUL_DIR}")
        return 0
    ERENGUL_DIR.mkdir(parents=True, exist_ok=True)
    known = _files(ERENGUL_DIR)
    for name, (item, remote) in SOURCES.items():
        path = ERENGUL_DIR / name
        if not args.force and path.is_file() and _sha256(path) == (known.get(name) or {}).get("sha256"):
            continue
        response = requests.get(DOWNLOAD.format(item, urllib.parse.quote(remote)), timeout=300,
                                headers={"User-Agent": "turkic-etymology-engine/3.0 (+research)"})
        if response.status_code != 200 or len(response.content) < 500_000:
            print(f"  ! {name} indirilemedi: HTTP {response.status_code}")
            return 1
        path.write_bytes(response.content)
        print(f"  + {name}: {len(response.content) / 1e6:.2f} MB")

    records = parse_dir(ERENGUL_DIR)
    write_jsonl(records, ERENGUL_DIR / JSONL_NAME)
    by_source = Counter(r["source"] for r in records)
    print(f"  + {JSONL_NAME}: {len(records)} madde {dict(by_source)}")

    def count(src: str, field: str, value: str | None = None) -> int:
        return sum(1 for r in records if r["source"] == src and (r.get(field) == value if value else r.get(field)))

    names = [*SOURCES, JSONL_NAME]
    provenance = {
        "_schema": "turkic-etymology-erengul-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "citations": {
            "eren": "Hasan Eren, Türk Dilinin Etimolojik Sözlüğü. Ankara: Bizim Büro 1999.",
            "gulensoy": "Tuncer Gülensoy, Türkiye Türkçesindeki Türkçe Sözcüklerin Köken Bilgisi Sözlüğü, "
                        "2 cilt. Ankara: TDK 2007.",
        },
        "landing_pages": {name: f"https://archive.org/details/{item}" for name, (item, _) in SOURCES.items()},
        "license": "telifli; archive.org tesseract OCR metni, repoya alınmaz, yalnız yerel kullanım",
        "note": (
            "Kaynaklar TANIK değildir: yalnız başlık/köken notu (engine/db/eren_gulensoy.py). Eren "
            "EDAL'dan (2003) önce yayımlandı; Räsänen/Clauson/ЭСТЯ'ya atıf yapar (Starling de). "
            "Gülensoy OCR'ı iki sütunlu düzen yüzünden gürültülü."
        ),
        "counts": {
            src: {
                "entries": by_source[src],
                "redirects": count(src, "redirect"),
                "with_root": count(src, "root"),
                "with_ot_form": count(src, "ot_form"),
                "loan": count(src, "origin", "loan"),
                "turkish": count(src, "origin", "turkish"),
                "unknown": count(src, "origin", "unknown"),
            }
            for src in ("eren", "gulensoy")
        },
        "files": {n: {"bytes": (ERENGUL_DIR / n).stat().st_size, "sha256": _sha256(ERENGUL_DIR / n)} for n in names},
    }
    write_if_changed(
        ERENGUL_DIR / "_provenance.json", provenance, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
