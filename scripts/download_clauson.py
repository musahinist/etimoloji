#!/usr/bin/env python3
"""
Clauson 1972, *An Etymological Dictionary of Pre-Thirteenth-Century Turkish*
(EDT) — ``make clauson``.

TurkicWorld'deki (N. Kisamov) düzeltilmiş Unicode HTML transkripsiyonunun 10
parçasını ``data/clauson/`` altına indirir ve ``engine.db.clauson`` ile
JSONL'e ayrıştırır. ⚠️ Metin OUP telifidir: HTML ve JSONL repoya ALINMAZ
(``.gitignore``); yalnız künye ve SHA-256'lar ``_provenance.json`` ile commit
edilir. Atlama kararı dosyaların kendisine (SHA-256) bakar; künye aynı veriyi
anlatıyorsa yeniden yazılmaz (indirme zamanı değişmez).

Yedek kaynak (kullanılmıyor): archive.org ``sir-gerard-clauson-an-etymological-
dictionary-of-pre-thirteenth-century-turkish-`` ``_djvu.txt``; OCR'da ``ŋ``
``p``/``n`` okunmuş (``tapsuk``), başlık/anlam ayrımı yok.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db.clauson import BASE_URL, CLAUSON_DIR, JSONL_NAME, PARTS, parse_dir, part_file, write_jsonl  # noqa: E402
from engine.utils.provenance import write_if_changed  # noqa: E402

LANDING = "http://s155239215.onlinehome.us/turkic/40_Language/ClausonEDT/"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(directory: Path) -> dict:
    provenance_path = directory / "_provenance.json"
    if not provenance_path.exists():
        return {}
    return json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}


def is_current(directory: Path) -> bool:
    """Künyedeki bütün dosyalar diskte VE SHA-256'ları aynı mı?"""
    files = _files(directory)
    return bool(files) and all(
        (directory / name).is_file() and _sha256(directory / name) == info.get("sha256")
        for name, info in files.items()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clauson EDT indirici/ayrıştırıcı")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir ve ayrıştır")
    args = ap.parse_args(argv)
    if not args.force and is_current(CLAUSON_DIR):
        print(f"Clauson: zaten var, SHA-256 künyeyle aynı (--force ile yeniden) -> {CLAUSON_DIR}")
        return 0
    CLAUSON_DIR.mkdir(parents=True, exist_ok=True)
    known = _files(CLAUSON_DIR)
    for part in PARTS:
        name = part_file(part)
        path = CLAUSON_DIR / name
        if not args.force and path.is_file() and _sha256(path) == (known.get(name) or {}).get("sha256"):
            continue
        response = requests.get(
            BASE_URL.format(part), timeout=120, headers={"User-Agent": "turkic-etymology-engine/3.0 (+research)"}
        )
        if response.status_code != 200 or b"<p" not in response.content or len(response.content) < 100_000:
            print(f"  ! {name} indirilemedi: HTTP {response.status_code}")
            return 1
        path.write_bytes(response.content)
        print(f"  + {name}: {len(response.content) / 1e6:.2f} MB")

    records = parse_dir(CLAUSON_DIR)
    jsonl = CLAUSON_DIR / JSONL_NAME
    write_jsonl(records, jsonl)
    prefixes = Counter(r.get("prefix") or "" for r in records)
    print(f"  + {JSONL_NAME}: {len(records)} madde")

    names = [part_file(p) for p in PARTS] + [JSONL_NAME]
    provenance = {
        "_schema": "turkic-etymology-clauson-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "citation": (
            "Sir Gerard Clauson, An Etymological Dictionary of Pre-Thirteenth-Century Turkish. "
            "Oxford: Clarendon Press 1972."
        ),
        "transcription": "N. Kisamov, TurkicWorld — substantially corrected and annotated Unicode HTML (10 parça)",
        "landing_page": LANDING,
        "url_pattern": BASE_URL,
        "license": "OUP telifi; metin repoya alınmaz, yalnız yerel kullanım",
        "note": (
            "Madde başı Eski Türkçe (8.-13. yy) biçimdir, Proto-Türkçe kök değildir. Mavi parantez "
            "anlamlar (gloss_tw) Kisamov'un eklemesidir. Starling turcet EDT'ye 1.504 kez atıf yapar "
            "(döngüsellik). JSONL, engine/db/clauson.py ayrıştırıcısının çıktısıdır."
        ),
        "counts": {
            "entries": len(records),
            "derived_D": sum(1 for r in records if "D" in (r.get("prefix") or "")),
            "with_base": sum(1 for r in records if r.get("base")),
            "cross_references": sum(1 for r in records if r.get("see")),
            "foreign_F": sum(1 for r in records if "F" in (r.get("prefix") or "")),
            "with_attestation": sum(1 for r in records if r.get("attestations")),
            "top_prefixes": dict(prefixes.most_common(8)),
        },
        "files": {n: {"bytes": (CLAUSON_DIR / n).stat().st_size, "sha256": _sha256(CLAUSON_DIR / n)} for n in names},
    }
    write_if_changed(
        CLAUSON_DIR / "_provenance.json", provenance, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
