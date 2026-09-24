#!/usr/bin/env python3
"""
Wilkens 2021, *Handwörterbuch des Altuigurischen* — ``make wilkens``.

Açık erişim PDF'i (Universitätsverlag Göttingen, CC BY-SA 4.0, DOI
10.17875/gup2021-1590) ``data/wilkens/`` altına indirir ve
``engine.db.wilkens`` ile JSONL'e ayrıştırır (``pdfminer.six`` gerekir:
``pip install -e ".[pdf]"``). PDF ve JSONL repoya alınmaz; künye ve iki
dosyanın SHA-256'sı ``_provenance.json`` ile commit edilir. Atlama kararı
dosyaların kendisine (SHA-256) bakar; künye aynı veriyi anlatıyorsa yeniden
yazılmaz (indirme zamanı değişmez).

⚠️ PDF'i ``data/books/`` altına KOYMAYIN: o dizin ``LocalPdfBooksFetcher``
tarafından tam metin taranır; 928 sayfalık sözlük her aramada serbest metin
eşleşmesi üretirdi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db.wilkens import JSONL_NAME, PDF_NAME, WILKENS_DIR, parse_pdf, write_jsonl  # noqa: E402
from engine.utils.provenance import write_if_changed  # noqa: E402

URL = (
    "https://univerlag.uni-goettingen.de/bitstream/handle/3/"
    "isbn-978-3-86395-481-9/Wilkens_handwoerterbuch.pdf?sequence=4&isAllowed=y"
)
LANDING = "https://univerlag.uni-goettingen.de/handle/3/isbn-978-3-86395-481-9"
DOI = "https://doi.org/10.17875/gup2021-1590"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_current(directory: Path) -> bool:
    """Künyedeki PDF ve JSONL diskte VE SHA-256'ları aynı mı?"""
    provenance_path = directory / "_provenance.json"
    if not provenance_path.exists():
        return False
    files = json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}
    return bool(files) and all(
        (directory / name).is_file() and _sha256(directory / name) == info.get("sha256")
        for name, info in files.items()
    )


def pdf_is_current(directory: Path) -> bool:
    provenance_path = directory / "_provenance.json"
    pdf = directory / PDF_NAME
    if not (provenance_path.exists() and pdf.is_file()):
        return False
    files = json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}
    return _sha256(pdf) == (files.get(PDF_NAME) or {}).get("sha256")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wilkens Eski Uygurca sözlüğü indirici/ayrıştırıcı")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir ve ayrıştır")
    args = ap.parse_args(argv)
    if not args.force and is_current(WILKENS_DIR):
        print(f"Wilkens: zaten var, SHA-256 künyeyle aynı (--force ile yeniden) -> {WILKENS_DIR}")
        return 0

    WILKENS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = WILKENS_DIR / PDF_NAME
    if args.force or not pdf_is_current(WILKENS_DIR):
        response = requests.get(
            URL, timeout=300, headers={"User-Agent": "turkic-etymology-engine/3.0 (+research)"}
        )
        if response.status_code != 200 or not response.content.startswith(b"%PDF"):
            print(f"  ! PDF indirilemedi: HTTP {response.status_code}")
            return 1
        pdf.write_bytes(response.content)
        print(f"  + {PDF_NAME}: {len(response.content) / 1e6:.1f} MB")

    records = parse_pdf(pdf)
    jsonl = WILKENS_DIR / JSONL_NAME
    write_jsonl(records, jsonl)
    real = [r for r in records if r.get("tr") and not r.get("error") and not r.get("see")]
    print(f"  + {JSONL_NAME}: {len(records)} madde başı, {len(real)} Türkçe anlamlı gerçek madde")

    provenance = {
        "_schema": "turkic-etymology-wilkens-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "citation": (
            "Jens Wilkens, Handwörterbuch des Altuigurischen. Altuigurisch – Deutsch – Türkisch. "
            "Göttingen: Universitätsverlag Göttingen 2021."
        ),
        "doi": DOI,
        "landing_page": LANDING,
        "url": URL,
        "license": "CC BY-SA 4.0",
        "note": (
            "Sözlük tanık yeri (metin/yıl) vermez; tanık dönem etiketiyle (9.-14. yy) tarihlenir. "
            "JSONL, engine/db/wilkens.py ayrıştırıcısının çıktısıdır."
        ),
        "counts": {
            "headwords": len(records),
            "usable_with_turkish_gloss": len(real),
            "errors_dagger": sum(1 for r in records if r.get("error")),
            "cross_references": sum(1 for r in records if r.get("see") and not r.get("error")),
            "with_donor_chain": sum(1 for r in real if r.get("donor_chain")),
            "subentries": sum(len(r.get("subentries") or []) for r in records),
        },
        "files": {
            PDF_NAME: {"bytes": pdf.stat().st_size, "sha256": _sha256(pdf)},
            JSONL_NAME: {"bytes": jsonl.stat().st_size, "sha256": _sha256(jsonl)},
        },
    }
    write_if_changed(
        WILKENS_DIR / "_provenance.json", provenance, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
