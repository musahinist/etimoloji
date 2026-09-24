#!/usr/bin/env python3
"""
Apertium iki dilli Türk dili sözlüklerini indirir — ``make apertium``.

Her sözlük Türkçe ile başka bir Türk dili arasında sözcükbirim eşleşmesidir
(Çuvaşça–Türkçe ~32 bin, Özbekçe–Türkçe …). Ham ``.dix`` dosyaları repoya
alınmaz; ``data/apertium/`` altına iner, künye ve SHA-256 commit edilir.

⚠️ Eşleşmeler ÇEVİRİ karşılığıdır, etimolojik denklik değil (Kırgızca
*терезе* 'pencere'). Tanık olarak kullanılmadan önce yazılış/ses denkliği
denetiminden geçer (bkz. ``engine/fetchers/apertium.py``).
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

from engine.config import PROJECT_ROOT  # noqa: E402
from engine.utils.provenance import write_if_changed  # noqa: E402

TARGET = PROJECT_ROOT / "data" / "apertium"
RAW = "https://raw.githubusercontent.com/apertium/apertium-{pair}/master/apertium-{pair}.{pair}.dix"

#: Depo çifti -> (sol dil, sağ dil) motor kodlarıyla. Sıra depo adındaki sıradır.
PAIRS = {
    "chv-tur": ("cv", "tr"),
    "crh-tur": ("crh", "tr"),
    "tur-kir": ("tr", "ky"),
    "tur-tat": ("tr", "tt"),
    "tur-uzb": ("tr", "uz"),
    "tur-aze": ("tr", "az"),
    "tuk-tur": ("tk", "tr"),
}


def is_current(directory: Path) -> bool:
    """Künyedeki tüm ``.dix`` dosyaları diskte VE SHA-256'ları aynı mı?

    ``_provenance.json`` commit edilir, ``.dix`` dosyaları edilmez; taze
    klonda künye olur ama veri olmaz. Atlama kararı dosyaların kendisine bakar.
    """
    provenance_path = directory / "_provenance.json"
    if not provenance_path.exists():
        return False
    files = json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}
    return bool(files) and all(
        (directory / f"{pair}.dix").is_file()
        and hashlib.sha256((directory / f"{pair}.dix").read_bytes()).hexdigest() == info.get("sha256")
        for pair, info in files.items()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apertium iki dilli sözlük indirici")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir")
    args = ap.parse_args(argv)
    if not args.force and is_current(TARGET):
        print(f"Apertium: zaten var, SHA-256 künyeyle aynı (--force ile yeniden indirilir) -> {TARGET}")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"
    files = {}
    for pair, (left, right) in PAIRS.items():
        url = RAW.format(pair=pair)
        response = session.get(url, timeout=120)
        if response.status_code != 200:
            print(f"  ! {pair}: HTTP {response.status_code}")
            continue
        path = TARGET / f"{pair}.dix"
        path.write_bytes(response.content)
        files[pair] = {
            "url": url, "left": left, "right": right,
            "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
        }
        print(f"  + {pair}: {len(response.content) / 1024:.0f} KB")
    provenance = {
        "_schema": "turkic-etymology-apertium-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Çeviri karşılığıdır, etimolojik denklik değil; tanık olmadan önce ses denkliği denetlenir.",
        "files": files,
    }
    # Sözlükler aynıysa commit edilmiş künye korunur (yalnız indirme zamanı değişirdi).
    write_if_changed(
        TARGET / "_provenance.json", provenance, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
