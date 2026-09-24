#!/usr/bin/env python3
"""
Zemberek kök sözlüğünü indirir (yalnız veri; Zemberek koduna bağımlılık yok).

Zemberek-NLP (Ahmet A. Akın, Apache License 2.0) Türkçe biçimbilim sözlüğünü
düz metin ``.dict`` dosyaları olarak yayımlıyor. Her satır bir madde başı ve
isteğe bağlı köşeli ayraç içinde öznitelikler::

    kitap
    burun [A:LastVowelDrop]
    hak [A:Doubling]
    kalp [A:Voicing, InverseHarmony]
    gitmek [A:Voicing]

Motorun kullandığı yalnız SES öznitelikleridir (``Voicing``, ``NoVoicing``,
``Doubling``, ``LastVowelDrop``, ``ProgressiveVowelDrop``) ve sözcük türü
(``P:``); bkz. ``engine/nlp/root_variants.py``. ``InverseHarmony`` dosyada
durur ama kök varyantında kullanılmaz (alıntı sinyali ayrı iştir).

Dosyalar sabit bir commit'ten iner (tekrarlanabilirlik); repoya alınmaz,
``data/zemberek/`` altına iner, künye ve SHA-256 commit edilir. Atlama kararı
dosyaların kendisine (SHA-256) bakar.
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

TARGET = PROJECT_ROOT / "data" / "zemberek"
REPO = "ahmetaa/zemberek-nlp"
#: Sabitlenmiş commit (2026-09-24'te master'ın ucu).
COMMIT = "a419ac2680352093e688b0110c1b8628f90bd807"
DICT_DIR = "morphology/src/main/resources/tr"
FILES = ("master-dictionary.dict", "non-tdk.dict")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_current(directory: Path) -> bool:
    """Künyedeki tüm dosyalar diskte VE SHA-256'ları aynı mı?"""
    provenance_path = directory / "_provenance.json"
    if not provenance_path.exists():
        return False
    files = json.loads(provenance_path.read_text(encoding="utf-8")).get("files") or {}
    return bool(files) and all(
        (directory / name).is_file() and _sha256(directory / name) == info.get("sha256")
        for name, info in files.items()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zemberek kök sözlüğü indirici")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir")
    args = ap.parse_args(argv)
    if not args.force and is_current(TARGET):
        print(f"Zemberek: zaten var, SHA-256 künyeyle aynı (--force ile yeniden indirilir) -> {TARGET}")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"
    files = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{DICT_DIR}/{name}"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        path = TARGET / name
        path.write_bytes(response.content)
        lines = response.content.decode("utf-8").splitlines()
        files[name] = {
            "url": url,
            "lines": len(lines),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        print(f"  + {name}: {len(lines)} satır")
    data = {
        "_schema": "turkic-etymology-zemberek-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "license": "Apache-2.0",
        "attribution": "Zemberek-NLP, Ahmet A. Akın ve katkıda bulunanlar (Apache License 2.0)",
        "note": "Yalnız kök ses öznitelikleri (Voicing/NoVoicing/Doubling/LastVowelDrop/"
                "ProgressiveVowelDrop) ve P: sözcük türü kullanılır; bkz. engine/nlp/root_variants.py.",
        "files": files,
    }
    write_if_changed(TARGET / "_provenance.json", data,
                     json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
