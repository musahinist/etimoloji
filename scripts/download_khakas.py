#!/usr/bin/env python3
"""
Hakasça sözlükleri indirir — ``make khakas``.

İki Hugging Face veri kümesi (CC-BY-4.0, Adeshkin):

* ``adeshkin/khakas-russian-dict`` (``draft`` yapılandırması, ~22 bin madde):
  Hakasça–Rusça sözlük; her maddede Rusça karşılık (``semgloss``), sözcük
  türü ve köken etiketi (``etym == 'rus'``: Rusça alıntı) var.
* ``adeshkin/khakas-explanatory-dict`` (~12,5 bin madde): Hakasça açıklamalı
  sözlük; açıklamanın ardından ``--`` ile Rusça çeviri gelir.

Veri kümeleri parquet olarak yayımlanıyor ama ortamda ``pyarrow`` yok; bu
yüzden Hugging Face veri kümesi sunucusunun satır API'si (100'er satır)
kullanılır ve yalnız gereken sütunlar JSONL olarak yazılır. Ham dosyalar
repoya alınmaz; ``data/khakas/`` altına iner, künye ve SHA-256 commit edilir.
Atlama kararı dosyaların kendisine (SHA-256) bakar, künyeye değil.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import PROJECT_ROOT  # noqa: E402

TARGET = PROJECT_ROOT / "data" / "khakas"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
#: Sayfalar arası bekleme (saniye); daha sık istek 429 alıyor.
PAUSE = 1.0
#: İstekler arası bekleme (sn); satır API'si hızlı ardışık istekte 429 veriyor.
PAUSE = 0.5

#: Yerel dosya adı -> (veri kümesi, yapılandırma, saklanan sütunlar).
DATASETS = {
    "khakas_russian.jsonl": (
        "adeshkin/khakas-russian-dict", "draft",
        ("headword", "word", "headnum", "semgloss", "part", "etym", "semtag", "dial", "rest"),
    ),
    "khakas_explanatory.jsonl": (
        "adeshkin/khakas-explanatory-dict", "default",
        ("headword_fix", "field_fix"),
    ),
}


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


def _get(session: requests.Session, params: dict, tries: int = 10) -> dict:
    """Sunucu hız sınırı koyuyor (HTTP 429); ``Retry-After``a ya da artan beklemeye uyar."""
    for attempt in range(tries):
        response = session.get(ROWS_API, params=params, timeout=120)
        if response.status_code == 200:
            time.sleep(PAUSE)
            return response.json()
        wait = response.headers.get("Retry-After")
        time.sleep(float(wait) if wait and wait.isdigit() else min(60, 2 ** attempt))
    response.raise_for_status()
    raise RuntimeError(f"HTTP {response.status_code}")


def download(session: requests.Session, dataset: str, config: str, columns: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    total = None
    offset = 0
    while total is None or offset < total:
        data = _get(session, {"dataset": dataset, "config": config, "split": "train",
                              "offset": offset, "length": PAGE})
        total = data["num_rows_total"]
        batch = data.get("rows") or []
        if not batch:
            break
        rows.extend({c: r["row"].get(c) for c in columns} for r in batch)
        offset += len(batch)
    if len(rows) != total:
        raise RuntimeError(f"{dataset}: {len(rows)}/{total} satır indi")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hakasça sözlük indirici")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir")
    args = ap.parse_args(argv)
    if not args.force and is_current(TARGET):
        print(f"Hakasça: zaten var, SHA-256 künyeyle aynı (--force ile yeniden indirilir) -> {TARGET}")
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"
    files = {}
    for name, (dataset, config, columns) in DATASETS.items():
        rows = download(session, dataset, config, columns)
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
        path = TARGET / name
        path.write_text(payload, encoding="utf-8")
        files[name] = {
            "dataset": dataset, "config": config, "split": "train",
            "url": f"https://huggingface.co/datasets/{dataset}",
            "license": "CC-BY-4.0", "rows": len(rows), "columns": list(columns),
            "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
        print(f"  + {name}: {len(rows)} satır")
    (TARGET / "_provenance.json").write_text(json.dumps({
        "_schema": "turkic-etymology-khakas-provenance/v1",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "via": ROWS_API,
        "note": "Yazılışla bulunan aday akraba değildir; tanık olmadan önce anlamla doğrulanır "
                "(bkz. engine/fetchers/khakas_dict.py). etym=='rus' maddeler Rusça alıntıdır, elenir.",
        "files": files,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
