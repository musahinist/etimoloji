#!/usr/bin/env python3
"""
CLDF veri kümesi indirici — Faz 1.

Karşılaştırmalı yöntemin altın standardını ve taban çizgisini kuran dört
Lexibank veri kümesini ``data/cldf/`` altına indirir. Her indirme **sürüm ve
tarih damgalıdır**: hangi commit'ten hangi dosyanın alındığı, SHA-256'sı ve
kayıt sayısı ``_provenance.json`` ile ``data/SOURCES.md``'ye yazılır.

Veri kümeleri ve rolleri (bkz. README "Bilimsel kaynakça")::

    savelyevturkic          birincil altın standart (uzman akrabalık kararları)
    hruschkaturkic          bağımsız çapraz kontrol
    starostinaltaic         yalnız karşılaştırma (Vovin 2005 eleştirisi)
    robbeetstriangulation   yalnız TEMAS çerçevesi (Tian ve ark. 2022 eleştirisi)

Kullanım::

    python scripts/download_cldf.py --all
    python scripts/download_cldf.py savelyevturkic
    python scripts/download_cldf.py --all --force     # yeniden indir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import CLDF_DIR  # noqa: E402

RAW = "https://raw.githubusercontent.com/lexibank/{repo}/{ref}/cldf/{name}"
API = "https://api.github.com/repos/lexibank/{repo}"

#: İndirilecek CLDF tabloları. Yıldızlılar zorunludur.
TABLES = ("forms.csv", "cognates.csv", "languages.csv", "parameters.csv")
REQUIRED = ("forms.csv",)
METADATA = "cldf-metadata.json"

#: repo adı -> (sabitlenmiş sürüm etiketi | None → varsayılan dal, rol notu)
DATASETS: dict[str, dict[str, str]] = {
    "savelyevturkic": {
        "ref": "v2.1",
        "role": "birincil altın standart",
        "citation": "Savelyev & Robbeets 2020, Journal of Language Evolution",
        "caveat": "",
    },
    "hruschkaturkic": {
        "ref": "v1.0",
        "role": "bağımsız çapraz kontrol",
        "citation": "Hruschka ve ark. 2015, Current Biology",
        "caveat": "",
    },
    "starostinaltaic": {
        "ref": "main",
        "role": "yalnız karşılaştırma",
        "citation": "Starostin, Dybo & Mudrak, Altaic Etymological Dictionary",
        "caveat": (
            "Altay hipotezi tartışmalıdır (Vovin 2005). Tek kaynak olarak kullanılmaz; akrabalık kanıtına katılmaz."
        ),
    },
    "robbeetstriangulation": {
        "ref": "v0.3",
        "role": "yalnız temas çerçevesi",
        "citation": "Robbeets & Bouckaert, Triangulation dataset",
        "caveat": (
            "Transeurasian verisi en az EDAL kadar tartışmalıdır (Tian ve ark. "
            "2022). YALNIZCA temas/ödünçleme analizinde kullanılır; akrabalık "
            "kanıtına asla katılmaz."
        ),
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_ref(repo: str, ref: str, session: requests.Session) -> tuple[str, str]:
    """Etiketi çözümleyip (ref, commit_sha) döndürür — sürüm damgası için."""
    url = f"{API.format(repo=repo)}/commits/{ref}"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return ref, resp.json().get("sha", "")[:12]
    except (requests.RequestException, ValueError) as exc:
        print(f"  ! commit sha çözümlenemedi ({exc}); ref olduğu gibi kullanılıyor")
        return ref, ""


def _count_rows(path: Path) -> int:
    if path.suffix != ".csv":
        return 0
    with path.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def download(name: str, *, force: bool = False, session: requests.Session | None = None) -> dict[str, Any]:
    """Tek bir veri kümesini indirir ve künyesini döndürür."""
    spec = DATASETS[name]
    session = session or requests.Session()
    target = CLDF_DIR / name
    target.mkdir(parents=True, exist_ok=True)

    provenance_path = target / "_provenance.json"
    if provenance_path.exists() and not force:
        print(f"[{name}] zaten var (yeniden indirmek için --force)")
        return json.loads(provenance_path.read_text(encoding="utf-8"))

    ref, sha = _resolve_ref(name, spec["ref"], session)
    print(f"[{name}] ref={ref} sha={sha or '?'}")

    files: dict[str, dict[str, Any]] = {}
    for table in (*TABLES, METADATA):
        url = RAW.format(repo=name, ref=ref, name=table)
        resp = session.get(url, timeout=60)
        if resp.status_code == 404:
            if table in REQUIRED:
                raise RuntimeError(f"{name}: zorunlu tablo bulunamadı: {table}")
            print(f"  - {table}: yok (atlandı)")
            continue
        resp.raise_for_status()
        out = target / table
        out.write_bytes(resp.content)
        rows = _count_rows(out)
        files[table] = {
            "sha256": _sha256(out),
            "bytes": len(resp.content),
            "rows": rows,
        }
        print(f"  + {table}: {len(resp.content):>9,} bayt" + (f" · {rows:,} kayıt" if rows else ""))
        time.sleep(0.2)

    provenance = {
        "_schema": "turkic-etymology-cldf-provenance/v1",
        "dataset": name,
        "repository": f"https://github.com/lexibank/{name}",
        "ref": ref,
        "commit": sha,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "citation": spec["citation"],
        "role": spec["role"],
        "caveat": spec["caveat"],
        "files": files,
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def write_sources_index(provenances: list[dict[str, Any]]) -> Path:
    """``data/SOURCES.md`` — insan-okunur künye tablosu."""
    out = CLDF_DIR.parent / "SOURCES.md"
    lines = [
        "# Veri kaynakları künyesi",
        "",
        "Bu dosya `scripts/download_*.py` tarafından **otomatik üretilir** — elle",
        "düzenlemeyin. Her satır hangi veri kümesinin hangi sürümünden, ne zaman",
        "indirildiğini ve içerik özetini gösterir.",
        "",
        "## CLDF veri kümeleri",
        "",
        "| Veri kümesi | Sürüm | Commit | İndirilme | Kayıt | Rol |",
        "|---|---|---|---|---|---|",
    ]
    for p in sorted(provenances, key=lambda x: x["dataset"]):
        forms = p["files"].get("forms.csv", {}).get("rows", 0)
        lines.append(
            f"| [{p['dataset']}]({p['repository']}) | `{p['ref']}` | `{p['commit'] or '—'}` "
            f"| {p['retrieved_at'][:10]} | {forms:,} biçim | {p['role']} |"
        )
    lines += ["", "### Künye ve uyarılar", ""]
    for p in sorted(provenances, key=lambda x: x["dataset"]):
        lines.append(f"**{p['dataset']}** — {p['citation']}")
        if p["caveat"]:
            lines.append(f"> ⚠️ {p['caveat']}")
        detail = " · ".join(f"`{name}` {info['rows']:,} kayıt" for name, info in p["files"].items() if info["rows"])
        lines += [f"> {detail}", ""]
    lines += [
        "### Doğrulama",
        "",
        "Her dosyanın SHA-256'sı ilgili `data/cldf/<ad>/_provenance.json` içindedir.",
        "Yeniden indirip aynı özetleri aldığınızda veri değişmemiştir.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Lexibank CLDF veri kümesi indirici")
    ap.add_argument("datasets", nargs="*", choices=[*DATASETS, []], help="indirilecek veri kümeleri")
    ap.add_argument("--all", action="store_true", help="hepsini indir")
    ap.add_argument("--force", action="store_true", help="mevcut olsa bile yeniden indir")
    args = ap.parse_args()

    names = list(DATASETS) if args.all else args.datasets
    if not names:
        ap.error("bir veri kümesi adı verin veya --all kullanın")

    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"

    results = []
    for name in names:
        try:
            results.append(download(name, force=args.force, session=session))
        except (requests.RequestException, RuntimeError) as exc:
            print(f"[{name}] BAŞARISIZ: {exc}", file=sys.stderr)

    # Daha önce indirilmiş olanları da künyeye kat
    for existing in sorted(CLDF_DIR.glob("*/_provenance.json")):
        data = json.loads(existing.read_text(encoding="utf-8"))
        if data["dataset"] not in {r["dataset"] for r in results}:
            results.append(data)

    if results:
        path = write_sources_index(results)
        print(f"\nKünye yazıldı: {path}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
