#!/usr/bin/env python3
"""
Türkçe -> İngilizce anlam köprüsü (9l S1, ``engine.db.sense_bridge``) — ``make sense-bridge``.

İki kaikki.org dökümünü indirir (Türkçe Vikisözlük ham dökümü ~44 MB,
İngilizce Vikisözlük İngilizce sözlüğü ~510 MB), künyedeki SHA-256'larla
karşılaştırır ve ``data/lexicons/sense_bridge/tr_en.db`` tablosunu kurar.
Dökümler ve tablo git-ignored; künye (``tr_en.provenance.json``: iki dökümün
SHA-256'sı ve tablonun içerik özeti ``content_sha256``) commit edilir.

Atlama kararı dosyanın kendisine bakar: tablo diskte VE içerik özeti künyeyle
aynıysa hiçbir şey indirilmez. Aynı SHA'lı dökümlerden kurulan tablo bayt
bayt aynıdır (kurulum belirlenimci); özet farklı çıkarsa uyarı basılır.

⚠️ kaikki dökümleri yerinde güncellenir: yeni indirme başka SHA verirse tablo
yine kurulur ama künye değişir (git diff'te görünür) ve ölçümler o tabloyla
aynı değildir. ``--raw-dir`` ile elde duran eski dökümler (SHA'ları künyeyle
aynıysa) indirmeden kullanılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.db.sense_bridge import BRIDGE_DB, build, provenance_path, table_digest  # noqa: E402

#: kaynak adı -> (URL, ham dosya adı)
SOURCES: dict[str, tuple[str, str]] = {
    "trwiktionary": ("https://kaikki.org/trwiktionary/raw-wiktextract-data.jsonl.gz", "trwikt.jsonl.gz"),
    "enwiktionary": ("https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz",
                     "en.jsonl.gz"),
}
RAW_DIR = BRIDGE_DB.parent / "raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(out: Path) -> dict:
    path = provenance_path(out)
    if not path.exists():
        # Doğrulama kurulumu başka dizine yazılıyorsa da commit'li künyeyle karşılaştır.
        path = provenance_path(BRIDGE_DB)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def is_current(out: Path, provenance: dict) -> bool:
    """Tablo diskte VE içerik özeti künyeyle aynı mı?"""
    expected = provenance.get("content_sha256")
    return bool(expected) and out.is_file() and table_digest(out) == expected


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "turkic-etymology-engine/3.0 (+research)"}) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    part.replace(target)


def fetch(name: str, expected_sha: str, raw_dirs: list[Path], force: bool) -> tuple[Path, bool]:
    """(döküm yolu, bu koşuda mı indirildi). SHA'sı künyeyle aynı yerel döküm varsa o kullanılır."""
    url, filename = SOURCES[name]
    if not force:
        for directory in raw_dirs:
            candidate = directory / filename
            if candidate.is_file() and expected_sha and _sha256(candidate) == expected_sha:
                print(f"  = {name}: yerel döküm, SHA künyeyle aynı -> {candidate}")
                return candidate, False
    target = RAW_DIR / filename
    print(f"  ↓ {name}: {url}")
    _download(url, target)
    sha = _sha256(target)
    if expected_sha and sha != expected_sha:
        print(f"  ! {name}: SHA künyeden FARKLI (kaikki dökümü güncellenmiş): {sha[:12]} != {expected_sha[:12]}; "
              "tablo ve ölçümler künyedekiyle aynı olmayacak")
    return target, True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Türkçe -> İngilizce anlam köprüsü indirici/kurucu")
    ap.add_argument("--force", action="store_true", help="tablo güncel olsa bile yeniden indir ve kur")
    ap.add_argument("--raw-dir", type=Path, action="append", default=[],
                    help="önce bu dizinde trwikt.jsonl.gz / en.jsonl.gz ara (SHA künyeyle aynıysa indirmez)")
    ap.add_argument("--out", type=Path, default=BRIDGE_DB,
                    help="tablo yolu (doğrulama için başka yere kurulabilir; künye yanına yazılır)")
    ap.add_argument("--keep-raw", action="store_true", help="indirilen dökümleri kurulumdan sonra silme (~550 MB)")
    args = ap.parse_args(argv)

    provenance = _provenance(args.out)
    if not args.force and is_current(args.out, provenance):
        print(f"Anlam köprüsü: zaten var, içerik özeti künyeyle aynı (--force ile yeniden) -> {args.out}")
        return 0

    sources = provenance.get("sources") or {}
    raw_dirs = [*args.raw_dir, RAW_DIR]
    paths: dict[str, Path] = {}
    downloaded: list[Path] = []
    for name in SOURCES:
        path, fresh = fetch(name, (sources.get(name) or {}).get("sha256", ""), raw_dirs, args.force)
        paths[name] = path
        if fresh:
            downloaded.append(path)

    expected = provenance.get("content_sha256")
    meta = build(paths["trwiktionary"], paths["enwiktionary"], out=args.out)
    digest = meta["content_sha256"]
    same_sources = all((sources.get(n) or {}).get("sha256") == meta["sources"][n]["sha256"] for n in SOURCES)
    if expected and same_sources and digest != expected:
        print(f"  ! aynı SHA'lı dökümlerden FARKLI tablo: {digest[:12]} != {expected[:12]} (kurulum belirlenimci değil?)")
    print(f"Anlam köprüsü: {meta['entries']:,} madde · {meta['en_vocab']:,} İngilizce başlık · "
          f"içerik {digest[:12]} -> {args.out}")

    if not args.keep_raw:
        for path in downloaded:
            path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
