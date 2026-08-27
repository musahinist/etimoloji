#!/usr/bin/env python3
"""
kaikki.org sözlük dökümü indirici — Faz 4.

Akraba araması için yerel sözlük indeksinin ham verisini indirir. İndirilen
her döküm ``data/lexicons/`` altına, sürüm ve tarih damgasıyla yazılır.

⚠️ Bu indeks **arama** içindir; akrabalık kararı buradan gelmez. Wiktionary
türevi akraba kümeleri altın standart ağaçlarla tutarsız çıkıyor
(Häuser & Stamatakis 2025), bu yüzden burada bulunanlar "aday" sayılır ve
kararı uzman verisi veya kümeleme katmanı verir.

Boyutlar çok farklı: Türkçe 410 MB, Kazakça 34 MB, Çuvaşça 1 MB. Bu yüzden
dil dil indirilebilir::

    python scripts/download_lexicons.py --small     # 50 MB altındakiler
    python scripts/download_lexicons.py Turkish Kazakh
    python scripts/download_lexicons.py --all       # ~761 MB
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import LEXICON_DIR  # noqa: E402

URL = "https://kaikki.org/dictionary/{name}/kaikki.org-dictionary-{name}.jsonl"

#: kaikki adı -> motorun dil kodu. kaikki dil adlarını İngilizce yazar.
LEXICONS: dict[str, str] = {
    "Turkish": "tr",
    "Azerbaijani": "az",
    "Turkmen": "tk",
    "Gagauz": "gag",
    "Kazakh": "kk",
    "Karakalpak": "kaa",
    "Kyrgyz": "ky",
    "Tatar": "tt",
    "Bashkir": "ba",
    "Nogai": "nog",
    "Kumyk": "kum",
    "Karachay-Balkar": "krc",
    "Crimean_Tatar": "crh",
    "Uzbek": "uz",
    "Uyghur": "ug",
    "Chuvash": "cv",
    "Yakut": "sah",
    "Tuvan": "tyv",
    "Southern_Altai": "alt",
    "Khakas": "khk",
    "Shor": "cjs",
    "Salar": "slq",
    "Old_Turkic": "otk",
    "Ottoman_Turkish": "ota",
    "Chagatai": "chg",
    "Khalaj": "klj",
    "Dolgan": "dlg",
}

#: **Verici** dil dökümleri. Bunlar Türki DEĞİLDİR ve akraba arama indeksine
#: ASLA karışmaz — ayrı dizine iner (``data/lexicons/donors/``).
#:
#: Gerekçe: alıntı tespitinin en güçlü ölçülmüş sinyali "verici dil
#: sözlüğüne yakınlık"tır (sabor, Miller & List 2023: F1 0,806, kesinlik
#: 0,931). Bu sinyal verici sözlüğü olmadan hesaplanamaz.
#:
#: Hangi vericiler? WOLD'daki Sakha alıntılarının kaynak dağılımı ölçüldü:
#: **Rusça 284 · Moğolca 253 · Evenkice 19** · Çince 4 · Arapça 3 · Farsça 3.
#: Türkçe için ölçüt farklıdır (Arapça/Farsça/Fransızca ağırlıklı) ve C6'da
#: kurulacak altın kümeyle birlikte inecektir.
DONORS: dict[str, str] = {
    "Russian": "ru",
    "Mongolian": "mn",
    "Evenki": "evn",
    "Arabic": "ar",
    "Persian": "fa",
    "Greek": "el",
    "Armenian": "hy",
    "French": "fr",
    "Italian": "it",
}

#: Verici dökümleri için ayrı dizin. Türki indeksle karışmaması **yapısal**
#: olarak garanti edilir: ``lexicon_index`` yalnız ``LEXICON_DIR``in kendi
#: köküne bakar.
DONOR_SUBDIR = "donors"

#: "küçük" sayılan eşik (bayt). Türkçe dökümü tek başına 410 MB.
SMALL_LIMIT = 50 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_size(name: str, session: requests.Session) -> int:
    try:
        response = session.head(URL.format(name=name), timeout=30, allow_redirects=True)
        return int(response.headers.get("content-length", 0))
    except (requests.RequestException, ValueError):
        return 0


def download(
    name: str,
    *,
    session: requests.Session,
    force: bool = False,
    compress: bool = True,
    donor: bool = False,
) -> dict[str, Any] | None:
    """Tek bir dilin dökümünü indirir ve künyesini döndürür.

    :param donor: verici dil mi? Öyleyse ``data/lexicons/donors/`` altına
        iner ve Türki arama indeksine **karışmaz**.
    """
    code = (DONORS if donor else LEXICONS)[name]
    directory = LEXICON_DIR / DONOR_SUBDIR if donor else LEXICON_DIR
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz" if compress else ".jsonl"
    target = directory / f"{code}{suffix}"
    provenance_path = directory / f"{code}.provenance.json"

    if provenance_path.exists() and not force:
        print(f"[{name}] zaten var (--force ile yeniden indirilir)")
        return json.loads(provenance_path.read_text(encoding="utf-8"))

    url = URL.format(name=name)
    print(f"[{name}] indiriliyor -> {target.name}")
    raw_bytes = 0
    lines = 0
    try:
        with session.get(url, stream=True, timeout=300) as response:
            if response.status_code == 404:
                print(f"  ! kaikki'de yok: {name}")
                return None
            response.raise_for_status()
            opener = gzip.open(target, "wb") if compress else target.open("wb")
            with opener as out:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    raw_bytes += len(chunk)
                    lines += chunk.count(b"\n")
                    out.write(chunk)
                    if raw_bytes % (50 << 20) < (1 << 20):
                        print(f"  … {raw_bytes / (1 << 20):.0f} MB")
    except requests.RequestException as exc:
        print(f"  ! indirme başarısız: {exc}")
        if target.exists():
            target.unlink()
        return None

    provenance = {
        "_schema": "turkic-etymology-lexicon-provenance/v1",
        "language": name,
        "code": code,
        "donor": donor,
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_bytes": raw_bytes,
        "stored_bytes": target.stat().st_size,
        "entries": lines,
        "compressed": compress,
        "sha256_stored": _sha256(target),
        "note": (
            "kaikki.org, Wiktionary'nin makine-okunur dökümüdür. Bu veri ARAMA "
            "İNDEKSİ olarak kullanılır; akrabalık veya ata biçim kararı buradan "
            "verilmez (Häuser & Stamatakis 2025)."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  + {lines:,} kayıt · ham {raw_bytes / (1 << 20):.1f} MB "
        f"· diskte {target.stat().st_size / (1 << 20):.1f} MB"
    )
    return provenance


def write_index_note(provenances: list[dict[str, Any]]) -> Path:
    """``data/lexicons/SOURCES.md`` — indirilen dökümlerin künyesi."""
    out = LEXICON_DIR / "SOURCES.md"
    total_entries = sum(p["entries"] for p in provenances)
    lines = [
        "# Sözlük dökümleri künyesi",
        "",
        "`scripts/download_lexicons.py` tarafından otomatik üretilir.",
        "",
        "⚠️ Bu dökümler **arama indeksi**dir. Akrabalık ve ata biçim kararı",
        "buradan verilmez; Wiktionary türevi akraba kümeleri altın standart",
        "ağaçlarla tutarsız çıkıyor (Häuser & Stamatakis 2025).",
        "",
        f"Toplam: **{len(provenances)} dil · {total_entries:,} kayıt**",
        "",
        "| Dil | Kod | Kayıt | Ham boyut | İndirilme |",
        "|---|---|---|---|---|",
    ]
    for prov in sorted(provenances, key=lambda p: p["code"]):
        lines.append(
            f"| [{prov['language']}]({prov['url']}) | `{prov['code']}` "
            f"| {prov['entries']:,} | {prov['raw_bytes'] / (1 << 20):.1f} MB "
            f"| {prov['retrieved_at'][:10]} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="kaikki.org sözlük dökümü indirici")
    ap.add_argument("languages", nargs="*", help="kaikki dil adları (ör. Turkish Kazakh)")
    ap.add_argument("--all", action="store_true", help="hepsini indir (~761 MB)")
    ap.add_argument(
        "--donors",
        nargs="*",
        metavar="DIL",
        help="verici dil dökümleri (ad verilmezse Sakha ölçütü için Russian Mongolian Evenki)",
    )
    ap.add_argument("--small", action="store_true", help=f"{SMALL_LIMIT >> 20} MB altındakiler")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-compress", action="store_true", help="gzip'lemeden sakla")
    ap.add_argument("--list", action="store_true", help="boyutları göster, indirme")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"

    if args.list:
        print(f"{'dil':22} {'kod':5} {'boyut':>10}")
        for name in LEXICONS:
            size = remote_size(name, session)
            print(f"{name:22} {LEXICONS[name]:5} {size / (1 << 20):>9.1f} MB")
        return 0

    if args.donors is not None:
        # Varsayılan: WOLD/Sakha birincil ölçütünün gerçek vericileri.
        donor_names = args.donors or ["Russian", "Mongolian", "Evenki"]
        unknown = [n for n in donor_names if n not in DONORS]
        if unknown:
            ap.error(f"bilinmeyen verici dil: {', '.join(unknown)}")
        results = [
            prov
            for name in donor_names
            if (
                prov := download(
                    name,
                    session=session,
                    force=args.force,
                    compress=not args.no_compress,
                    donor=True,
                )
            )
        ]
        for prov in results:
            print(f"  verici: {prov['language']} -> {prov['entries']:,} kayıt")
        return 0 if results else 1

    names: list[str]
    if args.all:
        names = list(LEXICONS)
    elif args.small:
        print(f"{SMALL_LIMIT >> 20} MB altındaki dökümler aranıyor…")
        names = [n for n in LEXICONS if 0 < remote_size(n, session) <= SMALL_LIMIT]
        print(f"  {len(names)} dil seçildi")
    else:
        names = args.languages
    unknown = [n for n in names if n not in LEXICONS]
    if unknown:
        ap.error(f"bilinmeyen dil: {', '.join(unknown)}")
    if not names:
        ap.error("bir dil adı verin, --small veya --all kullanın (--list ile boyutlara bakın)")

    if shutil.disk_usage(LEXICON_DIR.parent).free < (1 << 30):
        print("! uyarı: 1 GB'den az boş disk alanı var")

    results: list[dict[str, Any]] = []
    for name in names:
        prov = download(name, session=session, force=args.force, compress=not args.no_compress)
        if prov:
            results.append(prov)

    for existing in sorted(LEXICON_DIR.glob("*.provenance.json")):
        data = json.loads(existing.read_text(encoding="utf-8"))
        if data["code"] not in {r["code"] for r in results}:
            results.append(data)

    if results:
        path = write_index_note(results)
        print(f"\nKünye: {path}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
