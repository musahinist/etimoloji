#!/usr/bin/env python3
"""
TDK Derleme Sözlüğü toplayıcısı — ağız sözvarlığı (Faz 3)

Projenin ilan edilmiş asıl hedefi, etimolojisi yapılmamış ağız sözvarlığıdır.
Elde hiç ağız kelime listesi yoktu: `sozluk.gov.tr/derleme` ölü (SPA HTML
kabuğu döndürüyor) ve yerel bir liste bulunmuyordu.

Ölçüldü: **`eski.sozluk.gov.tr` ayakta ve JSON dönüyor.**

    eski.sozluk.gov.tr/derleme?ara=gaga     -> 9 kayıt (madde_id 85678)
    eski.sozluk.gov.tr/derleme?ara=yalak    -> 7 kayıt
    eski.sozluk.gov.tr/derleme?ara=cimcime  -> 10 kayıt

⚠️ LİSTELEME YOK. Ölçüldü: `derleme`, `derleme?id=5`, `derleme?sehir=Afyon`
hepsi aynı 15 kayıtlık varsayılanı döndürüyor (md5 özdeş). Yalnız TAM
eşleşmeli `ara=` çalışıyor (`gag` -> hata nesnesi, `gaga` -> 9 kayıt).
Bu yüzden dışarıdan bir sürücü liste ve içeriden bir yayılma gerekir:

1. **Tohum**: Vikisözlük `Kategori:Türkçe halk ağzı` (CC-BY-SA). Ölçüldü:
   6 sayfada 3.000 üye, 2.693'ü sözlükbirim süzgecinden geçiyor, devamı var.
2. **Yayılma**: dönen kayıtlardaki çapraz göndermeler yeni sorgu olur.

⚠️ Plan `bakin` ve `asilk` alanlarını yayılma kenarı sayıyordu; ölçüm bunu
düzeltti. `asilk` çapraz gönderme DEĞİL, aynı maddenin anlam numarasıdır
(`gaga (I) 1`, `12 yalak (I) 1`). Gerçek kenarlar:

    bakin      -> 'gakkı'            (çözüldü: 1 kayıt)
    asilkelim  -> 'cımbıt (II) 1'    (çözüldü: 4 kayıt)

⚠️ LİSANS. TDK'nın açık lisansı **yoktur**, telif TDK'dadır. Toplanan veri
yalnız yerel araştırma için önbelleğe alınır; `.gitignore` onu depodan
dışarıda tutar. Yalnızca künye (`_provenance.json`) commit edilir. Kurum
sunucusudur ve `robots.txt` yoktur (404) — bu yüzden istekler hız sınırlıdır.

Kullanım::

    python scripts/harvest_derleme.py --limit 500
    python scripts/harvest_derleme.py --limit 2000 --delay 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import PROJECT_ROOT  # noqa: E402
from engine.logging_setup import get_logger  # noqa: E402
from engine.utils.morphology import is_lexeme  # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "dialect"
RECORDS_PATH = OUTPUT_DIR / "derleme" / "records.jsonl"
WORDLIST_PATH = OUTPUT_DIR / "derleme" / "words.txt"
PROVENANCE_PATH = OUTPUT_DIR / "_provenance.json"

DERLEME_URL = "https://eski.sozluk.gov.tr/derleme?ara={word}"
WIKTIONARY_API = "https://tr.wiktionary.org/w/api.php"
SEED_CATEGORY = "Kategori:Türkçe halk ağzı"

#: ⚠️ SALT ASCII olmalı. HTTP başlıkları latin-1 ile kodlanır; Türkçe harf
#: (ör. "araştırma") `UnicodeEncodeError` fırlatır ve `http_json`'daki geniş
#: `except Exception` bunu sessizce `None`'a çevirir — tohum listesi boş
#: gelir, sebebi görünmez.
USER_AGENT = "turkic-etymology-engine/3.0 (+local research; rate-limited)"

#: Çapraz gönderme değerlerindeki süsler: baştaki cilt numarası, sondaki
#: anlam numarası ve roma rakamlı duyu işareti. `12 yalak (I) 1` -> `yalak`.
_REF_CLEAN = re.compile(r"^\s*\d+\s*|\s*\([IVX]+\)\s*|\s*\d+\s*$")
_HTML_TAG = re.compile(r"<[^>]+>")


def strip_html(value: str) -> str:
    """`sehir` alanı HTML taşır: `*Dinar -<b>Afyon</b><br>` gibi."""
    return _HTML_TAG.sub(" ", value or "").replace("&nbsp;", " ").strip()


def normalise_reference(value: str) -> str:
    """Çapraz gönderme değerinden çıplak madde başını çıkarır."""
    out = _REF_CLEAN.sub("", (value or "").strip())
    out = _REF_CLEAN.sub("", out).strip()
    return out


def http_json(url: str, *, timeout: float = 25.0) -> Any | None:
    """JSON çeker; gövde JSON değilse veya istek düşerse ``None``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        # Beklenen durum: madde yok, sunucu yavaş. Gürültü yapma.
        logger.debug("İstek düştü: %s (%s)", url, exc)
        return None
    except Exception:
        # Ağ hatası DEĞİL — programlama hatası. Sessizce `None` dönmek bunu
        # "kaynak veri döndürmedi" gibi gösterir. Ölçüldü: User-Agent'taki
        # Türkçe harf `UnicodeEncodeError` fırlatıyordu (HTTP başlıkları
        # latin-1 kodlanır) ve tohum listesi sebebi görünmeden boş geliyordu.
        logger.warning("İstek kurulamadı (ağ dışı hata): %s", url, exc_info=True)
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def seed_words(limit: int, *, delay: float) -> list[str]:
    """Vikisözlük ağız kategorisinden sürücü liste (CC-BY-SA)."""
    words: list[str] = []
    cont: str | None = None
    while len(words) < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": SEED_CATEGORY, "cmlimit": "500",
            "cmtype": "page", "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        payload = http_json(f"{WIKTIONARY_API}?{urllib.parse.urlencode(params)}")
        if not payload:
            break
        members = payload.get("query", {}).get("categorymembers", [])
        words.extend(m["title"] for m in members if is_lexeme(m["title"]))
        cont = (payload.get("continue") or {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(delay)
    return words[:limit]


def fetch_derleme(word: str) -> list[dict[str, Any]]:
    """Tek bir maddenin Derleme kayıtları. Hata nesnesi gelirse boş liste."""
    payload = http_json(DERLEME_URL.format(word=urllib.parse.quote(word)))
    return payload if isinstance(payload, list) else []


def expansion_targets(records: list[dict[str, Any]]) -> set[str]:
    """Kayıtlardaki çapraz göndermelerden yeni sorgu adayları."""
    targets: set[str] = set()
    for record in records:
        for field in ("bakin", "asilkelim"):
            candidate = normalise_reference(str(record.get(field) or ""))
            if candidate and is_lexeme(candidate):
                targets.add(candidate.lower())
    return targets


def harvest(
    seeds: list[str], *, max_words: int, delay: float
) -> tuple[list[dict], list[str], int]:
    """Tohumla-ve-yayıl. Görülen her madde bir kez sorgulanır."""
    queue: deque[str] = deque(dict.fromkeys(w.lower() for w in seeds))
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    found: list[str] = []

    while queue and len(seen) < max_words:
        word = queue.popleft()
        if word in seen:
            continue
        seen.add(word)

        rows = fetch_derleme(word)
        time.sleep(delay)
        if not rows:
            continue

        found.append(word)
        for row in rows:
            row["_query"] = word
            row["sehir_temiz"] = strip_html(str(row.get("sehir") or ""))
            records.append(row)

        for target in expansion_targets(rows) - seen:
            queue.append(target)

        if len(seen) % 50 == 0:
            print(f"  … {len(seen)} madde sorgulandı · {len(found)} tuttu · {len(records)} kayıt")

    # `seen` SORGULANAN, `found` TUTAN maddedir. İkisini karıştırmak isabet
    # oranını olduğundan iyi gösterir (ölçüldü: 2.000 sorgu -> 1.631 tuttu,
    # ama künyeye 1.631/1.631 yazılmıştı).
    return records, found, len(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description="TDK Derleme Sözlüğü ağız toplayıcısı")
    ap.add_argument("--limit", type=int, default=300, help="en çok kaç madde sorgulanacak")
    ap.add_argument("--seed-limit", type=int, default=0,
                    help="tohum listesi boyutu (0 = --limit ile aynı)")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="istekler arası bekleme (sn) — kurum sunucusu, nazik ol")
    args = ap.parse_args()

    seed_limit = args.seed_limit or args.limit
    print(f"Tohum listesi çekiliyor (Vikisözlük, en çok {seed_limit}) …")
    seeds = seed_words(seed_limit, delay=args.delay)
    if not seeds:
        print("Tohum alınamadı; Vikisözlük API'sine erişilemiyor olabilir.", file=sys.stderr)
        return 1
    print(f"  {len(seeds)} tohum madde")

    print(f"Derleme taranıyor (en çok {args.limit} madde, {args.delay}s bekleme) …")
    records, found, queried = harvest(seeds, max_words=args.limit, delay=args.delay)

    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECORDS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    WORDLIST_PATH.write_text("\n".join(found) + "\n", encoding="utf-8")

    PROVENANCE_PATH.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-dialect-provenance/v1",
                "source": "TDK Derleme Sözlüğü (eski.sozluk.gov.tr/derleme)",
                "seed_source": f"Vikisözlük {SEED_CATEGORY} (CC-BY-SA)",
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "queried_words": queried,
                "matched_words": len(found),
                "hit_rate": round(len(found) / queried, 3) if queried else 0.0,
                "records": len(records),
                "delay_seconds": args.delay,
                "license_note": (
                    "TDK'nın ilan edilmiş açık lisansı yoktur, telif TDK'dadır. "
                    "Veri bu depoda BİLEREK tutulur: depo private, çalışma "
                    "akademik ve ticari değil, kayıtlar atıflı ve künyeli. "
                    "Gerekçe yeniden üretilebilirliktir. ⚠️ Depo public'e "
                    "açılacak olursa `data/dialect/` önce çıkarılmalıdır. "
                    "Ayrıntı: data/dialect/README.md"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"  tuttu   : {len(found)} madde")
    print(f"  kayıt   : {len(records)}")
    print(f"  liste   : {WORDLIST_PATH}")
    print(f"  kayıtlar: {RECORDS_PATH}  (depoya girmez)")
    print(f"  künye   : {PROVENANCE_PATH}")
    print()
    print("Analiz için:")
    print(f"  .venv/bin/python scripts/analyse_dialect_words.py --words {WORDLIST_PATH} --limit 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
