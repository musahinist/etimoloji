#!/usr/bin/env python3
"""
Türkçe altın alıntı kümesi — WOLD'dan bağımsız ikinci ölçüt (Faz C6).

⚠️ **Neden gerekiyor.** Alıntı tespitinin birincil ölçütü WOLD'dur ama
WOLD'da tek Türki dil vardır: Sakha (n=769). Türkçe için elimizdeki tek
etiket kaynağı Wiktionary'ydi ve motorun zincir sinyali **zaten o etiketi
okuyor**; ona karşı ölçüm döngüseldir. Ayrıca o kümede alıntı oranı %72,9
olduğu için F'yi en yükselten karar "hepsine alıntı de"dir ve sistemler
çöküyor.

Bu betik **Wiktionary'ye hiç değmeden** iki bağımsız Türkçe kaynaktan
etiket toplar:

* **TDK Güncel Türkçe Sözlük** — ``sozluk.gov.tr/gts?ara=<kelime>``;
  ``lisan`` alanı kaynak dili verir ("Arapça kitāb").
* **Nişanyan Sözlük** — ``nisanyansozluk.com/api/words/<kelime>``;
  ``relation.name`` ``alıntı`` veya ``ses evrimi`` der.

⚠️ **İki kaynak da kusurlu; anlaşmaları şart koşulur.** Tek kaynağa
güvenmek o kaynağın sistematik yanını altın standarda taşır. Uyuşmayan
maddeler kümeye **alınmaz** ve uyuşmazlık oranı künyeye yazılır — o oran
kendi başına bir bulgudur (bkz. Faz E1 mantığı).

⚠️ **TDK'da boş ``lisan`` "Türkçe kökenli" demektir** ama sözlüğün
etimoloji vermediği maddeler de boştur. İkisi ayırt edilemediği için boş
``lisan`` **belirsiz** sayılır ve madde kümeye girmez.

⚠️ Bu kısıt yüzünden küme **miras yönünde eksik** kalır: TDK yalnız alıntı
maddelerde ``lisan`` yazdığı için miras kelimelerin çoğu belirsize düşer ve
kümedeki alıntı oranı gerçek dilden yüksek çıkar. Oran künyeye yazılır.

⚠️ **Lisans.** Nişanyan Sözlük'ün kullanım koşulları geliştiriciye açık bir
API lisansı ilan etmiyor. Bu veri **yalnız iç doğrulama** için, atıflı ve
sınırlı hacimde çekiliyor; yeniden dağıtılmıyor (``data/gold/`` .gitignore
altında). TDK Güncel Türkçe Sözlük kamuya açık bir kurum sözlüğüdür.

Kullanım::

    python scripts/build_turkish_loanword_gold.py --limit 1500
    python scripts/build_turkish_loanword_gold.py --limit 200 --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import PROJECT_ROOT  # noqa: E402

TDK_URL = "https://sozluk.gov.tr/gts?ara={word}"
NISANYAN_URL = "https://www.nisanyansozluk.com/api/words/{word}?session=1"

OUT_PATH = PROJECT_ROOT / "data" / "gold" / "turkish_loanwords.json"

#: Nişanyan'da doğrudan "miras" sayılan ilişki adları.
#:
#: ⚠️ Adlar ölçülerek alındı: gerçek değerler ``alıntı``, ``ses evrimi``,
#: ``türeme``, ``muhtemel türeme``. İlk sürüm ``türetme`` yazıyordu (yanlış
#: yazım) ve türemiş kelimelerin **hiçbiri** etiketlenemiyordu.
INHERITED_RELATIONS = {"ses evrimi"}

#: Doğrudan "alıntı" sayılan ilişki adları.
BORROWED_RELATIONS = {"alıntı"}

#: ⚠️ Türeme ilişkisi tek başına karar vermez: karar **taban dilin**
#: Türki olup olmamasına bakar. ``bakımlı`` (Türkiye Türkçesinden türeme)
#: miras, ``kitapçı`` (Arapça tabandan türeme) alıntı tabanlıdır.
DERIVATION_RELATIONS = {"türeme", "muhtemel türeme"}

#: ⚠️ Türki diller "verici" sayılmaz: Osmanlıcadan miras alınan bir kelime
#: alıntı değildir. TDK bazen "Eski Türkçe" yazar.
TURKIC_SOURCE_NAMES = (
    "türkçe", "türkiye türkçesi", "eski türkçe", "orta türkçe",
    "osmanlıca", "oğuzca", "proto-türkçe", "ana türkçe",
)

#: İstekler arası bekleme (saniye). ⚠️ İki kamuya açık servise nazik
#: davranmak gerekiyor; hız sınırı ilan edilmemiş.
DEFAULT_DELAY = 0.35


def turkish_word_list(limit: int) -> list[str]:
    """Sözlük indeksinden Türkçe kelime listesi.

    ⚠️ İndeks yalnız **kelime listesi** için kullanılıyor; etiket oradan
    GELMİYOR. Etiket TDK ve Nişanyan'dan gelir, yani Wiktionary'ye
    değilmiyor.
    """
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not index.exists:
        raise FileNotFoundError(
            "sözlük indeksi yok: python -m engine.db.lexicon_index --build"
        )
    with index.connect() as connection:
        rows = connection.execute(
            # ⚠️ Ekler ve çok kelimeli birimler elenmeli: ``-abilmek``,
            # ``-acak`` gibi maddeler sözlükte var ama alıntı/miras sorusunun
            # öznesi değiller. İlk sürüm alfabetik sıralayınca ilk 60 madde
            # tümüyle EK çıktı ve hiçbir etiket alınamadı.
            "SELECT DISTINCT word FROM entries WHERE lang_code = 'tr' "
            "AND length(comparison) >= 3 AND word NOT LIKE '% %' "
            "AND word NOT LIKE '-%' AND word NOT LIKE '%-' "
            "AND word NOT LIKE '%.%' AND pos IN ('noun', 'verb', 'adj') "
            "ORDER BY word",
        ).fetchall()
    words = [row["word"] for row in rows]
    if len(words) <= limit:
        return words
    # ⚠️ Örneklem **deterministik** olmalı: aynı komut aynı kümeyi kurmalı,
    # yoksa "aynı veride ölçtük" iddiası kurulamaz. Alfabetik ilk N almak da
    # yanlıdır (alfabenin başı Arapça alıntılarla doludur).
    step = len(words) / limit
    return [words[int(i * step)] for i in range(limit)]


def tdk_label(word: str, session: requests.Session) -> tuple[str | None, str]:
    """TDK etiketi: ``("alıntı"|"miras"|None, kaynak_dil)``."""
    try:
        response = session.get(TDK_URL.format(word=word), timeout=20)
        data = response.json()
    except (requests.RequestException, ValueError):
        return (None, "")
    if not isinstance(data, list) or not data:
        return (None, "")
    entry = data[0]
    source = (entry.get("lisan") or "").strip()
    if not source:
        # ⚠️ Boş `lisan` "Türkçe kökenli" DEMEK OLABİLİR ama sözlüğün
        # etimoloji vermediği maddeler de boştur. Karar tek başına
        # verilmez; Nişanyan'ın onayı aranır.
        return ("miras?", "")
    lowered = source.lower()
    if any(name in lowered for name in TURKIC_SOURCE_NAMES):
        return ("miras", source)
    return ("alıntı", source)


def nisanyan_label(word: str, session: requests.Session) -> tuple[str | None, str]:
    """Nişanyan etiketi: ``("alıntı"|"miras"|None, kaynak_dil)``."""
    try:
        response = session.get(NISANYAN_URL.format(word=word), timeout=20)
        data = response.json()
    except (requests.RequestException, ValueError):
        return (None, "")
    if data.get("isUnsuccessful") or not data.get("words"):
        return (None, "")
    etymologies = data["words"][0].get("etymologies") or []
    if not etymologies:
        return (None, "")
    first = etymologies[0]
    relation = ((first.get("relation") or {}).get("name") or "").strip().lower()
    languages = [
        (lang.get("name") or "").strip() for lang in (first.get("languages") or [])
    ]
    source = languages[0] if languages else ""
    turkic = any(name in source.lower() for name in TURKIC_SOURCE_NAMES)
    if relation in INHERITED_RELATIONS:
        return ("miras", source)
    if relation in BORROWED_RELATIONS:
        return ("miras" if turkic else "alıntı", source)
    if relation in DERIVATION_RELATIONS and source:
        # Türemenin TABANI belirleyicidir.
        return ("miras" if turkic else "alıntı", source)
    return (None, source)


def build(limit: int, delay: float) -> dict[str, Any]:
    words = turkish_word_list(limit)
    session = requests.Session()
    session.headers["User-Agent"] = "turkic-etymology-engine/3.0 (+research)"

    agreed: list[dict[str, str]] = []
    disagreed: list[dict[str, str]] = []
    unknown = 0

    for index, word in enumerate(words, start=1):
        tdk, tdk_source = tdk_label(word, session)
        time.sleep(delay)
        nisanyan, nisanyan_source = nisanyan_label(word, session)
        time.sleep(delay)

        # ⚠️ **Kanıt kuralı ASİMETRİKTİR ve bu gizlenemez.**
        #
        # TDK yalnız ALINTI maddelerde `lisan` yazar; miras kelimelerde alan
        # boştur ve "Türkçe kökenli" ile "etimoloji verilmemiş" ayırt
        # edilemez. Dolayısıyla:
        #
        #   alıntı  <- İKİ kaynak da alıntı diyor          (güçlü)
        #   miras   <- Nişanyan açıkça `ses evrimi` diyor
        #              VE TDK bir kaynak dil YAZMIYOR      (zayıf-orta)
        #
        # İlk sürüm TDK'nın sessizliğini Nişanyan'dan türetiyordu ve "iki
        # kaynak anlaştı" demek tautolojiydi: uyum %100 çıkıyordu. Sonraki
        # sürüm sessizliği tümden belirsiz saydı ve küme %100 ALINTI oldu —
        # ölçüt olarak işe yaramaz.
        tdk_silent = tdk == "miras?"
        label: str | None = None
        if nisanyan == "alıntı" and tdk == "alıntı":
            label = "alıntı"
        elif nisanyan == "miras" and (tdk_silent or tdk == "miras"):
            label = "miras"
        elif tdk == "alıntı" and nisanyan == "miras":
            disagreed.append(
                {"word": word, "tdk": "alıntı", "nisanyan": "miras",
                 "tdk_source": tdk_source, "nisanyan_source": nisanyan_source}
            )
        elif nisanyan == "alıntı" and tdk == "miras":
            disagreed.append(
                {"word": word, "tdk": "miras", "nisanyan": "alıntı",
                 "tdk_source": tdk_source, "nisanyan_source": nisanyan_source}
            )

        if label is None and not (
            (tdk == "alıntı" and nisanyan == "miras")
            or (nisanyan == "alıntı" and tdk == "miras")
        ):
            unknown += 1
        elif label is not None:
            agreed.append(
                {
                    "word": word,
                    "label": label,
                    "evidence": "iki kaynak" if not tdk_silent else "nişanyan+tdk sessiz",
                    "tdk_source": tdk_source,
                    "nisanyan_source": nisanyan_source,
                }
            )
        if index % 50 == 0:
            print(
                f"  … {index}/{len(words)} · anlaşma {len(agreed)} · "
                f"uyuşmazlık {len(disagreed)} · belirsiz {unknown}"
            )

    decided = len(agreed) + len(disagreed)
    borrowed = sum(1 for row in agreed if row["label"] == "alıntı")
    return {
        "_schema": "turkic-etymology-turkish-loanword-gold/v1",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": ["TDK Güncel Türkçe Sözlük", "Nişanyan Sözlük"],
        "n_queried": len(words),
        "n_agreed": len(agreed),
        "n_two_source": sum(1 for r in agreed if r["evidence"] == "iki kaynak"),
        "n_disagreed": len(disagreed),
        "n_unknown": unknown,
        "source_agreement_rate": round(len(agreed) / decided, 4) if decided else 0.0,
        "borrowed_rate": round(borrowed / len(agreed), 4) if agreed else 0.0,
        "caveats": [
            "⚠️ KANIT KURALI ASİMETRİKTİR. 'alıntı' etiketi İKİ kaynağın da "
            "alıntı demesini gerektirir (güçlü). 'miras' etiketi Nişanyan'ın "
            "açıkça `ses evrimi` demesi VE TDK'nın kaynak dil YAZMAMASIYLA "
            "verilir (zayıf-orta) — çünkü TDK yalnız alıntı maddelerde "
            "`lisan` yazar ve 'Türkçe kökenli' ile 'etimoloji verilmemiş' "
            "ayırt edilemez.",
            "Bu asimetri yüzünden yanlış-pozitif (miras'ı alıntı sanmak) "
            "hatası, yanlış-negatiften daha az olasıdır; ölçümde bu yön "
            "dikkate alınmalıdır.",
            "Kelime listesi sözlük indeksinden gelir ama ETİKET oradan "
            "gelmez; Wiktionary'ye değilmemiştir.",
            "Nişanyan Sözlük'ün API lisansı ilan edilmemiştir; veri yalnız "
            "iç doğrulama için, atıflı ve sınırlı hacimde çekilmiştir, "
            "yeniden dağıtılmaz.",
        ],
        "items": agreed,
        "disagreements": disagreed[:200],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Türkçe altın alıntı kümesi kur")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    payload = build(args.limit, args.delay)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\nsorgulanan {payload['n_queried']} · anlaşma {payload['n_agreed']} "
        f"· uyuşmazlık {payload['n_disagreed']} · belirsiz {payload['n_unknown']}"
    )
    print(f"kaynak uyum oranı: {payload['source_agreement_rate']:.4f}")
    print(f"alıntı oranı: {payload['borrowed_rate']:.4f}")
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
