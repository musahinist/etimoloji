"""
Kronoloji (A-HVP 2. aşama, zaman kilidi) — ``make eval-chronology``.

A-HVP'nin 2. aşaması kelimenin ilk tanıklama yılını kullanır
(``validation_report.stage_breakdown.stage2_time_lock.attestation_year``).
Bu yılın **var olup olmadığı** ve **doğru olup olmadığı** hiç ölçülmemişti.

Başvuru: Starling turcet tanıklarının kaynak etiketleri (Orkh. 732, KB 1069,
MK 1072, IM 1245, AH 1300) — ``StarlingEtymology.earliest_dated_source``.
Nişanyan ve EtimolojiTürkçe **kullanılmaz** (ikisi de ağ kaynağıdır ve bu
ölçümde kapalıdır; EtimolojiTürkçe Nişanyan türevidir).

Kelime kümesi: Starling TRK alanında TEK köke bağlanan (eşsesli belirsizliği
olmayan), tarihli tanığı bulunan ve Türk alfabesiyle yazılıp yerel indekste
Türkçe maddesi olan biçimler — yani Starling'e göre MİRAS Türkçe kelimeler.
Türkçe altın küme (TDK+Nişanyan) kullanılmaz.

İki yapılandırma (bkz. ``headline_eval.build_engine``):

* ``yerel`` — **DÖNGÜSEL**: ``StarlingFetcher`` ``first_attestation``
  alanını tam da bu etiketten üretir; motorun yılı başvurunun kopyasıdır.
  Yalnız "Starling yılı hatta ulaşıyor mu?" denetimi.
* ``starling_yok`` — Starling motorda okunmaz. Motorun yılı yalnız yerel
  kayıtlardaki eser adlarından (Orhun, Divânu Lugâti't-Türk, Kutadgu Bilig…;
  ``DATED_SOURCES``) gelir. Başvurudan bağımsızdır; ortak olan yalnız eserin
  kendisidir (DLT 1074 ~ MK 1072), bu da ölçülmek istenen şeydir.

Üretimde EtimolojiTürkçe de yıl verir; ağ kapalı olduğundan o katkı burada
ÖLÇÜLMEZ (ve ölçülse Nişanyan'a döngüsel olurdu).

Ölçüler: kapsam (yıl var mı), yıl ≤ başvuru+5 (motor daha GEÇ bir tarih
vermiyor), |fark| ≤ 100 (aynı yüzyıl), |fark| ≤ 5 (aynı eser), medyan |fark|.
Taban çizgisi: her kelimeye başvuru yıllarının medyanını vermek.
"""

from __future__ import annotations

import json
import random
from statistics import median
from typing import Any

from engine.evaluation.headline_eval import SEED, _head, is_turkish_word, run_engine, wilson
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SAMPLE = 200
#: Aynı eserin iki geleneksel tarihi arasındaki pay (MK 1072 ~ DLT 1074,
#: Orkh. 732 ~ Orhun 735).
SAME_SOURCE_TOLERANCE = 5
CENTURY = 100


def reference_items(n: int = SAMPLE, seed: int = SEED) -> list[tuple[str, int, str]]:
    """``(kelime, başvuru yılı, etiket)`` — sabit tohumlu örneklem."""
    from engine.db.starling import turkish_lookup

    table = turkish_lookup()
    rows: dict[str, tuple[int, str]] = {}
    for key in sorted(table):
        matches = table[key]
        if len(matches) != 1:
            continue
        dated = matches[0].earliest_dated_source()
        word = key.rstrip("-")
        if not dated or len(word) < 2 or word in rows:
            continue
        rows[word] = dated
    pool = [w for w in rows if is_turkish_word(w)]
    random.Random(seed).shuffle(pool)
    return sorted((w, rows[w][0], rows[w][1]) for w in pool[:n])


def agreement(engine_year: int | None, reference: int) -> dict[str, Any]:
    if engine_year is None:
        return {"has_year": False, "not_later": False, "within_century": False,
                "same_source": False, "diff": None}
    diff = engine_year - reference
    return {
        "has_year": True,
        "not_later": diff <= SAME_SOURCE_TOLERANCE,
        "within_century": abs(diff) <= CENTURY,
        "same_source": abs(diff) <= SAME_SOURCE_TOLERANCE,
        "diff": diff,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    covered = [r for r in rows if r["has_year"]]
    out: dict[str, Any] = {"n": n}
    k = len(covered)
    out["coverage"] = round(k / n, 4) if n else 0.0
    out["coverage_ci95"] = wilson(k, n)
    for name in ("not_later", "within_century", "same_source"):
        hits = sum(r[name] for r in covered)
        out[f"{name}|covered"] = round(hits / k, 4) if k else 0.0
        out[f"{name}|covered_ci95"] = wilson(hits, k)
        out[f"{name}|all"] = round(hits / n, 4) if n else 0.0
        out[f"{name}|all_ci95"] = wilson(hits, n)
    out["median_abs_diff|covered"] = median(abs(r["diff"]) for r in covered) if covered else None
    return out


CIRCULARITY = {
    "yerel": "DÖNGÜSEL — StarlingFetcher first_attestation alanını bu etiketten üretir.",
    "starling_yok": (
        "Starling okunmaz; yıl yerel kayıtlardaki eser adlarından. Nişanyan/EtimolojiTürkçe "
        "ağ kaynağı olduğu için kapalı (üretimdeki katkıları ölçülmedi)."
    ),
    "taban_medyan_yıl": "Her kelimeye başvuru yıllarının medyanı — bilgi taşımayan taban.",
}


def run(*, sample: int = SAMPLE, fresh: bool = False) -> dict[str, Any]:
    items = reference_items(sample)
    words = [w for w, _, _ in items]
    configs = {
        "yerel": run_engine(words, ablate_starling=False, fresh=fresh),
        "starling_yok": run_engine(words, ablate_starling=True, fresh=fresh),
    }
    constant = int(median(y for _, y, _ in items))
    systems: dict[str, Any] = {}
    detail: list[dict[str, Any]] = []
    for name, runs in configs.items():
        rows = [agreement(runs[w].get("attestation_year"), y) for w, y, _ in items]
        systems[name] = summarize(rows)
        systems[name]["engine_record_counts"] = _record_counts(runs[w] for w in words)
        systems[name].update(precision_split(rows, [runs[w] for w in words]))
    systems["taban_medyan_yıl"] = summarize([agreement(constant, y) for _, y, _ in items])
    for w, y, tag in items:
        detail.append({
            "word": w, "reference_year": y, "reference_tag": tag,
            **{f"{name}_year": configs[name][w].get("attestation_year") for name in configs},
            "starling_yok_record": configs["starling_yok"][w].get("attestation_record"),
            **{f"{name}_precision": configs[name][w].get("attestation_precision") for name in configs},
        })
    tags: dict[str, int] = {}
    for _, _, tag in items:
        tags[tag] = tags.get(tag, 0) + 1
    return {
        "_schema": "chronology_eval/v1",
        "commit": _head(),
        "reference": "Starling earliest_dated_source (ATU+KRH kaynak etiketleri)",
        "n": len(items),
        "reference_tags": tags,
        "constant_baseline_year": constant,
        "circularity": CIRCULARITY,
        "systems": systems,
        "items": detail,
    }


def precision_split(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Nokta tarih ile dönem düzeyindeki tanığı ayrı sayar.

    Wilkens (Eski Uygurca) tanığı "9.-14. yy" aralığıdır, yıl alanında
    yalnız üst sınır (1350) durur: yüzyıl isabetine katılması anlamsız.
    "nokta tarih kapsamı" ve "yüzyıl içi|nokta" yalnız nokta tarihleri,
    "dönem düzeyinde kapsam" yalnız aralıkları sayar.
    """
    n = len(rows)
    point = [r for r, run in zip(rows, runs, strict=True)
             if r["has_year"] and run.get("attestation_precision") != "period"]
    period = sum(1 for r, run in zip(rows, runs, strict=True)
                 if r["has_year"] and run.get("attestation_precision") == "period")
    hits = sum(r["within_century"] for r in point)
    return {
        "coverage_point": round(len(point) / n, 4) if n else 0.0,
        "coverage_period": round(period / n, 4) if n else 0.0,
        "within_century|point": round(hits / len(point), 4) if point else 0.0,
        "within_century|point_ci95": wilson(hits, len(point)),
    }


def _record_counts(runs) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in runs:
        key = str(r.get("attestation_year")) if r.get("attestation_year") is not None else "yok"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--fresh", action="store_true", help="önbelleği yok say")
    args = parser.parse_args()
    payload = run(sample=args.sample, fresh=args.fresh)
    print(f"\n=== kronoloji (A-HVP 2. aşama) · n={payload['n']} · commit {payload['commit']} ===")
    print(f"başvuru etiketleri: {payload['reference_tags']}")
    for name, s in payload["systems"].items():
        print(
            f"{name:18} kapsam {s['coverage']:.3f} {s['coverage_ci95']}  "
            f"≤başvuru {s['not_later|covered']:.3f}  yüzyıl içi {s['within_century|covered']:.3f} "
            f"{s['within_century|covered_ci95']} (tümünde {s['within_century|all']:.3f})  "
            f"aynı eser {s['same_source|covered']:.3f}  medyan|fark| {s['median_abs_diff|covered']}"
        )
        if "coverage_point" in s:
            print(f"{'':18} nokta tarih kapsamı {s['coverage_point']:.3f} (yüzyıl içi "
                  f"{s['within_century|point']:.3f})  dönem düzeyinde kapsam {s['coverage_period']:.3f}")
        print(f"{'':18} ⚠️ {payload['circularity'][name]}")
    out = EVAL_DIR / "chronology.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
