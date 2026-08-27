"""
Altın standartlar arası uyuşmazlık — motorun skoru neye göre okunmalı?

⚠️ Bu ölçüm olmadan hiçbir akraba tespiti skoru yorumlanamaz. "Motor
B-Cubed F 0,82 aldı" cümlesi, **uzmanların birbiriyle** ne kadar uyuştuğu
bilinmeden anlamsızdır: tavan 1,00 değildir.

List, Walworth ve ark. (2018, *Journal of Language Evolution*) bu boşluğu
açıkça ilan ediyor — akraba kümesi kararlarında uzmanlar arası uyumu
sistematik ölçen bir çalışma **yok**. Bu modül o ölçümü yapar.

## Yöntem

İki bağımsız uzman derlemesi karşılaştırılır:

* ``savelyevturkic`` — Savelyev & Robbeets 2020 (7.521 öğe, 650 küme)
* ``hruschkaturkic`` — Hruschka ve ark. 2015 (4.060 öğe, 219 küme)

⚠️ **Kavram köprüsü kurulamıyor**: ``hruschkaturkic``te Concepticon glossu
hiç yok, parametreler ``Etymon 2`` gibi künye etiketleri. Bu yüzden köprü
**öğe düzeyinde** kurulur: bir ``(dil, karşılaştırma biçmi)`` çifti iki
derlemede de geçiyorsa ortak öğedir. Ölçüldü: **1.833 ortak öğe, 26 dil**.

Her derleme bu ortak öğeler üzerinde bir **bölümleme** üretir. İki bölümleme
B-Cubed P/R/F ve Ayarlanmış Rand İndeksiyle karşılaştırılır.

⚠️ **Bu bir "doğruluk" ölçümü değildir**; hangi uzmanın haklı olduğu
sorulmuyor. Ölçülen, kararların ne kadar örtüştüğü — yani otomatik bir
sistemin ulaşabileceği **gerçekçi tavan**.

## ⚠️ Bu ölçümün kendi kısıtları

* Ortak öğe tanımı **biçim eşleşmesine** dayanır; çevriyazı farkları öğeyi
  ortaklıktan düşürür ve örneklem uzman uyuşmasının kolay tarafına kayabilir.
* İki derleme aynı kavram listesini kullanmıyor; kesişim rastgele değildir.
* Küme büyüklüğü dağılımları farklı (650 vs 219 küme) ve B-Cubed buna
  duyarlıdır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

DATASETS = ("savelyevturkic", "hruschkaturkic")


@dataclass(frozen=True)
class AgreementResult:
    """İki derlemenin ortak öğeler üzerindeki uyumu."""

    n_items: int
    n_languages: int
    clusters_a: int
    clusters_b: int
    bcubed: dict[str, float]
    adjusted_rand: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_languages": self.n_languages,
            "clusters_a": self.clusters_a,
            "clusters_b": self.clusters_b,
            "bcubed_precision": round(self.bcubed["precision"], 4),
            "bcubed_recall": round(self.bcubed["recall"], 4),
            "bcubed_fscore": round(self.bcubed["fscore"], 4),
            "adjusted_rand_index": round(self.adjusted_rand, 4),
        }


def adjusted_rand_index(a: dict[Any, Any], b: dict[Any, Any]) -> float:
    """Ayarlanmış Rand İndeksi — şansa göre düzeltilmiş bölümleme uyumu.

    ⚠️ Düzeltilmemiş Rand indeksi büyük veri kümelerinde şansla bile
    0,9'un üstüne çıkar; ayarlanmış sürüm beklenen değeri **sıfıra**
    çeker. 1,0 tam uyum, 0,0 şans düzeyi, negatif şanstan kötü.
    """
    shared = sorted(set(a) & set(b), key=repr)
    if len(shared) < 2:
        return 0.0

    contingency: dict[tuple[Any, Any], int] = {}
    rows: dict[Any, int] = {}
    columns: dict[Any, int] = {}
    for item in shared:
        key = (a[item], b[item])
        contingency[key] = contingency.get(key, 0) + 1
        rows[a[item]] = rows.get(a[item], 0) + 1
        columns[b[item]] = columns.get(b[item], 0) + 1

    def pairs(n: int) -> float:
        return n * (n - 1) / 2

    index = sum(pairs(count) for count in contingency.values())
    row_sum = sum(pairs(count) for count in rows.values())
    column_sum = sum(pairs(count) for count in columns.values())
    total = pairs(len(shared))
    expected = row_sum * column_sum / total if total else 0.0
    maximum = (row_sum + column_sum) / 2
    if maximum == expected:
        return 0.0
    return (index - expected) / (maximum - expected)


def _partition(dataset: str) -> dict[tuple[str, str], str]:
    """``(dil kodu, karşılaştırma biçmi) -> küme kimliği``."""
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.utils.orthography import to_comparison_form

    wordlist = CldfWordlist.load(dataset)
    mapping = build_mapping(wordlist)
    items: dict[tuple[str, str], str] = {}
    for cognate_set in wordlist.cognate_sets(min_languages=2):
        for language, form in cognate_set.forms_by_language().items():
            code = mapping.get(language)
            comparison = to_comparison_form(form)
            if code and comparison:
                items[(code, comparison)] = cognate_set.id
    return items


def measure(datasets: tuple[str, str] = DATASETS) -> AgreementResult | None:
    """İki uzman derlemesinin ortak öğeler üzerindeki uyumunu ölçer."""
    from engine.evaluation.metrics import bcubed_fscore

    try:
        first, second = (_partition(name) for name in datasets)
    except FileNotFoundError:
        logger.info("Uyum ölçümü için her iki CLDF veri kümesi de gerekiyor")
        return None

    shared = set(first) & set(second)
    if len(shared) < 2:
        return None

    a = {f"{lang}:{form}": first[(lang, form)] for lang, form in shared}
    b = {f"{lang}:{form}": second[(lang, form)] for lang, form in shared}
    return AgreementResult(
        n_items=len(shared),
        n_languages=len({lang for lang, _ in shared}),
        clusters_a=len(set(a.values())),
        clusters_b=len(set(b.values())),
        bcubed=bcubed_fscore(a, b),
        adjusted_rand=adjusted_rand_index(a, b),
    )


def pairwise_disagreements(
    datasets: tuple[str, str] = DATASETS, *, limit: int = 20
) -> list[dict[str, str]]:
    """İki derlemenin **ayrıştığı** somut çiftler — hükmü okunur kılar.

    Bir çift, bir derlemede aynı kümede diğerinde ayrı kümedeyse
    uyuşmazlıktır. Sayı değil, örnek döner: uzman uyuşmazlığı soyut bir
    bant değil, adı konabilir kararlardır.
    """
    try:
        first, second = (_partition(name) for name in datasets)
    except FileNotFoundError:
        return []
    shared = sorted(set(first) & set(second))
    out: list[dict[str, str]] = []
    for left, right in combinations(shared, 2):
        together_a = first[left] == first[right]
        together_b = second[left] == second[right]
        if together_a != together_b:
            out.append(
                {
                    "a": f"{left[0]} {left[1]}",
                    "b": f"{right[0]} {right[1]}",
                    "agrees": datasets[0] if together_a else datasets[1],
                }
            )
            if len(out) >= limit:
                break
    return out


def main() -> int:
    from engine.evaluation.report import EVAL_DIR

    result = measure()
    if result is None:
        print("Ölçüm yapılamadı: her iki CLDF veri kümesi de gerekiyor.")
        return 1

    print("=== ALTIN STANDARTLAR ARASI UYUM ===")
    print(f"{DATASETS[0]} × {DATASETS[1]}")
    print(
        f"ortak öğe {result.n_items} · {result.n_languages} dil · "
        f"küme {result.clusters_a} vs {result.clusters_b}"
    )
    print(
        f"\nB-Cubed  kesinlik {result.bcubed['precision']:.4f} "
        f"duyarlılık {result.bcubed['recall']:.4f} "
        f"**F {result.bcubed['fscore']:.4f}**"
    )
    print(f"Ayarlanmış Rand İndeksi  {result.adjusted_rand:.4f}")
    print(
        "\n⚠️ Bu bir doğruluk ölçümü DEĞİLDİR; hangi uzmanın haklı olduğu\n"
        "sorulmuyor. Ölçülen, otomatik bir sistemin ulaşabileceği gerçekçi\n"
        "TAVAN — akraba tespiti skorları bu bantla birlikte okunmalıdır."
    )
    examples = pairwise_disagreements(limit=6)
    if examples:
        print("\nörnek uyuşmazlıklar (biri birleştiriyor, öteki ayırıyor):")
        for row in examples:
            print(f"  {row['a']:22} ~ {row['b']:22} birleştiren: {row['agrees']}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "gold_agreement.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-gold-agreement/v1",
                "datasets": list(DATASETS),
                **result.as_dict(),
                "note": (
                    "Kavram köprüsü kurulamıyor (hruschkaturkic'te Concepticon "
                    "glossu yok); köprü öğe düzeyinde, (dil, karşılaştırma biçmi) "
                    "çiftleriyle kuruldu."
                ),
                "examples": pairwise_disagreements(limit=20),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
