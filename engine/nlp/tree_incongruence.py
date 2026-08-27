"""
Ağaç-uyumsuz dağılım — Türki-içi alıntı adayları (Faz C3).

⚠️ **Motorun göremediği bir sınıf var.** Bütün alıntı sinyallerimiz
*aileler arası* alıntıya bakıyor: verici sözlüğünde karşılık, fonotaktik
ihlal, zincir kaydı. Bir kelime bir Türki dilden başka bir Türki dile
geçmişse hiçbiri ateşlenmez — biçim "Türki görünür", verici sözlüğünde
yoktur, fonotaktiği bozmaz.

seabor (List & Forkel 2022) bunu **dağılımla** yakalar: bir kökün dil
ağacındaki dağılımı akrabalıkla açıklanamıyorsa (birbirine uzak iki kolda
var, aradaki kollarda yok) yatay geçiş şüphesi doğar.

## ⚠️ Bu modül DOĞRULANMADI ve doğrulanamıyor

Ölçüm için Türki-içi alıntı etiketi gerekiyor. Elimizdeki tek kaynak
WOLD'un Sakha kayıtlarında ``Source_languoid == "Turkic"`` olan
**18 madde**. O örneklemde F skoru hesaplanabilir ama güven aralığı
kullanılamayacak kadar geniş olur; "yöntem işliyor" iddiası kurulamaz.

Bu yüzden modül bir **sınıflandırıcı değil, aday üretecidir**: şüpheli
dağılımları işaretler ve uzman incelemesine (Faz E2) gönderir. Karar
katmanına **bağlanmamıştır**.

⚠️ Doğru ölçüm için gereken: Türki dil çiftleri arasında etiketli alıntı
verisi. NorthEuraLex 8 Türki dil içeriyor ama **alıntı etiketi yok** ve
çevriyazıları yazımdan otomatik üretilmiş.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.comparative_reconstruction import LANGUAGE_BRANCHES

logger = get_logger(__name__)

#: ⚠️ Modül karar katmanına **bağlı değildir**. Doğrulanamadığı için
#: (bkz. modül başlığı) yalnız aday üretir.
USE_AS_SIGNAL = False

#: Bir kolun "temsil edilmiş" sayılması için gereken asgari tanık sayısı.
MIN_BRANCH_WITNESSES = 1

#: Şüphe eşiği: dağılım boşluk oranı bunun üstündeyse aday sayılır.
GAP_SUSPICION = 0.5

#: ⚠️ Bu sayının altındaki kümelerde dağılım sorusu **sorulamaz**.
#:
#: İlk sürümde yoktu ve sonuç kullanılamazdı: 400 kümenin **106'sı** (%27)
#: aday işaretlendi çünkü 3 tanıklı bir küme her zaman düşük kol kapsamı
#: verir. Ölçülen şey uyumsuzluk değil, **seyreklikti**.
#:
#: Modül başlığı "az dilde görülmek tek başına şüphe değildir" diyordu ama
#: uygulama bunu zorlamıyordu.
MIN_LANGUAGES_FOR_DISTRIBUTION = 8


@dataclass(frozen=True)
class DistributionVerdict:
    """Bir kökün kollar arası dağılımı ve uyumsuzluk hükmü."""

    branches_present: tuple[str, ...]
    branches_absent: tuple[str, ...]
    languages_per_branch: dict[str, int]
    gap_ratio: float
    is_scattered: bool
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "branches_present": list(self.branches_present),
            "branches_absent": list(self.branches_absent),
            "gap_ratio": round(self.gap_ratio, 4),
            "is_scattered": self.is_scattered,
            "explanation": self.explanation,
        }


def _branch_counts(languages: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code in languages:
        branch = LANGUAGE_BRANCHES.get(code)
        if branch:
            counts[branch] = counts.get(branch, 0) + 1
    return counts


def analyse(languages: list[str], *, all_branches: set[str] | None = None) -> DistributionVerdict:
    """Bir kökün kol dağılımını inceler.

    ⚠️ "Az dilde görülmek" tek başına şüphe değildir: nadir bir miras kelime
    de az dilde kalır. Şüphe uyandıran şey **dağılımın biçimidir** — birden
    çok kolda var ama her kolda **tek tük** dilde. Miras bir kök, girdiği
    kolun içinde yayılır.
    """
    known = all_branches or {b for b in LANGUAGE_BRANCHES.values() if b}
    counts = _branch_counts(languages)
    present = tuple(sorted(b for b, n in counts.items() if n >= MIN_BRANCH_WITNESSES))
    absent = tuple(sorted(known - set(present)))

    if len(present) < 2:
        return DistributionVerdict(
            present, absent, counts, 0.0, False,
            "tek kolda görülüyor — dağılım uyumsuzluğu sorulamaz",
        )
    if len(languages) < MIN_LANGUAGES_FOR_DISTRIBUTION:
        return DistributionVerdict(
            present, absent, counts, 0.0, False,
            f"yalnız {len(languages)} tanık — bu kadar az tanıkla dağılım "
            f"uyumsuzluğu ile SEYREKLİK ayırt edilemez",
        )

    # Kol içi seyreklik: her kolda o kolun kaç dilinden kaçında var?
    branch_sizes: dict[str, int] = {}
    for branch in LANGUAGE_BRANCHES.values():
        if branch:
            branch_sizes[branch] = branch_sizes.get(branch, 0) + 1
    coverage = [
        counts[branch] / branch_sizes.get(branch, 1) for branch in present
    ]
    mean_coverage = sum(coverage) / len(coverage)
    gap_ratio = 1.0 - mean_coverage
    scattered = gap_ratio >= GAP_SUSPICION and len(present) >= 2

    if scattered:
        explanation = (
            f"{len(present)} kolda görülüyor ama her kolda seyrek "
            f"(ortalama kol kapsamı %{100 * mean_coverage:.0f}); miras bir kök "
            f"girdiği kolun içinde yayılır"
        )
    else:
        explanation = (
            f"{len(present)} kolda ve kol içinde yaygın "
            f"(ortalama kapsam %{100 * mean_coverage:.0f}) — miras dağılımıyla uyumlu"
        )
    return DistributionVerdict(present, absent, counts, gap_ratio, scattered, explanation)


def candidates(gold_items: list) -> list[dict[str, Any]]:
    """Altın standarttan Türki-içi alıntı **adaylarını** çıkarır.

    ⚠️ Çıktı bir karar değil, uzman incelemesine giden bir listedir
    (Faz E2). Bu modülün doğruluğu ölçülemedi.
    """
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping

    mapping = build_mapping(CldfWordlist.load("savelyevturkic"))
    out: list[dict[str, Any]] = []
    for item in gold_items:
        codes = [mapping[lang] for lang in item.witnesses if lang in mapping]
        if len(codes) < MIN_LANGUAGES_FOR_DISTRIBUTION:
            continue
        verdict = analyse(codes)
        if verdict.is_scattered:
            out.append(
                {
                    "set_id": item.set_id,
                    "concept": item.concepticon_gloss,
                    "gold_form": item.gold_form,
                    "n_languages": len(codes),
                    **verdict.as_dict(),
                }
            )
    out.sort(key=lambda row: -row["gap_ratio"])
    return out


def main() -> int:
    import json

    from engine.evaluation.gold import GoldStandard
    from engine.evaluation.report import EVAL_DIR

    gold = GoldStandard.build()
    rows = candidates(gold.items)
    print("=== TÜRKİ-İÇİ ALINTI ADAYLARI (ağaç-uyumsuz dağılım) ===")
    print(
        "⚠️ Bu bir SINIFLANDIRICI DEĞİL, aday üretecidir. Türki-içi alıntı\n"
        "   etiketi olmadığı için doğruluğu ölçülemedi (elimizdeki tek\n"
        "   kaynak WOLD/Sakha'da 18 madde — güven aralığı kullanılamaz).\n"
    )
    print(f"{len(rows)}/{len(gold.items)} küme şüpheli dağılım gösteriyor\n")
    for row in rows[:15]:
        print(
            f"  {row['gold_form']:14} {row['concept']:16} "
            f"{row['n_languages']:>2} dil · {len(row['branches_present'])} kol · "
            f"boşluk {row['gap_ratio']:.2f}"
        )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "tree_incongruence.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-tree-incongruence/v1",
                "note": (
                    "Aday üreteci; doğrulanmadı. Karar katmanına bağlı değil "
                    "(USE_AS_SIGNAL = False)."
                ),
                "n_candidates": len(rows),
                "n_items": len(gold.items),
                "candidates": rows,
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
