"""
İleri tahmin değerlendirmesi — ``make eval-prediction``.

Görev: bir dildeki biçim verildiğinde, **akraba** bir dildeki biçmi tahmin
etmek. Bu, etimolojisi yapılmamış kelimenin akrabasını bulmanın tek yoludur:
tahmin üretilir, sonra sözlükte aranır.

⚠️ Önceki taslakta bu metrik **15 kelimede** ölçülüyordu ve %47'den %85'e
çıkarmak hedefleniyordu. n=15'te %47 ile %85 arasındaki fark istatistiksel
olarak neredeyse anlamsızdır (±%25 güven aralığı). Burada test seti
``savelyevturkic``in dev kavramlarının **tamamı**dır ve sonuçlar bootstrap
güven aralığıyla verilir.

Taban çizgileri:

``identity``
    Kaynak biçmi olduğu gibi kopyala. Türki diller birbirine yakın olduğu
    için bu **yüksek** puan alır; geçilmesi zorunludur.
``learned``
    Öğrenilmiş denklik tabloları (:mod:`engine.nlp.cognate_prediction`).
``handwritten``
    Elle yazılmış eski kurallar (``sound_shifts``), en yakın adayı seçer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from engine.db.cldf_wordlist import CldfWordlist
from engine.db.language_mapping import build_mapping
from engine.evaluation.calibration import bootstrap_ci
from engine.evaluation.gold import assign_split
from engine.evaluation.metrics import edit_distance, normalized_edit_distance
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionCase:
    """Tek bir tahmin sorusu: kaynak biçimden hedef biçmi tahmin et."""

    concept: str
    source_lang: str
    source_form: str
    target_lang: str
    target_form: str


@dataclass
class PredictionScore:
    n: int = 0
    exact: int = 0
    total_ned: float = 0.0
    total_ed: int = 0
    within_one: int = 0
    per_item: list[bool] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def near_miss_rate(self) -> float:
        """Bir harf yanılma oranı — sözlükte aramak için hâlâ kullanışlıdır."""
        return self.within_one / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "within_one_edit": round(self.near_miss_rate, 4),
            "ED": round(self.total_ed / self.n, 4) if self.n else 0.0,
            "NED": round(self.total_ned / self.n, 4) if self.n else 0.0,
        }


def build_cases(
    dataset: str = "savelyevturkic",
    *,
    split: str = "dev",
    source_lang: str = "tr",
    min_length: int = 2,
) -> list[PredictionCase]:
    """Aynı akraba kümesindeki ``source_lang`` -> diğer diller çiftleri."""
    wordlist = CldfWordlist.load(dataset)
    mapping = build_mapping(wordlist)
    cases: list[PredictionCase] = []
    for cognate_set in wordlist.cognate_sets(min_languages=2):
        if assign_split(cognate_set.concept or cognate_set.id) != split:
            continue
        forms = {
            mapping[lang]: to_comparison_form(form)
            for lang, form in cognate_set.forms_by_language().items()
            if lang in mapping and len(to_comparison_form(form)) >= min_length
        }
        source_form = forms.get(source_lang)
        if not source_form:
            continue
        for target_lang, target_form in sorted(forms.items()):
            if target_lang == source_lang:
                continue
            cases.append(
                PredictionCase(
                    concept=cognate_set.concept,
                    source_lang=source_lang,
                    source_form=source_form,
                    target_lang=target_lang,
                    target_form=target_form,
                )
            )
    return cases


# --- Sistemler -------------------------------------------------------------


def predict_identity(case: PredictionCase) -> str:
    """Kaynak biçmi olduğu gibi kopyala — geçilmesi gereken çıta."""
    return case.source_form


def predict_learned(case: PredictionCase) -> str:

    return _shared_predictor().predict(case.source_form, case.source_lang, case.target_lang).form


_PREDICTOR: Any = None


def _shared_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from engine.nlp.cognate_prediction import CognatePredictor

        _PREDICTOR = CognatePredictor()
    return _PREDICTOR


def predict_handwritten(case: PredictionCase) -> str:
    """Elle yazılmış eski kurallardan **hedefe en yakın** adayı seçer.

    ⚠️ Bu, elle yazılmış kurallara **haksız bir avantaj** verir: gerçek
    kullanımda hangi adayın doğru olduğu bilinmez. Yine de öğrenilmiş
    tablolar bunu geçmelidir; geçemiyorsa öğrenme bir katkı sağlamıyor
    demektir.
    """
    from engine.utils.sound_shifts import generate_turkic_cognate_candidates

    candidates = generate_turkic_cognate_candidates(case.source_form) or [case.source_form]
    return min(candidates, key=lambda c: edit_distance(c, case.target_form))


SYSTEMS: dict[str, Callable[[PredictionCase], str]] = {
    "identity": predict_identity,
    "handwritten_oracle": predict_handwritten,
    "learned": predict_learned,
}


def score(system: Callable[[PredictionCase], str], cases: list[PredictionCase]) -> PredictionScore:
    result = PredictionScore()
    for case in cases:
        predicted = system(case) or ""
        distance = edit_distance(predicted, case.target_form)
        exact = predicted == case.target_form
        result.n += 1
        result.exact += exact
        result.within_one += distance <= 1
        result.total_ed += distance
        result.total_ned += normalized_edit_distance(predicted, case.target_form)
        result.per_item.append(exact)
    return result


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR
    from engine.evaluation.significance import compare_systems

    ap = argparse.ArgumentParser(description="İleri akraba tahmini ölçümü")
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--source", default="tr")
    args = ap.parse_args()

    cases = build_cases(args.dataset, split=args.split, source_lang=args.source)
    if not cases:
        print(f"{args.source} kaynaklı tahmin çifti bulunamadı.")
        return 1

    targets = sorted({c.target_lang for c in cases})
    print(f"\n=== ileri tahmin · {args.source} -> {len(targets)} dil · {args.split} ===")
    print(f"çift sayısı: {len(cases)} (önceki taslakta bu metrik 15 kelimede ölçülüyordu)")

    scores = {name: score(fn, cases) for name, fn in SYSTEMS.items()}
    print(f"\n{'sistem':22} {'tam':>8} {'%95 GA':>18} {'≤1 harf':>9} {'ED':>7} {'NED':>7}")
    print("-" * 76)
    rows: list[dict[str, Any]] = []
    for name, result in scores.items():
        low, high = bootstrap_ci(
            [1.0 if flag else 0.0 for flag in result.per_item],
            result.per_item,
            lambda values, _: sum(values) / len(values) if values else 0.0,
            iterations=2000,
        )
        data = result.as_dict()
        data["system"] = name
        data["accuracy_ci"] = [round(low, 4), round(high, 4)]
        rows.append(data)
        print(
            f"{name:22} {data['accuracy']:>8.4f} [{low:>7.4f},{high:>7.4f}] "
            f"{data['within_one_edit']:>9.4f} {data['ED']:>7.3f} {data['NED']:>7.4f}"
        )

    comparisons = compare_systems(
        {name: result.per_item for name, result in scores.items()}, reference="identity"
    )
    print("\nkimlik taban çizgisine karşı:")
    for row in comparisons:
        low, high = row["ci95"]
        verdict = "ANLAMLI" if row["significant_after_fdr"] else "anlamlı değil"
        print(
            f"  {row['system']:22} fark {row['difference']:+.4f} "
            f"[{low:+.4f}, {high:+.4f}] p={row['permutation_p']:.4f} -> {verdict}"
        )

    # Hedef dile göre döküm — tek ortalama, kolay ve zor dilleri gizler.
    learned = SYSTEMS["learned"]
    by_language: dict[str, PredictionScore] = {}
    for target in targets:
        subset = [c for c in cases if c.target_lang == target]
        by_language[target] = score(learned, subset)
    print("\nöğrenilmiş sistemin hedef dile göre dökümü (en iyi 10):")
    ordered = sorted(by_language.items(), key=lambda kv: -kv[1].accuracy)
    for target, result in ordered[:10]:
        print(f"  {target:5} n={result.n:<4} tam={result.accuracy:.3f} ≤1={result.near_miss_rate:.3f}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "prediction.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-prediction-eval/v1",
                "dataset": args.dataset,
                "split": args.split,
                "source_language": args.source,
                "n_cases": len(cases),
                "n_target_languages": len(targets),
                "systems": rows,
                "significance": comparisons,
                "by_target_language": {
                    target: result.as_dict() for target, result in by_language.items()
                },
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
