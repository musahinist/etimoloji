"""
Verici dil tanıma — ``make eval-donor``.

``make eval-borrowing`` yalnız "alıntı mı?" sorusunu ölçer. Alıntı dendiğinde
motorun gösterdiği VERİCİ dilin doğru olup olmadığı hiç ölçülmemişti.

Başvuru: WOLD Sakha (Pakendorf & Novgorodova 2009). ``borrowings.csv``
dosyasındaki ``Source_relation = immediate`` satırının ``Source_languoid``
alanı (Russian, Mongolic, Evenki, Turkic, …). Yalnız WOLD'un "kesinlikle" /
"muhtemelen alıntı" kademesindeki ve vericisi ``Unidentified`` OLMAYAN
maddeler ölçülür.

⚠️ ``borrowing_eval.load_wold_cases`` vericiyi ``row["Borrowed_base"]``
alanından okuyor; WOLD CLDF'de alan adı küçük harfli ``borrowed_base``tir ve
Sakha'da verici dili değil serbest not taşır. O yüzden ``BorrowingCase.donor``
Sakha'da hep boştur; burada verici ``borrowings.csv``'den okunur.

Sistemler
---------
* ``zincir``: ``BorrowingDetector.detect(...).donor_language`` — yerel sözlük
  indeksindeki (Wiktionary) alıntı kaydının verici kodu.
* ``yakınlık``: verici yakınlığı sinyalinin en yakın maddesinin dili
  (``ru``/``mn``/``evn`` sözlükleri, anlam kısıtlı SCA).
* ``motor``: önce zincir, yoksa yakınlık — kullanıcıya gösterilebilecek en
  iyi verici tahmini.
* ``motor|alıntı_dedi``: ``motor``, ama yalnız dedektör ``is_borrowed``
  dediyse (diğerleri "tahmin yok" sayılır).
* ``çoğunluk``: hep en sık verici (Rusça) — taban çizgisi.

Doğruluk **tüm maddeler** üzerinden verilir (tahmin yoksa yanlış); ayrıca
kapsam ve kapsanan maddelerdeki doğruluk ayrı raporlanır.

⚠️ Döngüsellik durumu
* Motorun verici tahmini WOLD'dan ÖĞRENİLMEZ. Zincir Wiktionary Sakha
  etimolojilerini, yakınlık Wiktionary ru/mn/evn sözlüklerini okur.
  Wiktionary editörleri WOLD ile aynı sözlük kaynaklarına (Pekarskij,
  Kałużyński, Anikin) dayanabilir — bağımsız ama ortak kaynaklı.
* ``SAKHA_DONORS = [ru, mn, evn]`` listesi WOLD'un verici dağılımından
  seçildi (bkz. ``borrowing_eval``). Yakınlık sinyali yalnız bu üç dili
  arar: **küçük bir önsel sızıntı** (aday kümesi WOLD'dan biliniyor).
* ``motor|alıntı_dedi`` kapısı eğitilmiş birleştiriciyi kullanır ve o
  birleştirici WOLD Sakha'nın ayar yarısında eğitildi; bu varyant için
  ayrıca yalnız değerlendirme yarısı (tek indeksler) raporlanır.
* Türkçe altın küme (TDK+Nişanyan) hiç kullanılmaz.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from engine.config import CLDF_DIR
from engine.evaluation.headline_eval import _head, wilson
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: WOLD Source_languoid -> sınıf.
WOLD_CLASS = {
    "Russian": "Rusça",
    "Mongolic": "Moğolca", "Mongolian": "Moğolca", "Khalkha": "Moğolca",
    "Buryat": "Moğolca", "Kalmyk": "Moğolca",
    "Evenki": "Tunguzca",
    "Turkic": "Türkçe", "Oghur": "Türkçe", "Tatar": "Türkçe",
}

#: Motorun verici kodu -> sınıf.
_ENGINE_CLASS = {
    "ru": "Rusça", "orv": "Rusça", "zle-ort": "Rusça", "zle-ono": "Rusça", "sla": "Rusça",
    "mn": "Moğolca", "xal": "Moğolca", "bua": "Moğolca", "khk": "Moğolca", "cmg": "Moğolca",
    "xng": "Moğolca", "mvf": "Moğolca", "xgn-pro": "Moğolca", "xgn": "Moğolca",
    "evn": "Tunguzca", "eve": "Tunguzca", "tuw": "Tunguzca", "tuw-pro": "Tunguzca",
}
_TURKIC_PREFIXES = ("trk", "otk", "tt", "ky", "kk", "tr", "ota", "chg", "cv", "xqa", "tyv", "alt", "kjh")

NO_PREDICTION = "—"


def wold_class(languoid: str) -> str:
    return WOLD_CLASS.get(languoid.strip(), "diğer")


def engine_class(code: str) -> str:
    code = (code or "").strip()
    if not code:
        return NO_PREDICTION
    if code in _ENGINE_CLASS:
        return _ENGINE_CLASS[code]
    if code.startswith("xgn") or code.startswith("mn"):
        return "Moğolca"
    if code.startswith("zle-") or code.startswith("zlw-"):
        return "Rusça" if code.startswith("zle-") else "diğer"
    if code.startswith("tuw"):
        return "Tunguzca"
    if code.split("-")[0] in _TURKIC_PREFIXES:
        return "Türkçe"
    return "diğer"


@dataclass(frozen=True)
class DonorCase:
    word: str
    sense: str
    gold: str
    languoid: str
    index: int


def load_cases(language: str = "Sakha") -> list[DonorCase]:
    directory = CLDF_DIR / "wold"
    forms_path, borrow_path = directory / "forms.csv", directory / "borrowings.csv"
    if not (forms_path.exists() and borrow_path.exists()):
        logger.info("WOLD indirilmemiş: python scripts/download_cldf.py wold")
        return []
    senses: dict[str, str] = {}
    with (directory / "parameters.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            senses[row["ID"]] = (row.get("Concepticon_Gloss") or row.get("Name") or "").strip()
    donors: dict[str, str] = {}
    with borrow_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Source_relation") == "immediate":
                donors.setdefault(row["Target_Form_ID"], row.get("Source_languoid") or "")
    borrowed = {"1. clearly borrowed", "2. probably borrowed"}
    cases: list[DonorCase] = []
    with forms_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Language_ID") != language:
                continue
            if (row.get("Borrowed") or "").strip().lower() not in borrowed:
                continue
            languoid = donors.get(row["ID"], "")
            word = (row.get("Form") or row.get("Value") or "").strip()
            if not word or not languoid or languoid == "Unidentified":
                continue
            cases.append(DonorCase(word, senses.get(row.get("Parameter_ID") or "", ""),
                                   wold_class(languoid), languoid, len(cases)))
    return cases


def predict(case: DonorCase, detector: Any) -> dict[str, Any]:
    from engine.evaluation.borrowing_eval import donors_for

    verdict = detector.detect(case.word, [], lang="sah", sense=case.sense, donors=donors_for("sah"))
    proximity_code = ""
    for signal in verdict.signals:
        if signal.name == "verici_yakınlığı" and signal.fired:
            proximity_code = str((signal.evidence or {}).get("donor_lang") or "")
    chain = engine_class(verdict.donor_language)
    proximity = engine_class(proximity_code)
    combined = chain if chain != NO_PREDICTION else proximity
    return {
        "zincir": chain,
        "yakınlık": proximity,
        "motor": combined,
        "motor|alıntı_dedi": combined if verdict.is_borrowed else NO_PREDICTION,
        "raw_chain": verdict.donor_language,
        "raw_proximity": proximity_code,
        "is_borrowed": verdict.is_borrowed,
    }


def score(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, predicted, strict=True))
    covered = [(g, p) for g, p in zip(gold, predicted, strict=True) if p != NO_PREDICTION]
    covered_correct = sum(g == p for g, p in covered)
    confusion: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, predicted, strict=True):
        confusion.setdefault(g, {}).setdefault(p, 0)
        confusion[g][p] += 1
    labels = sorted(set(gold))
    per_class_recall = {
        label: round(confusion[label].get(label, 0) / sum(confusion[label].values()), 4) for label in labels
    }
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "accuracy_ci95": wilson(correct, n),
        "coverage": round(len(covered) / n, 4) if n else 0.0,
        "accuracy_when_predicted": round(covered_correct / len(covered), 4) if covered else 0.0,
        "accuracy_when_predicted_ci95": wilson(covered_correct, len(covered)),
        "macro_recall": round(sum(per_class_recall.values()) / len(labels), 4) if labels else 0.0,
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


SYSTEMS = ("zincir", "yakınlık", "motor", "motor|alıntı_dedi")


def run() -> dict[str, Any]:
    from engine.nlp.borrowing_detector import BorrowingDetector

    cases = load_cases()
    if not cases:
        return {"error": "WOLD yok"}
    detector = BorrowingDetector()
    predictions = [predict(c, detector) for c in cases]
    gold = [c.gold for c in cases]
    majority = Counter(gold).most_common(1)[0][0]
    results = {name: score(gold, [p[name] for p in predictions]) for name in SYSTEMS}
    results["çoğunluk"] = score(gold, [majority] * len(gold))
    eval_half = [i for i, c in enumerate(cases) if c.index % 2 == 1]
    results["motor|alıntı_dedi (değerlendirme yarısı)"] = score(
        [gold[i] for i in eval_half], [predictions[i]["motor|alıntı_dedi"] for i in eval_half])
    from engine.evaluation.significance import mcnemar_test

    majority_hits = [g == majority for g in gold]
    significance = {
        name: mcnemar_test([p[name] == g for p, g in zip(predictions, gold, strict=True)],
                           majority_hits).as_dict()
        for name in ("motor", "yakınlık")
    }
    return {
        "_schema": "donor_id_eval/v1",
        "commit": _head(),
        "reference": "WOLD Sakha, borrowings.csv Source_relation=immediate, Borrowed ∈ {1,2}, Unidentified hariç",
        "n": len(cases),
        "gold_distribution": dict(Counter(gold).most_common()),
        "majority_class": majority,
        "raw_engine_codes": {
            "zincir": dict(Counter(p["raw_chain"] for p in predictions).most_common()),
            "yakınlık": dict(Counter(p["raw_proximity"] for p in predictions).most_common()),
        },
        "systems": results,
        "vs_majority_mcnemar": significance,
        "circularity": (
            "Motorun verici tahmini WOLD'dan öğrenilmez (Wiktionary indeksinden). Önsel sızıntı: "
            "yakınlık yalnız WOLD'dan seçilmiş ru/mn/evn sözlüklerini arar. 'alıntı_dedi' kapısının "
            "birleştiricisi WOLD Sakha ayar yarısında eğitildi. Türkçe altın küme kullanılmaz."
        ),
        "items": [
            {"word": c.word, "sense": c.sense, "gold": c.gold, "wold_languoid": c.languoid, **p}
            for c, p in zip(cases, predictions, strict=True)
        ],
    }


def main() -> int:
    from engine.evaluation.report import EVAL_DIR

    payload = run()
    if "error" in payload:
        print(payload["error"])
        return 1
    print(f"\n=== verici dil tanıma · WOLD Sakha · n={payload['n']} · commit {payload['commit']} ===")
    print(f"altın dağılım: {payload['gold_distribution']}")
    for name, s in payload["systems"].items():
        print(
            f"{name:42} doğruluk {s['accuracy']:.3f} {s['accuracy_ci95']}  kapsam {s['coverage']:.3f}  "
            f"kapsananda {s['accuracy_when_predicted']:.3f} {s['accuracy_when_predicted_ci95']}  "
            f"makro-duyarlılık {s['macro_recall']:.3f}"
        )
    for name, t in payload["vs_majority_mcnemar"].items():
        print(f"McNemar {name} vs çoğunluk: {t}")
    print("\nkarışıklık (motor):")
    for g, row in payload["systems"]["motor"]["confusion"].items():
        print(f"  {g:10} -> {row}")
    print(f"\n⚠️ {payload['circularity']}")
    out = EVAL_DIR / "donor_id.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
