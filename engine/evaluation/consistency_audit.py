"""
Arama çıktısı tutarlılık denetimi — ``make audit``.

Türkçe altın kümeden sabit tohumla seçilen kelimeler (varsayılan 30 alıntı +
30 miras) motorun TAM arama hattından geçirilir ve iki şey ölçülür:

1. **İç tutarlılık:** çıktının bölümleri birbirini yalanlıyor mu? (başlık ile
   sıralayıcı, A-HVP ile sıralayıcı, sınıflandırıcı ile sıralayıcı, rozet ile
   köken satırı, ham şablon artığı, tanık listesinde Türki olmayan kod…)
   Her biri bir kez elle bulunmuş gerçek bir hatadır (boncuk, faça, parça,
   gülüş).
2. **Sıralayıcı hükmü × altın etiket.**

⚠️ **2. madde bağımsız bir doğruluk ölçüsü DEĞİLDİR.** Arama hattı
Nişanyan'ı ve EtimolojiTürkçe'yi (Nişanyan türevi) okur; altın küme TDK +
Nişanyan'dan kuruldu. Uyum bu yüzden kısmen kaynak tekrarıdır. Bu araç
regresyon yakalamak içindir; motorun doğruluğu `make eval-borrowing`
(zincir sinyali KAPALI) ile ölçülür.

Canlı sözlük cevapları kalıcı önbellekten gelir (``data/cache/http.db``);
ilk koşudan sonra ağa çıkmaz ve tekrarlanabilir.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SEED = 21
PER_CLASS = 30
_ARTIFACT = re.compile(r"[=|{}]|\(der|^\s*-\s*$")


def sample_words(per_class: int = PER_CLASS, seed: int = SEED) -> dict[str, str]:
    from engine.evaluation.borrowing_eval import TURKISH_GOLD_PATH

    items = json.loads(TURKISH_GOLD_PATH.read_text(encoding="utf-8"))["items"]
    loans = sorted({r["word"] for r in items if r["label"] == "alıntı"})
    native = sorted({r["word"] for r in items if r["label"] != "alıntı"})
    rng = random.Random(seed)
    rng.shuffle(loans)
    rng.shuffle(native)
    return {**dict.fromkeys(loans[:per_class], "alıntı"), **dict.fromkeys(native[:per_class], "miras")}


def inspect(word: str, finding: dict[str, Any]) -> tuple[str, list[str]]:
    """``(sıralayıcı hükmü, tutarsızlık adları)``."""
    nlp = finding.get("nlp_analysis") or {}
    root = finding.get("root") or {}
    head = str(root.get("proto_turkic") or "")
    provenance = str(root.get("provenance") or "")
    kind = ((nlp.get("ranked_hypotheses") or {}).get("selected") or {}).get("kind", "")
    hypothesis = nlp.get("proven_hypothesis") or {}
    report = hypothesis.get("validation_report") or {}
    family = (nlp.get("loanword_classification") or {}).get("classification_key", "")
    hypothesis_type = str(hypothesis.get("hypothesis_type") or "")

    issues: list[str] = []
    if kind == "borrowed" and head.startswith("*"):
        issues.append("başlık yıldızlı kök, sıralayıcı alıntı")
    if kind == "inherited" and not head.startswith("*"):
        issues.append("başlık alıntı biçimi, sıralayıcı miras")
    if "alıntı" in hypothesis_type.lower() and kind == "inherited":
        issues.append("A-HVP alıntı, sıralayıcı miras")
    if "Proto-Türkçe" in hypothesis_type and kind == "borrowed":
        issues.append("A-HVP miras, sıralayıcı alıntı")
    if family == "native" and kind == "borrowed":
        issues.append("sınıflandırıcı öz Türkçe, sıralayıcı alıntı")
    if family not in ("native", "") and kind == "inherited":
        issues.append("sınıflandırıcı alıntı, sıralayıcı miras")
    if "doğrulanmış" in provenance and report.get("status_code") != "VALIDATED":
        issues.append("köken 'doğrulanmış' ama rozet değil")
    # Kök kelimenin kendisi olabilir (*kök, *gaga); ama ek taşıyan kelime
    # bütün olarak Proto-Türkçe kurulamaz (*kolaylaşmak: mastar eki kökte).
    from engine.utils.morphology import analyze_morphology

    if head.strip("*").lower() == word and analyze_morphology(word)[1]:
        issues.append("ekli kelime bütün olarak kök kurulmuş")
    for entry in finding.get("turkic_languages") or []:
        code = entry.get("lang_code")
        if _ARTIFACT.search(str(entry.get("word") or "")):
            issues.append("ham şablon artığı")
        if code not in TURKIC_LANGUAGES_MAP and code not in ("donor", "ai"):
            issues.append("Türki olmayan kod tanıkta")
    verdict = "alıntı" if kind == "borrowed" else ("miras" if kind in ("inherited", "modern_coinage") else "belirsiz")
    return verdict, sorted(set(issues))


def run(per_class: int = PER_CLASS) -> dict[str, Any]:
    from engine.search_engine import SearchEngine

    labels = sample_words(per_class)
    engine = SearchEngine()
    confusion: Counter = Counter()
    issues: dict[str, list[str]] = {}
    wrong: list[dict[str, str]] = []
    for word, label in labels.items():
        # DB önbelleği atlanır: eski bir bulgu denetimi yanıltmasın.
        finding = engine.search(word, save_to_db=False, use_qwen_agent=False, use_cache=False)
        verdict, found = inspect(word, finding)
        confusion[f"{label}->{verdict}"] += 1
        if verdict != label:
            wrong.append({"word": word, "gold": label, "verdict": verdict})
        for name in found:
            issues.setdefault(name, []).append(word)
    n = sum(confusion.values())
    agreement = sum(v for k, v in confusion.items() if k.split("->")[0] == k.split("->")[1]) / n if n else 0.0
    return {
        "n": n,
        "agreement_with_gold": round(agreement, 4),
        "confusion": dict(confusion),
        "disagreements": wrong,
        "issues": {k: {"count": len(v), "words": v} for k, v in sorted(issues.items(), key=lambda kv: -len(kv[1]))},
        "caveat": "Altın uyumu bağımsız doğruluk değildir: arama Nişanyan'ı okur, altın küme TDK+Nişanyan.",
    }


def main() -> int:
    from engine.evaluation.report import EVAL_DIR

    payload = run()
    print(f"\n=== tutarlılık denetimi · n={payload['n']} ===")
    print(f"sıralayıcı × altın uyumu: {payload['agreement_with_gold']:.3f}  (⚠️ bağımsız doğruluk değil)")
    print(f"karışıklık: {payload['confusion']}")
    for w in payload["disagreements"]:
        print(f"  ayrılık: {w['word']} altın={w['gold']} hüküm={w['verdict']}")
    if not payload["issues"]:
        print("iç tutarsızlık: yok")
    for name, row in payload["issues"].items():
        print(f"  [{row['count']:2}] {name}: {row['words'][:8]}")
    out = EVAL_DIR / "consistency_audit.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
