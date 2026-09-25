"""
A-HVP rozet kalibrasyonu — ``make eval-badge``.

Kullanıcı her aramada bir rozet görür (🟢 DOĞRULANDI, 🟡 İNCELEME GEREKLİ,
⚪ YETERSİZ KANIT, 🔴 REDDEDİLDİ). ``make eval-calibration`` (Platt) motorun
REKONSTRÜKSİYON güvenini ölçer; rozet ayrı bir skordur ve hiç ölçülmemişti.
Soru: rozet sınıfı yükseldikçe A-HVP'nin sınadığı hipotez daha sık doğru mu?

İki altın küme, iki doğruluk tanımı:

* **Türkçe alıntı/miras altını** (``data/gold/turkish_loanwords.json``,
  TDK + Nişanyan). Kümenin kendi bölmesi yok; savelyev ile AYNI kural
  (``gold.assign_split``, tuzlu özet) kelimeye uygulanır ve yalnız
  train+dev okunur. Test bölümü OKUNMAZ. Doğru: A-HVP hipotezinin yönü
  (miras: Proto-Türkçe/modern Türkçe türetme; alıntı: verici dil ya da
  fonotaktik alıntı adayı) altın etiketle aynı.
* **savelyev dev** (Türkçe tanıklı maddeler). Doğru: hipotez miras ve
  A-HVP'nin sınadığı ata biçim altın adaylardan biriyle kabul edilebilir
  eşleşiyor (``metrics.best_match``).

Arama hattı ``headline_eval.build_engine(ablate_starling=True)``: yalnız yerel
kaynaklar, Starling KAPALI, ağ kapalı.

⚠️ Döngüsellik:

* Türkçe altın TDK + Nişanyan'dan kuruldu. Ağ kapalı olduğu için motor
  ikisini okumuyor, ama sözlük indeksinin Wiktionary köken etiketi
  (``source_loan_step``, zincir sinyali) çoğu maddede aynı bilgiyi taşır:
  **yön doğruluğu kısmen Wiktionary↔TDK/Nişanyan uyumudur**, rozetin
  bağımsız doğruluğu değil. Rozetin sınıflar arası AYIRICILIĞI yine
  anlamlıdır (aynı kaynak her sınıfa girer).
* Sıralayıcının ağırlıkları (``borrowing_combiner``) Türkçe altının tamamında
  eğitildi; sıralayıcı hükmü A-HVP'nin tanık kümesini (paralel alıntılar)
  etkiliyor. train+dev içinde bu sızıntı var, test'te de olurdu.
* savelyev dev motorun örüntü tablolarına girmez (yalnız TRAIN); Wiktionary
  Proto-Türkçe biçimleriyle ortak Clauson/EDAL kaynakçası olasıdır.

Ölçüler: rozet sınıfı başına n, doğruluk ve Wilson %95 GA; doğru/yanlış
ayrımında AUC.

v2 — gösterilen rozet ``engine.nlp.verdict_badge`` (🟢 TUTARLI / 🔴 ŞÜPHELİ /
∅ DEĞERLENDİRİLMEDİ). Kuralı Türkçe altının TRAIN bölümünde seçildi; bu yüzden
Türkçe altın bölüm bölüm raporlanır: ``train`` = ayar (örneklem İÇİ, iyimser),
``dev`` = rapor. Eski A-HVP aşama kararı (``status_code``) karşılaştırma için
``stage_verdict`` altında kalır. Hipotezsiz kelime "değerlendirilmedi"dir ve
AUC'ye girmez. Sonuç (ön-kayıt ``data/cache/work/badge/PREREG.md``): ayar CV
AUC 0,915; dev AUC 0,984 ama dev'de hipotezli 62 kelimenin yalnız 1'i yanlış;
🟢/🔴 farkı anlamlılık şartını geçmedi (Fisher p ≈ 0,06).
"""

from __future__ import annotations

import json
import random
from typing import Any

from engine.evaluation.headline_eval import SEED, _head, run_engine, savelyev_dev_turkish, wilson
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Türkçe altın: train+dev'in tamamı (~573 madde; koşu ~15 dk).
TURKISH_SAMPLE = 10_000
#: Eski A-HVP aşama kararı sırası (AUC için): yüksek = daha emin.
BADGE_ORDER = {"REJECTED": 0, "INSUFFICIENT_EVIDENCE": 1, "NEEDS_REVIEW": 2, "VALIDATED": 3}
BADGES = ("VALIDATED", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE", "REJECTED", None)
VERDICTS = ("CONSISTENT", "SUSPECT", "NOT_EVALUATED")
_INHERITED_DONORS = ("Proto-Türkçe", "Türkçe (modern türetme)", "Ana Türkçe", "Öz Türkçe", "Eski Türkçe")


def turkish_items(n: int = TURKISH_SAMPLE, seed: int = SEED) -> list[tuple[str, bool]]:
    """``(kelime, altın alıntı mı)`` — YALNIZ train+dev, sabit tohumlu örneklem."""
    from engine.evaluation.borrowing_eval import TURKISH_GOLD_PATH
    from engine.evaluation.gold import assign_split

    data = json.loads(TURKISH_GOLD_PATH.read_text(encoding="utf-8"))
    pool: dict[str, bool] = {}
    for row in data.get("items") or []:
        word = str(row["word"]).strip().lower()
        if " " in word or not word or word in pool:
            continue
        if assign_split(f"tr-gold:{word}") == "test":
            continue
        pool[word] = row["label"] == "alıntı"
    words = sorted(pool)
    random.Random(seed).shuffle(words)
    return sorted((w, pool[w]) for w in words[:n])


def hypothesis_is_loan(run: dict[str, Any]) -> bool | None:
    donor = run.get("hypothesis_donor")
    if donor is None:
        return None
    return donor not in _INHERITED_DONORS


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUC (eşitlikler yarım sayılır)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def _table(rows: list[dict[str, Any]], key: str, classes: tuple, none_name: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for cls in classes:
        sub = [r for r in rows if r.get(key) == cls]
        k = sum(r["correct"] for r in sub)
        table[cls or none_name] = {"n": len(sub), "accuracy": round(k / len(sub), 4) if sub else None,
                                   "accuracy_ci95": wilson(k, len(sub))}
    return table


def fisher_p(a: int, n1: int, b: int, n2: int) -> float | None:
    """İki-yanlı Fisher kesin testi: ``a/n1`` ile ``b/n2`` doğruluk oranı."""
    from math import comb

    if not n1 or not n2:
        return None
    k, n = a + b, n1 + n2
    prob = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)  # noqa: E731
    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return round(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9))), 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``rows``: ``{"badge", "score", "verdict", "verdict_score", "correct"}``."""
    scored = [r for r in rows if r["badge"] is not None]
    labels = [r["correct"] for r in scored]
    stage = _table(rows, "badge", BADGES, "hipotez_yok")
    green, yellow = stage["VALIDATED"]["accuracy"], stage["NEEDS_REVIEW"]["accuracy"]
    verdict = _table(rows, "verdict", VERDICTS, "?")
    judged = [r for r in rows if r.get("verdict") in ("CONSISTENT", "SUSPECT")]
    ok = [r["correct"] for r in judged if r["verdict"] == "CONSISTENT"]
    bad = [r["correct"] for r in judged if r["verdict"] == "SUSPECT"]
    return {
        "n": len(rows),
        "n_with_hypothesis": len(scored),
        "accuracy_all": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None,
        "verdict_badge": {
            "by_badge": verdict,
            "auc_score": auc([float(r["verdict_score"] or 0.0) for r in judged], [r["correct"] for r in judged]),
            "consistent_above_suspect": (verdict["CONSISTENT"]["accuracy"] > verdict["SUSPECT"]["accuracy"])
            if ok and bad else None,
            "fisher_p": fisher_p(sum(ok), len(ok), sum(bad), len(bad)),
        },
        "stage_verdict": {
            "by_badge": stage,
            "auc_badge_order": auc([BADGE_ORDER[r["badge"]] for r in scored], labels),
            "auc_final_score": auc([float(r["score"] or 0.0) for r in scored], labels),
            "green_above_yellow": (green > yellow) if green is not None and yellow is not None else None,
        },
    }


CIRCULARITY = {
    "turkce_altin": (
        "Kısmen döngüsel: altın TDK+Nişanyan, motor ağ kapalıyken Wiktionary köken etiketini "
        "okuyor (çoğu maddede aynı bilgi); sıralayıcı ağırlıkları Türkçe altının tamamında eğitildi; "
        "rozet kuralı train bölümünde seçildi (train satırı örneklem içi)."
    ),
    "savelyev_dev": "Dev bölümü örüntü tablolarına girmez; Wiktionary ile ortak Clauson/EDAL kaynakçası olası.",
}


def run(*, sample: int = TURKISH_SAMPLE, fresh: bool = False) -> dict[str, Any]:
    from engine.evaluation.gold import assign_split
    from engine.evaluation.metrics import best_match

    tr_items = turkish_items(sample)
    sv_items = savelyev_dev_turkish()
    runs = run_engine([w for w, _ in tr_items] + [w for w, _ in sv_items], ablate_starling=True, fresh=fresh)

    def row(word: str, r: dict[str, Any], correct: bool, **extra: Any) -> dict[str, Any]:
        return {"word": word, "badge": r.get("badge"), "score": r.get("badge_score"),
                "verdict": r.get("verdict_badge"), "verdict_score": r.get("verdict_badge_score"),
                "correct": correct, **extra}

    tr_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for word, is_loan in tr_items:
        r = runs[word]
        predicted = hypothesis_is_loan(r)
        tr_rows[assign_split(f"tr-gold:{word}")].append(
            row(word, r, predicted is not None and predicted == is_loan,
                gold_loan=is_loan, hypothesis_donor=r.get("hypothesis_donor")))
    sv_rows = []
    for word, candidates in sv_items:
        r = runs[word]
        form = str(r.get("hypothesis_form") or "")
        ok = False
        if form and hypothesis_is_loan(r) is False:
            shown = form if form.startswith("*") else "*" + form
            ok = best_match(shown, candidates)[2]
        sv_rows.append(row(word, r, ok, hypothesis_form=form))
    subsets = {"turkce_altin/train (ayar, örneklem içi)": tr_rows["train"],
               "turkce_altin/dev (rapor)": tr_rows["dev"], "savelyev_dev": sv_rows}
    return {
        "_schema": "badge_eval/v2",
        "commit": _head(),
        "fetchers": "yalnız yerel, Starling KAPALI, ağ kapalı (headline_eval.build_engine)",
        "sample": {name: len(rows) for name, rows in subsets.items()} | {"seed": SEED},
        "correctness": {
            "turkce_altin": "A-HVP hipotezinin yönü (miras/alıntı) altın etiketle aynı",
            "savelyev_dev": "hipotez miras ve sınanan ata biçim altın adaylardan biriyle kabul edilebilir",
        },
        "circularity": CIRCULARITY,
        "subsets": {name: summarize(rows) for name, rows in subsets.items()},
        "items": subsets,
    }


_ICON = {"VALIDATED": "🟢", "NEEDS_REVIEW": "🟡", "INSUFFICIENT_EVIDENCE": "⚪", "REJECTED": "🔴",
         "hipotez_yok": "∅", "CONSISTENT": "🟢", "SUSPECT": "🔴", "NOT_EVALUATED": "∅", "?": "?"}


def _print_table(table: dict[str, Any]) -> None:
    for badge, cell in table.items():
        acc = "—" if cell["accuracy"] is None else f"{cell['accuracy']:.3f}"
        print(f"   {_ICON[badge]} {badge:22} n={cell['n']:3}  doğruluk {acc} {cell['accuracy_ci95']}")


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=TURKISH_SAMPLE)
    parser.add_argument("--fresh", action="store_true", help="önbelleği yok say")
    parser.add_argument("--out", default=str(EVAL_DIR / "badge.json"))
    args = parser.parse_args()
    payload = run(sample=args.sample, fresh=args.fresh)
    print(f"\n=== Hüküm rozeti kalibrasyonu · commit {payload['commit']} ===")
    for name, s in payload["subsets"].items():
        v, st = s["verdict_badge"], s["stage_verdict"]
        print(f"{name}  n={s['n']}  hipotezli={s['n_with_hypothesis']}  "
              f"AUC(rozet skoru)={v['auc_score']}  Fisher p(🟢 vs 🔴)={v['fisher_p']}")
        _print_table(v["by_badge"])
        if v["consistent_above_suspect"] is False:
            print("   ⚠️ 🟢 TUTARLI kutusu 🔴 ŞÜPHELİ'den doğru DEĞİL: rozet bu kümede ayırmıyor.")
        print(f"   eski A-HVP aşama kararı: AUC(sıra)={st['auc_badge_order']}  AUC(skor)={st['auc_final_score']}")
        circ = payload["circularity"]["savelyev_dev" if name.startswith("savelyev") else "turkce_altin"]
        print(f"   ⚠️ {circ}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
