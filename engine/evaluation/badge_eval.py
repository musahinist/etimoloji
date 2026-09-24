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
ayrımında AUC (rozet sırası 🔴 < ⚪ < 🟡 < 🟢 ve ``final_confidence_score``).
"""

from __future__ import annotations

import json
import random
from typing import Any

from engine.evaluation.headline_eval import SEED, _head, run_engine, savelyev_dev_turkish, wilson
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Türkçe altından (train+dev) örneklem. Tamamı ~560 madde; 300 ile koşu
#: ~6 dk.
TURKISH_SAMPLE = 300
#: Rozet sırası (AUC için): yüksek = daha emin.
BADGE_ORDER = {"REJECTED": 0, "INSUFFICIENT_EVIDENCE": 1, "NEEDS_REVIEW": 2, "VALIDATED": 3}
BADGES = ("VALIDATED", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE", "REJECTED", None)
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``rows``: ``{"badge", "score", "correct"}``."""
    table: dict[str, Any] = {}
    for badge in BADGES:
        sub = [r for r in rows if r["badge"] == badge]
        k = sum(r["correct"] for r in sub)
        table[badge or "hipotez_yok"] = {
            "n": len(sub),
            "accuracy": round(k / len(sub), 4) if sub else None,
            "accuracy_ci95": wilson(k, len(sub)),
        }
    scored = [r for r in rows if r["badge"] is not None]
    labels = [r["correct"] for r in scored]
    green, yellow = table["VALIDATED"]["accuracy"], table["NEEDS_REVIEW"]["accuracy"]
    return {
        "n": len(rows),
        "n_with_hypothesis": len(scored),
        "accuracy_all": round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None,
        "by_badge": table,
        "auc_badge_order": auc([BADGE_ORDER[r["badge"]] for r in scored], labels),
        "auc_final_score": auc([float(r["score"] or 0.0) for r in scored], labels),
        "green_above_yellow": (green > yellow) if green is not None and yellow is not None else None,
    }


CIRCULARITY = {
    "turkce_altin/train+dev": (
        "Kısmen döngüsel: altın TDK+Nişanyan, motor ağ kapalıyken Wiktionary köken etiketini "
        "okuyor (çoğu maddede aynı bilgi); sıralayıcı ağırlıkları Türkçe altının tamamında eğitildi."
    ),
    "savelyev_dev": "Dev bölümü örüntü tablolarına girmez; Wiktionary ile ortak Clauson/EDAL kaynakçası olası.",
}


def run(*, sample: int = TURKISH_SAMPLE, fresh: bool = False) -> dict[str, Any]:
    from engine.evaluation.metrics import best_match

    tr_items = turkish_items(sample)
    sv_items = savelyev_dev_turkish()
    runs = run_engine([w for w, _ in tr_items] + [w for w, _ in sv_items], ablate_starling=True, fresh=fresh)

    tr_rows = []
    for word, is_loan in tr_items:
        r = runs[word]
        predicted = hypothesis_is_loan(r)
        tr_rows.append({"word": word, "gold_loan": is_loan, "badge": r.get("badge"),
                        "score": r.get("badge_score"), "hypothesis_donor": r.get("hypothesis_donor"),
                        "correct": predicted is not None and predicted == is_loan})
    sv_rows = []
    for word, candidates in sv_items:
        r = runs[word]
        form = str(r.get("hypothesis_form") or "")
        ok = False
        if form and hypothesis_is_loan(r) is False:
            shown = form if form.startswith("*") else "*" + form
            ok = best_match(shown, candidates)[2]
        sv_rows.append({"word": word, "badge": r.get("badge"), "score": r.get("badge_score"),
                        "hypothesis_form": form, "correct": ok})
    return {
        "_schema": "badge_eval/v1",
        "commit": _head(),
        "fetchers": "yalnız yerel, Starling KAPALI, ağ kapalı (headline_eval.build_engine)",
        "sample": {"turkce_altin_train+dev": len(tr_items), "savelyev_dev": len(sv_items), "seed": SEED},
        "correctness": {
            "turkce_altin/train+dev": "A-HVP hipotezinin yönü (miras/alıntı) altın etiketle aynı",
            "savelyev_dev": "hipotez miras ve sınanan ata biçim altın adaylardan biriyle kabul edilebilir",
        },
        "circularity": CIRCULARITY,
        "subsets": {"turkce_altin/train+dev": summarize(tr_rows), "savelyev_dev": summarize(sv_rows)},
        "items": {"turkce_altin/train+dev": tr_rows, "savelyev_dev": sv_rows},
    }


_ICON = {"VALIDATED": "🟢", "NEEDS_REVIEW": "🟡", "INSUFFICIENT_EVIDENCE": "⚪", "REJECTED": "🔴",
         "hipotez_yok": "∅"}


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=TURKISH_SAMPLE)
    parser.add_argument("--fresh", action="store_true", help="önbelleği yok say")
    parser.add_argument("--out", default=str(EVAL_DIR / "badge.json"))
    args = parser.parse_args()
    payload = run(sample=args.sample, fresh=args.fresh)
    print(f"\n=== A-HVP rozet kalibrasyonu · commit {payload['commit']} ===")
    for name, s in payload["subsets"].items():
        print(f"{name}  n={s['n']}  hipotezli={s['n_with_hypothesis']}  "
              f"AUC(rozet)={s['auc_badge_order']}  AUC(skor)={s['auc_final_score']}")
        for badge, cell in s["by_badge"].items():
            acc = "—" if cell["accuracy"] is None else f"{cell['accuracy']:.3f}"
            print(f"   {_ICON[badge]} {badge:22} n={cell['n']:3}  doğruluk {acc} {cell['accuracy_ci95']}")
        if s["green_above_yellow"] is False:
            print("   ⚠️ 🟢 kutusunun doğruluğu 🟡'dan YÜKSEK DEĞİL: rozet bu kümede güveni sıralamıyor.")
        print(f"   ⚠️ {payload['circularity'][name]}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
