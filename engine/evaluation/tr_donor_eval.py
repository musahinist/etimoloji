"""
Türkçe verici dil tanıma — ``make eval-tr-donor`` (plan C4, GAPS G1).

``make eval-donor`` yalnız WOLD Sakha'yı ölçer; Türkçede motorun "ALINTI —
Arapça" derken vericiyi doğru bulup bulmadığı hiç ölçülmemişti.

Altın
-----
``data/gold/turkish_loanwords.json`` (TDK + Nişanyan): ``label == "alıntı"``
olup ``tdk_source`` ile ``nisanyan_source``'un **ilk dil adı aynı** olan
maddeler (eşanlamlılar: Rumca = Yunanca, Venedikçe/Cenevizce = İtalyanca).
Sınıflar: Arapça, Farsça, Fransızca, İtalyanca, Yunanca, diğer.

⚠️ Test bölümü OKUNMAZ: ``gold.assign_split(f"tr-gold:{kelime}")`` (``badge_eval``
ile aynı anahtar) ``test`` diyen maddeler yüklemede atılır; yalnız train+dev.

Sistemler
---------
``(a) etiket``
    Yalnız etiket adımı: ``donor_proximity.attribute_donor`` her maddeye,
    kapısız (yakınlık sinyali ateşlenmese de). Sözlük indeksine bakmaz.
``(b) arama``
    Tam motor, zincir açık: ``SearchEngine.search`` (yalnız yerel kaynaklar,
    ``headline_eval.build_engine``) sıralayıcısının seçtiği hüküm
    ``ALINTI — <dil>`` ise o dil. Köken zinciri sözlük kaydından (Wiktionary).
``(b) zincir`` / ``(b) dedektör``
    ``BorrowingDetector`` tam indeksle: zincirin verici kodu / önce zincir,
    yoksa (ateşlendiyse) yakınlık etiketi.
``(c) kör``
    Kör indeks (``ETY_LEXICON_INDEX=data/cache/work/xtr/index_blind.db``,
    köken sütunları boş) + zincir kapalı: dedektörün vericisi, yoksa yalnız
    yakınlık sinyali ATEŞLENDİYSE etiketi. Bağımsız çıkarım budur.
``çoğunluk``
    Hep en sık altın sınıf.

⚠️ **Adı "Wiktionary↔TDK uyumu"dur, verici doğruluğu değil.** (b) döngüseldir:
zincir Wiktionary ``donor_lang`` kaydını okur; Wiktionary Türkçe etimolojileri
çoğunlukla Nişanyan/TDK'ya dayanır. (c) ve (a) kaydı görmez, ama verici
havuzu (``TURKISH_DONORS``) Türkçe alıntı dağılımına bakılarak seçilmişti.

A2 bayrakları (``ETY_DONOR_CLEAN`` + ``ETY_DONOR_RAMP_CHANCE``) ortamdan okunur;
``capture`` her durum için ayrı önbellek yazar, ``report`` ikisini de raporlar.

Kullanım::

    python -m engine.evaluation.tr_donor_eval capture --index full --a2 off
    ETY_LEXICON_INDEX=data/cache/work/xtr/index_blind.db \\
        python -m engine.evaluation.tr_donor_eval capture --index blind --a2 off
    python -m engine.evaluation.tr_donor_eval report
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.evaluation.donor_id_eval import NO_PREDICTION, score
from engine.logging_setup import get_logger

logger = get_logger(__name__)

GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "turkish_loanwords.json"
WORK_DIR = PROJECT_ROOT / "data" / "cache" / "work" / "trdonor"
BLIND_INDEX = PROJECT_ROOT / "data" / "cache" / "work" / "xtr" / "index_blind.db"
OUT_NAME = "tr_donor.json"

CLASSES = ("Arapça", "Farsça", "Fransızca", "İtalyanca", "Yunanca", "diğer")

#: Altın dil adı eşanlamlıları (TDK "Rumca", Nişanyan "Yunanca" aynı vericidir).
_NAME_SYNONYMS = {"Rumca": "Yunanca", "Venedikçe": "İtalyanca", "Cenevizce": "İtalyanca"}

#: Motor kodu -> sınıf (önek eşleşmesi ``engine_class``'ta).
_CODE_CLASS = {
    "ar": "Arapça", "arb": "Arapça", "acm": "Arapça", "apc": "Arapça", "ajp": "Arapça", "arz": "Arapça",
    "fa": "Farsça", "fa-cls": "Farsça", "pal": "Farsça", "peo": "Farsça", "xpr": "Farsça", "ira-mid": "Farsça",
    "fr": "Fransızca", "frm": "Fransızca", "fro": "Fransızca",
    "it": "İtalyanca", "vec": "İtalyanca", "lij": "İtalyanca",
    "el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "el-kal": "Yunanca",
}

A2_ENV = {"off": {"ETY_DONOR_CLEAN": "0", "ETY_DONOR_RAMP_CHANCE": "0"},
          "on": {"ETY_DONOR_CLEAN": "1", "ETY_DONOR_RAMP_CHANCE": "1"}}


def gold_name(source: str) -> str:
    """Kaynak alanının ilk dil adı ("Arapça ʿasker" -> "Arapça")."""
    first = (source or "").strip().split(" ")[0].strip(",;:.")
    return _NAME_SYNONYMS.get(first, first)


def name_class(name: str) -> str:
    """Dil ADI -> sınıf ("Orta Farsça" -> Farsça, "Eski Yunanca" -> Yunanca)."""
    name = (name or "").strip()
    if not name:
        return NO_PREDICTION
    name = _NAME_SYNONYMS.get(name, name)
    for cls in CLASSES[:-1]:
        if cls in name:
            return cls
    if "Rumca" in name or "Venedikçe" in name or "Cenevizce" in name:
        return _NAME_SYNONYMS[next(k for k in _NAME_SYNONYMS if k in name)]
    return "diğer"


def engine_class(value: str) -> str:
    """Motorun verici KODU ya da ADI -> sınıf."""
    value = (value or "").strip()
    if not value:
        return NO_PREDICTION
    if value in _CODE_CLASS:
        return _CODE_CLASS[value]
    head = value.split("-")[0]
    if head in _CODE_CLASS and value.isascii():
        return _CODE_CLASS[head]
    if value.isascii() and value.replace("-", "").isalpha() and value.islower() and len(head) <= 3:
        return "diğer"  # bilinmeyen dil kodu
    return name_class(value)


def search_donor(summary: dict[str, Any]) -> str:
    """Sıralayıcının seçtiği hükümden verici adı ("ALINTI — Arapça")."""
    if summary.get("kind") != "borrowed":
        return ""
    claim = str(summary.get("claim") or "")
    return claim.split("—", 1)[1].strip() if "—" in claim else ""


@dataclass(frozen=True)
class TrDonorCase:
    word: str
    gold: str
    gold_name: str
    tdk_source: str
    nisanyan_source: str
    split: str


def load_cases(path: Path = GOLD_PATH) -> list[TrDonorCase]:
    """İki kaynağın aynı vericiyi verdiği alıntılar, train+dev (test atılır)."""
    from engine.evaluation.gold import assign_split

    if not path.exists():
        logger.info("Türkçe altın yok: python scripts/build_turkish_loanword_gold.py")
        return []
    items = json.loads(path.read_text(encoding="utf-8")).get("items") or []
    cases: list[TrDonorCase] = []
    for row in items:
        split = assign_split(f"tr-gold:{row['word']}")
        if split == "test":
            continue  # mühürlü — içeriğine hiç bakılmaz
        if row.get("label") != "alıntı":
            continue
        tdk, nis = gold_name(row.get("tdk_source", "")), gold_name(row.get("nisanyan_source", ""))
        if not tdk or not nis or tdk != nis:
            continue
        cases.append(TrDonorCase(row["word"], name_class(tdk), tdk, row.get("tdk_source", ""),
                                 row.get("nisanyan_source", ""), split))
    return cases


# --- yakalama -------------------------------------------------------------------


def cache_path(index: str, a2: str) -> Path:
    return WORK_DIR / f"{index}_a2{a2}.jsonl"


def _check_env(index: str, a2: str) -> None:
    for key, value in A2_ENV[a2].items():
        if os.environ.get(key) != value:
            raise SystemExit(f"--a2 {a2} için {key}={value} ortamda olmalı")
    blind = Path(os.environ.get("ETY_LEXICON_INDEX") or "").resolve() == BLIND_INDEX.resolve()
    if (index == "blind") != blind:
        raise SystemExit("--index blind yalnız ETY_LEXICON_INDEX=<kör indeks> ile; full onsuz")


def capture(index: str, a2: str, *, limit: int = 0) -> Path:
    """Madde başına verici tahminlerini önbelleğe yazar (kaldığı yerden sürer)."""
    _check_env(index, a2)
    from engine.evaluation.borrowing_eval import _turkish_glosses
    from engine.nlp.borrowing_detector import TURKISH_DONORS, BorrowingDetector
    from engine.nlp.donor_proximity import attribute_donor
    from engine.utils.orthography import to_comparison_form

    cases = load_cases()[: limit or None]
    out = cache_path(index, a2)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["word"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    glosses = _turkish_glosses([c.word for c in cases])
    detector = BorrowingDetector()
    engine = None
    if index == "full":
        from engine.evaluation.headline_eval import build_engine

        engine = build_engine(ablate_starling=False)
    with out.open("a", encoding="utf-8") as handle:
        for n, case in enumerate(cases):
            if case.word in done:
                continue
            sense = glosses.get(case.word, "")
            query = case.word
            verdict = detector.detect(query, [], lang="tr", sense=sense, donors=TURKISH_DONORS)
            proximity = ""
            for signal in verdict.signals:
                if signal.name == "verici_yakınlığı" and signal.fired:
                    evidence = signal.evidence or {}
                    proximity = str(evidence.get("attributed_lang") or evidence.get("donor_lang") or "")
            label = attribute_donor(to_comparison_form(query), sense, languages=TURKISH_DONORS)
            row: dict[str, Any] = {
                "word": case.word, "sense": bool(sense),
                "chain": verdict.donor_language, "proximity": proximity,
                "label": label.lang_code if label is not None else "",
                "is_borrowed": verdict.is_borrowed,
            }
            if engine is not None:
                try:
                    finding = engine.search(query, save_to_db=False, use_qwen_agent=False, use_cache=False)
                    nlp = finding.get("nlp_analysis") or {}
                    ranked = nlp.get("ranked_hypotheses") or finding.get("ranked_hypotheses") or {}
                    selected = ranked.get("selected") or {}
                    row["search"] = search_donor(selected)
                    row["search_kind"] = selected.get("kind")
                except Exception as exc:  # tek kelime hattı durdurmasın
                    logger.warning("Arama başarısız: %s", case.word, exc_info=True)
                    row["search"], row["search_error"] = "", f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 50 == 0:
                print(f"{index}/a2{a2}: {n}/{len(cases)}", flush=True)
    return out


# --- rapor ----------------------------------------------------------------------


def predictions(row: dict[str, Any], index: str) -> dict[str, str]:
    """Önbellek satırından sistem tahminleri (sınıf)."""
    chain = engine_class(row.get("chain", ""))
    proximity = engine_class(row.get("proximity", ""))
    out = {"(a) etiket": engine_class(row.get("label", ""))}
    if index == "full":
        out["(b) arama"] = engine_class(row.get("search", ""))
        out["(b) zincir"] = chain
        out["(b) dedektör"] = chain if chain != NO_PREDICTION else proximity
    else:
        # Kör indeks zincir kaydı taşımaz; yine de gelirse sayılmaz (zincir kapalı).
        out["(c) kör"] = proximity
    return out


def top_errors(gold: list[str], pred: list[str], k: int = 5) -> list[dict[str, Any]]:
    pairs = Counter((g, p) for g, p in zip(gold, pred, strict=True) if g != p)
    return [{"gold": g, "pred": p, "n": n} for (g, p), n in pairs.most_common(k)]


def evaluate(cases: list[TrDonorCase], rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """``rows[index][kelime]`` -> sistem skorları (bir A2 durumu için)."""
    from engine.evaluation.significance import mcnemar_test

    gold = [c.gold for c in cases]
    majority = Counter(gold).most_common(1)[0][0]
    preds: dict[str, list[str]] = {}
    for index in ("full", "blind"):
        if not rows.get(index):
            continue
        per = [predictions(rows[index].get(c.word, {}), index) for c in cases]
        for name in per[0]:
            if name == "(a) etiket" and name in preds:
                continue  # sözlük indeksinden bağımsız; tam koşudaki alınır
            preds[name] = [p[name] for p in per]
    preds["çoğunluk"] = [majority] * len(gold)
    systems = {name: {**score(gold, p), "top_errors": top_errors(gold, p)} for name, p in preds.items()}
    majority_hits = [g == majority for g in gold]
    vs_majority = {
        name: mcnemar_test([p == g for p, g in zip(pr, gold, strict=True)], majority_hits).as_dict()
        for name, pr in preds.items() if name != "çoğunluk"
    }
    return {"systems": systems, "vs_majority_mcnemar": vs_majority, "_preds": preds}


def load_rows(index: str, a2: str) -> dict[str, dict[str, Any]]:
    path = cache_path(index, a2)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["word"]: r for r in rows}


CIRCULARITY = (
    "Bu ölçüm 'Wiktionary↔TDK uyumu'dur, verici doğruluğu değil. (b) DÖNGÜSEL: arama/zincir "
    "sözlük indeksindeki Wiktionary donor_lang kaydını okur ve Wiktionary Türkçe etimolojileri "
    "büyük ölçüde Nişanyan/TDK'ya dayanır — (b) iki sözlükçü geleneğin uyumunu ölçer. (c) kör "
    "indeks + zincir kapalı ve (a) yalnız etiket adımı kaydı görmez: bağımsız çıkarım. Önsel "
    "sızıntı: verici havuzu TURKISH_DONORS (ar, fa, el, hy, fr, it) Türkçe alıntı dağılımına "
    "bakılarak seçildi; 'diğer' sınıfını (a)/(c) yapısal olarak hemen hiç üretemez."
)


#: En sık hata sınıfının incelemesi (A2 kapalı, (a) etiketinin ilk 15
#: Arapça->Farsça hatası; düzeltme YAPILMADI).
ERROR_NOTE = (
    "Bağımsız sistemlerde (a)/(c) en sık hata Arapça->Farsça. İncelenen 15 maddenin 15'inde "
    "Farsça havuzdaki eşleşme Farsçadaki ARAPÇA ALINTININ kendisi (davet~دعوت, kısmet~قسمت, "
    "gazap~غضب, masum~معصوم, azamet~عظمت): Osmanlıca biçim Arapça sözcüğün Farsça/Osmanlı "
    "okunuşuna (-et, v/w) daha yakın düştüğü için Farsça kaydı Arapça kayıttan (ör. عظمة) daha "
    "kısa mesafede. Etiket adımı 'en yakın biçimin dili'ni seçer; vericinin vericisini (Farsçanın "
    "da Arapçadan aldığını) bilmez. İkinci sık hata Fransızca->Arapça/İtalyanca/Farsça (anlam "
    "kısıtlı havuzda Fransızca karşılığın uzak düşmesi)."
)


def report() -> dict[str, Any]:
    from engine.evaluation.headline_eval import _head
    from engine.evaluation.significance import mcnemar_test

    cases = load_cases()
    gold = [c.gold for c in cases]
    payload: dict[str, Any] = {
        "_schema": "tr_donor_eval/v1",
        "name": "Türkçe verici dil — Wiktionary↔TDK uyumu",
        "commit": _head(),
        "gold": "turkish_loanwords.json: alıntı, tdk_source ile nisanyan_source ilk dil adı aynı; "
                "train+dev (assign_split('tr-gold:<kelime>') test hariç)",
        "n": len(cases),
        "split_counts": dict(Counter(c.split for c in cases)),
        "gold_distribution": dict(Counter(gold).most_common()),
        "gold_other_names": dict(Counter(c.gold_name for c in cases if c.gold == "diğer").most_common()),
        "majority_class": Counter(gold).most_common(1)[0][0] if gold else "",
        "circularity_warning": CIRCULARITY,
        "a2_flags": A2_ENV,
        "dominant_error_note": ERROR_NOTE,
        "by_a2": {},
    }
    kept_preds: dict[str, dict[str, list[str]]] = {}
    for a2 in ("off", "on"):
        rows = {index: load_rows(index, a2) for index in ("full", "blind")}
        if not any(rows.values()):
            continue
        missing = {index: sum(c.word not in r for c in cases) for index, r in rows.items() if r}
        result = evaluate(cases, rows)
        kept_preds[a2] = result.pop("_preds")
        result["missing_rows"] = missing
        result["sense_coverage"] = round(
            sum(bool(rows["full" if rows["full"] else "blind"].get(c.word, {}).get("sense")) for c in cases)
            / max(len(cases), 1), 4)
        payload["by_a2"][a2] = result
    if "off" in kept_preds and "on" in kept_preds:
        payload["a2_on_vs_off_mcnemar"] = {
            name: mcnemar_test([p == g for p, g in zip(kept_preds["on"][name], gold, strict=True)],
                               [p == g for p, g in zip(kept_preds["off"][name], gold, strict=True)]).as_dict()
            for name in kept_preds["off"] if name in kept_preds["on"] and name != "çoğunluk"
        }
    reference = kept_preds.get("off") or next(iter(kept_preds.values()), {})
    payload["items"] = [
        {"word": c.word, "gold": c.gold, "gold_name": c.gold_name, "split": c.split,
         **{f"{a2}:{name}": kept_preds[a2][name][i] for a2 in kept_preds for name in kept_preds[a2]
            if name != "çoğunluk"}}
        for i, c in enumerate(cases)
    ] if reference else []
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--index", choices=("full", "blind"), required=True)
    cap.add_argument("--a2", choices=("off", "on"), required=True)
    cap.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "capture":
        print(capture(args.index, args.a2, limit=args.limit))
        return 0
    from engine.evaluation.report import EVAL_DIR

    payload = report()
    print(f"\n=== {payload['name']} · n={payload['n']} · commit {payload['commit']} ===")
    print(f"altın: {payload['gold_distribution']}  (çoğunluk {payload['majority_class']})")
    for a2, result in payload["by_a2"].items():
        print(f"\n--- A2 {a2} ---")
        for name, s in result["systems"].items():
            print(f"{name:16} doğruluk {s['accuracy']:.3f} {s['accuracy_ci95']}  kapsam {s['coverage']:.3f}  "
                  f"kapsananda {s['accuracy_when_predicted']:.3f}  makro-duyarlılık {s['macro_recall']:.3f}")
            print(f"{'':16} en sık hata: {s['top_errors'][:3]}")
    print(f"\n⚠️ {payload['circularity_warning']}")
    out = EVAL_DIR / OUT_NAME
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
