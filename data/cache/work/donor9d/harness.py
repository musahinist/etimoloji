"""9d hafif yol: yalnız etiket adımını (attribute_donor) kurallarla yeniden oynatır.

python data/cache/work/donor9d/harness.py tr --split train
python data/cache/work/donor9d/harness.py xt
"""
import json, os, sys, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("ETY_DONOR_CLEAN", "0")
os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
OUT = Path(__file__).parent
RULES = ("off", "d1", "d2", "d3")
import engine.nlp.donor_proximity as _dp  # noqa: E402  (varsayılan ne olursa olsun kural açıkça atanır)


def tr(split: str) -> None:
    from engine.evaluation.borrowing_eval import _turkish_glosses
    from engine.evaluation.tr_donor_eval import engine_class, load_cases, load_rows
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.utils.orthography import to_comparison_form

    cases = [c for c in load_cases() if c.split == split]
    glosses = _turkish_glosses([c.word for c in cases])
    cached = load_rows("full", "off")
    rows = []
    t = time.time()
    for n, c in enumerate(cases):
        sense = glosses.get(c.word, "")
        row = {"word": c.word, "gold": c.gold}
        for rule in RULES:
            dp.ARABIC_VIA_RULE = rule
            a = dp.attribute_donor(to_comparison_form(c.word), sense, languages=TURKISH_DONORS)
            row[rule] = engine_class(a.lang_code) if a else "—"
            row[rule + "_w"] = f"{a.word}/{a.comparison}/{a.via}" if a else ""
        row["cached"] = engine_class(cached.get(c.word, {}).get("label", ""))
        rows.append(row)
        if n % 50 == 0:
            print(n, len(cases), f"{time.time()-t:.0f}s", flush=True)
    (OUT / f"tr_{split}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    summarize(rows)


def summarize(rows):
    from engine.evaluation.significance import mcnemar_test

    n = len(rows)
    print("n", n, "cached==off", sum(r["cached"] == r["off"] for r in rows))
    base = [r["off"] == r["gold"] for r in rows]
    for rule in RULES:
        hits = [r[rule] == r["gold"] for r in rows]
        err = Counter((r["gold"], r[rule]) for r in rows if r[rule] != r["gold"]).most_common(6)
        mc = mcnemar_test(hits, base).as_dict() if rule != "off" else {}
        print(rule, f"{sum(hits)/n:.3f}", sum(hits), mc, err)


def xt(split: str = "tune") -> None:
    from engine.evaluation import xborrowing_eval as xb
    from engine.evaluation.xturkic_gold import donor_macro
    from engine.nlp import donor_proximity as dp
    from engine.utils.orthography import to_comparison_form

    xb.apply_variant("sca")
    rows = xb.load_cache(split)
    gold = {g["id"]: g for g in xb.load_split(split)}
    rows = [r for r in rows if r["id"] in gold]
    pred = xb.crossfit(rows)[0]["engine_trained"] if split == "tune" else None
    out = {}
    t = time.time()
    for rule in RULES:
        dp.ARABIC_VIA_RULE = rule
        new = []
        for r in rows:
            r2 = dict(r)
            if r["attributed"] is not None:
                a = dp.attribute_donor(to_comparison_form(r["query"]), gold[r["id"]]["gloss"],
                                       languages=list(gold[r["id"]]["donors"]))
                r2["attributed"] = a.lang_code if a else None
            new.append(r2)
        if rule == "off":
            print("cached==off", sum(a["attributed"] == b["attributed"] for a, b in zip(rows, new)), len(rows))
        out[rule] = {"engine_trained": xb.donor_identification(new, pred) if pred else None,
                     "all_fired": xb.donor_identification(new, {r["id"]: True for r in new}),
                     "attributed": {r["id"]: r["attributed"] for r in new}}
        print(rule, {k: v for k, v in out[rule].items() if k != "attributed"}, f"{time.time()-t:.0f}s", flush=True)
    (OUT / f"xt_{split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    if sys.argv[1] == "tr":
        tr(sys.argv[3] if len(sys.argv) > 3 else "train")
    else:
        xt(sys.argv[2] if len(sys.argv) > 2 else "tune")
