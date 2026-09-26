"""9e — yalnız etiket adımı (attribute_donor), Fransızca kuralları (off/f1/f2).

Anlam KÖR indeks kopyasından (data/cache/work/donor9e/index_blind.db; köken
sütunları NULL). ARABIC_VIA_RULE üretimdeki gibi d1.

python harness.py gold ayar|rapor   # yeni altın (rapor bir kez!)
python harness.py k2                # sızıntı akıl sağlığı (ayar)
python harness.py tr                # Türkçe TDK+Nişanyan train+dev (koruma)
python harness.py xt                # xturkic ayar verici tanıma (koruma)
python harness.py saha              # make eval-donor, her kural
"""
import json, os, sqlite3, sys, time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
BLIND = OUT / "index_blind.db"
FULL = OUT.parents[2] / "lexicons" / "index.db"
if sys.argv[1:2] != ["saha"]:  # Saha korumasi make eval-donor ortamiyla (tam indeks)
    os.environ["ETY_LEXICON_INDEX"] = str(BLIND)
os.environ.setdefault("ETY_DONOR_CLEAN", "0")
os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
RULES = ("off", "f1", "f2", "g1", "g2")
CLS = {"fr": "Fransızca", "ar": "Arapça", "fa": "Farsça", "it": "İtalyanca", "el": "Yunanca"}


def glosses(db: Path, words):
    con = sqlite3.connect(db)
    out = {}
    for i in range(0, len(words), 400):
        b = words[i:i + 400]
        for w, g in con.execute("SELECT word, gloss FROM entries WHERE lang_code='tr' AND word IN "
                                f"({','.join('?' * len(b))}) AND gloss IS NOT NULL AND gloss != '' ORDER BY id", b):
            out.setdefault(w, g)
    return out


def label_rows(cases, gl, rules=RULES):
    from engine.evaluation.tr_donor_eval import engine_class
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.utils.orthography import to_comparison_form

    assert dp.ARABIC_VIA_RULE == "d1"
    rows, t = [], time.time()
    for n, (word, gold) in enumerate(cases):
        row = {"word": word, "gold": gold}
        for rule in rules:
            dp.FRENCH_RULE = rule
            a = dp.attribute_donor(to_comparison_form(word), gl.get(word, ""), languages=TURKISH_DONORS)
            row[rule] = engine_class(a.lang_code) if a else "—"
            row[rule + "_w"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.via}" if a else ""
        rows.append(row)
        if n % 100 == 0:
            print(n, len(cases), f"{time.time() - t:.0f}s", flush=True)
    return rows


def summarize(rows, rules=RULES):
    from engine.evaluation.significance import mcnemar_test

    n = len(rows)
    base = [r["off"] == r["gold"] for r in rows]
    out = {"n": n}
    for rule in rules:
        hits = [r[rule] == r["gold"] for r in rows]
        err = Counter(f"{r['gold']}->{r[rule]}" for r in rows if r[rule] != r["gold"]).most_common(8)
        mc = mcnemar_test(hits, base).as_dict() if rule != "off" else {}
        per = {c: f"{sum(1 for r in rows if r['gold'] == c and r[rule] == c)}/{sum(1 for r in rows if r['gold'] == c)}"
               for c in CLS.values()}
        out[rule] = {"acc": round(sum(hits) / n, 4), "hits": sum(hits), "mcnemar": mc, "per_class": per, "errors": err}
        print(rule, json.dumps(out[rule], ensure_ascii=False), flush=True)
    return out


def gold_cases(split):
    items = json.loads((OUT / "gold.json").read_text())["items"]
    return [(i["word"], i["gold"]) for i in items if i["split"] == split]


def cmd_gold(split):
    cases = gold_cases(split)
    gl = glosses(BLIND, [w for w, _ in cases])
    rows = label_rows(cases, gl)
    res = summarize(rows)
    (OUT / f"gold_{split}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    cases = gold_cases("ayar")
    words = [w for w, _ in cases]
    con = sqlite3.connect(BLIND)
    leaked = con.execute("SELECT count(*) FROM entries WHERE lang_code='tr' AND word IN "
                         f"({','.join('?' * len(words))}) AND (coalesce(origin,'')!='' OR coalesce(donor_lang,'')!=''"
                         " OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!='' OR coalesce(cognates,'')!='')",
                         words).fetchone()[0]
    gb, gf = glosses(BLIND, words), glosses(FULL, words)
    same_gloss = sum(gb.get(w) == gf.get(w) for w in words)
    lang_words = ("french", "arabic", "persian", "italian", "greek", "borrowed", "from ")
    gloss_hint = [w for w in words if any(k in (gb.get(w) or "").lower() for k in lang_words)]
    rb = label_rows(cases, gb, ("off",))
    rf = label_rows(cases, gf, ("off",))
    same = sum(a["off_w"] == b["off_w"] for a, b in zip(rb, rf))
    res = {"n": len(words), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "gloss_with_language_hint": gloss_hint,
           "pass": leaked == 0 and same == len(words)}
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_tr():
    from engine.evaluation.tr_donor_eval import load_cases

    cases = [(c.word, c.gold) for c in load_cases() if c.split in ("train", "dev")]
    rows = label_rows(cases, glosses(BLIND, [w for w, _ in cases]))
    res = summarize(rows)
    (OUT / "tr_traindev.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_xt():
    from engine.evaluation import xborrowing_eval as xb
    from engine.nlp import donor_proximity as dp
    from engine.utils.orthography import to_comparison_form

    xb.apply_variant("sca")
    rows = xb.load_cache("tune")
    gold = {g["id"]: g for g in xb.load_split("tune")}
    rows = [r for r in rows if r["id"] in gold]
    pred = xb.crossfit(rows)[0]["engine_trained"]
    out = {}
    for rule in RULES:
        dp.FRENCH_RULE = rule
        new = []
        for r in rows:
            r2 = dict(r)
            if r["attributed"] is not None:
                a = dp.attribute_donor(to_comparison_form(r["query"]), gold[r["id"]]["gloss"],
                                       languages=list(gold[r["id"]]["donors"]))
                r2["attributed"] = a.lang_code if a else None
            new.append(r2)
        out[rule] = {"engine_trained": xb.donor_identification(new, pred),
                     "all_fired": xb.donor_identification(new, {r["id"]: True for r in new})}
        print(rule, json.dumps(out[rule], ensure_ascii=False)[:400], flush=True)
    (OUT / "xt_tune.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_saha():
    from engine.evaluation import donor_id_eval
    from engine.nlp import donor_proximity as dp

    out = {}
    for rule in RULES:
        dp.FRENCH_RULE = rule
        dp.reset_cache()
        out[rule] = donor_id_eval.run()
        print(rule, json.dumps(out[rule], ensure_ascii=False, default=str)[:600], flush=True)
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0, default=str))


if __name__ == "__main__":
    c = sys.argv[1]
    {"gold": lambda: cmd_gold(sys.argv[2]), "k2": cmd_k2, "tr": cmd_tr, "xt": cmd_xt, "saha": cmd_saha}[c]()
