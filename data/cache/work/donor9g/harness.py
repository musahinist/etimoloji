"""9g — yalnız etiket adımı: eski dil havuzları (G1) / çekim süzgeci (G2) / ikisi (G3).

Kör indeks (xtr/index_blind.db; ETY_LEXICON_INDEX), ETY_DONOR_CLEAN=0, üretim
kuralları (ARABIC_VIA d1, FRENCH g2, WESTERN h1). Koşullar: off (üretim), g1, g2, g3.

python harness.py gold ayar          # 9g ayar (karışıklık matrisi dahil)
python harness.py gold rapor         # 9g YENİ rapor (bir kez!)
python harness.py k2                 # sızıntı akıl sağlığı (9g altının tümü; doğruluk hesaplanmaz)
python harness.py gold9f             # 9f rapor (bilgi)
python harness.py tr|xt|saha         # korumalar
"""
import json, os, sqlite3, sys, time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
BLIND = OUT.parent / "xtr" / "index_blind.db"
FULL = OUT.parents[2] / "lexicons" / "index.db"
if sys.argv[1:2] != ["saha"]:
    os.environ["ETY_LEXICON_INDEX"] = str(BLIND)
os.environ.setdefault("ETY_DONOR_CLEAN", "0")
os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
CONDS = {"off": (False, False, "separate"), "g1": (True, False, "separate"), "g2": (False, True, "separate"),
         "g3": (True, True, "separate")}
if os.environ.get("NINEG_MERGE"):  # ayar tanısı: eski dil ailenin grubuna katılır
    CONDS = {"off": (False, False, "separate"), "g1m": (True, False, "merge"), "g3m": (True, True, "merge")}
CLS9G = {"el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "hy": "Ermenice", "xcl": "Ermenice"}


def setc(cond):
    from engine.nlp import donor_proximity as dp
    dp.OLD_DONOR_LABELS, dp.LABEL_FORM_FILTER, dp.OLD_DONOR_MODE = CONDS[cond]
    return dp


def check_prod():
    from engine.nlp import donor_proximity as dp
    assert (dp.ARABIC_VIA_RULE, dp.FRENCH_RULE, dp.WESTERN_RULE) == ("d1", "g2", "h1")


def cls(code):
    from engine.evaluation.tr_donor_eval import engine_class
    return CLS9G.get(code) or engine_class(code)


def glosses(db, pairs):
    con = sqlite3.connect(db)
    out = {}
    for lang, w in pairs:
        r = con.execute("SELECT gloss FROM entries WHERE lang_code=? AND word=? AND gloss IS NOT NULL AND gloss!=''"
                        " ORDER BY id LIMIT 1", (lang, w)).fetchone()
        if r:
            out[(lang, w)] = r[0]
    return out


def label_rows(items, gl, conds=tuple(CONDS), classify=cls):
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    check_prod()
    rows, t = [], time.time()
    for n, it in enumerate(items):
        row = {"lang": it["lang"], "word": it["word"], "gold": it["gold"], "dialectal": it.get("dialectal")}
        for c in conds:
            dp = setc(c)
            a = dp.attribute_donor(it["comparison"], gl.get((it["lang"], it["word"]), ""), languages=TURKISH_DONORS)
            row[c] = classify(a.lang_code) if a else "—"
            row[c + "_w"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.source}/{a.via}" if a else ""
        rows.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    setc("off")
    return rows


def summarize(rows, conds=tuple(CONDS), classes=("Yunanca", "Ermenice")):
    from engine.evaluation.significance import mcnemar_test
    n = len(rows)
    base = [r["off"] == r["gold"] for r in rows]
    out = {"n": n}
    for c in conds:
        hits = [r[c] == r["gold"] for r in rows]
        mc = mcnemar_test(hits, base).as_dict() if c != "off" else {}
        per = {k: f"{sum(1 for r in rows if r['gold'] == k and r[c] == k)}/{sum(1 for r in rows if r['gold'] == k)}"
               for k in classes}
        conf = Counter(f"{r['gold']}->{r[c]}" for r in rows)
        out[c] = {"acc": round(sum(hits) / n, 4), "hits": sum(hits), "mcnemar": mc, "per_class": per,
                  "confusion": dict(conf.most_common())}
        print(c, json.dumps(out[c], ensure_ascii=False), flush=True)
    return out


def items_of(split):
    return [i for i in json.loads((OUT / "gold.json").read_text())["items"] if split in ("all", i["split"])]


def cmd_gold(split):
    items = items_of(split)
    rows = label_rows(items, glosses(BLIND, [(i["lang"], i["word"]) for i in items]))
    res = summarize(rows)
    res["dialectal"] = summarize([r for r in rows if r["dialectal"]]) if any(r["dialectal"] for r in rows) else {}
    (OUT / f"gold_{split}{os.environ.get('NINEG_SUFFIX', '')}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    items = items_of("all")
    pairs = [(i["lang"], i["word"]) for i in items]
    con = sqlite3.connect(BLIND)
    leaked = sum(con.execute(
        "SELECT count(*) FROM entries WHERE lang_code=? AND word=? AND (coalesce(origin,'')!='' OR coalesce(donor_lang,'')!=''"
        " OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!='' OR coalesce(cognates,'')!='')", p).fetchone()[0]
        for p in pairs)
    gb, gf = glosses(BLIND, pairs), glosses(FULL, pairs)
    same_gloss = sum(gb.get(p) == gf.get(p) for p in pairs)
    hints = ("greek", "armenian", "byzantine", "borrowed", "from ")
    hint = [(p, gb.get(p)) for p in pairs if any(k in (gb.get(p) or "").lower() for k in hints)]
    rb = label_rows(items, gb, ("off", "g3"))
    rf = label_rows(items, gf, ("off", "g3"))
    same = sum(a["off_w"] == b["off_w"] and a["g3_w"] == b["g3_w"] for a, b in zip(rb, rf))
    res = {"n": len(items), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "gloss_with_language_hint": hint,
           "pass": leaked == 0 and same == len(items)}
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_gold9f():
    from engine.evaluation.tr_donor_eval import engine_class
    items = json.loads((OUT.parent / "donor9f" / "gold.json").read_text())["items"]
    from engine.utils.orthography import to_comparison_form
    items = [dict(i, lang="tr", comparison=to_comparison_form(i["word"])) for i in items]
    rows = label_rows(items, glosses(BLIND, [("tr", i["word"]) for i in items]), classify=engine_class)
    res = summarize(rows, classes=("Fransızca", "İtalyanca"))
    (OUT / "gold9f_rapor.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_tr():
    from engine.evaluation.tr_donor_eval import engine_class, load_cases
    from engine.utils.orthography import to_comparison_form
    cases = [c for c in load_cases() if c.split in ("train", "dev")]
    items = [{"lang": "tr", "word": c.word, "gold": c.gold, "comparison": to_comparison_form(c.word)} for c in cases]
    rows = label_rows(items, glosses(BLIND, [("tr", c.word) for c in cases]), classify=engine_class)
    res = summarize(rows, classes=sorted({c.gold for c in cases}))
    (OUT / f"tr_traindev{os.environ.get('NINEG_SUFFIX', '')}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_xt():
    from engine.evaluation import xborrowing_eval as xb
    from engine.utils.orthography import to_comparison_form
    check_prod()
    xb.apply_variant("sca")
    rows = xb.load_cache("tune")
    gold = {g["id"]: g for g in xb.load_split("tune")}
    rows = [r for r in rows if r["id"] in gold]
    pred = xb.crossfit(rows)[0]["engine_trained"]
    out = {}
    for c in CONDS:
        dp = setc(c)
        new = []
        for r in rows:
            r2 = dict(r)
            if r["attributed"] is not None:
                a = dp.attribute_donor(to_comparison_form(r["query"]), gold[r["id"]]["gloss"],
                                       languages=list(gold[r["id"]]["donors"]))
                r2["attributed"] = a.lang_code if a else None
            new.append(r2)
        out[c] = {"engine_trained": xb.donor_identification(new, pred)["accuracy"]}
        print(c, out[c], flush=True)
    setc("off")
    (OUT / "xt_tune.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_saha():
    from engine.evaluation import donor_id_eval
    check_prod()
    out = {}
    for c in CONDS:
        dp = setc(c)
        dp.reset_cache()
        r = donor_id_eval.run()
        out[c] = {k: v.get("accuracy") for k, v in r["systems"].items()}
        print(c, out[c], flush=True)
    setc("off")
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    c = sys.argv[1]
    {"gold": lambda: cmd_gold(sys.argv[2]), "k2": cmd_k2, "gold9f": cmd_gold9f,
     "tr": cmd_tr, "xt": cmd_xt, "saha": cmd_saha}[c]()
