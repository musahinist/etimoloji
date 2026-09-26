"""9f — yalnız etiket adımı, Batı alıntısında Fransızca önceliği (WESTERN_RULE).

9e harness'ının yardımcılarını kullanır (kör indeks, ETY_DONOR_CLEAN=0,
FRENCH_RULE=g2 üretim, ARABIC_VIA_RULE=d1). Kurallar: off (üretim), h1, h2,
h12 (bilgi).

python harness.py gold9e ayar|rapor   # 9e altını (ayar = doğrulama; rapor görüldü, bilgi)
python harness.py gold9f              # YENİ rapor (bir kez!)
python harness.py k2                  # sızıntı akıl sağlığı, 9f altını
python harness.py tr|xt|saha          # korumalar
"""
import json, os, sqlite3, sys
from pathlib import Path

OUT = Path(__file__).parent
W9E = OUT.parent / "donor9e"
sys.path.insert(0, str(W9E))
import harness as H  # noqa: E402  (ETY_LEXICON_INDEX'i kör kopyaya ayarlar; saha hariç)

RULES = ("off", "h1", "h2", "h12")


def _patch():
    from engine.nlp import donor_proximity as dp

    assert dp.FRENCH_RULE == "g2"


    def label_rows(cases, gl, rules=RULES):
        from engine.evaluation.tr_donor_eval import engine_class
        from engine.nlp.borrowing_detector import TURKISH_DONORS
        from engine.utils.orthography import to_comparison_form

        rows = []
        for word, gold in cases:
            row = {"word": word, "gold": gold}
            for rule in rules:
                dp.WESTERN_RULE = rule
                a = dp.attribute_donor(to_comparison_form(word), gl.get(word, ""), languages=TURKISH_DONORS)
                row[rule] = engine_class(a.lang_code) if a else "—"
                row[rule + "_w"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.via}" if a else ""
            rows.append(row)
        dp.WESTERN_RULE = "off"
        return rows

    H.label_rows = label_rows
    H.RULES = RULES
    return dp


def cmd_gold(path, split, out):
    dp = _patch()
    items = json.loads(path.read_text())["items"]
    cases = [(i["word"], i["gold"]) for i in items if i["split"] == split]
    gl = H.glosses(H.BLIND, [w for w, _ in cases])
    rows = H.label_rows(cases, gl)
    res = H.summarize(rows, RULES)
    (OUT / out).write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    _patch()
    items = json.loads((OUT / "gold.json").read_text())["items"]
    cases = [(i["word"], i["gold"]) for i in items]
    words = [w for w, _ in cases]
    con = sqlite3.connect(H.BLIND)
    leaked = con.execute("SELECT count(*) FROM entries WHERE lang_code='tr' AND word IN "
                         f"({','.join('?' * len(words))}) AND (coalesce(origin,'')!='' OR coalesce(donor_lang,'')!=''"
                         " OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!='' OR coalesce(cognates,'')!='')",
                         words).fetchone()[0]
    gb, gf = H.glosses(H.BLIND, words), H.glosses(H.FULL, words)
    same_gloss = sum(gb.get(w) == gf.get(w) for w in words)
    lang_words = ("french", "italian", "borrowed", "from ")
    hint = [(w, gb.get(w)) for w in words if any(k in (gb.get(w) or "").lower() for k in lang_words)]
    rb = H.label_rows(cases, gb, ("off", "h12"))
    rf = H.label_rows(cases, gf, ("off", "h12"))
    same = sum(a["off_w"] == b["off_w"] and a["h12_w"] == b["h12_w"] for a, b in zip(rb, rf))
    res = {"n": len(words), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "gloss_with_language_hint": hint,
           "pass": leaked == 0 and same == len(words)}
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_tr():
    from engine.evaluation.tr_donor_eval import load_cases

    _patch()
    cases = [(c.word, c.gold) for c in load_cases() if c.split in ("train", "dev")]
    rows = H.label_rows(cases, H.glosses(H.BLIND, [w for w, _ in cases]))
    res = H.summarize(rows, RULES)
    (OUT / "tr_traindev.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_xt():
    from engine.evaluation import xborrowing_eval as xb
    from engine.utils.orthography import to_comparison_form

    dp = _patch()
    xb.apply_variant("sca")
    rows = xb.load_cache("tune")
    gold = {g["id"]: g for g in xb.load_split("tune")}
    rows = [r for r in rows if r["id"] in gold]
    pred = xb.crossfit(rows)[0]["engine_trained"]
    out = {}
    for rule in RULES:
        dp.WESTERN_RULE = rule
        new = []
        for r in rows:
            r2 = dict(r)
            if r["attributed"] is not None:
                a = dp.attribute_donor(to_comparison_form(r["query"]), gold[r["id"]]["gloss"],
                                       languages=list(gold[r["id"]]["donors"]))
                r2["attributed"] = a.lang_code if a else None
            new.append(r2)
        out[rule] = {"engine_trained": xb.donor_identification(new, pred)["accuracy"]}
        print(rule, out[rule], flush=True)
    dp.WESTERN_RULE = "off"
    (OUT / "xt_tune.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_saha():
    from engine.evaluation import donor_id_eval

    dp = _patch()
    out = {}
    for rule in RULES:
        dp.WESTERN_RULE = rule
        dp.reset_cache()
        r = donor_id_eval.run()
        out[rule] = {k: v.get("accuracy") for k, v in r["systems"].items()}
        print(rule, out[rule], flush=True)
    dp.WESTERN_RULE = "off"
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    c = sys.argv[1]
    {"gold9e": lambda: cmd_gold(W9E / "gold.json", sys.argv[2], f"gold9e_{sys.argv[2]}.json"),
     "gold9f": lambda: cmd_gold(OUT / "gold.json", "rapor9f", "gold9f_rapor.json"),
     "k2": cmd_k2, "tr": cmd_tr, "xt": cmd_xt, "saha": cmd_saha}[c]()
