"""9j — yalnız etiket adımı: İtalyanca imla normalizasyonu (I1) / Venedikçe+Cenevizce
yalnız-etiket havuzu (I2) / eski dil havuzu yalnız SCA <= 0,35 iken (G1').

Kör indeks (xtr/index_blind.db; ETY_LEXICON_INDEX), ETY_DONOR_CLEAN=0, üretim kuralları
(ARABIC_VIA d1, FRENCH g2, WESTERN h1). AĞ KAPALI: socket.connect engellenir ve koşu
sonunda hiçbir getirici (engine.fetchers.*) yüklenmemiş olmalı (TDK etimonu motora girmez).

python harness.py gold tdk|tettl ayar|rapor    # rapor: bir kez!
python harness.py diag tdk|tettl                # ayar İtalyanca tanısı (yalnız off)
python harness.py k2                            # sızıntı akıl sağlığı (iki altının tümü; doğruluk yok)
python harness.py gold9f|gold9e|gold9g          # önceki raporlar (bilgi)
python harness.py tr|xt|saha                    # korumalar
"""
import json, os, socket, sqlite3, sys, time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
W = OUT.parent
BLIND = W / "xtr" / "index_blind.db"
FULL = OUT.parents[2] / "lexicons" / "index.db"
if sys.argv[1:2] != ["saha"]:
    os.environ["ETY_LEXICON_INDEX"] = str(BLIND)
os.environ.setdefault("ETY_DONOR_CLEAN", "0")
os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")


def _no_net(*a, **k):
    raise RuntimeError("9j: ağ kapalı (ölçümde hiçbir getirici çalışmamalı)")


if sys.argv[1:2] != ["saha"]:
    socket.socket.connect = _no_net
    socket.create_connection = _no_net

#: koşul -> (ITALIAN_ORTHO, VENETAN_LABELS, OLD_DONOR_LABELS, OLD_DONOR_MAX)
CONDS = {"off": (False, False, False, None), "i1": (True, False, False, None),
         "i2": (False, True, False, None), "g1p": (False, False, True, 0.35)}
CLS9 = {"el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "hy": "Ermenice", "xcl": "Ermenice"}


def setc(cond):
    from engine.nlp import donor_proximity as dp
    dp.ITALIAN_ORTHO, dp.VENETAN_LABELS, dp.OLD_DONOR_LABELS, dp.OLD_DONOR_MAX = CONDS[cond]
    dp.OLD_DONOR_MODE = "separate"
    dp.LABEL_FORM_FILTER = False
    return dp


def check_prod():
    from engine.nlp import donor_proximity as dp
    assert (dp.ARABIC_VIA_RULE, dp.FRENCH_RULE, dp.WESTERN_RULE) == ("d1", "g2", "h1")


def check_no_fetch():
    bad = sorted(m for m in sys.modules if m.startswith("engine.fetchers") or "tdk" in m.lower())
    assert not bad, f"getirici yüklendi: {bad}"


def cls(code):
    from engine.evaluation.tr_donor_eval import engine_class
    return CLS9.get(code) or engine_class(code)


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
        row = {"lang": it["lang"], "word": it["word"], "gold": it["gold"]}
        for c in conds:
            dp = setc(c)
            a = dp.attribute_donor(it["comparison"], gl.get((it["lang"], it["word"]), ""), languages=TURKISH_DONORS)
            row[c] = classify(a.lang_code) if a else "—"
            row[c + "_w"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}/{a.source}/{a.via}" if a else ""
        rows.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    setc("off")
    check_no_fetch()
    return rows


def summarize(rows, conds=tuple(CONDS), classes=None):
    from engine.evaluation.significance import mcnemar_test
    classes = classes or sorted({r["gold"] for r in rows})
    n = len(rows)
    base = [r["off"] == r["gold"] for r in rows]
    out = {"n": n}
    for c in conds:
        hits = [r[c] == r["gold"] for r in rows]
        mc = mcnemar_test(hits, base).as_dict() if c != "off" else {}
        per = {k: f"{sum(1 for r in rows if r['gold'] == k and r[c] == k)}/{sum(1 for r in rows if r['gold'] == k)}"
               for k in classes}
        conf = Counter(f"{r['gold']}->{r[c]}" for r in rows if r[c] != r["gold"])
        out[c] = {"acc": round(sum(hits) / n, 4), "hits": sum(hits), "mcnemar": mc, "per_class": per,
                  "errors": dict(conf.most_common())}
        print(c, json.dumps(out[c], ensure_ascii=False), flush=True)
    return out


def items_of(gold, split):
    items = json.loads((OUT / f"gold_{gold}.json").read_text())["items"]
    return [dict(i, lang="tr") for i in items if split in ("all", i["split"])]


def cmd_gold(gold, split):
    items = items_of(gold, split)
    rows = label_rows(items, glosses(BLIND, [("tr", i["word"]) for i in items]))
    res = summarize(rows)
    (OUT / f"res_{gold}_{split}{os.environ.get('NINEJ_SUFFIX', '')}.json").write_text(
        json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_diag(gold):
    """Ayar İtalyanca maddeleri: TDK/TETTL etimonu donors.db 'it'de var mı, anlam aramasına giriyor mu,
    SCA'sı kazanana göre ne? (yalnız off; kod değişmez)"""
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.utils.orthography import to_comparison_form
    setc("off")
    if gold == "gold9f":  # 9f raporu (açılmış, bilgi): İngilizce anlamlı Türkçe Wiktionary İtalyancaları
        items = [dict(i, lang="tr", comparison=to_comparison_form(i["word"]))
                 for i in json.loads((W / "donor9f" / "gold.json").read_text())["items"] if i["donor"] == "it"]
    else:
        items = [i for i in items_of(gold, "ayar") if i["donor"] == "it"]
    gl = glosses(BLIND, [("tr", i["word"]) for i in items])
    idx = dp._index()
    con = sqlite3.connect(idx.path)
    cat, rows = Counter(), []
    for i in items:
        g = gl.get(("tr", i["word"]), "")
        a = dp.attribute_donor(i["comparison"], g, languages=TURKISH_DONORS)
        ety = to_comparison_form(i.get("etymon") or "")
        in_db = bool(ety) and con.execute("SELECT 1 FROM donor_entries WHERE lang_code='it' AND comparison=? LIMIT 1",
                                          (ety,)).fetchone() is not None
        sense_rows = idx.by_sense(g, languages=["it"], limit=200)
        in_sense = any(r["comparison"] == ety for r in sense_rows)
        d_ety = dp.sca_distance(i["comparison"], ety) if ety else None
        d_i1 = dp.sca_distance(i["comparison"], dp.italian_phonetic(ety)) if ety else None
        fr_best = min((dp.sca_distance(i["comparison"], r["comparison"]) for r in idx.by_sense(g, languages=["fr"], limit=200)
                       if r["comparison"]), default=None)
        win = a.lang_code if a else "—"
        if cls(win) == "İtalyanca":
            k = "doğru"
        elif not ety:
            k = "etimon-yok"
        elif not in_db:
            k = "(b) etimon havuzda yok"
        elif not in_sense:
            k = "(a1) havuzda var, anlam aramasına girmiyor"
        else:
            k = "(a2) anlam aramasında var, yenildi"
        cat[k] += 1
        rows.append({"word": i["word"], "etymon": i.get("etymon"), "cat": k, "pred": row_w(a),
                     "sca_etymon": round(d_ety, 3) if d_ety is not None else None,
                     "sca_etymon_i1": round(d_i1, 3) if d_i1 is not None else None,
                     "sca_best_fr": round(fr_best, 3) if fr_best is not None else None,
                     "gloss_lang": i.get("gloss_lang"), 
                     "venetian": i.get("venetian")})
    check_no_fetch()
    print(json.dumps(dict(cat), ensure_ascii=False))
    (OUT / f"diag_{gold}.json").write_text(json.dumps({"counts": dict(cat), "rows": rows}, ensure_ascii=False, indent=0))


def row_w(a):
    return f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}" if a else ""


def cmd_k2():
    items = items_of("tdk", "all") + items_of("tettl", "all")
    pairs = [("tr", i["word"]) for i in items]
    con = sqlite3.connect(BLIND)
    leaked = sum(con.execute(
        "SELECT count(*) FROM entries WHERE lang_code=? AND word=? AND (coalesce(origin,'')!='' OR coalesce(donor_lang,'')!=''"
        " OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!='' OR coalesce(cognates,'')!='')", p).fetchone()[0]
        for p in pairs)
    gb, gf = glosses(BLIND, pairs), glosses(FULL, pairs)
    same_gloss = sum(gb.get(p) == gf.get(p) for p in pairs)
    hints = ("italian", "french", "greek", "arabic", "persian", "armenian", "venetian", "borrowed", "from ")
    hint = [(p[1], gb.get(p)) for p in pairs if any(k in (gb.get(p) or "").lower() for k in hints)]
    conds = ("off", "i1", "i2", "g1p")
    rb = label_rows(items, gb, conds)
    rf = label_rows(items, gf, conds)
    same = sum(all(a[c + "_w"] == b[c + "_w"] for c in conds) for a, b in zip(rb, rf))
    # TDK etimonları verici/etiket havuzlarında "TDK" kaynağıyla yok: havuzlar yalnız kaikki
    from engine.db.donor_index import DEFAULT_DB, LABEL_DB
    srcs = {}
    for db in (DEFAULT_DB, LABEL_DB):
        c2 = sqlite3.connect(db)
        srcs[db.name] = [r[0] for r in c2.execute("SELECT DISTINCT lang_code FROM donor_entries")]
    res = {"n": len(items), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "gloss_with_language_hint": hint, "pool_languages": srcs,
           "fetchers_loaded": sorted(m for m in sys.modules if m.startswith("engine.fetchers")),
           "network": "socket.connect engelli", "pass": leaked == 0 and same == len(items)}
    print(json.dumps({k: v for k, v in res.items() if k != "gloss_with_language_hint"}, ensure_ascii=False),
          len(hint))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_prev(which):
    from engine.evaluation.tr_donor_eval import engine_class
    from engine.utils.orthography import to_comparison_form
    if which == "gold9g":
        items = [i for i in json.loads((W / "donor9g" / "gold.json").read_text())["items"] if i["split"] == "rapor"]
        classify = cls
    else:
        d, sp = {"gold9f": ("donor9f", "rapor9f"), "gold9e": ("donor9e", "rapor")}[which]
        items = [dict(i, lang="tr", comparison=to_comparison_form(i["word"]))
                 for i in json.loads((W / d / "gold.json").read_text())["items"] if i["split"] == sp]
        classify = engine_class
    rows = label_rows(items, glosses(BLIND, [(i["lang"], i["word"]) for i in items]), classify=classify)
    res = summarize(rows)
    (OUT / f"{which}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_tr():
    from engine.evaluation.tr_donor_eval import engine_class, load_cases
    from engine.utils.orthography import to_comparison_form
    cases = [c for c in load_cases() if c.split in ("train", "dev")]
    items = [{"lang": "tr", "word": c.word, "gold": c.gold, "comparison": to_comparison_form(c.word)} for c in cases]
    rows = label_rows(items, glosses(BLIND, [("tr", c.word) for c in cases]), classify=engine_class)
    res = summarize(rows, classes=sorted({c.gold for c in cases}))
    (OUT / "tr_traindev.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


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
    {"gold": lambda: cmd_gold(sys.argv[2], sys.argv[3]), "diag": lambda: cmd_diag(sys.argv[2]), "k2": cmd_k2,
     "gold9f": lambda: cmd_prev("gold9f"), "gold9e": lambda: cmd_prev("gold9e"), "gold9g": lambda: cmd_prev("gold9g"),
     "tr": cmd_tr, "xt": cmd_xt, "saha": cmd_saha}[c]()
