"""9l — yalnız etiket adımı: S1 Türkçe anlam -> İngilizce köprü / S2 ayrı dil havuz sorgusu
(G2 genellemesi) / S3 tam eşleşmede ham mesafe. (9j harness'ından türetildi.)

Kör indeks (xtr/index_blind.db; ETY_LEXICON_INDEX), ETY_DONOR_CLEAN=0, üretim kuralları
(ARABIC_VIA d1, FRENCH g2, WESTERN h1). AĞ KAPALI: socket.connect engellenir ve koşu
sonunda hiçbir getirici (engine.fetchers.*) yüklenmemiş olmalı (TDK etimonu motora girmez).

python harness.py gold tdk ayar                 # 9j TDK ayar (seçim/doğrulama)
python harness.py gold 9l rapor                 # YENİ rapor: bir kez!
python harness.py k2                            # sızıntı akıl sağlığı (yeni altın; doğruluk yok)
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
    raise RuntimeError("9l: ağ kapalı (ölçümde hiçbir getirici çalışmamalı)")


if sys.argv[1:2] != ["saha"]:
    socket.socket.connect = _no_net
    socket.create_connection = _no_net

#: koşul -> (SENSE_BRIDGE, EXTRA_POOL_LANGS, EXACT_MATCH_EPS)
ALL = {"off": (False, (), None), "s1": (True, (), None), "s2": (False, ("it",), None),
       "s2b": (False, ("it", "el", "hy"), None), "s3": (False, (), 0.0), "s3b": (False, (), 0.05),
       "s13": (True, (), 0.05)}
CONDS = {k: ALL[k] for k in os.environ.get("NINEL_CONDS", "off,s1,s2,s2b,s3,s3b").split(",")}
CLS9 = {"el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "hy": "Ermenice", "xcl": "Ermenice"}


def setc(cond):
    from engine.nlp import donor_proximity as dp
    dp.SENSE_BRIDGE, dp.EXTRA_POOL_LANGS, dp.EXACT_MATCH_EPS = ALL[cond]
    assert not (dp.ITALIAN_ORTHO or dp.VENETAN_LABELS or dp.OLD_DONOR_LABELS or dp.LABEL_FORM_FILTER)
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
    path = OUT / "gold.json" if gold == "9l" else W / "donor9j" / f"gold_{gold}.json"
    items = json.loads(path.read_text())["items"]
    return [dict(i, lang="tr") for i in items if split in ("all", i["split"])]


def cmd_gold(gold, split):
    items = items_of(gold, split)
    rows = label_rows(items, glosses(BLIND, [("tr", i["word"]) for i in items]))
    res = summarize(rows)
    res["strata"] = strata(rows, items)
    print("strata", json.dumps(res["strata"], ensure_ascii=False), flush=True)
    (OUT / f"res_{gold}_{split}{os.environ.get('NINEL_SUFFIX', '')}.json").write_text(
        json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def strata(rows, items):
    """Bilgi: anlam dili ve sınıf katmanları."""
    gl = {i["word"]: i.get("gloss_lang") for i in items}
    out = {}
    for c in CONDS:
        for k in ("en", "tr"):
            sub = [r for r in rows if gl.get(r["word"]) == k]
            if sub:
                out[f"{c}/{k}"] = f"{sum(r[c] == r['gold'] for r in sub)}/{len(sub)}"
        out[f"{c}/etiketsiz"] = sum(r[c] == "—" for r in rows)
    return out


def row_w(a):
    return f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}" if a else ""


def cmd_k2():
    items = items_of("9l", "all")
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
    # S1 köprü anlamı (kelimenin Vikisözlük çeviri karşılıkları) dil adı/köken ipucu içeriyor mu?
    from engine.nlp import donor_proximity as dp
    from engine.db.sense_bridge import english_sense
    setc("s1")
    bridged = {i["word"]: dp.bridged_sense(i["comparison"], gb.get(("tr", i["word"]), "")) for i in items}
    setc("off")
    bridge_used = sum(bridged[i["word"]] != gb.get(("tr", i["word"]), "") for i in items)
    bridge_hint = [(w, b) for w, b in bridged.items() if any(k in b.lower() for k in hints)]
    conds = tuple(CONDS)
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
           "bridge_used": bridge_used, "bridge_with_language_hint": bridge_hint,
           "fetchers_loaded": sorted(m for m in sys.modules if m.startswith("engine.fetchers")),
           "network": "socket.connect engelli", "pass": leaked == 0 and same == len(items)}
    print(json.dumps({k: v for k, v in res.items() if k not in ("gloss_with_language_hint", "bridge_with_language_hint")},
                     ensure_ascii=False), len(hint), bridge_hint)
    (OUT / f"k2{os.environ.get('NINEL_SUFFIX', '')}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


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
    res["strata"] = strata(rows, items) if which == "gold9g" else {}
    (OUT / f"{which}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_9j():
    """9j raporları (açılmış, bilgi): TDK rapor + TETTL rapor."""
    for g in ("tdk", "tettl"):
        items = items_of(g, "rapor")
        rows = label_rows(items, glosses(BLIND, [("tr", i["word"]) for i in items]))
        res = summarize(rows)
        res["strata"] = strata(rows, items)
        print("strata", json.dumps(res["strata"], ensure_ascii=False), flush=True)
        (OUT / f"gold9j_{g}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_tr():
    from engine.evaluation.tr_donor_eval import engine_class, load_cases
    from engine.utils.orthography import to_comparison_form
    cases = [c for c in load_cases() if c.split in ("train", "dev")]
    items = [{"lang": "tr", "word": c.word, "gold": c.gold, "comparison": to_comparison_form(c.word)} for c in cases]
    rows = label_rows(items, glosses(BLIND, [("tr", c.word) for c in cases]), classify=engine_class)
    res = summarize(rows, classes=sorted({c.gold for c in cases}))
    (OUT / f"tr_traindev{os.environ.get('NINEL_SUFFIX', '')}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


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
    (OUT / f"xt_tune{os.environ.get('NINEL_SUFFIX', '')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


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
    (OUT / f"saha{os.environ.get('NINEL_SUFFIX', '')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    c = sys.argv[1]
    {"gold": lambda: cmd_gold(sys.argv[2], sys.argv[3]), "k2": cmd_k2,
     "gold9f": lambda: cmd_prev("gold9f"), "gold9e": lambda: cmd_prev("gold9e"), "gold9g": lambda: cmd_prev("gold9g"),
     "gold9j": lambda: cmd_9j(),
     "tr": cmd_tr, "xt": cmd_xt, "saha": cmd_saha}[c]()
