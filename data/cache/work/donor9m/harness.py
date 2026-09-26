"""9m — Fransızca kurallarının (G2, H1) Arapça bedeli ve doğal dağılım ağırlıklı doğruluk.

Yalnız etiket adımı (`donor_proximity.attribute_donor`). Kör indeks (xtr/index_blind.db;
ETY_LEXICON_INDEX), ETY_DONOR_CLEAN=0, S1 anlam köprüsü açık (üretim). AĞ KAPALI:
socket.connect engellenir ve koşu sonunda hiçbir getirici (engine.fetchers.*) yüklenmemiş olmalı.

Koşullar (FRENCH_RULE, WESTERN_RULE, R1 düşülen sonekler, R2 Arapça iskelet koruması):
  base  off/off     — G2 ve H1 kapalı (4.3.0 öncesi Fransızca kuralı yok)
  g1    g1/off      — yalnız G2'nin ayrı 200'lük Fransızca havuzu (Fransızca aracılı yok)
  g2    g2/off      — G2 tam
  h1    off/h1      — yalnız H1
  prod  g2/h1       — üretim (4.3.1)
  r1    prod + H1'den R1_DROP sonekleri çıkarılmış
  r2    prod + Arapça iskelet koruması (iskelet >= 3 ünsüz)
  r3    r1 + r2

python harness.py gold <altın>      # altın: tr tdk_ayar tettl_ayar tdk_rapor tettl_rapor 9l 9e_ayar
                                    #        9e_rapor 9f 9g_ayar 9g_rapor 9m (YENİ rapor: bir kez!)
python harness.py k2                # sızıntı akıl sağlığı (yeni altın; doğruluk yok)
python harness.py xt|saha           # korumalar
"""
import json, os, socket, sqlite3, sys, time
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
    raise RuntimeError("9m: ağ kapalı (ölçümde hiçbir getirici çalışmamalı)")


if sys.argv[1:2] != ["saha"]:
    socket.socket.connect = _no_net
    socket.create_connection = _no_net

#: R1'de H1'den çıkarılan sonekler (ön kayıtta sabit; bkz. PREREG.md).
R1_DROP = tuple(os.environ.get("NINEM_R1_DROP", "ik").split(","))
#: koşul -> (FRENCH_RULE, WESTERN_RULE, WESTERN_SUFFIX_DROP, FRENCH_ARABIC_GUARD)
ALL = {"base": ("off", "off", (), False), "g1": ("g1", "off", (), False), "g2": ("g2", "off", (), False),
       "h1": ("off", "h1", (), False), "prod": ("g2", "h1", (), False),
       "r1": ("g2", "h1", R1_DROP, False), "r2": ("g2", "h1", (), True), "r3": ("g2", "h1", R1_DROP, True)}
CONDS = [k for k in os.environ.get("NINEM_CONDS", ",".join(ALL)).split(",")]
CLS9 = {"el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "hy": "Ermenice", "xcl": "Ermenice"}


#: R2 "güçlü" eşleşme ayarı: (iskelet en az, Arapça aday en çok SCA, Arapçanın Batı alıntısı sayılmaz)
R2_VARIANTS = {"a": (3, None, False), "b": (3, 0.35, False), "c": (3, 0.35, True), "d": (3, None, True)}
R2_VARIANT = os.environ.get("NINEM_R2", "d")


def setc(cond):
    from engine.nlp import donor_proximity as dp
    dp.FRENCH_RULE, dp.WESTERN_RULE, dp.WESTERN_SUFFIX_DROP, dp.FRENCH_ARABIC_GUARD = ALL[cond]
    dp.FRENCH_ARABIC_GUARD_MIN, dp.FRENCH_ARABIC_GUARD_MAX, dp.FRENCH_ARABIC_GUARD_SKIP_WESTERN = \
        R2_VARIANTS[R2_VARIANT]
    assert not (dp.ITALIAN_ORTHO or dp.VENETAN_LABELS or dp.OLD_DONOR_LABELS or dp.LABEL_FORM_FILTER)
    assert dp.SENSE_BRIDGE and not dp.EXTRA_POOL_LANGS and dp.EXACT_MATCH_EPS is None
    assert dp.ARABIC_VIA_RULE == "d1"
    return dp


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
    rows, t = [], time.time()
    for n, it in enumerate(items):
        row = {"lang": it["lang"], "word": it["word"], "gold": it["gold"]}
        for c in conds:
            dp = setc(c)
            a = dp.attribute_donor(it["comparison"], gl.get((it["lang"], it["word"]), ""), languages=TURKISH_DONORS)
            row[c] = classify(a.lang_code) if a else "—"
            row[c + "_w"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}/{a.via}" if a else ""
        rows.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    setc("prod")
    check_no_fetch()
    return rows


def items_of(gold):
    """Altın -> maddeler (lang, word, comparison, gold sınıfı)."""
    from engine.utils.orthography import to_comparison_form
    if gold == "tr":
        from engine.evaluation.tr_donor_eval import load_cases
        return [{"lang": "tr", "word": c.word, "gold": c.gold, "comparison": to_comparison_form(c.word)}
                for c in load_cases() if c.split in ("train", "dev")]
    if gold in ("tdk_ayar", "tettl_ayar", "tdk_rapor", "tettl_rapor"):
        g, sp = gold.split("_")
        items = json.loads((W / "donor9j" / f"gold_{g}.json").read_text())["items"]
        return [dict(i, lang="tr") for i in items if i["split"] == sp]
    if gold in ("9l", "9m"):
        items = json.loads((W / f"donor{gold}" / "gold.json").read_text())["items"]
        return [dict(i, lang="tr") for i in items]
    if gold in ("9e_ayar", "9e_rapor", "9f"):
        d, sp = {"9e_ayar": ("donor9e", "ayar"), "9e_rapor": ("donor9e", "rapor"), "9f": ("donor9f", "rapor9f")}[gold]
        return [dict(i, lang="tr", comparison=to_comparison_form(i["word"]))
                for i in json.loads((W / d / "gold.json").read_text())["items"] if i["split"] == sp]
    if gold in ("9g_ayar", "9g_rapor"):
        sp = gold.split("_")[1]
        return [i for i in json.loads((W / "donor9g" / "gold.json").read_text())["items"] if i["split"] == sp]
    raise SystemExit(f"bilinmeyen altın {gold}")


def cmd_gold(gold):
    from engine.evaluation.tr_donor_eval import engine_class
    items = items_of(gold)
    classify = engine_class if gold in ("tr", "9e_ayar", "9e_rapor", "9f") else cls
    rows = label_rows(items, glosses(BLIND, [(i["lang"], i["word"]) for i in items]), classify=classify)
    from natural import summarize
    res = summarize(rows, CONDS)
    for c in CONDS:
        print(c, json.dumps(res[c], ensure_ascii=False), flush=True)
    (OUT / f"res_{gold}{os.environ.get('NINEM_SUFFIX', '')}.json").write_text(
        json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    items = items_of("9m")
    pairs = [("tr", i["word"]) for i in items]
    con = sqlite3.connect(BLIND)
    leaked = sum(con.execute(
        "SELECT count(*) FROM entries WHERE lang_code=? AND word=? AND (coalesce(origin,'')!='' OR coalesce(donor_lang,'')!=''"
        " OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!='' OR coalesce(cognates,'')!='')", p).fetchone()[0]
        for p in pairs)
    gb, gf = glosses(BLIND, pairs), glosses(FULL, pairs)
    same_gloss = sum(gb.get(p) == gf.get(p) for p in pairs)
    hints = ("italian", "french", "greek", "arabic", "persian", "armenian", "borrowed", "from ")
    hint = [(p[1], gb.get(p)) for p in pairs if any(k in (gb.get(p) or "").lower() for k in hints)]
    conds = ("prod", "r1", "r2", "r3")
    rb = label_rows(items, gb, conds)
    rf = label_rows(items, gf, conds)
    same = sum(all(a[c + "_w"] == b[c + "_w"] for c in conds) for a, b in zip(rb, rf))
    from engine.db.donor_index import DEFAULT_DB
    pool = [r[0] for r in sqlite3.connect(DEFAULT_DB).execute("SELECT DISTINCT lang_code FROM donor_entries")]
    res = {"n": len(items), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "gloss_with_language_hint": hint, "pool_languages": pool,
           "fetchers_loaded": sorted(m for m in sys.modules if m.startswith("engine.fetchers")),
           "network": "socket.connect engelli", "pass": leaked == 0 and same == len(items)}
    print(json.dumps({k: v for k, v in res.items() if k != "gloss_with_language_hint"}, ensure_ascii=False), len(hint))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_xt():
    from engine.evaluation import xborrowing_eval as xb
    from engine.utils.orthography import to_comparison_form
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
    setc("prod")
    (OUT / "xt_tune.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_saha():
    from engine.evaluation import donor_id_eval
    out = {}
    for c in CONDS:
        dp = setc(c)
        dp.reset_cache()
        r = donor_id_eval.run()
        out[c] = {k: v.get("accuracy") for k, v in r["systems"].items()}
        print(c, out[c], flush=True)
    setc("prod")
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    sys.path.insert(0, str(OUT))
    c = sys.argv[1]
    {"gold": lambda: cmd_gold(sys.argv[2]), "k2": cmd_k2, "xt": cmd_xt, "saha": cmd_saha}[c]()
