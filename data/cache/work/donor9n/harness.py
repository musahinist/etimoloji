"""9n — şans düzeyindeki verici etiketinde dürüst gösterim (A1 aile / A2 önsel / A3 önsel-aile).

Yalnız etiket adımı: `attribute_donor` (D1, G2, H1, S1; üretim) bir kez, sonra
`donor_proximity.honest_label(mode)` her koşul için. Kör indeks (xtr/index_blind.db), AĞ KAPALI;
koşu sonunda hiçbir getirici yüklenmemiş olmalı. Etimon referansı (yalnız PUANLAMA):
altının etimon çevriyazısı + TAM indeksin Wiktionary şablon argümanı/etimoloji metni (`diag.py`).

Koşullar: prod (DONOR_HONEST=off) · a1 · a2 · a3.
Ölçütler (PREREG.md):
  (i)  biçim kesinliği = gösterilen biçimlerden doğru etimon olanların payı (referanslı maddeler;
       "doğru etimon" = biçim eşleşmesi VE dil sınıfı altınla aynı);
  (ii) acc_nat = doğal dağılım ağırlıklı dil doğruluğu; aile etiketi altın sınıf ailedeyse 1/2 puan;
  (iii) kapsama = kesin (biçimli) etiket verilen madde payı.

python harness.py gold <altın>     # tr tdk_ayar tettl_ayar | 9n (YENİ rapor: bir kez!)
python harness.py k2               # sızıntı akıl sağlığı (yeni altın; doğruluk yok)
"""
import itertools, json, math, os, random, sqlite3, sys, time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
W = OUT.parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(W / "donor9m"))
import diag  # noqa: E402  (ortamı kurar: kör indeks, ağ kapalı)
from natural import item_weights  # noqa: E402

CONDS = ["prod", "a1", "a2", "a3"]
MODE = {"prod": "off", "a1": "a1", "a2": "a2", "a3": "a3"}
FAMSIZE = 2


def classify_for(gold):
    from engine.evaluation.tr_donor_eval import engine_class
    if gold == "tr":
        return engine_class
    return lambda code: diag.CLS9.get(code) or engine_class(code)


def items_of(gold):
    if gold in ("tr", "tdk_ayar", "tettl_ayar"):
        return [i for i in diag.items_all() if i["set"] == gold]
    if gold == "9n":
        return [dict(i, set="9n") for i in json.loads((OUT / "gold.json").read_text())["items"]]
    raise SystemExit(f"bilinmeyen altın {gold}")


def label_rows(items, blind_gloss=True):
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    assert dp.DONOR_HONEST == "off" and dp.FRENCH_RULE == "g2" and dp.WESTERN_RULE == "h1"
    assert dp.ARABIC_VIA_RULE == "d1" and dp.SENSE_BRIDGE and not dp.FRENCH_ARABIC_GUARD
    gl = diag.glosses(diag.BLIND if blind_gloss else diag.FULL, {i["word"] for i in items})
    full = sqlite3.connect(diag.FULL)
    rows, t = [], time.time()
    for n, it in enumerate(items):
        a = dp.attribute_donor(it["comparison"], gl.get(it["word"], ""), languages=TURKISH_DONORS)
        refs = diag.refs_for(it["word"], full)
        tr = diag.gold_translit(it)
        row = {"word": it["word"], "gold": it["gold"], "has_ref": bool(refs or tr)}
        if a is None:
            row.update({c: None for c in CONDS})
        else:
            row["form"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}/{a.via}"
            row["etym"] = diag.match_class({"word": a.word, "comparison": a.comparison}, refs, tr) == "biçim"
            for c in CONDS:
                h = dp.honest_label(a, it["comparison"], MODE[c])
                row[c] = {"certain": h.certain, "lang": h.lang_code, "family": list(h.family), "basis": h.basis}
        rows.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    bad = sorted(m for m in sys.modules if m.startswith("engine.fetchers"))
    assert not bad, f"getirici yüklendi: {bad}"
    return rows


def score(row, c, classify):
    """(dil puanı, aile doğru mu)."""
    lab = row[c]
    if lab is None:
        return 0.0, False
    if lab["family"]:
        fam = {classify(x) for x in lab["family"]}
        return (1.0 / FAMSIZE if row["gold"] in fam else 0.0), row["gold"] in fam
    ok = classify(lab["lang"]) == row["gold"]
    return float(ok), ok


def shown_correct(row, c, classify):
    """Gösterilen biçim doğru etimon mu? (None: biçim gösterilmedi)"""
    lab = row[c]
    if lab is None or not lab["certain"]:
        return None
    return bool(row.get("etym")) and classify(lab["lang"]) == row["gold"]


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]] kesin Fisher testi, iki yönlü."""
    n1, n2, m1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(n, m1)
    p0 = p(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def signflip(diffs, draws=200000, seed=9):
    d = [x for x in diffs if abs(x) > 1e-15]
    obs = sum(d)
    if not d:
        return {"D": 0.0, "p": 1.0, "discordant": 0}
    tol = 1e-12
    if len(d) <= 20:
        hit = sum(abs(sum(s * abs(x) for s, x in zip(signs, d))) >= abs(obs) - tol
                  for signs in itertools.product((1, -1), repeat=len(d)))
        p = hit / 2 ** len(d)
    else:
        rng = random.Random(seed)
        ab = [abs(x) for x in d]
        hit = sum(abs(sum(x if rng.random() < 0.5 else -x for x in ab)) >= abs(obs) - tol for _ in range(draws))
        p = (hit + 1) / (draws + 1)
    return {"D": round(obs, 5), "p": round(p, 6), "discordant": len(d),
            "a_better": sum(x > 0 for x in d), "b_better": sum(x < 0 for x in d)}


def summarize(rows, classify):
    w = item_weights([r["gold"] for r in rows], "tum")
    n = len(rows)
    out = {"n": n, "classes": dict(Counter(r["gold"] for r in rows)),
           "n_ref": sum(r["has_ref"] for r in rows)}
    ref_rows = [r for r in rows if r["has_ref"]]
    for c in CONDS:
        sc = [score(r, c, classify) for r in rows]
        shown = [shown_correct(r, c, classify) for r in ref_rows]
        s = [x for x in shown if x is not None]
        res = {"acc_nat": round(sum(a * x for a, (x, _) in zip(w, sc)), 4),
               "acc": round(sum(x for x, _ in sc) / n, 4),
               "family_acc": round(sum(f for _, f in sc) / n, 4),
               "precision": round(sum(s) / len(s), 4) if s else None, "shown": len(s), "shown_correct": sum(s),
               "shown_wrong": len(s) - sum(s),
               "coverage": round(sum(1 for r in rows if r[c] is not None and r[c]["certain"]) / n, 4),
               "labelled": round(sum(1 for r in rows if r[c] is not None) / n, 4),
               "per_class": {g: round(sum(x for r, (x, _) in zip(rows, sc) if r["gold"] == g)
                                      / max(1, sum(1 for r in rows if r["gold"] == g)), 4)
                             for g in sorted(out["classes"])}}
        if c != "prod":
            # (i) gizlenen / kalan x doğru etimon (prod'da gösterilen, referanslı maddeler)
            kept = [shown_correct(r, "prod", classify) for r in ref_rows
                    if r["prod"] is not None and r[c]["certain"]]
            hid = [shown_correct(r, "prod", classify) for r in ref_rows
                   if r["prod"] is not None and not r[c]["certain"]]
            res["vs_prod"] = {
                "precision_fisher": {"kept": [sum(kept), len(kept) - sum(kept)],
                                     "hidden": [sum(hid), len(hid) - sum(hid)],
                                     "p": round(fisher_two_sided(sum(kept), len(kept) - sum(kept),
                                                                 sum(hid), len(hid) - sum(hid)), 8)},
                "nat_signflip": signflip([a * (score(r, c, classify)[0] - score(r, "prod", classify)[0])
                                          for a, r in zip(w, rows)])}
        out[c] = res
    return out


def cmd_gold(gold):
    rows = label_rows(items_of(gold))
    res = summarize(rows, classify_for(gold))
    for c in CONDS:
        print(c, json.dumps({k: v for k, v in res[c].items() if k != "per_class"}, ensure_ascii=False), flush=True)
        print("   per_class", res[c]["per_class"], flush=True)
    (OUT / f"res_{gold}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    """Sızıntı: kör indekste köken sütunu yok; etiket + dürüst etiket kör/tam anlamla aynı."""
    items = items_of("9n")
    pairs = [("tr", i["word"]) for i in items]
    con = sqlite3.connect(diag.BLIND)
    leaked = sum(con.execute(
        "SELECT count(*) FROM entries WHERE lang_code=? AND word=? AND (coalesce(origin,'')!='' OR"
        " coalesce(donor_lang,'')!='' OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!=''"
        " OR coalesce(cognates,'')!='')", p).fetchone()[0] for p in pairs)
    gb, gf = diag.glosses(diag.BLIND, [w for _, w in pairs]), diag.glosses(diag.FULL, [w for _, w in pairs])
    same_gloss = sum(gb.get(w) == gf.get(w) for _, w in pairs)
    rb, rf = label_rows(items, True), label_rows(items, False)
    same = sum(all(a.get(k) == b.get(k) for k in ["form", *CONDS]) for a, b in zip(rb, rf))
    from engine.nlp import donor_prior
    model = json.loads(donor_prior.MODEL_PATH.read_text())
    skip = donor_prior.excluded_words()
    in_train = sum(1 for i in items if i["word"].casefold() in skip and i["comparison"] in skip)
    res = {"n": len(items), "blind_label_columns_nonnull": leaked, "gloss_identical_blind_full": same_gloss,
           "labels_identical_blind_full": same, "cue_model_excludes_9n_gold": "data/cache/work/donor9n/gold.json"
           in model["excluded_golds"], "gold_words_in_exclusion": in_train,
           "fetchers_loaded": sorted(m for m in sys.modules if m.startswith("engine.fetchers")),
           "network": "socket.connect engelli"}
    res["pass"] = leaked == 0 and same == len(items) and res["cue_model_excludes_9n_gold"] and in_train == len(items)
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_saha():
    """Saha `eval-donor` (WOLD, Rusça/Moğolca/Evenkçe verici kümesi) her DONOR_HONEST değerinde."""
    from engine.evaluation import donor_id_eval
    from engine.nlp import donor_proximity as dp
    out = {}
    for c in CONDS:
        dp.DONOR_HONEST = MODE[c]
        dp.reset_cache()
        r = donor_id_eval.run()
        out[c] = {k: v.get("accuracy") for k, v in r["systems"].items() if k in ("motor", "yakınlık")}
        print(c, out[c], flush=True)
    dp.DONOR_HONEST = "off"
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_xt():
    """xturkic ayar verici tanıma (9m harness'ının `xt` yolu; `attribute_donor` dil kodu)."""
    from engine.evaluation import xborrowing_eval as xb
    from engine.nlp import donor_proximity as dp
    from engine.utils.orthography import to_comparison_form
    xb.apply_variant("sca")
    rows = xb.load_cache("tune")
    gold = {g["id"]: g for g in xb.load_split("tune")}
    rows = [r for r in rows if r["id"] in gold]
    pred = xb.crossfit(rows)[0]["engine_trained"]
    new = []
    for r in rows:
        r2 = dict(r)
        if r["attributed"] is not None:
            a = dp.attribute_donor(to_comparison_form(r["query"]), gold[r["id"]]["gloss"],
                                   languages=list(gold[r["id"]]["donors"]))
            r2["attributed"] = a.lang_code if a else None
        new.append(r2)
    out = {"prod": {"engine_trained": xb.donor_identification(new, pred)["accuracy"]},
           "turkish_donor_set_items": sum(1 for r in rows if set(gold[r["id"]]["donors"]) == {"ar", "fa", "el", "hy", "fr", "it"})}
    print(out, flush=True)
    (OUT / "xt_tune.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    {"gold": lambda: cmd_gold(sys.argv[2]), "k2": cmd_k2, "saha": cmd_saha, "xt": cmd_xt}[sys.argv[1]]()
