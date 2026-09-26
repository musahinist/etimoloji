"""9o — biçim-öncelikli ikinci arama (C1 / C2 / C3): kesinliği koruyarak kapsamayı artırmak.

Üretim (4.3.2 + 9o ön kayıt modeli): `attribute_donor` (D1, G2, H1, S1) bir kez, sonra
`donor_proximity.honest_label(a, comp, "a2", sense=anlam, form_first=koşul)`. Kör indeks KOPYASI
(donor9o/index_blind.db), AĞ KAPALI; hiçbir getirici yüklenmemeli. Etimon referansı (yalnız PUANLAMA):
altının etimon çevriyazısı + TAM indeks kopyasının Wiktionary şablon argümanı/etimoloji metni (9n `diag`).

Koşullar: prod (DONOR_FORM_FIRST=off) · c1 · c2 · c3. Ölçütler (PREREG.md):
  (i)   biçim kesinliği = gösterilen biçimlerden doğru etimon olanların payı (referanslı maddeler);
  (ii)  acc_nat = doğal dağılım ağırlıklı dil doğruluğu (gösterilen dil);
  (iii) kapsama = kesin (biçimli) etiket verilen madde payı; McNemar (kesin exact binom).

python harness.py gold <altın>     # tr tdk_ayar tettl_ayar | 9o (YENİ rapor: bir kez!)
python harness.py k2               # sızıntı akıl sağlığı (yeni altın; doğruluk yok)
python harness.py saha | xt        # korumalar
"""
import json, math, sqlite3, sys, time
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
W = OUT.parent
sys.path.insert(0, str(OUT))
import common  # noqa: E402  (ortam: kör indeks kopyası, ağ kapalı)
from common import diag  # noqa: E402
sys.path.insert(0, str(W / "donor9n"))
import harness as h9n  # noqa: E402  (fisher, signflip)
from natural import item_weights  # noqa: E402

CONDS = ["prod", "c1", "c2", "c3"]
MODE = {"prod": "off", "c1": "c1", "c2": "c2", "c3": "c3"}


def classify_for(gold):
    return h9n.classify_for("tr" if gold == "tr" else gold)


def items_of(gold):
    if gold in ("tr", "tdk_ayar", "tettl_ayar"):
        return [i for i in diag.items_all() if i["set"] == gold]
    if gold == "9o":
        return [dict(i, set="9o") for i in json.loads((OUT / "gold.json").read_text())["items"]]
    raise SystemExit(f"bilinmeyen altın {gold}")


def label_rows(items, blind_gloss=True):
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    assert dp.DONOR_HONEST == "a2" and dp.DONOR_FORM_FIRST == "off"
    assert dp.FRENCH_RULE == "g2" and dp.WESTERN_RULE == "h1" and dp.ARABIC_VIA_RULE == "d1" and dp.SENSE_BRIDGE
    gl = diag.glosses(diag.BLIND if blind_gloss else diag.FULL, {i["word"] for i in items})
    full = sqlite3.connect(diag.FULL)
    rows, t = [], time.time()
    for n, it in enumerate(items):
        comp, sense = it["comparison"], gl.get(it["word"], "")
        a = dp.attribute_donor(comp, sense, languages=TURKISH_DONORS)
        refs = diag.refs_for(it["word"], full)
        tr = diag.gold_translit(it)
        row = {"word": it["word"], "gold": it["gold"], "has_ref": bool(refs or tr), "sense": bool(sense)}
        row["form"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}/{a.via}" if a else None
        for c in CONDS:
            lab = dp.honest_label(a, comp, "a2", sense=sense, form_first=MODE[c])
            if lab is None:
                row[c] = None
                continue
            f = lab.form or a
            row[c] = {"certain": lab.certain, "lang": lab.lang_code, "family": list(lab.family), "basis": lab.basis,
                      "shown": f"{f.lang_code}:{f.word}/{f.comparison}/{f.distance:.3f}/{f.via}" if lab.certain else "",
                      "etym": bool(lab.certain) and diag.match_class(
                          {"word": f.word, "comparison": f.comparison}, refs, tr) == "biçim"}
        rows.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    bad = sorted(m for m in sys.modules if m.startswith("engine.fetchers"))
    assert not bad, f"getirici yüklendi: {bad}"
    return rows


def score(row, c, classify):
    lab = row[c]
    if lab is None:
        return 0.0
    return float(classify(lab["lang"]) == row["gold"])


def shown_correct(row, c, classify):
    lab = row[c]
    if lab is None or not lab["certain"]:
        return None
    return bool(lab["etym"]) and classify(lab["lang"]) == row["gold"]


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows, classify):
    w = item_weights([r["gold"] for r in rows], "tum")
    n = len(rows)
    out = {"n": n, "classes": dict(Counter(r["gold"] for r in rows)), "n_ref": sum(r["has_ref"] for r in rows),
           "n_sense": sum(r["sense"] for r in rows), "n_attr": sum(r["form"] is not None for r in rows)}
    ref_rows = [r for r in rows if r["has_ref"]]
    cert = lambda r, c: r[c] is not None and r[c]["certain"]  # noqa: E731
    for c in CONDS:
        sc = [score(r, c, classify) for r in rows]
        s = [x for x in (shown_correct(r, c, classify) for r in ref_rows) if x is not None]
        res = {"acc_nat": round(sum(a * x for a, x in zip(w, sc)), 4), "acc": round(sum(sc) / n, 4),
               "precision": round(sum(s) / len(s), 4) if s else None, "shown": len(s), "shown_correct": sum(s),
               "shown_wrong": len(s) - sum(s),
               "coverage": round(sum(cert(r, c) for r in rows) / n, 4),
               "labelled": round(sum(r[c] is not None for r in rows) / n, 4),
               "per_class": {g: round(sum(x for r, x in zip(rows, sc) if r["gold"] == g)
                                      / max(1, sum(1 for r in rows if r["gold"] == g)), 4)
                             for g in sorted(out["classes"])}}
        if c != "prod":
            b = sum(1 for r in rows if cert(r, c) and not cert(r, "prod"))
            d = sum(1 for r in rows if cert(r, "prod") and not cert(r, c))
            added = [shown_correct(r, c, classify) for r in ref_rows if cert(r, c) and not cert(r, "prod")]
            res["vs_prod"] = {
                "coverage_mcnemar": {"gained": b, "lost": d, "p": binom_two_sided(b, d)},
                "added_precision": [sum(added), len(added)],
                "form_changed_in_prod_certain": sum(1 for r in rows if cert(r, "prod") and r[c]["shown"] != r["prod"]["shown"]),
                "nat_signflip": h9n.signflip([a * (score(r, c, classify) - score(r, "prod", classify))
                                              for a, r in zip(w, rows)])}
        out[c] = res
    ps = sorted((out[c]["vs_prod"]["coverage_mcnemar"]["p"], c) for c in CONDS[1:])
    m, prev = len(ps), 0.0
    for i, (p, c) in enumerate(ps):
        prev = max(prev, min(1.0, (m - i) * p))
        out[c]["vs_prod"]["coverage_mcnemar"]["p_holm"] = prev
    return out


def cmd_gold(gold):
    rows = label_rows(items_of(gold))
    res = summarize(rows, classify_for(gold))
    for c in CONDS:
        print(c, json.dumps({k: v for k, v in res[c].items() if k != "per_class"}, ensure_ascii=False), flush=True)
        print("   per_class", res[c]["per_class"], flush=True)
    (OUT / f"res_{gold}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


def cmd_k2():
    """Sızıntı: kör indekste köken sütunu yok; kelime kesişimi 0; etiketler kör/tam anlamla aynı."""
    import build_gold
    items = items_of("9o")
    words, groups = build_gold.used()
    overlap = sum(1 for i in items if i["word"].casefold() in words or i["comparison"] in words)
    pairs = [("tr", i["word"]) for i in items]
    con = sqlite3.connect(diag.BLIND)
    leaked = sum(con.execute(
        "SELECT count(*) FROM entries WHERE lang_code=? AND word=? AND (coalesce(origin,'')!='' OR"
        " coalesce(donor_lang,'')!='' OR coalesce(donor_form,'')!='' OR coalesce(etymology,'')!=''"
        " OR coalesce(cognates,'')!='')", p).fetchone()[0] for p in pairs)
    gb, gf = diag.glosses(diag.BLIND, [w for _, w in pairs]), diag.glosses(diag.FULL, [w for _, w in pairs])
    same_gloss = sum(gb.get(w) == gf.get(w) for _, w in pairs)
    rb, rf = label_rows(items, True), label_rows(items, False)
    keys = ["form", *CONDS]
    same = sum(all((a.get(k) or {}).get("shown") == (b.get(k) or {}).get("shown") if k != "form" else a[k] == b[k]
                   for k in keys) for a, b in zip(rb, rf))
    from engine.nlp import donor_prior
    model = json.loads(donor_prior.MODEL_PATH.read_text())
    skip = donor_prior.excluded_words()
    in_excl = sum(1 for i in items if i["word"].casefold() in skip and i["comparison"] in skip)
    res = {"n": len(items), "word_overlap_previous_golds": overlap, "blind_label_columns_nonnull": leaked,
           "gloss_identical_blind_full": same_gloss, "labels_identical_blind_full": same,
           "cue_model_excludes_9o_gold": "data/cache/work/donor9o/gold.json" in model["excluded_golds"],
           "gold_words_in_exclusion": in_excl,
           "fetchers_loaded": sorted(m for m in sys.modules if m.startswith("engine.fetchers")),
           "network": "socket.connect engelli"}
    res["pass"] = (overlap == 0 and leaked == 0 and same == len(items) and res["cue_model_excludes_9o_gold"]
                   and in_excl == len(items))
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "k2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_saha():
    """Saha `eval-donor` (WOLD; ru/mn/evn verici kümesi) her DONOR_FORM_FIRST değerinde."""
    from engine.evaluation import donor_id_eval
    from engine.nlp import donor_proximity as dp
    out = {}
    for c in CONDS:
        dp.DONOR_FORM_FIRST = MODE[c]
        dp.reset_cache()
        r = donor_id_eval.run()
        out[c] = {k: v.get("accuracy") for k, v in r["systems"].items() if k in ("motor", "yakınlık")}
        print(c, out[c], flush=True)
    dp.DONOR_FORM_FIRST = "off"
    (OUT / "saha.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    {"gold": lambda: cmd_gold(sys.argv[2]), "k2": cmd_k2, "saha": cmd_saha}[sys.argv[1]]()
