"""9o öntarama (yalnız ayar): biçim-öncelikli ikinci arama (C1 taslağı) kesin olmayan maddelerde ne kazandırır?

Aday = TURKISH_DONORS maddeleri, ünsüz iskeleti sorgunun iskeletlerinden biri, label_distance ≤ T,
anlam eşi (varyant). En küçük mesafe seçilir. Çıktı: sim_rows.json (madde başına prod + aday listesi).
"""
import json, pickle, sqlite3, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from common import diag  # noqa: E402

OUT = Path(__file__).parent


def skel_map(dbpath, langs):
    from engine.nlp.donor_proximity import consonant_skeleton
    p = OUT / "skel_map.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    m = defaultdict(list)
    con = sqlite3.connect(dbpath)
    q = ",".join("?" * len(langs))
    for i, lang, w, c, g in con.execute(f"SELECT id, lang_code, word, comparison, gloss FROM donor_entries"
                                        f" WHERE from_turkic=0 AND lang_code IN ({q})", langs):
        s = consonant_skeleton(c)
        if len(s) >= 2:
            m[s].append((lang, w, c, g or ""))
    m = dict(m)
    p.write_bytes(pickle.dumps(m))
    return m


def main():
    from engine.evaluation.tr_donor_eval import engine_class
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.db.sense_bridge import english_sense, ottoman_sense
    t = time.time()
    sm = skel_map(str(dp._index().path), TURKISH_DONORS)
    print("skel", len(sm), f"{time.time() - t:.0f}s", flush=True)
    items = diag.items_all()
    gl = diag.glosses(diag.BLIND, {i["word"] for i in items})
    full = sqlite3.connect(diag.FULL)
    out = []
    for n, it in enumerate(items):
        comp = it["comparison"]
        sense = gl.get(it["word"], "")
        a = dp.attribute_donor(comp, sense, languages=TURKISH_DONORS)
        h = dp.honest_label(a, comp) if a else None
        refs = diag.refs_for(it["word"], full)
        tr = diag.gold_translit(it)
        cls = (lambda c: diag.CLS9.get(c) or engine_class(c)) if it["set"] != "tr" else engine_class
        row = {k: it.get(k) for k in ("set", "word", "gold", "comparison")}
        row["has_ref"] = bool(refs or tr)
        row["prod"] = None if a is None else {
            "certain": h.certain, "lang": h.lang_code, "form": a.word, "d": round(a.distance, 3),
            "ok": cls(h.lang_code) == it["gold"],
            "etym": diag.match_class({"word": a.word, "comparison": a.comparison}, refs, tr) == "biçim"}
        bs = dp.bridged_sense(comp, sense) if sense else ""
        row["sense"] = sense[:120]
        row["bs"] = bs[:120]
        row["bridge_terms"] = english_sense(comp)
        row["ota"] = ottoman_sense(comp)[:120]
        cands = []
        for s in dp._query_skeletons(comp):
            for lang, w, c, g in sm.get(s, ()):
                if abs(len(c) - len(comp)) > max(3, len(comp) // 2):
                    continue
                d = dp.label_distance(comp, c)
                if d > 0.30:
                    continue
                cands.append({"lang": lang, "word": w, "comp": c, "gloss": g[:160], "d": round(d, 3),
                              "cls": cls(lang),
                              "etym": diag.match_class({"word": w, "comparison": c}, refs, tr) == "biçim"})
        cands.sort(key=lambda x: x["d"])
        row["cands"] = cands[:300]
        row["n_cands"] = len(cands)
        out.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    (OUT / "sim_rows.json").write_text(json.dumps(out, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
