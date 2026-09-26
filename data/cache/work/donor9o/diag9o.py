"""9o tanı — kesin (biçimli) etiket verilmeyen maddelerde doğru etimon neden havuza girmiyor?

Yalnız ayar: Türkçe TDK+Nişanyan train+dev, 9j TDK ayar, 9j TETTL ayar. Üretim (4.3.2, a2), kör indeks, ağ kapalı.
Her kesin OLMAYAN madde için sınıf:
  ref_yok        — etimon referansı yok (puanlanamaz)
  anlam_yok      — kör indekste Türkçe maddenin anlamı yok (etiket hiç yok)
  sozluk_yok (b) — doğru etimon (altın dil, referansla biçim eşi) donors.db'de yok
  kopru_yok (a)  — etimon sözlükte var; sorgu anlamı Türkçe kaldı (köprü yok), ortak anlam sözcüğü yok
  anlam_farkli (d) — etimon var, sorgu anlamı İngilizce ama etimonun anlamıyla ortak eşleşme sözcüğü yok
  sinir (a)      — etimon FTS'yle eşleşiyor ama 200'lük havuza girmiyor (sıralamasız LIMIT)
  havuzda_*      — havuzda: secilmedi (başka dil/biçim kazandı) / kesin_degil (seçildi ama kural reddetti)
Ayrıca etimon biçim mesafesi (label_distance; (c) biçim karşılaştırması) ve romanizasyon durumu yazılır.

python data/cache/work/donor9o/diag.py  -> diag_rows.json
"""
import json, re, sqlite3, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from common import CLASS_CODES, diag  # noqa: E402

OUT = Path(__file__).parent
ARS = re.compile(r"[؀-ۿ]")


def etymon_entries(con, codes, refs, translits, query_comp):
    """donors.db'de altın dil(ler)inde referansla 'biçim' eşi olan maddeler."""
    from engine.nlp.donor_proximity import script_skeleton
    from engine.utils.orthography import to_comparison_form
    cands = {}
    q = ",".join("?" * len(codes))
    latin = {to_comparison_form(f) for _, f in refs if not ARS.search(f)} | {to_comparison_form(t) for t in translits}
    latin = {x for x in latin if x} | {query_comp}
    for x in latin:
        # eşitlik + aynı ilk iki harf ve ±2 uzunluk (düzenleme uzaklığı ≤ 0,20 için)
        pre = x[:2]
        for r in con.execute(f"SELECT id, lang_code, word, comparison, gloss FROM donor_entries WHERE from_turkic=0"
                             f" AND lang_code IN ({q}) AND comparison >= ? AND comparison < ? AND length BETWEEN ? AND ?",
                             (*codes, pre, pre + "￿", len(x) - 2, len(x) + 2)):
            cands[r[0]] = r
    if {"ar", "fa"} & set(codes):
        sks = {script_skeleton(f) for _, f in refs if ARS.search(f)}
        sks = {s for s in sks if s}
        if sks:
            for r in con.execute(f"SELECT id, lang_code, word, comparison, gloss FROM donor_entries WHERE from_turkic=0"
                                 f" AND lang_code IN ({q})", codes):
                if ARS.search(r[2]):
                    s = script_skeleton(r[2])
                    if s in sks or s.removeprefix("ال") in sks:
                        cands[r[0]] = r
    out = []
    for r in cands.values():
        if diag.match_class({"word": r[2], "comparison": r[3]}, refs, translits) == "biçim":
            out.append({"id": r[0], "lang": r[1], "word": r[2], "comparison": r[3], "gloss": r[4] or ""})
    return out


def main():
    from engine.db.donor_index import FUNCTION_WORDS, _sense_tokens, sense_tokens_for_match
    from engine.evaluation.tr_donor_eval import engine_class
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.db.sense_bridge import is_english_sense
    assert dp.DONOR_HONEST == "a2"
    items = diag.items_all()
    gl = diag.glosses(diag.BLIND, {i["word"] for i in items})
    full = sqlite3.connect(diag.FULL)
    index = dp._index()
    dcon = sqlite3.connect(str(index.path))
    out, t = [], time.time()
    for n, it in enumerate(items):
        comp = it["comparison"]
        sense = gl.get(it["word"], "")
        a = dp.attribute_donor(comp, sense, languages=TURKISH_DONORS)
        h = dp.honest_label(a, comp) if a else None
        refs = diag.refs_for(it["word"], full)
        tr = diag.gold_translit(it)
        cls = (lambda c: diag.CLS9.get(c) or engine_class(c)) if it["set"] != "tr" else engine_class
        row = {k: it.get(k) for k in ("set", "word", "gold", "comparison")}
        row["sense"] = sense[:100]
        row["certain"] = bool(h and h.certain)
        if a:
            row["form"] = f"{a.lang_code}:{a.word}/{a.comparison}/{a.distance:.3f}"
            row["etym"] = diag.match_class({"word": a.word, "comparison": a.comparison}, refs, tr) == "biçim"
            row["shown_ok"] = row["certain"] and row["etym"] and cls(a.lang_code) == it["gold"]
        codes = CLASS_CODES.get(it["gold"], ())
        bs = dp.bridged_sense(comp, sense) if sense else ""
        row["bridged"] = bs[:100]
        row["bridge_en"] = bool(bs) and is_english_sense(bs)
        if row["certain"]:
            row["cls"] = "kesin"
            out.append(row)
            continue
        ety = etymon_entries(dcon, codes, refs, tr, comp) if codes and (refs or tr) else []
        row["ety"] = [f"{e['lang']}:{e['word']}/{e['comparison']}" for e in ety][:5]
        if ety:
            ds = sorted((dp.label_distance(comp, e["comparison"]), e["comparison"]) for e in ety if e["comparison"])
            row["ety_dist"] = round(ds[0][0], 3) if ds else None
            row["ety_skel"] = any(dp.consonant_skeleton(e["comparison"]) in dp._query_skeletons(comp) for e in ety)
        if not (refs or tr):
            row["cls"] = "ref_yok"
        elif not sense:
            row["cls"] = "anlam_yok" + ("+sozluk_var" if ety else "+sozluk_yok")
        elif not ety:
            row["cls"] = "sozluk_yok"
        else:
            qtok = set(sense_tokens_for_match(bs))
            match = [e for e in ety if qtok & set(_sense_tokens(e["gloss"]))]
            cmatch = [e for e in match if {x for x in qtok & set(_sense_tokens(e["gloss"])) if x not in FUNCTION_WORDS}]
            row["ety_content_overlap"] = bool(cmatch)
            if not match:
                row["cls"] = "anlam_farkli" if row["bridge_en"] else "kopru_yok"
                row["ety_gloss"] = ety[0]["gloss"][:80]
            else:
                pool = index.by_sense(bs, languages=TURKISH_DONORS, limit=200)
                keys = {(r["lang_code"], r["word"]) for r in pool}
                if "fr" in codes:
                    keys |= {(r["lang_code"], r["word"]) for r in index.by_sense(bs, languages=["fr"], limit=200)}
                inpool = [e for e in match if (e["lang"], e["word"]) in keys]
                if not inpool:
                    row["cls"] = "sinir"
                elif row.get("etym") and cls(a.lang_code) == it["gold"]:
                    row["cls"] = "havuzda_kesin_degil"
                else:
                    row["cls"] = "havuzda_secilmedi"
        out.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    (OUT / "diag_rows.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    for s in ("tr", "tdk_ayar", "tettl_ayar", None):
        rs = [r for r in out if s is None or r["set"] == s]
        print(s or "toplam", len(rs), dict(Counter(r["cls"] for r in rs).most_common()))
    print("DONE")


if __name__ == "__main__":
    main()
