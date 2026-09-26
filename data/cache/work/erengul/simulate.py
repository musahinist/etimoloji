"""Ön kayıtlı simülasyon (PREREG.md): 'kök yok' başlığına Eren kökü.

Veri: commit e42af76'daki data/eval/headline.json (koşu ebb1c62). Gülensoy
ayrıştırma denetiminde 14/30 (< 0,90) — ölçüme alınmaz; yalnız bilgi sütunu.
"""
import json
import re
import subprocess
import sys
import unicodedata
from math import comb

from engine.db.eren_gulensoy import clean_root, load_index
from engine.evaluation.headline_eval import score_item, split_halves
from engine.evaluation.negative_controls import GENERATED_FAKES
from engine.utils.proto_notation import same_root_across_traditions

DATA = json.loads(subprocess.run(["git", "show", "e42af76:data/eval/headline.json"],
                                 capture_output=True, text=True, check=True).stdout)
INDEX = load_index()
CLASS23 = set("besim bürgü büz dala darga duma evin göbelek güce koçkar obuz savur sümek sümter tansık "
              "tatu toru tumağan tın uçarı yaprağı çaput öke".split())


def to_starling(form: str) -> str:
    """PREREG gösterim denkliği (Türkiye Türkolojisi -> Starling/EDAL)."""
    f = unicodedata.normalize("NFC", clean_root(form)).replace(":", "")
    f = f.replace("ğ", "g").replace("ñ", "ŋ").replace("é", "e")
    f = f.translate(str.maketrans("âîû", "aiu"))
    f = re.sub(r"ng(?=$|[^aeıioöuü])", "ŋ", f)
    return "*" + f


def records(word, sources):
    out = []
    for src in sources:
        recs = [r for r in INDEX.get(word, []) if r["source"] == src and not r.get("redirect")]
        recs.sort(key=lambda r: r["key"] != word)  # tam anahtar önce, fiil mastarı sonra
        out += recs
    return out


def source_form(word, sources, mode):
    recs = records(word, sources)
    usable = [r for r in recs if r["origin"] not in ("loan", "unknown")]
    for r in usable:
        if r["root"] and clean_root(r["root"]):
            return r, clean_root(r["root"]), "kök"
    if mode == "RO":
        for r in usable:
            if r["ot_form"] and clean_root(r["ot_form"]):
                return r, clean_root(r["ot_form"]), "OT"
    return None, "", ""


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def simulate(sources, mode, subset, half_name):
    s = DATA["subsets"][subset]
    half = split_halves(i["word"] for i in s["items"])
    rows = [i for i in s["items"] if half_name == "all" or half[i["word"]] == half_name]
    st = {"n": len(rows), "exact": [0, 0], "trad": [0, 0], "win": {"exact": 0, "trad": 0},
          "loss": {"exact": 0, "trad": 0}, "changed": []}
    for i in rows:
        new = {"exact": i["exact"], "tradition_equivalent": i["tradition_equivalent"]}
        if "kök belirlenemedi" in i["provenance"]:
            rec, form, kind = source_form(i["word"], sources, mode)
            if form:
                new = score_item(to_starling(form), i["reference"])
                st["changed"].append((i["word"], rec["source"], kind, form, to_starling(form), i["reference"],
                                      i["exact"], new["exact"], i["tradition_equivalent"],
                                      new["tradition_equivalent"]))
        for key, k2 in (("exact", "exact"), ("trad", "tradition_equivalent")):
            st[key][0] += i[k2]
            st[key][1] += new[k2]
            st["win"][key] += (not i[k2]) and new[k2]
            st["loss"][key] += i[k2] and not new[k2]
    for key in ("exact", "trad"):
        st[key + "_p"] = mcnemar(st["win"][key], st["loss"][key])
    return st


def fmt(st):
    return (f"n={st['n']} tam {st['exact'][0]}->{st['exact'][1]} (+{st['win']['exact']} -{st['loss']['exact']}, "
            f"p={st['exact_p']:.3g}) gelenek {st['trad'][0]}->{st['trad'][1]} (+{st['win']['trad']} "
            f"-{st['loss']['trad']}, p={st['trad_p']:.3g})")


def main():
    result = {"data_commit": DATA["commit"]}
    print("koşu", DATA["commit"])
    # seçim (A, Starling kapalı): Eren tek kaynak (Gülensoy denetimde elendi) -> yalnız M
    sel = {}
    for mode in ("R", "RO"):
        st = simulate(("eren",), mode, "starling/starling_yok", "A")
        sel[mode] = st
        print("SEÇİM A", mode, fmt(st))
    best = max(("R", "RO"), key=lambda m: (sel[m]["trad"][1], sel[m]["exact"][1], m == "R"))
    print("seçilen M =", best)
    result["selection"] = {m: {k: v for k, v in st.items() if k != "changed"} for m, st in sel.items()}
    result["chosen_mode"] = best
    rep = {}
    for subset in DATA["subsets"]:
        for h in ("A", "B", "all"):
            st = simulate(("eren",), best, subset, h)
            rep[f"{subset}|{h}"] = st
            print(f"{subset:28} {h:3} {fmt(st)}")
    result["report"] = rep
    # sınıflar (Starling kapalı, iki yarı)
    noroot = [i for i in DATA["subsets"]["starling/starling_yok"]["items"] if "kök belirlenemedi" in i["provenance"]]
    for name, items in (("kök_yok", noroot), ("yalnız_starling_23", [i for i in noroot if i["word"] in CLASS23])):
        cov = [(i["word"], *source_form(i["word"], ("eren",), best)[1:]) for i in items]
        cov = [c for c in cov if c[1]]
        ok = [(w, f, score_item(to_starling(f), next(i for i in items if i["word"] == w)["reference"])) for w, f, _ in cov]
        print(f"{name}: n={len(items)} kapsam {len(cov)} tam {sum(o[2]['exact'] for o in ok)} "
              f"gelenek {sum(o[2]['tradition_equivalent'] for o in ok)}  {[(w, f) for w, f, _ in cov]}")
        result[name] = {"n": len(items), "covered": [(w, f, o["exact"], o["tradition_equivalent"]) for w, f, o in ok]}
    # bağımsızlık: 240 kelimede Starling başvurusuyla uyuşma (kaynak başına, M=R: yalnız açık kök)
    for src in ("eren", "gulensoy"):
        rows = []
        for i in DATA["subsets"]["starling/yerel/tümü"]["items"]:
            rec, form, kind = source_form(i["word"], (src,), "R")
            if form:
                sc = score_item(to_starling(form), i["reference"])
                rows.append((i["word"], form, i["reference"], sc["exact"], sc["tradition_equivalent"]))
        agree = sum(r[4] for r in rows)
        print(f"uyuşma {src}: kök verilen {len(rows)}/240, Starling ile gelenek-denk {agree} "
              f"({agree / max(1, len(rows)):.3f}); ayrışanlar {[(r[0], r[1], r[2][0]) for r in rows if not r[4]]}")
        result[f"agreement_{src}"] = rows
    fakes = [(w, source_form(w, ("eren",), best)[1]) for w in GENERATED_FAKES]
    fakes = [f for f in fakes if f[1]]
    print("sahte kelimeye kök:", len(fakes), fakes)
    result["fakes"] = fakes
    json.dump(result, open("data/cache/work/erengul/analysis.json", "w"), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
