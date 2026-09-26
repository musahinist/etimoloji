import json
from math import comb
from engine.evaluation.headline_eval import score_item, split_halves
from engine.search_engine import _index_source_proto
from engine.nlp import derivation
from engine.evaluation.negative_controls import GENERATED_FAKES

def fallback(word, order):
    ana = list(derivation.analyze(word))
    if order == "derin":
        ana.sort(key=lambda a: (len(a.segments[0]), -len(a.suffixes)))
    for a in ana:
        p, _ = _index_source_proto(a.root)
        if p:
            return p, a.formula
    return "", ""

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

d = json.load(open('data/eval/headline.json'))
res = {}
for order in ("sığ", "derin"):
    for name, s in d['subsets'].items():
        half = split_halves(i['word'] for i in s['items'])
        for h in ("A", "B", "all"):
            rows = [i for i in s['items'] if h == "all" or half[i['word']] == h]
            st = {"n": len(rows), "exact": [0, 0], "trad": [0, 0], "win": {"exact": 0, "trad": 0}, "loss": {"exact": 0, "trad": 0}, "changed": []}
            for i in rows:
                new = dict(exact=i['exact'], tradition_equivalent=i['tradition_equivalent'])
                if 'kök belirlenemedi' in i['provenance']:
                    p, f = fallback(i['word'], order)
                    if p:
                        new = score_item(p, i['reference'])
                        st["changed"].append((i['word'], p, f, new['exact'], new['tradition_equivalent']))
                for key, k2 in (("exact", "exact"), ("trad", "tradition_equivalent")):
                    st[key][0] += i[k2]; st[key][1] += new[k2]
                    st["win"][key] += (not i[k2]) and new[k2]; st["loss"][key] += i[k2] and not new[k2]
            for key in ("exact", "trad"):
                st[key + "_p"] = mcnemar(st["win"][key], st["loss"][key])
            res[f"{order}|{name}|{h}"] = st
for k, st in res.items():
    print(k, "n", st["n"], "tam", st["exact"], "p", round(st["exact_p"], 3), "gelenek", st["trad"], "p", round(st["trad_p"], 3), st["changed"] if k.endswith("|all") else "")
for order in ("sığ", "derin"):
    hits = [(w, *fallback(w, order)) for w in GENERATED_FAKES]
    hits = [h for h in hits if h[1]]
    print("sahte", order, len(hits), hits)
