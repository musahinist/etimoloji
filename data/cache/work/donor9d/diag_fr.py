import json, os
os.environ.setdefault("ETY_DONOR_CLEAN", "0"); os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
from collections import Counter
from engine.evaluation.borrowing_eval import _turkish_glosses
from engine.nlp import donor_proximity as dp
from engine.nlp.borrowing_detector import TURKISH_DONORS
from engine.utils.orthography import to_comparison_form
rows = json.load(open("data/cache/work/donor9d/tr_train.json"))
fr = [r for r in rows if r["gold"] == "Fransızca" and r["d1"] != "Fransızca"]
g = _turkish_glosses([r["word"] for r in fr])
idx = dp._index()
kinds = Counter()
for r in fr:
    sense = g.get(r["word"], ""); q = to_comparison_form(r["word"])
    pool = [x for x in idx.by_sense(sense, languages=TURKISH_DONORS, limit=200)]
    frp = [x for x in pool if x["lang_code"] == "fr"]
    langs = Counter(x["lang_code"] for x in pool)
    exact = [x["comparison"] for x in idx._connect().execute("select comparison from donor_entries where lang_code='fr' and comparison=?", (q,)).fetchall()]
    best = dp.best_label(q, [x["comparison"] for x in frp]) if frp else (None, "")
    a = dp.attribute_donor(q, sense, languages=TURKISH_DONORS)
    kind = "fr-havuz-yok" if not frp else ("fr-yakın-ama-null" if best[0] is not None and best[0] <= a.distance + 0.05 else "fr-uzak")
    if not sense: kind = "anlam-yok"
    kinds[kind] += 1
    print(r["word"], "|", sense[:40], "|", r["d1"], r["d1_w"], "| fr", best, "fr_exact_in_db", bool(exact), dict(langs), kind)
print(kinds)
