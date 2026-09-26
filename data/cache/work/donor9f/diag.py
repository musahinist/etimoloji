"""9f tanı: 9e ayar bölümünde g2 altında Fransızca->İtalyanca hataları ve fr/it adaylarının mesafeleri."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "donor9e"))
W9E = Path(__file__).resolve().parents[1] / "donor9e"
os.environ["ETY_LEXICON_INDEX"] = str(W9E / "index_blind.db")
os.environ.setdefault("ETY_DONOR_CLEAN", "0"); os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
import harness as H
from engine.nlp import donor_proximity as dp
from engine.nlp.borrowing_detector import TURKISH_DONORS
from engine.utils.orthography import to_comparison_form
dp.FRENCH_RULE = "g2"
cases = H.gold_cases("ayar")
gl = H.glosses(W9E / "index_blind.db", [w for w, _ in cases])
out = []
for w, g in cases:
    c = to_comparison_form(w)
    a = dp.attribute_donor(c, gl.get(w, ""), languages=TURKISH_DONORS)
    if not a: out.append({"word": w, "gold": g, "pred": None}); continue
    alts = {l: (round(d, 3), round(n, 3)) for l, d, n in a.alternatives}
    out.append({"word": w, "cmp": c, "gold": g, "pred": a.lang_code, "form": a.comparison, "d": round(a.distance, 3),
                "null": round(a.null_distance, 3), "via": a.via, "alts": alts})
json.dump(out, open(Path(__file__).parent / "diag_ayar.json", "w"), ensure_ascii=False, indent=0)
for r in out:
    if r["gold"] == "Fransızca" and r.get("pred") == "it":
        print(r["word"], r["form"], r["d"], r["null"], "fr:", r["alts"].get("fr"))
