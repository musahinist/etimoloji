"""9n ayar: kapı (şans düzeyi) + A1/A2/A3 etiket puanlaması (diag_rows.json)."""
import json, sys
from collections import Counter
from pathlib import Path
OUT = Path(__file__).parent
sys.path.insert(0, str(OUT.parent / "donor9m"))
R = json.loads((OUT / "diag_rows.json").read_text())
FAM = {"Arapça": "if", "Farsça": "if", "Fransızca": "bt", "İtalyanca": "bt", "Yunanca": "ea", "Ermenice": "ea"}
FAMSIZE = Counter(FAM.values())
from natural import item_weights

def certain(r, T=0.25, sk=0.35):
    if r["pred"] == "—":
        return False
    base = r["distance"] <= T and r["content_overlap"] >= 1 and r["chance_pct"] == 0
    ske = r["skeleton_eq"] and r["distance"] <= sk and r["content_overlap"] >= 1
    return base or ske

def score(r, label):
    """label: sınıf adı ya da ('aile', f)."""
    if isinstance(label, tuple):
        return (1 / FAMSIZE[label[1]]) if FAM.get(r["gold"]) == label[1] else 0.0
    return float(label == r["gold"])

def acc_nat(rows, labels):
    a = item_weights([r["gold"] for r in rows], "tum")
    return sum(w * score(r, l) for w, r, l in zip(a, rows, labels))

def run(sets, label_fns):
    for s in sets:
        rows = [r for r in R if r["set"] == s]
        # 9m sınıf eşlemesi (Yunanca/Ermenice ayrı) — tr altını engine_class (Ermenice diğer)
        out = {}
        for name, fn in label_fns.items():
            labs = [fn(r) for r in rows]
            out[name] = round(acc_nat(rows, labs), 4)
        g = [r for r in rows if certain(r)]
        print(s, len(rows), "kesin", len(g), out)

prod = lambda r: r["pred"]
a1 = lambda r: r["pred"] if certain(r) or r["pred"] == "—" or r["pred"] not in FAM else ("aile", FAM[r["pred"]])
always_ar = lambda r: r["pred"] if certain(r) or r["pred"] == "—" else "Arapça"
if __name__ == "__main__":
    run(["tr", "tdk_ayar", "tettl_ayar"], {"prod": prod, "A1": a1, "hep_ar": always_ar})
    rows = [r for r in R if r["pred"] != "—" and not certain(r)]
    print("belirsiz dilimde etiket doğruluğu (ham):", sum(r["pred"] == r["gold"] for r in rows), "/", len(rows))
    print("belirsiz dilimde karışıklık (gold -> pred):", Counter((r["gold"][:3], r["pred"][:3]) for r in rows).most_common(20))
    rows = [r for r in R if certain(r)]
    print("kesin dilimde etiket doğruluğu (ham):", sum(r["pred"] == r["gold"] for r in rows), "/", len(rows))

# --- A2 / A3 (τ taraması) ---
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from engine.nlp import donor_prior
NAME = {"ar": "Arapça", "fa": "Farsça", "fr": "Fransızca", "it": "İtalyanca", "el": "Yunanca", "hy": "Ermenice"}
CODE_FAM = {"ar": "if", "fa": "if", "fr": "bt", "it": "bt", "el": "ea", "hy": "ea"}
A2_LANGS = ["ar", "fa", "fr"]

def a2(tau):
    def f(r):
        if certain(r) or r["pred"] == "—":
            return r["pred"]
        return NAME[donor_prior.best(r["comparison"], A2_LANGS, tau)[0]]
    return f

def a3(tau):
    def f(r):
        if certain(r) or r["pred"] == "—":
            return r["pred"]
        post = donor_prior.posterior(r["comparison"], A2_LANGS, tau)
        mass = Counter()
        for k, v in post.items():
            mass[CODE_FAM[k]] += v
        return ("aile", mass.most_common(1)[0][0])
    return f

if __name__ == "__main__" and "a2" in sys.argv:
    fns = {"prod": prod, "A1": a1}
    for tau in (0.0, 0.25, 0.5, 1.0):
        fns[f"A2 τ={tau}"] = a2(tau)
        fns[f"A3 τ={tau}"] = a3(tau)
    run(["tr", "tdk_ayar", "tettl_ayar"], fns)
