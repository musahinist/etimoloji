"""9m tanı: base (G2/H1 kapalı) doğru, prod yanlış Arapça maddeler ve nedeni (res_<altın>.json'dan)."""
import json, sys
from pathlib import Path
OUT = Path(__file__).parent


def cause(r):
    if r["g1"] != "Arapça":
        return "G2 ayrı Fransızca havuzu (kazanan doğrudan havuzdan)"
    if r["g2"] != "Arapça":
        return "G2 Fransızca aracılı (_french_via)"
    if r["h1"] != "Arapça":
        return "H1 soneki"
    return "G2+H1 etkileşimi"


res = {}
for g in sys.argv[1:]:
    rows = json.loads((OUT / f"res_{g}.json").read_text())["rows"]
    lost = [r for r in rows if r["gold"] == "Arapça" and r["base"] == "Arapça" and r["prod"] != "Arapça"]
    gain = [r for r in rows if r["gold"] == "Arapça" and r["base"] != "Arapça" and r["prod"] == "Arapça"]
    res[g] = {"n_ar": sum(r["gold"] == "Arapça" for r in rows), "lost": [
        {"word": r["word"], "base": r["base_w"], "prod": r["prod_w"], "cause": cause(r),
         "r2": r["r2_w"]} for r in lost], "gained": [r["word"] for r in gain]}
    print(f"== {g}: Arapça {res[g]['n_ar']}, kayıp {len(lost)}, kazanç {len(gain)} {res[g]['gained']}")
    for x in res[g]["lost"]:
        print(f"  {x['word']:14} base {x['base']:40} prod {x['prod']:40} | {x['cause']}")
(OUT / "diag_ar.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
