"""9n tanı özeti (diag_rows.json -> diag_summary.json)."""
import json, sys
from collections import Counter
from pathlib import Path
OUT = Path(__file__).parent
sys.path.insert(0, str(OUT))
from gate_eval import certain
R = json.loads((OUT / "diag_rows.json").read_text())
res = {}
for s in ("tr", "tdk_ayar", "tettl_ayar", "tümü"):
    rows = [r for r in R if s in ("tümü", r["set"])]
    def cat(r):
        if r["pred"] == "—":
            return "etiket_yok"
        if r["pred"] != r["gold"]:
            return "yanlış_dil"
        if not (r["refs"] or r["translit"]):
            return "doğru_dil_etimon_bilinmiyor"
        return "doğru_etimon" if r["etym"] == "biçim" else "şans_doğru_dil"
    c = Counter(cat(r) for r in rows)
    lab = [r for r in rows if r["pred"] != "—" and (r["refs"] or r["translit"])]
    k = [r for r in lab if certain(r)]; u = [r for r in lab if not certain(r)]
    def dist(rs):
        cc = Counter(cat(r) for r in rs); return {x: cc[x] for x in ("doğru_etimon", "şans_doğru_dil", "yanlış_dil")}
    res[s] = {"n": len(rows), "kategori": dict(c), "etiketli_referanslı": len(lab),
              "kesin": {"n": len(k), **dist(k)}, "belirsiz": {"n": len(u), **dist(u)},
              "şans_doğru_dil_yol": dict(Counter(r["path"] + "/" + r["pool"] for r in lab if cat(r) == "şans_doğru_dil")),
              "şans_doğru_dil_içerik0": sum(1 for r in lab if cat(r) == "şans_doğru_dil" and r["content_overlap"] == 0),
              "şans_etimon_havuzda": sum(1 for r in lab if cat(r) == "şans_doğru_dil" and r["etym_in_pool"])}
    print(s, json.dumps(res[s], ensure_ascii=False))
(OUT / "diag_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
