"""D8 ayar seçimi (YALNIZ tune, kat dışı) + R4 bileşimli MDE.
Kullanım: tune_select.py  -> d8/tune_select.json"""
import json, math, random, sys
from engine.evaluation import xborrowing_eval as x
x.forbid_model_writes()
W = x.PROJECT_ROOT / "data/cache/work/d8"
base_rows, base = x._variant_predictions("tune", "d8")
ids = [r["id"] for r in base_rows]
def summ(rows, p):
    inh = [r for r in rows if not r["y"]]
    prf = x.prf(rows, p["engine_trained"]).as_dict()
    return {"F": prf["fscore"], "P": prf["precision"], "R": prf["recall"], "acc": prf["accuracy"],
            "spec_inh": round(sum(1 for r in inh if not p["engine_trained"][r["id"]]) / len(inh), 4),
            "ramp_inh": round(sum(x._ramp(r) for r in inh) / len(inh), 4)}
out = {"n": len(base_rows), "n_inh": sum(1 for r in base_rows if not r["y"]), "base": summ(base_rows, base), "cands": {}}
# FP'lerin kaçı rampa (tanı)
fp = [r for r in base_rows if not r["y"] and base["engine_trained"][r["id"]]]
out["base_fp"] = {"n": len(fp), "ramp": sum(x._ramp(r) for r in fp),
                  "close": sum(1 for r in fp if r["signals"].get("verici_yakınlığı", 0) >= 1.0),
                  "phon": sum(1 for r in fp if r["signals"].get("fonotaktik_ihlal", 0) > 0)}
tags = [f"d8@hn{w}:{rule}{':thr' if thr else ''}" for w in (2, 3, 5) for rule in ("ramp", "ramp_phon") for thr in (False, True)]
tags.append("d8pred")
def strat_se(rows_a, a, rows_b, b, nb, ni, it=2000):
    A = {r["id"]: r for r in rows_a}
    bor = [r["id"] for r in rows_b if r["y"]]; inh = [r["id"] for r in rows_b if not r["y"]]
    rng = random.Random(x.BOOTSTRAP_SEED); vals = []
    def F(pick, pred):
        tp = sum(1 for i in pick if pred[i] and i in yset); fp_ = sum(1 for i in pick if pred[i] and i not in yset)
        fn = sum(1 for i in pick if not pred[i] and i in yset)
        return x._f(tp, fp_, fn)
    yset = set(bor)
    for _ in range(it):
        pick = [bor[rng.randrange(len(bor))] for _ in range(nb)] + [inh[rng.randrange(len(inh))] for _ in range(ni)]
        vals.append(F(pick, a) - F(pick, b))
    m = sum(vals) / it
    return round(m, 4), round(math.sqrt(sum((v - m) ** 2 for v in vals) / (it - 1)), 4)
for tag in tags:
    if tag == "d8pred" and not (x.cache_path("tune", "d8pred")).exists():
        continue
    rows, p = x._variant_predictions("tune", tag)
    assert [r["id"] for r in rows] == ids or set(r["id"] for r in rows) == set(ids)
    s = summ(rows, p)
    s["dF"] = x.paired_bootstrap(rows, p["engine_trained"], base_rows, base["engine_trained"], "F")
    inh = [r for r in base_rows if not r["y"]]
    s["fp_mcnemar"] = x.mcnemar_one_sided([base["engine_trained"][r["id"]] for r in inh], [p["engine_trained"][r["id"]] for r in inh])
    s["mde"] = {}
    for label, nb, ni in (("cap300", 1300, 238), ("cap350", 1350, 238)):
        mean, se = strat_se(rows, p["engine_trained"], base_rows, base["engine_trained"], nb, ni)
        s["mde"][label] = {"dF_at_comp": mean, "se": se, "mde": round(2.8016 * se, 4)}
    out["cands"][tag] = s
    print(tag, json.dumps({k: s[k] for k in ("F", "spec_inh", "acc")}), s["dF"]["value"], s["dF"]["ci95"], s["fp_mcnemar"]["before"], s["fp_mcnemar"]["after"], s["mde"], flush=True)
print("base", out["base"], out["base_fp"])
(W / "tune_select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
