"""9m — doğal dağılım ağırlıklı (sınıf-ağırlıklı) doğruluk + eşleştirilmiş ağırlıklı işaret testi.

Doğal ağırlık w_c: TDK GTS `lisan` sayımı (natural.json; `tum` birincil, `altin` duyarlılık).
Bir altında: acc_nat = Σ_c w_c · duyarlılık_c / Σ_c w_c (yalnız altında bulunan sınıflar; altında
"Ermenice" sınıfı yoksa Ermenice payı "diğer"e katılır). Madde ağırlığı a_i = w_c / (n_c · Σ w),
acc_nat = Σ a_i · isabet_i. Test: iki koşulun farkı D = Σ a_i (isabet_A − isabet_B); H0 altında
uyuşmayan maddelerin işaretleri değiştirilebilir -> kesin (≤ 20 uyuşmazlık) ya da 200.000 örnekli
(tohum 9) işaret çevirme, iki yönlü p = P(|D*| ≥ |D|).
"""
import itertools, json, random
from collections import Counter
from pathlib import Path

NAT = json.loads((Path(__file__).parent / "natural.json").read_text())


def weights(classes, kind="tum"):
    base = dict(NAT[kind])
    if "Ermenice" not in classes:
        base["diğer"] = base.get("diğer", 0) + base.pop("Ermenice", 0)
    w = {c: base.get(c, 0.0) for c in classes}
    t = sum(w.values())
    return {c: v / t for c, v in w.items()}


def item_weights(golds, kind="tum"):
    n = Counter(golds)
    w = weights(sorted(n), kind)
    return [w[g] / n[g] for g in golds]


def nat_acc(rows, cond, kind="tum"):
    a = item_weights([r["gold"] for r in rows], kind)
    return sum(x for x, r in zip(a, rows) if r[cond] == r["gold"])


def signflip(rows, ca, cb, kind="tum", draws=200000, seed=9):
    a = item_weights([r["gold"] for r in rows], kind)
    d = [x * ((r[ca] == r["gold"]) - (r[cb] == r["gold"])) for x, r in zip(a, rows)]
    d = [x for x in d if x != 0]
    obs = sum(d)
    if not d:
        return {"D": 0.0, "p": 1.0, "discordant": 0}
    tol = 1e-12
    if len(d) <= 20:
        tot = hit = 0
        for signs in itertools.product((1, -1), repeat=len(d)):
            tot += 1
            hit += abs(sum(s * abs(x) for s, x in zip(signs, d))) >= abs(obs) - tol
        p = hit / tot
    else:
        rng = random.Random(seed)
        ab = [abs(x) for x in d]
        hit = sum(abs(sum(x if rng.random() < 0.5 else -x for x in ab)) >= abs(obs) - tol for _ in range(draws))
        p = (hit + 1) / (draws + 1)
    return {"D": round(obs, 5), "p": round(p, 6), "discordant": len(d),
            "a_only": sum(x > 0 for x in d), "b_only": sum(x < 0 for x in d)}


def summarize(rows, conds, ref="prod"):
    from engine.evaluation.significance import mcnemar_test
    classes = sorted({r["gold"] for r in rows})
    n = len(rows)
    out = {"n": n, "classes": dict(Counter(r["gold"] for r in rows)),
           "weights_tum": {k: round(v, 4) for k, v in weights(classes, "tum").items()},
           "weights_altin": {k: round(v, 4) for k, v in weights(classes, "altin").items()}}
    for c in conds:
        hits = [r[c] == r["gold"] for r in rows]
        per = {k: f"{sum(1 for r in rows if r['gold'] == k and r[c] == k)}/{sum(1 for r in rows if r['gold'] == k)}"
               for k in classes}
        rec = {k: round(sum(1 for r in rows if r['gold'] == k and r[c] == k) / max(1, sum(1 for r in rows if r['gold'] == k)), 4)
               for k in classes}
        res = {"acc": round(sum(hits) / n, 4), "acc_nat": round(nat_acc(rows, c, "tum"), 4),
               "acc_nat_altin": round(nat_acc(rows, c, "altin"), 4), "hits": sum(hits),
               "per_class": per, "recall": rec,
               "errors": dict(Counter(f"{r['gold']}->{r[c]}" for r in rows if r[c] != r["gold"]).most_common(12))}
        if c != ref and ref in conds:
            res["vs_" + ref] = {"mcnemar": mcnemar_test(hits, [r[ref] == r["gold"] for r in rows]).as_dict(),
                                "nat_signflip": signflip(rows, c, ref, "tum")}
        out[c] = res
    return out
