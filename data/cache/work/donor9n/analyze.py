"""9n tanı çözümlemesi (diag_rows.json): doğru-etimon olasılığını ayıran özellikler."""
import json, sys
from collections import Counter
from pathlib import Path
R = json.loads((Path(__file__).parent / "diag_rows.json").read_text())
R = [r for r in R if r["pred"] != "—" and (r["refs"] or r["translit"])]
y = [r["etym"] == "biçim" for r in R]
print("n (biçim gösterilen, referanslı)", len(R), "doğru etimon", sum(y), round(sum(y)/len(R), 3))

def auc(xs, ys):
    pos = [x for x, t in zip(xs, ys) if t]; neg = [x for x, t in zip(xs, ys) if not t]
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))

feats = {"-distance": lambda r: -r["distance"], "-margin": lambda r: -r["margin"],
         "-chance_pct": lambda r: -(r["chance_pct"] if r["chance_pct"] is not None else 1),
         "content_overlap": lambda r: r["content_overlap"], "gap2": lambda r: r["gap2"] or 0,
         "-sense_tokens": lambda r: -r["sense_tokens"], "-pool_size": lambda r: -r["pool_size"],
         "-gloss_len": lambda r: -r["gloss_len"], "len": lambda r: r["len"], "skeleton_eq": lambda r: r["skeleton_eq"],
         "-null": lambda r: -r["null"]}
for k, f in feats.items():
    print(f"AUC {k:18} {auc([f(r) for r in R], y):.3f}")
print("path", {p: (sum(1 for r, t in zip(R, y) if r['path']==p and t), sum(1 for r in R if r['path']==p)) for p in set(r['path'] for r in R)})
print("pool", {p: (sum(1 for r, t in zip(R, y) if r['pool']==p and t), sum(1 for r in R if r['pool']==p)) for p in set(r['pool'] for r in R)})
print("lang", {p: (sum(1 for r, t in zip(R, y) if r['lang']==p and t), sum(1 for r in R if r['lang']==p)) for p in set(r['lang'] for r in R)})
# eşik taraması: distance
for th in (0.10, 0.15, 0.2, 0.25, 0.3, 0.35):
    k = [t for r, t in zip(R, y) if r["distance"] <= th]; o = [t for r, t in zip(R, y) if r["distance"] > th]
    print(f"d<= {th}: kesin {len(k)} doğru {sum(k)} ({sum(k)/max(1,len(k)):.3f}) | üstü {len(o)} doğru {sum(o)} ({sum(o)/max(1,len(o)):.3f})")
for th in (-0.40, -0.35, -0.30, -0.25, -0.2, -0.15, -0.1):
    k = [t for r, t in zip(R, y) if r["margin"] <= th]; o = [t for r, t in zip(R, y) if r["margin"] > th]
    print(f"m<= {th}: kesin {len(k)} doğru {sum(k)} ({sum(k)/max(1,len(k)):.3f}) | üstü {len(o)} doğru {sum(o)} ({sum(o)/max(1,len(o)):.3f})")
# 2B tablo: mesafe x yüzdelik
print("distance bins x doğru oranı")
for lo, hi in ((0, .1), (.1, .15), (.15, .2), (.2, .25), (.25, .3), (.3, .35), (.35, .45), (.45, 1)):
    b = [(r, t) for r, t in zip(R, y) if lo < r["distance"] <= hi or (lo == 0 and r["distance"] == 0)]
    print(f"  ({lo},{hi}] n={len(b):4} doğru={sum(t for _, t in b):4} oran={sum(t for _, t in b)/max(1,len(b)):.3f}",
          "| content_ov=0:", sum(1 for r, _ in b if r['content_overlap'] == 0), sum(t for r, t in b if r['content_overlap'] == 0))
print("\nkural taraması (kesin = koşul): kesin n / doğru / kesinlik ; gizlenen doğru")
def rule(T, co, cp, sk):
    def f(r):
        ok = r["distance"] <= T
        if co: ok = ok and r["content_overlap"] >= 1
        if cp is not None: ok = ok and (r["chance_pct"] is not None and r["chance_pct"] <= cp)
        if sk: ok = ok or (r["skeleton_eq"] and r["distance"] <= 0.35 and r["content_overlap"] >= 1)
        return ok
    return f
for T in (0.15, 0.2, 0.25, 0.3):
    for co in (False, True):
        for cp in (None, 0.0):
            for sk in (False, True):
                f = rule(T, co, cp, sk)
                k = [t for r, t in zip(R, y) if f(r)]
                print(f"T={T} içerik={co:d} şans%={cp} iskelet={sk:d}: kesin {len(k):4} doğru {sum(k):4} kesinlik {sum(k)/max(1,len(k)):.3f} | gizlenen doğru {sum(y)-sum(k)}")
for s in ("tr", "tdk_ayar", "tettl_ayar"):
    f = rule(0.2, True, None, False)
    rs = [(r, t) for r, t in zip(R, y) if r["set"] == s]
    k = [t for r, t in rs if f(r)]
    print(s, "T=.2+içerik", len(rs), sum(t for _, t in rs), "->", len(k), sum(k), round(sum(k)/max(1,len(k)),3))
if len(sys.argv) > 1:
    f = rule(0.25, True, 0.0, True)
    bad = [r for r, t in zip(R, y) if f(r) and not t]
    print("\nkesin ama etimon değil:", len(bad), Counter(r["cat"] for r in bad))
    for r in bad:
        print(f"  {r['set'][:3]} {r['word']:14} gold={r['gold'][:3]} {r['lang']} {r['form']} ({r['form_comp']}) d={r['distance']} | {r['translit'][:2]} {[f for _, f in r['refs']][:3]}")
