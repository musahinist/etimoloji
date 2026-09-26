"""sim_rows.json üzerinde C1 varyant taraması (yalnız ayar)."""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/Users/mshn/Documents/etimoloji")
from engine.db.donor_index import FUNCTION_WORDS, _sense_tokens, is_form_of  # noqa: E402

OUT = Path(__file__).parent
rows = json.loads((OUT / "sim_rows.json").read_text())


def ctoks(s):
    return {t for t in _sense_tokens(s) if len(t) > 2 and t not in FUNCTION_WORDS}


def qtoks(r, v):
    base = ctoks(r["bs"])
    if v == "S0":
        return {t for t in [t for t in _sense_tokens(r["bs"]) if len(t) > 2][:6] if t not in FUNCTION_WORDS}
    if v == "S1":
        return base
    s = base | ctoks(r["bridge_terms"]) | ctoks(r["ota"]) | ctoks(r["sense"])
    if v == "S3":
        s |= {r["comparison"]}
    return s


def pick(r, v, T, formof=True, minsk=2):
    q = qtoks(r, v) if v != "SN" else None
    for c in r["cands"]:
        if c["d"] > T:
            break
        if formof and is_form_of(c["gloss"]):
            continue
        if q is not None and not (q & ctoks(c["gloss"])):
            continue
        return c
    return None


def run(v, T, only_uncertain=True, sets=None):
    add = ok = etym_only = wrong_lang = 0
    base_cert = base_ok = 0
    ex = []
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        p = r["prod"]
        if p and p["certain"]:
            base_cert += 1
            base_ok += p["etym"] and p["ok"]
            continue
        if not r["has_ref"]:
            continue
        c = pick(r, v, T)
        if c is None:
            continue
        add += 1
        good = c["etym"] and c["cls"] == r["gold"]
        ok += good
        wrong_lang += c["cls"] != r["gold"]
        if len(ex) < 400:
            ex.append((good, r["set"], r["word"], r["gold"], c["lang"], c["word"], c["comp"], c["d"], c["gloss"][:50]))
    return add, ok, wrong_lang, base_cert, base_ok, ex


if __name__ == "__main__":
    for sets in (["tr"], ["tdk_ayar", "tettl_ayar"], None):
        print("==", sets or "hepsi")
        for v in ("S0", "S1", "S2", "S3", "SN"):
            for T in (0.10, 0.15, 0.20, 0.25):
                add, ok, wl, bc, bo, _ = run(v, T, sets=sets)
                print(f"{v} T={T:.2f} eklenen {add:4d} doğru {ok:4d} ({ok / max(add, 1):.2f}) yanlış_dil {wl:3d}"
                      f" | toplam kesinlik {(bo + ok) / max(bc + add, 1):.3f} (taban {bo / max(bc, 1):.3f}) kesin {bc}+{add}")
    if len(sys.argv) > 1:
        v, T = sys.argv[1], float(sys.argv[2])
        for e in run(v, T)[5]:
            print(e)


def pick_lang(r, v, T, eps=0.05, mode="L1"):
    """Dil seçimi: eşiği geçen adaylar arasında en küçük mesafenin eps yakınındaki diller -> sonsal."""
    from engine.nlp import donor_prior
    q = qtoks(r, v) if v != "SN" else None
    ok = []
    for c in r["cands"]:
        if c["d"] > T:
            break
        if is_form_of(c["gloss"]):
            continue
        if q is not None and not (q & ctoks(c["gloss"])):
            continue
        ok.append(c)
    if not ok:
        return None
    dmin = ok[0]["d"]
    near = [c for c in ok if c["d"] <= dmin + eps]
    langs = sorted({c["lang"] for c in near})
    if mode == "L0" or len(langs) == 1:
        return near[0]
    if mode == "L1":
        post = donor_prior.posterior(r["comparison"], [l for l in langs if l in ("ar", "fa", "fr")] or langs)
    else:  # L2: bütün adaylar (it/el/hy modeli zayıf)
        post = donor_prior.posterior(r["comparison"], langs)
    lang = max(post, key=lambda k: (post[k], k))
    return next(c for c in near if c["lang"] == lang)


def run2(v, T, eps, mode, sets=None):
    add = ok = wl = bc = bo = 0
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        p = r["prod"]
        if p and p["certain"]:
            bc += 1
            bo += p["etym"] and p["ok"]
            continue
        if not r["has_ref"]:
            continue
        c = pick_lang(r, v, T, eps, mode)
        if c is None:
            continue
        add += 1
        good = c["etym"] and c["cls"] == r["gold"]
        ok += good
        wl += c["cls"] != r["gold"]
    return add, ok, wl, bc, bo


def main2():
    for sets in (["tr"], ["tdk_ayar", "tettl_ayar"], None):
        print("==", sets or "hepsi")
        for mode in ("L1", "L2"):
            for v in ("S0", "S2", "S3"):
                for T in (0.10, 0.15, 0.25):
                    for eps in (0.05, 0.10):
                        add, ok, wl, bc, bo = run2(v, T, eps, mode, sets)
                        print(f"{mode} {v} T={T:.2f} eps={eps:.2f} eklenen {add:4d} doğru {ok:4d} ({ok / max(add, 1):.2f})"
                              f" yanlış_dil {wl:3d} | kesinlik {(bo + ok) / max(bc + add, 1):.3f} (taban {bo / max(bc, 1):.3f})")


ORDERS = {"H1": ["fr", "it", "ar", "fa", "el", "hy"], "H2": ["ar", "fr", "it", "fa", "el", "hy"]}


def pick3(r, v, T, eps, mode):
    from engine.nlp import donor_prior
    q = qtoks(r, v) if v != "SN" else None
    F = [c for c in r["cands"] if c["d"] <= T and not is_form_of(c["gloss"])]
    Fs = [c for c in F if q is None or (q & ctoks(c["gloss"]))]
    if not Fs:
        return None
    anchor = Fs[0]
    near = [c for c in F if c["d"] <= anchor["d"] + eps]
    langs = sorted({c["lang"] for c in near})
    if len(langs) == 1:
        lang = langs[0]
    elif mode == "P":
        cand = [l for l in langs if l in ("ar", "fa", "fr")]
        if cand:
            post = donor_prior.posterior(r["comparison"], cand)
            lang = max(post, key=lambda k: (post[k], k))
        else:
            lang = anchor["lang"]
    elif mode == "Pall":
        post = donor_prior.posterior(r["comparison"], langs)
        lang = max(post, key=lambda k: (post[k], k))
    else:
        lang = min(langs, key=ORDERS[mode].index)
    same = [c for c in near if c["lang"] == lang]
    same.sort(key=lambda c: (not (q is None or (q & ctoks(c["gloss"]))), c["d"]))
    return same[0]


def run3(v, T, eps, mode, sets=None, show=False):
    add = ok = wl = bc = bo = 0
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        p = r["prod"]
        if p and p["certain"]:
            bc += 1
            bo += p["etym"] and p["ok"]
            continue
        if not r["has_ref"]:
            continue
        c = pick3(r, v, T, eps, mode)
        if c is None:
            continue
        add += 1
        good = c["etym"] and c["cls"] == r["gold"]
        ok += good
        wl += c["cls"] != r["gold"]
        if show and not good:
            print("  ", r["set"], r["word"], r["gold"], c["lang"], c["word"], c["comp"], c["d"], c["gloss"][:40])
    return add, ok, wl, bc, bo


def main3():
    for sets in (["tr"], ["tdk_ayar", "tettl_ayar"]):
        print("==", sets)
        for mode in ("P", "Pall", "H1", "H2"):
            for v in ("S2", "S3"):
                for T in (0.10, 0.15, 0.25):
                    for eps in (0.0, 0.05, 0.10):
                        add, ok, wl, bc, bo = run3(v, T, eps, mode, sets)
                        print(f"{mode} {v} T={T:.2f} eps={eps:.2f} eklenen {add:4d} doğru {ok:4d} ({ok / max(add, 1):.2f})"
                              f" yanlış_dil {wl:3d} | kesinlik {(bo + ok) / max(bc + add, 1):.3f} (taban {bo / max(bc, 1):.3f})")


def pick4(r, v, T, mode, remap=True):
    from engine.nlp import donor_prior
    from engine.nlp.donor_proximity import script_skeleton
    q = qtoks(r, v) if v != "SN" else None
    post = donor_prior.posterior(r["comparison"], ["ar", "fa", "fr"])
    L = max(post, key=lambda k: (post[k], k))
    F = [c for c in r["cands"] if c["d"] <= T and not is_form_of(c["gloss"])]
    Fs = [c for c in F if q is None or (q & ctoks(c["gloss"]))]
    hit = [c for c in Fs if c["lang"] == L]
    if hit:
        return hit[0]
    if not remap or not Fs:
        return None
    for c in Fs:
        if L == "ar" and c["lang"] == "fa":
            sk = script_skeleton(c["word"])
            m = [x for x in r["cands"] if x["lang"] == "ar" and script_skeleton(x["word"]) == sk and not is_form_of(x["gloss"])]
            if m:
                return m[0]
        if L == "fr" and c["lang"] in ("fa", "el", "hy", "it", "ar") and mode == "fr":
            m = [x for x in F if x["lang"] == "fr" and x["d"] <= c["d"] + 0.02]
            if m:
                return m[0]
    return None


def run4(v, T, mode, remap, sets=None, show=False):
    add = ok = wl = bc = bo = 0
    for r in rows:
        if sets and r["set"] not in sets:
            continue
        p = r["prod"]
        if p and p["certain"]:
            bc += 1
            bo += p["etym"] and p["ok"]
            continue
        if not r["has_ref"]:
            continue
        c = pick4(r, v, T, mode, remap)
        if c is None:
            continue
        add += 1
        good = c["etym"] and c["cls"] == r["gold"]
        ok += good
        wl += c["cls"] != r["gold"]
        if show:
            print("  ", "+" if good else "-", r["set"], r["word"], r["gold"], c["lang"], c["word"], c["comp"], c["d"], c["gloss"][:40])
    return add, ok, wl, bc, bo


def main4():
    for sets in (["tr"], ["tdk_ayar", "tettl_ayar"]):
        print("==", sets)
        for mode, remap in (("x", False), ("x", True), ("fr", True)):
            for v in ("S0", "S2", "S3"):
                for T in (0.10, 0.15, 0.20, 0.25):
                    add, ok, wl, bc, bo = run4(v, T, mode, remap, sets)
                    print(f"{mode}{int(remap)} {v} T={T:.2f} eklenen {add:4d} doğru {ok:4d} ({ok / max(add, 1):.2f})"
                          f" yanlış_dil {wl:3d} | kesinlik {(bo + ok) / max(bc + add, 1):.3f} (taban {bo / max(bc, 1):.3f})")


_TR = {}


def tr_translate(sense):
    """Türkçe anlam metninin içerik sözcükleri -> köprü (tr_en) İngilizce terimleri."""
    from engine.db.sense_bridge import english_sense
    from engine.utils.orthography import to_comparison_form
    out = set()
    for t in _sense_tokens(sense):
        if len(t) < 3:
            continue
        c = to_comparison_form(t)
        if c not in _TR:
            _TR[c] = ctoks(english_sense(c))
        out |= _TR[c]
    return out


_qt_orig = qtoks


def qtoks(r, v):  # noqa: F811
    if v == "S4":
        return _qt_orig(r, "S2") | tr_translate(r["sense"])
    return _qt_orig(r, v)


def main5():
    for sets in (["tr"], ["tdk_ayar", "tettl_ayar"]):
        print("==", sets)
        for v in ("S2", "S4"):
            for T in (0.10, 0.15, 0.20, 0.25):
                add, ok, wl, bc, bo = run4(v, T, "x", True, sets)
                print(f"x1 {v} T={T:.2f} eklenen {add:4d} doğru {ok:4d} ({ok / max(add, 1):.2f})"
                      f" yanlış_dil {wl:3d} | kesinlik {(bo + ok) / max(bc + add, 1):.3f} (taban {bo / max(bc, 1):.3f})")


def pick5(r, v, T, minlen=0, remap=True):
    if len(r["comparison"]) < minlen:
        return None
    c = pick4(r, v, T, "x", remap)
    if c is not None and c["d"] > T:
        return None
    return c


def main6():
    import itertools
    rs = [r for r in rows if r["gold"] in ("Arapça", "Farsça", "Fransızca")]
    for v, T, ml in itertools.product(("S2", "S4"), (0.10, 0.15, 0.20), (0, 4, 5)):
        res = {}
        for s in ("tr", "tj"):
            sub = [r for r in rs if (r["set"] == "tr") == (s == "tr")]
            bc = sum(1 for r in sub if r["prod"] and r["prod"]["certain"])
            bo = sum(1 for r in sub if r["prod"] and r["prod"]["certain"] and r["prod"]["etym"] and r["prod"]["ok"])
            add = ok = 0
            for r in sub:
                if (r["prod"] and r["prod"]["certain"]) or not r["has_ref"]:
                    continue
                c = pick5(r, v, T, ml)
                if c:
                    add += 1
                    ok += c["etym"] and c["cls"] == r["gold"]
            res[s] = f"{s}: n={len(sub)} +{add} ({ok / max(add, 1):.2f}) kesinlik {bo / bc:.3f}->{(bo + ok) / (bc + add):.3f} kapsama {bc / len(sub):.3f}->{(bc + add) / len(sub):.3f}"
        print(v, T, ml, " | ".join(res.values()))
