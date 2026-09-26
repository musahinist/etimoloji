"""9j — TETTL (Tietze) ayrıştırmasından (raw/tettl_parsed.json) YENİ verici altını + TDK uyumu.

Madde: yakın verici it/fr/el/ar/fa/hy (Yun. = el; EYun./Lat./İng. vb. dışarıda), şüphe
işareti ("?"/"veya" etimondan hemen sonra) yok, cilt 1-9 (her biri denetimde >= %90);
başlığın varyantlarından kör indekste Türkçe anlamı olan ilki; aynı kelime için çelişen
verici -> düşer; TDK+Nişanyan, 9e/9f/9g ve 9j-TDK altınlarındaki kelimeler ve (verici,
etimon) grupları dışarıda. Bölme grup (verici, etimon; Yunanca/Ermenice OCR bozuk -> kelime)
tuz `donor9j-tettl-v1`, ayar 0,35. TETTL etimonları verici havuzuna KONMAZ.

python data/cache/work/donor9j/build_tettl_gold.py
"""
import json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
sys.path.insert(0, str(OUT))
import build_tdk_gold as T  # noqa: E402

SALT = "donor9j-tettl-v1"
CODES = ("it", "fr", "el", "ar", "fa", "hy")
CAP = {"rapor": {"it": 150, "fr": 150, "el": 150, "ar": 100, "fa": 100, "hy": 100},
       "ayar": {"it": 80, "fr": 60, "el": 80, "ar": 60, "fa": 60, "hy": 60}}


def main():
    rows = json.loads((OUT / "raw" / "tettl_parsed.json").read_text())
    words_used, groups_used = T.used_sets()
    tdk = json.loads((OUT / "gold_tdk.json").read_text())["items"]
    words_used |= {i["word"].casefold() for i in tdk}
    groups_used |= {(i["donor"], i["group"]) for i in tdk}
    blind = sqlite3.connect(T.BLIND)
    comp, glang = {}, {}
    for i, w, c in blind.execute("SELECT id, word, comparison FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL"
                                 " AND gloss != '' ORDER BY id"):
        comp.setdefault(w, c)
        glang.setdefault(w, "en" if i < T.EN_MAX else "tr")
    # TDK GTS lisan (tam) — uyum için
    tdk_lang = defaultdict(set)
    for line in open(OUT / "raw" / "gts.json", encoding="utf-8"):
        r = json.loads(line)
        lis = (r.get("lisan") or "").strip()
        if lis and "+" not in lis:
            tdk_lang[T.plain(r["madde"]).casefold()].add(T.LANG.get(lis.split()[0], "diğer"))
    by_word = defaultdict(list)
    drop = Counter()
    for r in rows:
        if r["donor"] not in CODES:
            drop["verici-kapsam-dışı"] += 1
            continue
        if r["doubt"]:
            drop["şüpheli"] += 1
            continue
        w = next((h for h in r["head"] if h in comp or T.plain(h) in comp), None)
        if w is None:
            drop["kör-indekste-anlam-yok"] += 1
            continue
        by_word[w].append(r)
    agree = Counter()
    pool = []
    for w, rs in sorted(by_word.items()):
        codes = {r["donor"] for r in rs}
        if len(codes) != 1:
            drop["çelişen-verici"] += 1
            continue
        code = codes.pop()
        r = rs[0]
        t = tdk_lang.get(T.plain(w).casefold())
        if t and len(t) == 1:
            agree[(code, next(iter(t)))] += 1
        c = comp.get(w) or comp.get(T.plain(w))
        if not re.fullmatch(r"[a-zçğıöşü]{3,}", c or ""):
            drop["biçim"] += 1
            continue
        if w.casefold() in words_used or T.plain(w).casefold() in words_used or c in words_used:
            drop["önceki/TDK-altında-kelime"] += 1
            continue
        ety = r["etymon"] if code in ("it", "fr", "ar", "fa") and re.fullmatch(r"[A-Za-zÀ-ÿāīūṣḍṭẓḥġḫšžčʿʾ'’\-]+", r["etymon"]) else ""
        key = T.to_comparison_form(ety) if ety else c
        if (code, key) in groups_used:
            drop["önceki/TDK-altında-etimon"] += 1
            continue
        split = "ayar" if T.h(f"{SALT}:{code}:{key}") % 1000 < T.AYAR_SHARE * 1000 else "rapor"
        pool.append({"word": w, "comparison": c, "donor": code, "gold": T.CLASS[code], "etymon": ety[:40],
                     "group": key, "gloss_lang": glang.get(w) or glang.get(T.plain(w)), "venetian": r["venetian"], "vol": r["vol"], "split": split})
    avail = Counter(f"{p['split']}/{p['donor']}" for p in pool)
    items = []
    for split, caps in CAP.items():
        for code, cap in caps.items():
            cand = sorted((p for p in pool if p["split"] == split and p["donor"] == code),
                          key=lambda p: T.h(f"{SALT}:pick:{p['word']}"))
            items += cand[:cap]
    items.sort(key=lambda p: (p["split"], p["donor"], p["word"]))
    tot = sum(agree.values())
    same = sum(v for (a, b), v in agree.items() if a == b)
    per = {c: f"{agree[(c, c)]}/{sum(v for (a, _), v in agree.items() if a == c)}" for c in CODES}
    meta = {"salt": SALT, "source": "Tietze TETTL cilt 1-9 (archive.org OCR); TDK+Nişanyan, 9e/9f/9g, 9j-TDK dışı",
            "available": dict(sorted(avail.items())), "dropped": dict(drop),
            "counts": dict(Counter(f"{p['split']}/{p['donor']}" for p in items)),
            "counts_en_gloss": dict(Counter(f"{p['split']}/{p['donor']}" for p in items if p["gloss_lang"] == "en")),
            "venetian": dict(Counter(f"{p['split']}" for p in items if p["venetian"] and p["donor"] == "it")),
            "tdk_agreement": {"n": tot, "same": same, "rate": round(same / max(tot, 1), 4), "per_tettl_class": per,
                              "confusion": {f"{a}->{b}": v for (a, b), v in agree.most_common()}}}
    (OUT / "gold_tettl.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
