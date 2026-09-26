"""9l — TDK GTS (12. baskı) `lisan` alanından YENİ rapor altını (9j'nin kullanmadığı maddeler).

9j `build_tdk_gold.py` ile aynı madde kuralı (tek sözcük, özel ad değil, `lisan` tek dil
it/fr/el/ar/fa/hy, aynı başlığın kayıtları aynı dil, kör indekste anlamı var, karşılaştırma
biçimi >= 3 harf) + dışlama: TDK+Nişanyan (items + disagreements), 9e, 9f, 9g, 9j-TDK (ayar +
rapor), 9j-TETTL (ayar + rapor) altınlarının KELİMELERİ ve (verici, etimon) GRUPLARI yok.
Tek bölüm `rapor` (seçim/doğrulama 9j TDK ayarında). Tavan: it/el/hy tümü, fr 150, ar 130,
fa 130; seçim tuzlu hash sırası (`donor9l-tdk-v1`). Altın dosyasında yalnız kelime, sınıf,
kısa etimon. Kaynak: ../donor9j/raw/gts.json (git-ignored; künye 9j PREREG).

python data/cache/work/donor9l/build_gold.py
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT.parent / "donor9j"))
import build_tdk_gold as b9j  # noqa: E402
from engine.utils.orthography import to_comparison_form  # noqa: E402

SALT = "donor9l-tdk-v1"
CAP = {"it": 10**6, "el": 10**6, "hy": 10**6, "fr": 150, "ar": 130, "fa": 130}


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def main():
    words, groups = b9j.used_sets()
    for g in ("gold_tdk", "gold_tettl"):
        for i in json.loads((OUT.parent / "donor9j" / f"{g}.json").read_text())["items"]:
            words |= {i["word"].casefold(), i["comparison"]}
            groups.add((i["donor"], i["group"]))
            if i.get("etymon"):
                groups.add((i["donor"], to_comparison_form(i["etymon"])))
    # 9j kuralıyla havuzu yeniden üret (b9j.main'in iç mantığı; yazmadan)
    import sqlite3
    from collections import defaultdict
    blind = sqlite3.connect(b9j.BLIND)
    comp, glang = {}, {}
    for i, w, c in blind.execute("SELECT id, word, comparison FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL"
                                 " AND gloss != '' ORDER BY id"):
        comp.setdefault(w, c)
        glang.setdefault(w, "en" if i < b9j.EN_MAX else "tr")
    recs = defaultdict(set)
    for line in open(OUT.parent / "donor9j" / "raw" / "gts.json", encoding="utf-8"):
        r = json.loads(line)
        lisan = (r.get("lisan") or "").strip()
        w = (r.get("madde") or "").strip()
        if not lisan or not w:
            continue
        if " " in w or r.get("ozel_mi") == "1" or w[0].isupper() or "+" in lisan or "," in lisan:
            recs[w].add(None)
            continue
        head, _, ety = lisan.partition(" ")
        code = b9j.LANG.get(head)
        recs[w].add((code, ety.strip()) if code else None)
    import re
    drop, pool = Counter(), []
    for w, ds in sorted(recs.items()):
        if None in ds or len({d[0] for d in ds}) != 1:
            continue
        code, ety = sorted(ds)[0]
        c = comp.get(w) or comp.get(b9j.plain(w))
        if c is None:
            drop["kör-indekste-anlam-yok"] += 1
            continue
        if not re.fullmatch(r"[a-zçğıöşü]{3,}", c or ""):
            drop["biçim"] += 1
            continue
        if w.casefold() in words or b9j.plain(w).casefold() in words or c in words:
            drop["önceki-altında-kelime"] += 1
            continue
        key = to_comparison_form(ety) if ety else c
        if (code, key) in groups or (code, c) in groups:
            drop["önceki-altında-etimon"] += 1
            continue
        pool.append({"word": w, "comparison": c, "donor": code, "gold": b9j.CLASS[code], "etymon": ety[:40],
                     "group": key, "gloss_lang": glang.get(w) or glang.get(b9j.plain(w)), "split": "rapor"})
    # aynı grup bir kez (grup içi bağımlılık yok)
    seen, uniq = set(), []
    for p in sorted(pool, key=lambda p: h(f"{SALT}:pick:{p['word']}")):
        if (p["donor"], p["group"]) in seen:
            drop["aynı-grup"] += 1
            continue
        seen.add((p["donor"], p["group"]))
        uniq.append(p)
    avail = Counter(p["donor"] for p in uniq)
    items = []
    for code, cap in CAP.items():
        items += [p for p in uniq if p["donor"] == code][:cap]
    items.sort(key=lambda p: (p["donor"], p["word"]))
    meta = {"salt": SALT, "source": "TDK GTS v12 lisan; TDK+Nişanyan, 9e/9f/9g, 9j-TDK, 9j-TETTL dışı",
            "available": dict(avail), "dropped": dict(drop), "counts": dict(Counter(p["donor"] for p in items)),
            "counts_en_gloss": dict(Counter(p["donor"] for p in items if p["gloss_lang"] == "en"))}
    (OUT / "gold.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
