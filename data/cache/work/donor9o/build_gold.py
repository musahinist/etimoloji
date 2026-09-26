"""9o — TDK GTS (12. baskı) `lisan` alanından YENİ rapor altını, DOĞAL oranlarda (9n kuralı + 9n altını dışarıda).

9j/9l madde kuralı (tek sözcük, özel ad değil, `lisan` tek dil it/fr/el/ar/fa/hy, aynı başlığın
kayıtları aynı dil, kör indekste anlamı var, karşılaştırma biçimi >= 3 harf) + dışlama:
TDK+Nişanyan (items + disagreements), 9e, 9f, 9g, 9j-TDK (ayar + rapor), 9j-TETTL (ayar + rapor),
9l, 9m, 9n altınlarının KELİMELERİ ve (verici, etimon) GRUPLARI yok; grup başına tek madde.
Kota: N = 600, sınıf payları natural.json `tum` (TDK `lisan` doğal dağılımı) kalan sınıflar
üzerinde yeniden normalleştirilmiş, en büyük kalan yöntemi; mevcut sayıyı aşan sınıf tümüyle alınır
ve fazlası diğerlerine dağıtılmaz (oran korunur). Seçim tuzlu hash sırası (`donor9o-tdk-v1`).
Altın dosyasında yalnız kelime, sınıf, kısa etimon. Tek bölüm `rapor`.

python data/cache/work/donor9o/build_gold.py
"""
import hashlib, json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT.parent / "donor9j"))
import build_tdk_gold as b9j  # noqa: E402
from engine.utils.orthography import to_comparison_form  # noqa: E402

SALT = "donor9o-tdk-v1"
N = 600
NAT = json.loads((OUT.parent / "donor9m" / "natural.json").read_text())["tum"]


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def used():
    words, groups = b9j.used_sets()
    extra = [OUT.parent / "donor9j" / "gold_tdk.json", OUT.parent / "donor9j" / "gold_tettl.json",
             OUT.parent / "donor9l" / "gold.json", OUT.parent / "donor9m" / "gold.json",
             OUT.parent / "donor9n" / "gold.json"]
    for p in extra:
        for i in json.loads(p.read_text())["items"]:
            words |= {i["word"].casefold(), i["comparison"]}
            groups.add((i["donor"], i["group"]))
            if i.get("etymon"):
                groups.add((i["donor"], to_comparison_form(i["etymon"])))
    return words, groups


def main():
    words, groups = used()
    blind = sqlite3.connect(OUT / "index_blind.db")  # kopya (xtr/index_blind.db ile aynı sha)
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
    seen, uniq = set(), []
    for p in sorted(pool, key=lambda p: h(f"{SALT}:pick:{p['word']}")):
        if (p["donor"], p["group"]) in seen:
            drop["aynı-grup"] += 1
            continue
        seen.add((p["donor"], p["group"]))
        uniq.append(p)
    avail = Counter(p["donor"] for p in uniq)
    share = {code: NAT[b9j.CLASS[code]] for code in avail}
    t = sum(share.values())
    raw = {code: N * v / t for code, v in share.items()}
    quota = {code: int(v) for code, v in raw.items()}
    for code in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[: N - sum(quota.values())]:
        quota[code] += 1
    items = []
    for code, q in quota.items():
        items += [p for p in uniq if p["donor"] == code][: min(q, avail[code])]
    items.sort(key=lambda p: (p["donor"], p["word"]))
    meta = {"salt": SALT, "N": N, "source": "TDK GTS v12 lisan; TDK+Nişanyan, 9e/9f/9g, 9j-TDK, 9j-TETTL, 9l, 9m, 9n dışı",
            "shares_tum": {k: round(v / t, 4) for k, v in share.items()}, "quota": quota,
            "available": dict(avail), "dropped": dict(drop), "counts": dict(Counter(p["donor"] for p in items)),
            "counts_en_gloss": dict(Counter(p["donor"] for p in items if p["gloss_lang"] == "en"))}
    (OUT / "gold.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
