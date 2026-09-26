"""9j — TDK GTS (12. baskı, ogun/guncel-turkce-sozluk) `lisan` alanından YENİ verici altını.

Kaynak (git-ignored, künye PREREG'de): raw/gts.json. Madde: tek sözcük, özel ad değil,
`lisan` tek dil (+ yok) ve dil it/fr/el(Rumca, Yunanca)/ar/fa/hy; aynı başlığın tüm
kayıtları aynı dili veriyor; TDK+Nişanyan altınında (items+disagreements) YOK; 9e/9f/9g
altınlarındaki kelimeler ve (verici, etimon) grupları YOK; kör indekste (xtr/index_blind.db)
anlamı var (ilk anlam; `gloss_lang` en/tr), karşılaştırma biçimi >= 3 harf. Bölme (verici, etimon) grubuna göre,
tuz `donor9j-tdk-v1`, ayar payı 0,35. Altın dosyasında yalnız kelime, sınıf, kısa etimon.

python data/cache/work/donor9j/build_tdk_gold.py
"""
import hashlib, json, re, sqlite3, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from engine.utils.orthography import to_comparison_form  # noqa: E402

SALT = "donor9j-tdk-v1"
BLIND = OUT.parent / "xtr" / "index_blind.db"
LANG = {"İtalyanca": "it", "Fransızca": "fr", "Rumca": "el", "Yunanca": "el", "Arapça": "ar", "Farsça": "fa",
        "Ermenice": "hy"}
CLASS = {"it": "İtalyanca", "fr": "Fransızca", "el": "Yunanca", "ar": "Arapça", "fa": "Farsça", "hy": "Ermenice"}
CAP = {"rapor": {"it": 200, "fr": 150, "el": 150, "ar": 100, "fa": 100, "hy": 100},
       "ayar": {"it": 200, "fr": 80, "el": 200, "ar": 60, "fa": 60, "hy": 100}}
AYAR_SHARE = 0.35
#: Kör indekste Türkçe maddeler iki kaynaktan: en-Wiktionary (kaikki tr; id <= 115.854, İngilizce anlam)
#: ve Türkçe Vikisözlük (tr_edition; id >= 230.491, Türkçe anlam — çoğu TDK tanımının kopyası).
#: Motor (tr_donor_eval, borrowing_detector) ilk anlamı kullanır (en varsa en). Verici havuzlarının
#: anlamları İngilizce: Türkçe anlamla anlam araması çoğunlukla boş döner (ayar tanısı). Altın
#: üretim koşulunu korur (ilk anlam); `gloss_lang` katman olarak taşınır.
EN_MAX = 200000


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def plain(w):
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn" or c in "̧̇̈̆")


def used_sets():
    """TDK+Nişanyan, 9e, 9f, 9g altınlarının kelimeleri ve (verici, etimon) grupları."""
    gold = json.loads((ROOT / "data/gold/turkish_loanwords.json").read_text())
    words = {r["word"].casefold() for r in gold["items"]}
    words |= {(r.get("word") or "").casefold() for r in gold.get("disagreements") or [] if isinstance(r, dict)}
    groups = set()
    for d in ("donor9e", "donor9f", "donor9g"):
        for i in json.loads((OUT.parent / d / "gold.json").read_text())["items"]:
            words.add(i["word"].casefold())
            if i.get("etymon"):
                groups.add((i["donor"], to_comparison_form(i["etymon"])))
    return words, groups


def main():
    words_used, groups_used = used_sets()
    blind = sqlite3.connect(BLIND)
    comp, glang = {}, {}
    for i, w, c in blind.execute("SELECT id, word, comparison FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL"
                                 " AND gloss != '' ORDER BY id"):
        comp.setdefault(w, c)
        glang.setdefault(w, "en" if i < EN_MAX else "tr")
    recs = defaultdict(set)
    for line in open(OUT / "raw" / "gts.json", encoding="utf-8"):
        r = json.loads(line)
        lisan = (r.get("lisan") or "").strip()
        w = (r.get("madde") or "").strip()
        if not lisan or not w:
            continue
        if " " in w or r.get("ozel_mi") == "1" or w[0].isupper() or "+" in lisan or "," in lisan:
            recs[w].add(None)
            continue
        head, _, ety = lisan.partition(" ")
        code = LANG.get(head)
        recs[w].add((code, ety.strip()) if code else None)
    drop, pool = Counter(), []
    for w, ds in sorted(recs.items()):
        if None in ds or len({d[0] for d in ds}) != 1:
            drop["çok-dilli/uygunsuz"] += 1 if any(d for d in ds if d) else 0
            continue
        code, ety = sorted(ds)[0]
        c = comp.get(w) or comp.get(plain(w))
        if c is None:
            drop["kör-indekste-anlam-yok"] += 1
            continue
        if not re.fullmatch(r"[a-zçğıöşü]{3,}", c or ""):
            drop["biçim"] += 1
            continue
        if w.casefold() in words_used or plain(w).casefold() in words_used or c in words_used:
            drop["önceki-altında-kelime"] += 1
            continue
        key = to_comparison_form(ety) if ety else c
        if (code, key) in groups_used:
            drop["önceki-altında-etimon"] += 1
            continue
        split = "ayar" if h(f"{SALT}:{code}:{key}") % 1000 < AYAR_SHARE * 1000 else "rapor"
        pool.append({"word": w, "comparison": c, "donor": code, "gold": CLASS[code], "etymon": ety[:40],
                     "group": key, "gloss_lang": glang.get(w) or glang.get(plain(w)), "split": split})
    avail = Counter(f"{p['split']}/{p['donor']}" for p in pool)
    items = []
    for split, caps in CAP.items():
        for code, cap in caps.items():
            cand = sorted((p for p in pool if p["split"] == split and p["donor"] == code),
                          key=lambda p: h(f"{SALT}:pick:{p['word']}"))
            items += cand[:cap]
    items.sort(key=lambda p: (p["split"], p["donor"], p["word"]))
    meta = {"salt": SALT, "source": "TDK GTS v12 (ogun/guncel-turkce-sozluk) lisan; TDK+Nişanyan, 9e/9f/9g dışı",
            "available": dict(sorted(avail.items())), "dropped": dict(drop),
            "counts": dict(Counter(f"{p['split']}/{p['donor']}" for p in items)),
            "counts_en_gloss": dict(Counter(f"{p['split']}/{p['donor']}" for p in items if p["gloss_lang"] == "en"))}
    (OUT / "gold_tdk.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
