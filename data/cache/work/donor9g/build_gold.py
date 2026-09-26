"""9g — Rumca/Yunanca ve Ermenice alıntılar için YENİ verici altını (dokunulmamış).

Kaynak: kaikki en dökümleri tr, ota, az, crh (data/lexicons/*.jsonl.gz). Madde:
ilk verici şablonu (bor/bor+/lbor/der/...; tr'de önünde en çok bir `inh|tr|ota`)
el/grc/gkm/pnt (Yunanca ailesi) ya da hy/xcl/axm (Ermenice ailesi); önünde
türetme şablonu yok (9e `donor_of` birebir); aynı (dil, kelime) kayıtları
vericide anlaşıyor; özel ad/ek/deyim değil; karşılaştırma biçimi ≥3 harf;
TDK+Nişanyan altınında (items + disagreements) YOK (kelime ya da karşılaştırma
biçimi); 9e ve 9f altınlarındaki kelimeler ve (aile, etimon) grupları YOK; kör
indekste (xtr/index_blind.db) o dilde anlamı var. Aynı (aile, etimon,
karşılaştırma biçimi) birden çok dilde varsa bir kez (öncelik tr > ota > az > crh).
Bölme etimona göre (tuz `donor9g-v1`, (aile, etimon) grubu): ayar / rapor.
Ağız etiketi (`dialectal`) bilgi olarak taşınır.

python data/cache/work/donor9g/build_gold.py
"""
import gzip, hashlib, json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
W9E = OUT.parent / "donor9e"
sys.path.insert(0, str(W9E))
import build_gold as B  # noqa: E402

ROOT = B.ROOT
SALT = "donor9g-v1"
BLIND = OUT.parent / "xtr" / "index_blind.db"
FAM = {"el": "el", "grc": "el", "gkm": "el", "pnt": "el", "el-kal": "el", "el-pap": "el",
       "hy": "hy", "xcl": "hy", "axm": "hy"}
CLASS = {"el": "Yunanca", "hy": "Ermenice"}
LANGS = ("tr", "ota", "az", "crh")
CAP = {"rapor": {"el": 150, "hy": 150}, "ayar": {"el": 80, "hy": 80}}
AYAR_SHARE = 0.35
BAD_POS = ("name", "suffix", "prefix", "phrase", "proverb", "affix", "abbrev", "symbol", "character",
           "punct", "infix", "interfix", "circumfix", "romanization")


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def raw_code(ety):
    for t in ety:
        if t.get("name") in B.DONOR_T and t.get("args", {}).get("2") in FAM:
            return t["args"]["2"]
    return ""


def main():
    B.FAMILY = FAM
    gold = json.loads((ROOT / "data/gold/turkish_loanwords.json").read_text())
    known = {r["word"].casefold() for r in gold["items"]}
    known |= {(r.get("word") or "").casefold() for r in gold.get("disagreements") or [] if isinstance(r, dict)}
    used = json.loads((W9E / "gold.json").read_text())["items"] + \
        json.loads((OUT.parent / "donor9f" / "gold.json").read_text())["items"]
    used_words = {i["word"].casefold() for i in used}
    used_groups = {(i["donor"], i["etymon"].casefold()) for i in used}
    blind = sqlite3.connect(BLIND)
    comp = {}
    for lang, w, c in blind.execute(
            f"SELECT lang_code, word, comparison FROM entries WHERE lang_code IN ({','.join('?' * len(LANGS))})"
            " AND gloss IS NOT NULL AND gloss != '' ORDER BY id", LANGS):
        comp.setdefault((lang, w), c)
    recs = defaultdict(list)
    dial = defaultdict(bool)
    for lang in LANGS:
        with gzip.open(ROOT / f"data/lexicons/{lang}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                w = (r.get("word") or "").strip()
                if not w or r.get("pos") in BAD_POS or " " in w:
                    continue
                ety = r.get("etymology_templates") or []
                d = B.donor_of(ety) if ety else "none"
                recs[(lang, w)].append((d, raw_code(ety) if d not in (None, "none") else ""))
                tags = set(r.get("tags") or []) | {t for s in r.get("senses") or [] for t in s.get("tags") or []}
                dial[(lang, w)] |= "dialectal" in tags
    drop = Counter()
    pool, seen = [], set()
    for (lang, w), ds in sorted(recs.items(), key=lambda kv: (LANGS.index(kv[0][0]), kv[0][1])):
        ds2 = {d for d, _ in ds if d != "none"}
        if not ds2 or None in ds2:
            continue
        fams = {d[0] for d in ds2}
        if len(fams) != 1:
            drop["kayıtlar-çelişik"] += 1
            continue
        fam, etymon = sorted(ds2)[0]
        etymon = etymon or w
        c = comp.get((lang, w))
        if c is None:
            drop["anlam-yok"] += 1
            continue
        if not re.fullmatch(r"[a-zçğıöşüâîûəxq]{3,}", c or ""):
            drop["biçim"] += 1
            continue
        if w.casefold() in known or c in known:
            drop["tdk-nişanyan-altınında"] += 1
            continue
        if w.casefold() in used_words or c in used_words or (fam, etymon.casefold()) in used_groups:
            drop["9e/9f-altınında"] += 1
            continue
        key = (fam, etymon.casefold(), c)
        if key in seen:
            drop["dil-arası-yinelenen"] += 1
            continue
        seen.add(key)
        split = "ayar" if h(f"{SALT}:{fam}:{etymon.casefold()}") % 1000 < AYAR_SHARE * 1000 else "rapor"
        codes = sorted({rc for _, rc in ds if rc})
        pool.append({"lang": lang, "word": w, "comparison": c, "donor": fam, "gold": CLASS[fam],
                     "etymon": etymon, "donor_code": codes[0] if codes else fam,
                     "dialectal": dial[(lang, w)], "split": split})
    avail = Counter(f"{p['split']}/{p['donor']}" for p in pool)
    items = []
    for split, caps in CAP.items():
        for fam, cap in caps.items():
            cand = sorted((p for p in pool if p["split"] == split and p["donor"] == fam),
                          key=lambda p: h(f"{SALT}:pick:{p['lang']}:{p['word']}"))
            items += cand[:cap]
    items.sort(key=lambda p: (p["split"], p["donor"], p["lang"], p["word"]))
    meta = {"salt": SALT, "source": "kaikki en tr/ota/az/crh; 9e+9f altını ve TDK+Nişanyan dışı",
            "available": dict(sorted(avail.items())), "dropped": dict(drop),
            "counts": dict(Counter(f"{p['split']}/{p['donor']}" for p in items)),
            "by_lang": dict(Counter(f"{p['split']}/{p['donor']}/{p['lang']}" for p in items)),
            "by_code": dict(Counter(f"{p['donor']}/{p['donor_code']}" for p in items)),
            "dialectal": dict(Counter(f"{p['split']}/{p['donor']}" for p in items if p["dialectal"]))}
    (OUT / "gold.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
