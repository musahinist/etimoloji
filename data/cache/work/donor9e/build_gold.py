"""9e — Türkçe Wiktionary alıntı kayıtlarından YENİ verici altını (dokunulmamış).

Kaynak: data/lexicons/tr.jsonl.gz (kaikki en, tr). Madde: ilk verici şablonu
(bor/bor+/lbor/der; önünde en çok `inh|tr|ota`) fr/ar/fa/it/el ailesinden;
önünde türetme şablonu (af/suffix/prefix/compound/blend/cal/…) yok; aynı
kelimenin kayıtları vericide anlaşıyor; özel ad/ek/deyim değil; TDK+Nişanyan
altınında (items + disagreements, TÜM bölümler) YOK; kör indekste Türkçe anlamı
var. Bölme etimona göre (tuz `donor9e-v1`): ayar / rapor. Dengeli örnekleme
(sınıf başına tavan, hash sırası).

python data/cache/work/donor9e/build_gold.py
"""
import gzip, hashlib, json, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
SALT = "donor9e-v1"
FAMILY = {"fr": "fr", "frm": "fr", "ar": "ar", "fa": "fa", "fa-cls": "fa",
          "it": "it", "vec": "it", "el": "el", "gkm": "el"}
CLASS = {"fr": "Fransızca", "ar": "Arapça", "fa": "Farsça", "it": "İtalyanca", "el": "Yunanca"}
DONOR_T = {"bor", "bor+", "lbor", "der", "der+", "ubor", "uder"}
BLOCK_T = {"af", "affix", "suffix", "prefix", "confix", "compound", "com", "blend", "cal", "calque",
           "sl", "semantic loan", "pcal", "psm", "clipping", "back-form", "univerbation", "surf", "l"}
# Rapor (≥300) ve ayar tavanları, sınıf başına.
CAP = {"rapor": {"fr": 200, "ar": 160, "fa": 100, "it": 50, "el": 50},
       "ayar": {"fr": 100, "ar": 80, "fa": 50, "it": 30, "el": 30}}
AYAR_SHARE = 0.35


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def donor_of(templates):
    seen_ota = False
    for t in templates:
        name, args = t.get("name", ""), t.get("args", {})
        if name == "inh" and args.get("2") == "ota" and not seen_ota:
            seen_ota = True
            continue
        if name in ("der", "bor", "bor+", "lbor", "der+", "ubor", "uder") and args.get("2") == "ota" and not seen_ota:
            seen_ota = True
            continue
        if name in BLOCK_T:
            return None
        if name in DONOR_T:
            fam = FAMILY.get(args.get("2", ""))
            return (fam, (args.get("3") or "").strip()) if fam else None
        if name in ("etydate", "glossary", "m", "mention", "cog", "noncog", "ncog", "q", "qualifier", "w",
                    "lang", "IPAchar", "nb...", "rfe", "defdate", "inh"):
            if name == "inh":
                return None
            continue
    return None


def main():
    gold = json.loads((ROOT / "data/gold/turkish_loanwords.json").read_text())
    known = {r["word"].casefold() for r in gold["items"]}
    known |= {(r.get("word") or "").casefold() for r in gold.get("disagreements") or [] if isinstance(r, dict)}
    by_word = defaultdict(list)
    with gzip.open(ROOT / "data/lexicons/tr.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word") or ""
            if r.get("pos") in ("name", "suffix", "prefix", "phrase", "proverb", "affix", "abbrev", "symbol",
                                "character", "punct", "infix", "interfix", "circumfix", "romanization"):
                continue
            if not re.fullmatch(r"[a-zçğıöşüâîû]{3,}", w):
                continue
            ety = r.get("etymology_templates") or []
            by_word[w].append(donor_of(ety) if ety else "none")
    blind = sqlite3.connect(OUT / "index_blind.db")
    has_gloss = {row[0] for row in blind.execute(
        "SELECT DISTINCT word FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL AND gloss != ''")}
    drop = Counter()
    pool = []
    for w, ds in by_word.items():
        ds2 = {d for d in ds if d not in ("none",)}
        if not ds2 or None in ds2:
            drop["şablon-yok/türetme"] += 1; continue
        fams = {d[0] for d in ds2}
        if len(fams) != 1:
            drop["kayıtlar-çelişik"] += 1; continue
        if w.casefold() in known:
            drop["tdk-nişanyan-altınında"] += 1; continue
        if w not in has_gloss:
            drop["anlam-yok"] += 1; continue
        fam, etymon = sorted(ds2)[0]
        etymon = etymon or w
        split = "ayar" if h(f"{SALT}:{fam}:{etymon.casefold()}") % 1000 < AYAR_SHARE * 1000 else "rapor"
        pool.append({"word": w, "donor": fam, "gold": CLASS[fam], "etymon": etymon, "split": split})
    avail = Counter((p["split"], p["donor"]) for p in pool)
    items = []
    for split, caps in CAP.items():
        for fam, cap in caps.items():
            cand = sorted((p for p in pool if p["split"] == split and p["donor"] == fam),
                          key=lambda p: h(f"{SALT}:pick:{p['word']}"))
            items += cand[:cap]
    items.sort(key=lambda p: (p["split"], p["donor"], p["word"]))
    meta = {"salt": SALT, "source": "data/lexicons/tr.jsonl.gz (kaikki en, tr)", "available": {f"{s}/{d}": n for (s, d), n in sorted(avail.items())},
            "dropped": dict(drop), "counts": dict(Counter(f"{p['split']}/{p['donor']}" for p in items))}
    (OUT / "gold.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
