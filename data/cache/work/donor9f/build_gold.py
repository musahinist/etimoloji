"""9f — yeni rapor altını: 9e'de seçilmeyen Türkçe Wiktionary alıntıları (it/fr).

Havuz 9e `build_gold.py` ile AYNI süzgeçler (ilk verici şablonu, türetme yok,
kayıtlar vericide anlaşıyor, TDK+Nişanyan altınında yok, kör indekste anlamı
var; yardımcılar 9e betiğinden içe aktarılır). Sonra: 9e altında (ayar+rapor,
850 madde) kullanılmış HER (dil, etimon) grubu ve her kelime atılır; yalnız
fr ve it; bölme yok (tümü rapor). Dengeli: it tümü (tavan 120), fr aynı sayıda
(hash sırası, tuz `donor9f-v1`).

python data/cache/work/donor9f/build_gold.py
"""
import gzip, hashlib, json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
W9E = OUT.parent / "donor9e"
sys.path.insert(0, str(W9E))
import build_gold as B  # noqa: E402

SALT = "donor9f-v1"
CAP = 120


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def pool9e():
    gold = json.loads((B.ROOT / "data/gold/turkish_loanwords.json").read_text())
    known = {r["word"].casefold() for r in gold["items"]}
    known |= {(r.get("word") or "").casefold() for r in gold.get("disagreements") or [] if isinstance(r, dict)}
    by_word = defaultdict(list)
    with gzip.open(B.ROOT / "data/lexicons/tr.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w = r.get("word") or ""
            if r.get("pos") in ("name", "suffix", "prefix", "phrase", "proverb", "affix", "abbrev", "symbol",
                                "character", "punct", "infix", "interfix", "circumfix", "romanization"):
                continue
            if not re.fullmatch(r"[a-zçğıöşüâîû]{3,}", w):
                continue
            ety = r.get("etymology_templates") or []
            by_word[w].append(B.donor_of(ety) if ety else "none")
    blind = sqlite3.connect(W9E / "index_blind.db")
    has_gloss = {row[0] for row in blind.execute(
        "SELECT DISTINCT word FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL AND gloss != ''")}
    pool = []
    for w, ds in by_word.items():
        ds2 = {d for d in ds if d != "none"}
        if not ds2 or None in ds2 or len({d[0] for d in ds2}) != 1 or w.casefold() in known or w not in has_gloss:
            continue
        fam, etymon = sorted(ds2)[0]
        pool.append({"word": w, "donor": fam, "gold": B.CLASS[fam], "etymon": etymon or w})
    return pool


def main():
    used = json.loads((W9E / "gold.json").read_text())["items"]
    used_words = {i["word"] for i in used}
    used_groups = {(i["donor"], i["etymon"].casefold()) for i in used}
    pool = pool9e()
    fresh = [p for p in pool if p["donor"] in ("fr", "it") and p["word"] not in used_words
             and (p["donor"], p["etymon"].casefold()) not in used_groups]
    avail = Counter(p["donor"] for p in fresh)
    items = []
    n = min(CAP, avail["it"])
    for fam in ("it", "fr"):
        cand = sorted((p for p in fresh if p["donor"] == fam), key=lambda p: h(f"{SALT}:pick:{p['word']}"))
        items += [dict(p, split="rapor9f") for p in cand[:n]]
    items.sort(key=lambda p: (p["donor"], p["word"]))
    meta = {"salt": SALT, "source": "data/lexicons/tr.jsonl.gz (kaikki en, tr); 9e havuzu eksi 9e altındaki (dil, etimon) grupları",
            "pool_9e": len(pool), "available_fresh": dict(avail), "counts": dict(Counter(p["donor"] for p in items))}
    (OUT / "gold.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=0))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
