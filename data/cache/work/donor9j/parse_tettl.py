"""9j — Tietze TETTL (archive.org OCR, raw/tNN.txt; git-ignored) ayrıştırıcısı.

Madde = boş satırla ayrılmış paragraf; başlık = (EOsm./Osm./AD. vb. işaretinden ve *'dan
sonra) ilk küçük harfli sözcük ("/" varyantları); yakın verici = paragrafın ilk 450
karakterindeki ilk "< Dil." (OCR'de "<" bazı ciltlerde “ € c — olarak bozulmuş: cilde göre
izinli işaretler). Etimon = dil kısaltmasından (ve varsa "(Venedik diyal.)" gibi ayraçtan)
sonraki ilk sözcük. Çıktı raw/tettl_parsed.json (git-ignored; OCR metni içermez, yalnız
kelime/verici/etimon/cilt/paragraf no).

python data/cache/work/donor9j/parse_tettl.py
"""
import json, re
from collections import Counter
from pathlib import Path

RAW = Path(__file__).parent / "raw"
ABBR = {"İt": "it", "It": "it", "Ît": "it", "Fr": "fr", "Ar": "ar", "Fa": "fa", "Yun": "el", "OYun": "el",
        "Erm": "hy", "İng": "en", "Ing": "en", "Alm": "de", "EYun": "grc", "Lat": "la", "İsp": "es", "Rus": "ru"}
# cilt -> "<" yerine OCR'de görülen işaretler
MARK = {1: "<€", 2: "<€", 3: "<€", 4: "<€", 6: "<€", 5: "<c€—", 7: "<“€", 8: "<“€", 9: "<“€"}
PREFIX = r"(?:(?:EOsm|YOsm|Osm|AD|Ttü|Çağ|Rum|Ağ|Tkm|Az|Kırım)\.\s*)*\*?\s*"
_W = r"[a-zçğıöşüâîûäêô](?:[a-zçğıöşüâîûäêô\-]|[’'”‘`](?=[a-zçğıöşüâîûäêô]))*"
HEAD = re.compile(r"^" + PREFIX + r"(" + _W + r"(?:\s*/\s*" + _W + r")*)")
LANGS = "|".join(sorted(ABBR, key=len, reverse=True))


def donor_re(vol):
    """İlk köken işareti: gerçek "<" her şeyle; bozuk OCR işareti yalnız büyük harfli kısaltmayla."""
    bad = re.escape(MARK[vol].replace("<", ""))
    abbr = r"([A-ZÇĞİÖŞÜÎ][A-Za-zçğıöşüİ]{0,5})\.\s*(\([^)]{1,40}\)\s*)?([^\s,;.“”'\"()<]+)"
    alt = r"|(?:^|[\s(])[" + bad + r"]\s*" + abbr if bad else ""
    return re.compile(r"(?:^|[\s(\w])<\s*(?:" + abbr + r"|(\S+))" + alt)


def paragraphs(text):
    for n, block in enumerate(re.split(r"\n\s*\n", text)):
        p = " ".join(l.strip() for l in block.splitlines() if l.strip())
        p = re.sub(r"-\s+(?=[a-zçğıöşü])", "", p)  # satır sonu tireleri
        if p:
            yield n, p


def parse(vol):
    out = []
    rx = donor_re(vol)
    for n, p in paragraphs((RAW / f"t{vol:02d}.txt").read_text(encoding="utf-8")):
        m = HEAD.match(p)
        if not m or len(p) < 25:
            continue
        head = re.sub(r"[’'”‘`]", "'", m.group(1))
        d = rx.search(p[:450], m.end())
        if not d:
            continue
        g = d.groups()
        ab, par, ety = (g[0], g[1], g[2]) if g[0] else (g[4], g[5], g[6]) if len(g) > 4 and g[4] else (None, None, None)
        if ab not in ABBR:
            continue  # ilk köken işareti Türkçe/başka dile ya da bir türetmeye gidiyor
        between = p[m.end():d.start()]
        # çok sözcüklü başlık (deyim/birleşik: "sabahın köründe", "pilas tahtası") -> atla
        pre = re.split(r"[‘'“\"(<€—]", between, maxsplit=1)[0]
        if any(re.fullmatch(r"[a-zçğıöşüâîû][a-zçğıöşüâîû\-]+", t) for t in pre.split() if "/" not in t):
            continue
        paren = (par or "").strip()
        out.append({"vol": vol, "para": n, "head": [h.strip() for h in head.split("/")],
                    "donor": ABBR[ab], "abbr": ab, "etymon": ety[:40],
                    "venetian": bool(re.search(r"Venedik|Ceneviz", paren + p[d.start():d.start() + 120])),
                    "gap": len(between), "doubt": "?" in p[d.end():d.end() + 25] or " veya " in p[d.end():d.end() + 25]})
    return out


def main():
    allr = []
    for vol in MARK:
        rs = parse(vol)
        allr += rs
        print(vol, len(rs), dict(Counter(r["donor"] for r in rs).most_common(8)))
    (RAW / "tettl_parsed.json").write_text(json.dumps(allr, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
