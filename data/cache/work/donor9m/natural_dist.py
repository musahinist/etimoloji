"""9m — TDK GTS (12. baskı) `lisan` alanından Türkçe alıntıların DOĞAL köken dağılımı.

Kaynak ../donor9j/raw/gts.json (git-ignored; künye 9j PREREG). Yalnız SAYI yazılır (natural.json).
İki sayım:
  tum     — `lisan` dolu her başlık; ilk dil adı (başlık birden çok dilse eşit pay).
  altin   — altınların madde kuralı: tek sözcük, özel ad değil, `lisan` tek dil (+ , yok),
            aynı başlığın tüm kayıtları aynı dil. BİRİNCİL ağırlık bu (altınlar bu kuralla kuruldu).
Sınıflar: Arapça, Fransızca, Farsça, İtalyanca (Venedikçe/Cenevizce dahil), Yunanca (Rumca dahil),
Ermenice, diğer (Türkçe hariç tüm diğer diller).

python data/cache/work/donor9m/natural_dist.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent
SRC = OUT.parent / "donor9j" / "raw" / "gts.json"
CLASS = {"Arapça": "Arapça", "Fransızca": "Fransızca", "Farsça": "Farsça", "İtalyanca": "İtalyanca",
         "Venedikçe": "İtalyanca", "Cenevizce": "İtalyanca", "Rumca": "Yunanca", "Yunanca": "Yunanca",
         "Ermenice": "Ermenice"}


def cls(head):
    return CLASS.get(head, "diğer")


def main():
    heads_all, recs = defaultdict(set), defaultdict(set)
    names = Counter()
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        lisan = (r.get("lisan") or "").strip()
        w = (r.get("madde") or "").strip()
        if not lisan or not w:
            continue
        head = lisan.split(" ")[0].strip(",;:.")
        if head.startswith("(") or not head[:1].isupper():
            head = "?"
        heads_all[w].add(head)
        if " " in w or r.get("ozel_mi") == "1" or w[0].isupper() or "+" in lisan or "," in lisan:
            recs[w].add(None)
            continue
        recs[w].add(head)
    tum = Counter()
    for w, hs in heads_all.items():
        hs = {h for h in hs if h not in ("Türkçe", "?")}
        for h in hs:
            tum[cls(h)] += 1 / len(hs)
            names[h] += 1 / len(hs)
    altin = Counter()
    for w, hs in recs.items():
        if None in hs or len(hs) != 1:
            continue
        h = next(iter(hs))
        if h in ("Türkçe", "?"):
            continue
        altin[cls(h)] += 1
    res = {"source": "TDK GTS v12 lisan (../donor9j/raw/gts.json)",
           "altin": dict(altin.most_common()), "tum": {k: round(v, 1) for k, v in tum.most_common()},
           "diger_names_tum": {k: round(v, 1) for k, v in names.most_common() if cls(k) == "diğer"}}
    for k in ("altin", "tum"):
        t = sum(res[k].values())
        res[k + "_share"] = {c: round(v / t, 4) for c, v in res[k].items()}
    (OUT / "natural.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
