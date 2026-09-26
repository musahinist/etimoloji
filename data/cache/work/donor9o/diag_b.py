"""(b) sözlükte yok alt tanısı: from_turkic=1 ile elenmiş mi? ham kaikki dökümünde var mı?"""
import gzip, json, re, sqlite3, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa
from common import CLASS_CODES, diag
import diag9o

OUT = Path(__file__).parent
rows = [r for r in json.loads((OUT / "diag_rows.json").read_text()) if r["cls"] == "sozluk_yok"]
items = {(i["set"], i["word"]): i for i in diag.items_all()}
full = sqlite3.connect(diag.FULL)
con = sqlite3.connect("/Users/mshn/Documents/etimoloji/data/lexicons/donors/donors.db")
src = diag9o.etymon_entries.__code__
# from_turkic=1 dahil
import types
code = open(OUT / "diag9o.py").read()
ns = {"__file__": str(OUT / "diag9o.py")}
exec(code.replace("WHERE from_turkic=0", "WHERE 1=1").split("def main")[0], ns)
out = Counter()
need = {}
for r in rows:
    it = items[(r["set"], r["word"])]
    refs = diag.refs_for(it["word"], full); tr = diag.gold_translit(it)
    codes = CLASS_CODES.get(it["gold"], ())
    if not codes:
        out["altin_diger"] += 1
        continue
    e = ns["etymon_entries"](con, codes, refs, tr, it["comparison"])
    if e:
        out["from_turkic_ile_elenmis"] += 1
        print("FT", r["word"], [x["word"] for x in e][:3])
    else:
        need[r["word"]] = (codes, refs, tr)
print(out, len(need))
# ham döküm: Arap yazılı referansı olanlar için kelime eşitliği; Latin için romanizasyon
from engine.nlp.donor_proximity import script_skeleton
from engine.utils.orthography import to_comparison_form
found = Counter()
for lang in ("ar", "fa", "fr", "it", "el", "hy"):
    targets = {w: v for w, v in need.items() if lang in v[0]}
    if not targets:
        continue
    sks = {}
    for w, (codes, refs, tr) in targets.items():
        for _, f in refs:
            if re.search(r"[؀-ۿͰ-Ͽ԰-֏]", f):
                sks.setdefault(script_skeleton(f) if lang in ("ar", "fa") else diag.strip_marks(f), set()).add(w)
        for t in list(tr) + [f for _, f in refs if not re.search(r"[^\x00-ɏḀ-ỿ]", f)]:
            sks.setdefault("L:" + to_comparison_form(t), set()).add(w)
    with gzip.open(f"/Users/mshn/Documents/etimoloji/data/lexicons/donors/{lang}.jsonl.gz", "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            w = rec.get("word") or ""
            k = script_skeleton(w) if lang in ("ar", "fa") else diag.strip_marks(w)
            hits = sks.get(k, set()) | sks.get("L:" + to_comparison_form(w), set())
            rom = ""
            for f in rec.get("forms") or []:
                if "romanization" in (f.get("tags") or []):
                    rom = f.get("form") or ""
                    break
            if rom:
                hits |= sks.get("L:" + to_comparison_form(rom), set())
            for h in hits:
                found[h] = (lang, w, rom, (rec.get("senses") or [{}])[0].get("glosses", [""])[:1])
print("ham dökümde bulunan", len(found), "/", len(need))
for k, v in list(found.items())[:30]:
    print(" ", k, v)
print("hiç yok:", [w for w in need if w not in found][:60])
