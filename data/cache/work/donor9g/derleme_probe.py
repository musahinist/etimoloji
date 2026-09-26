"""9g keşif (ALTIN DEĞİL): TDK Derleme ağız maddeleri Rumca/Ermenice verici havuzuyla eşleşiyor mu?

"Ağızlarda gizli kalan temas" için kaba bir TAHMİN. Derleme köken etiketi
taşımaz. Yöntem: madde anlamının ilk Türkçe karşılığı tam indeksin Türkçe
maddesiyle İngilizce anlama çevrilir; etiket adımı (G3 açık: grc/xcl + çekim
süzgeci; TURKISH_DONORS) koşulur. "Güçlü" = etiket el/hy VE SCA ≤ 0,35.
Şans denetimi: aynı kelimeler, anlamlar karıştırılmış (döngüsel kaydırma).
Wiktionary kökeni (tam indeks) varsa çapraz tablo.

python data/cache/work/donor9g/derleme_probe.py
"""
import json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
FULL = ROOT / "data/lexicons/index.db"


def main():
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    from engine.utils.orthography import to_comparison_form

    con = sqlite3.connect(FULL)
    gl = {}
    for w, g in con.execute("SELECT word, gloss FROM entries WHERE lang_code='tr' AND gloss IS NOT NULL AND gloss!='' ORDER BY id"):
        gl.setdefault(w, g)
    origin = {}
    for w, o, d in con.execute("SELECT word, origin, donor_lang FROM entries WHERE lang_code='tr' AND origin IS NOT NULL"):
        origin.setdefault(w, f"{o}:{d or ''}")
    senses = defaultdict(list)
    for line in open(ROOT / "data/dialect/derleme/records.jsonl"):
        r = json.loads(line)
        a = (r.get("anlam") or "").strip()
        if not a or a.startswith("[") or not r.get("madde"):
            continue
        senses[r["madde"]].append(re.sub(r"^\d+\.\s*", "", a))
    items = []
    for w, al in senses.items():
        eng = ""
        for a in al:
            for part in re.split(r"[,.;:()]", a):
                p = part.strip().lower()
                if p and p in gl:
                    eng = gl[p]
                    break
            if eng:
                break
        c = to_comparison_form(w)
        if eng and len(c) >= 3:
            items.append({"word": w, "comparison": c, "sense_tr": al[0][:80], "sense_en": eng[:120],
                          "wikt": origin.get(w, "")})
    dp.OLD_DONOR_LABELS = dp.LABEL_FORM_FILTER = True
    shuffled = [it["sense_en"] for it in items[7:] + items[:7]]

    def run(sense_of):
        out = []
        for it, s in zip(items, sense_of):
            a = dp.attribute_donor(it["comparison"], s, languages=TURKISH_DONORS)
            out.append((a.lang_code, a.word, round(a.distance, 3), a.source) if a else None)
        return out

    real = run([it["sense_en"] for it in items])
    ctrl = run(shuffled)
    strong = lambda x: bool(x) and x[0] in ("el", "hy") and x[2] <= dp.DONOR_DISTANCE_THRESHOLD
    rows = []
    for it, r, c in zip(items, real, ctrl):
        rows.append(dict(it, label=r, strong=strong(r), ctrl=c, ctrl_strong=strong(c)))
    res = {"derleme_maddeler": len(senses), "ingilizce_anlam_bulunan": len(items),
           "guclu": dict(Counter(r["label"][0] for r in rows if r["strong"])),
           "guclu_sans_denetimi": dict(Counter(r["ctrl"][0] for r in rows if r["ctrl_strong"])),
           "guclu_wikt": dict(Counter(r["wikt"] or "yok" for r in rows if r["strong"])),
           "tum_wikt": dict(Counter((r["wikt"] or "yok").split(":")[0] for r in rows)),
           "wikt_el_hy_yakalanan": f"{sum(1 for r in rows if r['strong'] and r['wikt'].split(':')[-1] in ('el','hy','grc','xcl','gkm'))}/"
                                   f"{sum(1 for r in rows if r['wikt'].split(':')[-1] in ('el','hy','grc','xcl','gkm'))}"}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "derleme_probe.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
