"""9n tanı — gösterilen verici BİÇİMİ doğru etimon mu? (yalnız ayar/görülmüş veri)

Altınlar: Türkçe TDK+Nişanyan train+dev (test OKUNMAZ; `load_cases`), 9j TDK ayar, 9j TETTL ayar.
Etiket: üretim (`attribute_donor`, D1+G2+H1+S1), kör indeks (xtr/index_blind.db), AĞ KAPALI.
Etimon referansı (yalnız DEĞERLENDİRME için; etiketleyici görmez):
  (1) altının etimon çevriyazısı (TDK/TETTL `etymon`; Türkçe altında tdk_source/nisanyan_source'un dil
      adından sonraki kısmı),
  (2) TAM indeksteki (data/lexicons/index.db) Türkçe maddenin Wiktionary şablon argümanı `donor_form`
      ve etimoloji metnindeki dil adı + biçim çiftleri (Arabic عَسْكَر (ʿaskar) …).
Her maddeye etiketin iç özellikleri (mesafe, null, marj, ikinci dile fark, kural yolu, havuz kaynağı,
anlam eşleşme türü, şans yüzdeliği) ve referansla eşleşme sınıfı yazılır -> diag_rows.json.

python data/cache/work/donor9n/diag.py
"""
import json, os, re, socket, sqlite3, sys, time, unicodedata
from pathlib import Path

OUT = Path(__file__).parent
W = OUT.parent
ROOT = OUT.parents[3]
BLIND = W / "xtr" / "index_blind.db"
FULL = ROOT / "data" / "lexicons" / "index.db"
_SAHA = sys.argv[1:2] == ["saha"]  # harness.py saha: tam indeks, ağ serbest (9m ile aynı)
if not _SAHA:
    os.environ["ETY_LEXICON_INDEX"] = str(BLIND)
os.environ.setdefault("ETY_DONOR_CLEAN", "0")
os.environ.setdefault("ETY_DONOR_RAMP_CHANCE", "0")
sys.path.insert(0, str(ROOT))


def _no_net(*a, **k):
    raise RuntimeError("9n: ağ kapalı")


if not _SAHA:
    socket.socket.connect = _no_net
    socket.create_connection = _no_net

#: 9j/9m altınlarının sınıfları (Ermenice ayrı; harness.py ile aynı).
CLS9 = {"el": "Yunanca", "grc": "Yunanca", "gkm": "Yunanca", "hy": "Ermenice", "xcl": "Ermenice"}
LANG_NAMES = {
    "Arabic": "ar", "Arapça": "ar", "Persian": "fa", "Farsça": "fa", "French": "fr", "Fransızca": "fr",
    "Italian": "it", "İtalyanca": "it", "Venetian": "it", "Venedikçe": "it", "Genoese": "it",
    "Greek": "el", "Yunanca": "el", "Rumca": "el", "Armenian": "hy", "Ermenice": "hy",
}
_REF = re.compile(r"(Ancient Greek|Byzantine Greek|Medieval Greek|Modern Greek|Classical Persian|"
                  + "|".join(sorted(LANG_NAMES, key=len, reverse=True))
                  + r")\s+([^\s(),;:.]+)(?:\s+\(([^,);]+))?")


def lang_of(name):
    for k, v in LANG_NAMES.items():
        if k in name:
            return v
    return ""


def strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if not unicodedata.combining(c)).casefold()


def refs_for(word, full):
    """(dil, biçim) referans kümesi: Wiktionary donor_form + etimoloji metni."""
    out = set()
    for lang, form, ety in full.execute(
            "SELECT donor_lang, donor_form, etymology FROM entries WHERE lang_code='tr' AND word=?", (word,)):
        if form:
            out.add((lang or "", re.sub(r"<[^<>]*>", "", form).strip()))
        for m in _REF.finditer(ety or ""):
            code = lang_of(m.group(1))
            out.add((code, m.group(2)))
            if m.group(3):
                out.add((code, m.group(3).strip()))
    return {(l, f) for l, f in out if f and f not in ("from", "word", "term")}


def gold_translit(item):
    """Altının etimon çevriyazısı (dil adı atılmış)."""
    if "etymon" in item:
        return [item["etymon"]] if item.get("etymon") else []
    out = []
    for src in (item.get("tdk_source", ""), item.get("nisanyan_source", "")):
        parts = src.split(" ", 1)
        if len(parts) == 2:
            out += [p.strip() for p in re.split(r"\s*\+\s*|,", parts[1]) if p.strip()]
    return out


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_class(a, refs, translits):
    """'biçim' (aynı sözcük), 'iskelet' (yalnız ünsüz iskeleti), '' (eşleşme yok)."""
    from engine.nlp.donor_proximity import consonant_skeleton, script_skeleton
    from engine.utils.orthography import to_comparison_form
    word, comp = a["word"], a["comparison"]
    arabic = bool(re.search(r"[؀-ۿ]", word))
    sk = script_skeleton(word) if arabic else ""
    for _, f in refs:
        if arabic and re.search(r"[؀-ۿ]", f):
            rs = script_skeleton(f)
            if rs and (rs == sk or rs.removeprefix("ال") == sk or sk.removeprefix("ال") == rs):
                return "biçim"
        elif not arabic and strip_marks(f) == strip_marks(word):
            return "biçim"
    latin = [to_comparison_form(f) for _, f in refs if not re.search(r"[^\x00-ɏḀ-ỿ]", f)]
    latin += [to_comparison_form(t) for t in translits]
    latin = [x for x in latin if x]
    for x in latin:
        if x == comp or (len(x) >= 4 and lev(x, comp) / max(len(x), len(comp)) <= 0.2):
            return "biçim"
    cs = consonant_skeleton(comp)
    for x in latin:
        xs = consonant_skeleton(x)
        if len(cs) >= 2 and (xs == cs or (x.endswith(("et", "at")) and consonant_skeleton(x[:-1]) == cs)):
            return "iskelet"
    return ""


def items_all():
    from engine.evaluation.tr_donor_eval import load_cases
    from engine.utils.orthography import to_comparison_form
    out = []
    for c in load_cases():
        if c.split in ("train", "dev"):
            out.append({"set": "tr", "word": c.word, "gold": c.gold, "comparison": to_comparison_form(c.word),
                        "tdk_source": c.tdk_source, "nisanyan_source": c.nisanyan_source})
    for g in ("tdk", "tettl"):
        for i in json.loads((W / "donor9j" / f"gold_{g}.json").read_text())["items"]:
            if i["split"] == "ayar":
                out.append(dict(i, set=f"{g}_ayar"))
    return out


def glosses(db, words):
    con = sqlite3.connect(db)
    out = {}
    for w in words:
        r = con.execute("SELECT gloss FROM entries WHERE lang_code='tr' AND word=? AND gloss IS NOT NULL AND gloss!=''"
                        " ORDER BY id LIMIT 1", (w,)).fetchone()
        if r:
            out[w] = r[0]
    return out


def features(comparison, sense, a):
    """Etiketin iç özellikleri (etiketleme anında bilinen)."""
    from engine.db.donor_index import FUNCTION_WORDS, _sense_tokens, sense_tokens_for_match
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    index = dp._index()
    bsense = dp.bridged_sense(comparison, sense)
    rows = index.by_sense(bsense, languages=TURKISH_DONORS, limit=200)
    shared = {(r["lang_code"], r["word"], r["comparison"]) for r in rows}
    fr_extra = [r for r in index.by_sense(bsense, languages=["fr"], limit=200)
                if (r["lang_code"], r["word"], r["comparison"]) not in shared]
    groups = {}
    for r in list(rows) + fr_extra:
        groups.setdefault(r["lang_code"], []).append(r)
    scored = []
    for lang, members in groups.items():
        forms = sorted({r["comparison"] for r in members if r["comparison"]})
        if not forms:
            continue
        d, f = dp.best_label(comparison, forms)
        null = dp._null_distance(len(comparison), tuple(forms))
        scored.append((d - null, d, lang, len(forms)))
    scored.sort()
    top = scored[0] if scored else None
    second = scored[1] if len(scored) > 1 else None
    win_pool = tuple(sorted({r["comparison"] for r in groups.get(a.lang_code, []) if r["comparison"]}))
    prof = dp._control_profile(len(comparison), win_pool) if win_pool else ()
    pct = sum(1 for x in prof if x <= a.distance) / len(prof) if prof else None
    qtok = set(sense_tokens_for_match(bsense))
    gtok = set(_sense_tokens(a.gloss))
    ov = qtok & gtok
    content_ov = {t for t in ov if t not in FUNCTION_WORDS}
    key = (a.lang_code, a.word, a.comparison)
    if a.via == "fa" and a.lang_code == "ar":
        path = "d1"
    elif a.lang_code == "fr" and a.via:
        path = "frvia"
    elif top and a.lang_code == "fr" and top[2] != "fr":
        path = "h1"
    else:
        path = "direct"
    return {
        "distance": round(a.distance, 4), "null": round(a.null_distance, 4),
        "margin": round(a.distance - a.null_distance, 4),
        "gap2": round(second[0] - top[0], 4) if second else None,
        "second_lang": second[2] if second else "", "top_lang": top[2] if top else "",
        "path": path, "pool": "shared" if key in shared else ("fr_extra" if a.lang_code == "fr" else "other"),
        "pool_size": len(win_pool), "n_langs": len(scored), "chance_pct": pct,
        "sense_tokens": len(qtok), "overlap": sorted(ov), "content_overlap": len(content_ov),
        "gloss_len": len(gtok), "bridged": bsense != sense, "len": len(comparison),
        "skeleton_eq": dp.consonant_skeleton(a.comparison) in dp._query_skeletons(comparison),
        "groups": {lang: [r["comparison"] for r in m] for lang, m in groups.items()},
        "group_words": {lang: [r["word"] for r in m] for lang, m in groups.items()},
    }


def main():
    from engine.evaluation.tr_donor_eval import engine_class
    from engine.nlp import donor_proximity as dp
    from engine.nlp.borrowing_detector import TURKISH_DONORS
    assert dp.FRENCH_RULE == "g2" and dp.WESTERN_RULE == "h1" and dp.ARABIC_VIA_RULE == "d1" and dp.SENSE_BRIDGE
    assert not dp.FRENCH_ARABIC_GUARD and not dp.WESTERN_SUFFIX_DROP
    items = items_all()[: int(os.environ.get("NINEN_LIMIT", "0")) or None]
    gl = glosses(BLIND, {i["word"] for i in items})
    full = sqlite3.connect(FULL)
    out, t = [], time.time()
    for n, it in enumerate(items):
        sense = gl.get(it["word"], "")
        a = dp.attribute_donor(it["comparison"], sense, languages=TURKISH_DONORS)
        refs = refs_for(it["word"], full)
        tr = gold_translit(it)
        row = {k: it.get(k) for k in ("set", "word", "gold", "comparison")}
        row.update({"sense": sense, "refs": sorted(map(list, refs)), "translit": tr})
        if a is None:
            row.update({"pred": "—", "etym": "", "cat": "yok"})
        else:
            f = features(it["comparison"], sense, a)
            pa = {"word": a.word, "comparison": a.comparison}
            m = match_class(pa, refs, tr)
            pred = engine_class(a.lang_code) if it["set"] == "tr" else (CLS9.get(a.lang_code) or engine_class(a.lang_code))
            # havuzda doğru etimon var mı (tanı; referansla herhangi bir dil grubunda biçim eşi)
            in_pool = sorted({lang for lang, comps in f["groups"].items()
                              for w, c in zip(f["group_words"][lang], comps)
                              if match_class({"word": w, "comparison": c}, refs, tr) == "biçim"})
            del f["groups"], f["group_words"]
            row.update({"pred": pred, "lang": a.lang_code, "form": a.word, "form_comp": a.comparison,
                        "gloss": a.gloss[:80], "via": a.via, "etym": m, "etym_in_pool": in_pool, **f})
            row["cat"] = ("etimon" if m == "biçim" else "iskelet" if m == "iskelet" else "şans") \
                if pred == it["gold"] else "yanlış_dil"
        out.append(row)
        if n % 100 == 0:
            print(n, len(items), f"{time.time() - t:.0f}s", flush=True)
    bad = sorted(m for m in sys.modules if m.startswith("engine.fetchers"))
    assert not bad, bad
    (OUT / "diag_rows.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print("DONE", len(out))


if __name__ == "__main__":
    main()
