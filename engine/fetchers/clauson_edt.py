"""
Eski Türkçe tanık ve taban biçimi — Clauson 1972, *EDT* (yerel, ``make clauson``).

Clauson ~9.100 madde verir (ayrıştırıcı: ``engine/db/clauson.py``): madde başı
Eski Türkçe (8.-13. yy) biçim, türemişlerde taban (``taŋsuk`` < ``2 taŋ``,
``učar`` < ``uč-``, ``čapğut`` < ``čap-``), İngilizce anlam ve tanıklık
(Türkü vııı / Uyğ. vııı ff. / Xak. xı, Kaš., KB …).

⚠️ Proto-Türkçe kök DEĞİLDİR. Motor Clauson biçimini yalnız "Eski Türkçe
tabanı: uč- (Clauson)" diye ayrı alanda gösterir (``old_turkic_base``);
başlık kökünün (``proto_turkic``) yerine koymaz.

⚠️ Döngüsellik: Starling turcet EDT'ye 1.504 kez atıf yapar; Clauson ile
Starling uyumu bağımsız doğrulama değildir, aynı kanıtın ham hâlidir.

Eşleşme (yalnız ikisi birlikte tutarsa):

1. **Biçim** — madde başı (ya da yazım varyantı; ``S X See Y`` göndermesi
   izlenir) sorguyla kaba ses sınıflarında (``wilkens_old_uyghur.coarse``:
   k/g/ğ, t/d, b/p, ŋ/n, ünlü yuvarlaklık/yükseklik) ``FORM_THRESHOLD``
   kadar benzer.
2. **Anlam** — sorgunun yerel indeksteki Türkçe/İngilizce anlamlarından biri
   ile maddenin İngilizce anlamı (Clauson'un ilk tırnaklı anlamı ya da
   Kisamov'un kısa anlamı) çok dilli MiniLM'de ``MEANING_THRESHOLD`` kadar
   yakın. Anlam ZORUNLU: yalnız biçim benzerliği (``petek`` ~ ``büt-``
   dersi) eşsesli ve ilgisiz maddeleri bağlar. Sorgunun anlamı yoksa eşleşme
   yok.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from functools import lru_cache
from typing import Any

from engine.fetchers.base import BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

LANG = "otk"
SOURCE_LABEL = "Clauson 1972, An Etymological Dictionary of Pre-Thirteenth-Century Turkish"
#: Kaba ses sınıflarında en düşük yazılış benzerliği (1 - Levenshtein / uzunluk).
FORM_THRESHOLD = 0.75
#: Çok dilli MiniLM (``paraphrase-multilingual-MiniLM-L12-v2``) kosinüs tabanı.
MEANING_THRESHOLD = 0.50
MAX_WITNESSES = 2
#: Kök zincirinde en çok adım (``uč-`` <- ``učar`` <- ``učarlığ``).
MAX_CHAIN = 4
#: Taban modu: ``head`` madde başı, ``base`` doğrudan taban, ``chain`` kök
#: zinciri. Seçim ön kayıtta A yarısında (``data/cache/work/clauson/PREREG.md``):
#: gelenek-denk head 69, base 68, chain 68 -> ``head``.
BASE_MODE = "head"

#: Clauson çevriyazısı -> Türkiye Türkçesi karşılaştırma harfleri.
_TO_TR = str.maketrans({"č": "ç", "š": "ş", "ŋ": "n", "ñ": "n", "x": "h", "ä": "e", "é": "e", "ẹ": "e",
                        "ā": "a", "ī": "i", "ū": "u", "ō": "o", "ē": "e", "İ": "i", "I": "ı"})


def clauson_comparison(form: str) -> str:
    """``kočŋa:r`` -> ``koçnar``; ``uč-`` -> ``uç``; ``*ina:-`` -> ``ina``."""
    text = (form or "").strip().lstrip("*").translate(_TO_TR).lower()
    text = re.sub(r"[:\-'’()?.]", "", text)
    return to_comparison_form(text) or text


def _coarse(form: str) -> str:
    from engine.fetchers.wilkens_old_uyghur import coarse

    return coarse(form)


def _similarity(a: str, b: str) -> float:
    from engine.fetchers.northeuralex import _similarity as similarity

    return similarity(_coarse(a), _coarse(b))


@lru_cache(maxsize=1)
def _records() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    from engine.db.clauson import load_records

    records = load_records()
    by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        forms = []
        for raw in [record.get("headword") or "", *(record.get("variants") or [])]:
            comparison = clauson_comparison(raw)
            if comparison and comparison not in forms:
                forms.append(comparison)
        record["_forms"] = forms
        for form in forms:
            by_form[form].append(record)
    return records, dict(by_form)


def _usable(record: dict[str, Any]) -> bool:
    prefix = record.get("prefix") or ""
    return "E" not in prefix and " " not in (record.get("headword") or "").strip()


def resolve(record: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """``S čaput See čapğut`` -> ``čapğut`` maddesi (kaba biçim ve eşsesli no ile)."""
    target = record.get("see")
    if not target or depth > 2:
        return record
    _, by_form = _records()
    wanted = clauson_comparison(target)

    def others(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pool = [r for r in pool if r is not record]
        return [r for r in pool if not r.get("see")] or pool

    pool = others(by_form.get(wanted) or [])
    if not pool:  # yazım farkı: kaba sınıfta eşit, yoksa en yakın (≥0,8; "kočga:r" ~ "kočŋa:r")
        coarse = _coarse(wanted)
        pool = others([r for form, rs in by_form.items() if _coarse(form) == coarse for r in rs])
    if not pool:
        scored = sorted(((_similarity(wanted, form), form) for form in by_form
                         if abs(len(form) - len(wanted)) <= 1 and form[:1] == wanted[:1]), reverse=True)
        for score, form in scored:
            if score < 0.8:
                break
            pool = others(list(by_form[form]))
            if pool:
                break
    if record.get("see_homonym"):
        pool = [r for r in pool if r.get("homonym") == record["see_homonym"]] or pool
    return resolve(pool[0], depth + 1) if pool else record


def form_candidates(word: str) -> list[tuple[float, dict[str, Any]]]:
    """Biçimce eşleşen (göndermesi çözülmüş) maddeler, benzerliğe göre."""
    query = to_comparison_form((word or "").strip().lower().replace("I", "ı").replace("İ", "i"))
    if not query or " " in query:
        return []
    stem = re.sub(r"m[ae]k$", "", query)
    targets = {query, stem} if len(stem) >= 2 else {query}
    records, by_form = _records()
    found: dict[int, tuple[float, dict[str, Any]]] = {}
    coarse_targets = {_coarse(t) for t in targets}
    for form, pool in by_form.items():
        if abs(len(form) - len(query)) > 3:
            continue
        score = 1.0 if _coarse(form) in coarse_targets else max(_similarity(t, form) for t in targets)
        if score < FORM_THRESHOLD:
            continue
        for record in pool:
            if not _usable(record):
                continue
            target = resolve(record)
            if not _usable(target):
                continue
            previous = found.get(id(target))
            if previous is None or previous[0] < score:
                found[id(target)] = (score, target)
    return sorted(found.values(), key=lambda item: (-item[0], item[1].get("id", 0)))


def record_glosses(record: dict[str, Any]) -> list[str]:
    out = []
    for gloss in (record.get("gloss"), record.get("gloss_tw")):
        gloss = re.sub(r"\s+", " ", str(gloss or "")).strip(" ,;")
        if gloss and gloss not in out:
            out.append(gloss)
    return out


def query_gloss_sets(word: str) -> tuple[list[str], list[str]]:
    """Sorgunun yerel indeksteki (``tr``/``ota``) anlamları, ``(ad, fiil)``:
    ad anlamları kelimenin kendi (fiil olmayan) kayıtlarından, fiil anlamları
    mastarın (``savur`` -> ``savurmak``) ve fiil kayıtlarından. ``taş`` ~
    *taš-* "taşmak" fiil maddesine ad anlamıyla, ``taşmak`` anlamıyla da
    *ta:š* "stone" maddesine bağlanmasın. Çekim, gönderme, ek ve özel ad
    kayıtları hariç."""
    nouns: list[str] = []
    verbs: list[str] = []
    try:
        from engine.db.lexicon_index import LexiconIndex
        from engine.utils.morphology import is_inflection_gloss
        from engine.utils.reference_resolver import is_cross_reference

        index = LexiconIndex()
        if not index.exists:
            return nouns, verbs
        query = to_comparison_form(word)
        stem = re.sub(r"m[ae]k$", "", query) if len(query) > 4 else query
        vowels = [c for c in stem if c in "aıoueiöü"]
        infinitive = stem + ("mak" if vowels and vowels[-1] in "aıou" else "mek")
        keys = {stem, infinitive}
        for key in (stem, infinitive):
            for row in index.lookup(key, languages=["tr", "ota"], limit=30):
                gloss = re.sub(r"\s+", " ", str(row.get("gloss") or "")).strip()
                raw_word = str(row.get("word") or "")
                form = to_comparison_form(raw_word)
                if (raw_word.startswith("-") or row.get("pos") in ("suffix", "prefix", "infix", "affix", "name")
                        or form not in keys or not gloss or is_inflection_gloss(gloss)
                        or is_cross_reference(gloss) or re.search(r"\b(?:form|spelling) of\b", gloss, re.I)):
                    continue
                bucket = verbs if (form == infinitive or row.get("pos") == "verb") else nouns
                gloss = gloss[:160]
                if gloss not in bucket:
                    bucket.append(gloss)
    except Exception:
        logger.debug("Clauson: sorgu anlamı okunamadı: %s", word, exc_info=True)
    return nouns[:8], verbs[:8]


def query_glosses(word: str) -> list[str]:
    nouns, verbs = query_gloss_sets(word)
    return [*nouns, *(v for v in verbs if v not in nouns)]


_ENCODE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _cpu_model():
    """Motorun çok dilli MiniLM'inin AYRI, CPU'daki kopyası. Fetcher iş
    parçacığında çalışır: ortak (MPS) modeli başka bir iş parçacığıyla aynı
    anda kullanmak Metal komut tamponu hatasıyla süreci düşürdü (ölçüldü,
    2026-09-26). Kısa anlam metinlerinde CPU yeterince hızlı."""
    try:
        from sentence_transformers import SentenceTransformer

        from engine.nlp.diachronic_semantic_engine import _ST_MODEL_NAME

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:
            pass
        try:
            return SentenceTransformer(_ST_MODEL_NAME, device="cpu", local_files_only=True)
        except Exception:
            return SentenceTransformer(_ST_MODEL_NAME, device="cpu")
    except Exception:
        logger.warning("Clauson: anlam modeli yüklenemedi", exc_info=True)
        return None


def _meaning_scores(glosses: list[str], candidates: list[dict[str, Any]]) -> list[float] | None:
    texts = [g for record in candidates for g in record_glosses(record)]
    if not glosses or not texts:
        return [0.0] * len(candidates)
    try:
        model = _cpu_model()
        if model is None:
            return None
        from sentence_transformers.util import cos_sim

        with _ENCODE_LOCK:
            sims = cos_sim(model.encode(glosses, show_progress_bar=False),
                           model.encode(texts, show_progress_bar=False)).tolist()
    except Exception:
        logger.warning("Clauson: anlam benzerliği hesaplanamadı", exc_info=True)
        return None
    out: list[float] = []
    col = 0
    for record in candidates:
        n = len(record_glosses(record))
        out.append(max((sims[i][j] for i in range(len(glosses)) for j in range(col, col + n)), default=0.0))
        col += n
    return out


def matches(word: str, glosses: list[str] | None = None) -> list[tuple[float, float, dict[str, Any]]]:
    """``(biçim, anlam, madde)``: ikisi de eşiği geçen maddelerden yalnız en iyi
    biçim benzerliğindekiler (``göl`` ~ *kö:l* varken türev ya da eşsesli
    aday alınmaz), anlama göre sıralı. ``glosses`` verilirse her maddeye o."""
    candidates = form_candidates(word)
    if not candidates:
        return []
    if glosses is not None:
        noun_glosses = verb_glosses = [g for g in glosses if g]
    else:
        noun_glosses, verb_glosses = query_gloss_sets(word)
    # Mastar sorgusu yalnız fiil maddesine; çıplak sorgu önce ad olarak, fiil
    # maddesi yalnız ad eşleşmesi yoksa (``taş`` ~ *ta:š*, ``savur`` ~ *savur-*).
    infinitive_query = bool(re.search(r"m[ae]k$", to_comparison_form(word))) and len(word) > 4
    kept: list[tuple[float, float, dict[str, Any]]] = []
    for is_verb, pool_glosses in ((False, noun_glosses), (True, verb_glosses)):
        if kept or (infinitive_query and not is_verb):
            continue
        pool = [(form, record) for form, record in candidates
                if (record.get("headword") or "").endswith("-") == is_verb]
        if not pool or not pool_glosses:
            continue
        scores = _meaning_scores(pool_glosses, [record for _, record in pool])
        if scores is None:
            return []
        kept.extend((form, round(meaning, 3), record)
                    for (form, record), meaning in zip(pool, scores, strict=True)
                    if meaning >= MEANING_THRESHOLD)
    if not kept:
        return []
    best = max(form for form, _, _ in kept)
    kept = [item for item in kept if item[0] >= best - 1e-9]
    kept.sort(key=lambda item: -item[1])
    return kept[:MAX_WITNESSES]


def _lookup_base(form: str, homonym: int | None) -> dict[str, Any] | None:
    _, by_form = _records()
    pool = [r for r in by_form.get(clauson_comparison(form), []) if _usable(r)]
    # Fiil tabanı fiil maddesine, ad tabanı ad maddesine (``uč-`` ≠ ``u:č``).
    verb = form.rstrip().endswith("-")
    pool = [r for r in pool if (r.get("headword") or "").endswith("-") == verb] or pool
    if homonym:
        pool = [r for r in pool if r.get("homonym") == homonym] or pool
    return resolve(pool[0]) if pool else None


def base_form(record: dict[str, Any], mode: str = BASE_MODE) -> str:
    """Seçilen moda göre Eski Türkçe taban biçimi (Clauson yazımıyla).

    Yabancı (``F``) maddenin Türkçe tabanı yoktur: madde başının kendisi."""
    head = record.get("headword") or ""
    if mode == "head" or "F" in (record.get("prefix") or "") or not record.get("base"):
        return head
    if mode == "base":
        return record["base"]
    current, seen = record, {id(record)}
    form = head
    for _ in range(MAX_CHAIN):
        if not current.get("base"):
            break
        form = current["base"]
        nxt = _lookup_base(current["base"], current.get("base_homonym"))
        if nxt is None or id(nxt) in seen or "F" in (nxt.get("prefix") or ""):
            break
        seen.add(id(nxt))
        current = nxt
    return form


#: Eser -> ``attestation_dates`` anahtarı.
_WORK_KEYS = ("orhun", "kb", "dlt")


def earliest_attestation(record: dict[str, Any]) -> dict[str, Any] | None:
    """Maddenin en erken NOKTA tarihli eseri (Orhun 732, KB 1069, DLT 1074);
    yoksa en erken lehçe/yüzyıl etiketi dönem olarak (yalnız üst sınır)."""
    from engine.utils.attestation_dates import work

    works = [a["work"] for a in record.get("attestations") or [] if a.get("work") in _WORK_KEYS]
    if works:
        dated = min((work(k) for k in works), key=lambda w: w.year)
        return {"year": dated.year, "precision": "point", "label": dated.label, "key": dated.key}
    periods = [a for a in record.get("attestations") or [] if a.get("century")]
    if not periods:
        return None
    first = min(periods, key=lambda a: (a["century"], a.get("ff", False)))
    century = first["century"]
    # "vııı ff." = 8. yy'dan itibaren: yıl yalnız ÜST SINIR. Runik (Türkü) ve
    # Uygur yazılı dönemleri haritadaki dönem tanımlarıyla (1000 / 1350).
    if first.get("ff") and first["dialect"] == "Türkü":
        span = work("otk_runic")
        return {"year": span.year, "precision": "period", "range": list(span.range or (700, span.year)),
                "label": "Türkü vııı ff. (runik yazıt dönemi, Clauson)"}
    if first.get("ff") and first["dialect"] == "Uyğ":
        span = work("oui")
        return {"year": span.year, "precision": "period", "range": list(span.range or (800, span.year)),
                "label": "Uyğ. vııı ff. (Eski Uygurca dönemi, Clauson)"}
    return {"year": century * 100, "precision": "period", "range": [century * 100 - 99, century * 100],
            "label": f"{first['dialect']}. {century}. yy{' ve sonrası' if first.get('ff') else ''} (Clauson)"}


class ClausonEDTFetcher(BaseFetcher):
    """Clauson EDT'den biçim + anlamla eşleşen Eski Türkçe madde (tanık + tarih)."""

    is_seed_source = False
    is_local = True
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return f"{SOURCE_LABEL} (yerel, Eski Türkçe)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        try:
            matched = matches(word)
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
            return result
        if not matched:
            return result
        for form_score, meaning_score, record in matched:
            meaning = "; ".join(record_glosses(record)[:2])
            entry = self.make_entry(LANG, record["headword"], meaning)
            entry["comparison"] = record["_forms"][0] if record.get("_forms") else ""
            entry["form_similarity"] = round(form_score, 3)
            entry["meaning_similarity"] = meaning_score
            entry["clauson_prefix"] = record.get("prefix") or ""
            if record.get("base"):
                entry["etymology"] = f"Clauson: {record['headword']} < {record['base']}"
            dated = earliest_attestation(record)
            if dated:
                entry["attestation_precision"] = dated["precision"]
            result["turkic_languages"].append(entry)
        best = matched[0][2]
        result["root"]["old_turkic_base"] = base_form(best)
        result["root"]["reconstruction_notes"] = "Eski Türkçe (Clauson 1972): " + ", ".join(
            f"{r['headword']} “{(record_glosses(r) or [''])[0]}”" for _, _, r in matched
        )
        dated = earliest_attestation(best)
        if dated:
            first = {"form": best["headword"], "meaning": (record_glosses(best) or [""])[0],
                     "source": f"{dated['label']}; {SOURCE_LABEL}", "year": dated["year"],
                     "precision": dated["precision"]}
            if dated["precision"] == "period":
                first["range"] = dated["range"]
                first["label"] = f"{dated['label']} içinde tanıklı; kesin yer yok"
            result["first_attestation"] = first
        return result
