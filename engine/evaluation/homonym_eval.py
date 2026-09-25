"""
Eşsesli ayrımı — ``make eval-homonym``.

Wiktionary'de aynı başlığın ayrı "Etymology 1/2/…" bölümleri bedava eşsesli
etiketidir (``yüz`` 'face' / 'hundred'; ``kaymak`` 'cream' / 'to slide').
kaikki dökümü her bölümü ayrı kayıt olarak ``etymology_number`` alanıyla
verir. Motor bunları hiç ayırt etmek zorunda kalmadığı için eşsesli ayrımı
şimdiye dek ölçülmedi (negatif kontrollerde ``eşadlı`` n=3).

Altın küme
----------
``data/lexicons/{tr,ota}.jsonl.gz`` (kaikki, İngilizce sürüm). Biçim
göndermesi (``form_of`` / ``form-of`` etiketi) ve "See the etymology of the
corresponding lemma form" kayıtları atılır. Aynı ``word`` için en az iki
ayrı ``etymology_number`` taşıyan başlık eşseslidir. Her etimoloji için:
anlamlar (İngilizce), köken (``lexicon_index._origin_from_templates``: miras /
alıntı + verici), Proto-Türkçe kök(ler) (``trk-pro`` şablonları + metindeki
"Proto-(Common) Turkic *…").

Ölçütler (ölçümden ÖNCE tanımlandı)
-----------------------------------
Motor ``headline_eval.build_engine`` ile koşar: yalnız yerel kaynaklar, ağ
kapalı; Starling AÇIK (``yerel``) ve KAPALI (``starling_yok``).

Anlam atama (``assign_gloss``): bir İngilizce anlam dizgisi, içerik
sözcükleri (durak sözcükler atılır, sondaki -s budanır) EN ÇOK örtüşen
etimolojiye atanır; eşitlik ya da sıfır örtüşme → atanamaz (``None``).
Türkçe anlamlar (TDK) atanamaz; oran çıktıda görünür.

Kök atama (``assign_root``): bir ``*``-biçim, ``metrics.is_acceptable`` ya
da ``normalize_proto`` eşitliğiyle kökünü tuttuğu etimolojilere atanır;
birden çok etimoloji tutarsa ``belirsiz``.

(a) **Fark etme.** Rapor eşsesliliği şu yollardan biriyle gösteriyor mu:
    * ``uyari_tanik`` — tanık süzgeci eşsesli akraba ayıkladı
      (``etymology_mentions.homonym_cognates`` boş değil; CLI "eşsesli,
      tanık sayılmadı" diye basar);
    * ``uyari_metin`` — kök notu / damga / hüküm gerekçesinde "eşsesli" ya
      da "eşadlı" geçiyor;
    * ``cok_koken`` — raporda görünen ``*``-kökler (başlık, kaynak kökleri,
      rekonstrüksiyon, hipotezler) EN AZ İKİ ayrı etimolojiye atanıyor.
    ``fark_ediyor`` = üçünden biri. Yalnız ``cok_koken`` "iki etimolojiyi
    ayrı ayrı sunuyor" demektir; uyarılar yalnız farkındalıktır.
(b) **Başlık etimolojisi.** Başlık kökü (``root.proto_turkic``) hangi
    etimolojiye atanıyor (E1, E2, …, ``belirsiz``, ``hiçbiri``). Ayrıca
    gösterilen İngilizce anlam (``root.historical_meaning``) hangi
    etimolojiye atanıyor. ``tutarsiz_baslik`` = ikisi de atandı ve FARKLI
    (kök bir etimolojinin, anlam ötekinin). Şans tabanı: başlık rastgele bir
    etimolojiden seçilseydi E1 payı = ortalama 1/k.
(c) **Tanık karışması.** Tanık listesi (``turkic_languages``) anlamı
    atanabilen tanıklar üzerinden: başlık etimolojisi belliyse ondan
    FARKLI etimolojiye atanan tanık "yabancı"dır. ``karisik`` = en az bir
    yabancı tanık; ``baslik_tanik_catismasi`` = atanan tanıkların
    çoğunluğu başka etimolojide (``yüz``: kök *yǖŕ 'face', tanıklar
    'hundred').

⚠️ Döngüsellik: motorun sözlük indeksi (``data/lexicons/index.db``) AYNI
kaikki dökümünden kurulur. Motorun gördüğü Türkçe/Osmanlıca kayıtların
kökleri ve anlamları altın kümeyle aynı kaynaktır; ölçülen şey motorun bu
bilgiyi AYIRIP AYIRAMADIĞIDIR, bağımsız bir doğruluk değildir. Starling açık
düzende başlık Starling kaydından da gelebilir (ayrıca döngüsel değil, ama
Wiktionary Proto-Türkçe biçimleri kısmen EDAL/Starostin soyundandır).
"""

from __future__ import annotations

import gzip
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SEED = 20260925
#: Ağır arama koşusu sınırı (kelime başına ~3 sn, iki düzen).
MAX_ENGINE_WORDS = 150
LEXICON_DIR = PROJECT_ROOT / "data" / "lexicons"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "eval_engine_runs"
#: Görevde adı geçen örnekler; örnekleme her zaman dahil.
ANCHORS = ("yüz", "kaymak", "ben", "baş", "kol", "dolu", "kar")

CIRCULARITY = (
    "DÖNGÜSEL KAYNAK: motorun sözlük indeksi (index.db) altın kümeyle AYNI kaikki "
    "dökümünden kurulur. Ölçülen, motorun aynı kaynaktaki iki etimolojiyi ayırıp "
    "ayıramadığıdır; bağımsız doğruluk değildir. Starling açık düzende başlık "
    "Starling'den de gelebilir (Wiktionary Proto-Türkçe biçimleri kısmen EDAL soyundan)."
)

_PROTO_CODES = frozenset({"trk-pro", "trk-pcm", "trk-cmn-pro"})
_PROTO_TEXT = re.compile(r"Proto-(?:Common )?Turkic \*([^\s,;.()“”\"]+)")
_LEMMA_POINTER = "See the etymology of the corresponding lemma"


# --- altın küme: kaikki "Etymology N" -----------------------------------------------


#: Vericisi olan köken sınıfları: yabancı alıntı ve Türk dili içi diriltme
#: (`lexicon_index.REVIVAL_ORIGIN`).
_DONOR_ORIGINS = ("alıntı", "diriltme")


def _is_form_of(record: dict[str, Any]) -> bool:
    senses = record.get("senses") or []
    if senses and all(s.get("form_of") or "form-of" in (s.get("tags") or []) for s in senses):
        return True
    return str(record.get("etymology_text", "")).startswith(_LEMMA_POINTER)


def proto_roots(record: dict[str, Any]) -> list[str]:
    """Kaydın Proto-Türkçe kök(ler)i: şablonlar, yoksa metin."""
    out: list[str] = []
    for template in record.get("etymology_templates") or []:
        args = template.get("args") or {}
        if str(args.get("2", "")).strip() in _PROTO_CODES:
            form = str(args.get("3", "")).strip()
            if form and form not in out:
                out.append(form if form.startswith("*") else "*" + form)
    if not out:
        # Şablonsuz kayıt: metindeki İLK Proto-Türkçe biçim (sonrakiler çoğu
        # zaman "Compare …" / akraba anmasıdır).
        match = _PROTO_TEXT.search(str(record.get("etymology_text") or ""))
        if match:
            out.append("*" + match.group(1))
    return out


def _glosses(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for sense in record.get("senses") or []:
        for gloss in sense.get("glosses") or []:
            gloss = str(gloss).strip()
            if gloss and gloss not in out:
                out.append(gloss)
    return out


def group_etymologies(records: Iterable[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """``{başlık: {etimoloji no: özet}}`` — yalnız numaralı, lema kayıtları."""
    from engine.db.lexicon_index import _origin_from_templates

    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        number = record.get("etymology_number")
        word = str(record.get("word", "")).strip()
        if number is None or not word or _is_form_of(record):
            continue
        slot = grouped[word].setdefault(int(number), {
            "pos": [], "glosses": [], "origin": None, "donor_lang": "", "donor_form": "",
            "proto_roots": [], "etymology": "",
        })
        pos = str(record.get("pos", ""))
        if pos and pos not in slot["pos"]:
            slot["pos"].append(pos)
        for gloss in _glosses(record):
            if gloss not in slot["glosses"]:
                slot["glosses"].append(gloss)
        for root in proto_roots(record):
            if root not in slot["proto_roots"]:
                slot["proto_roots"].append(root)
        origin, donor_lang, donor_form = _origin_from_templates(record)
        if origin and not slot["origin"]:
            slot.update(origin=origin, donor_lang=donor_lang, donor_form=donor_form)
        text = str(record.get("etymology_text") or "").strip()
        if text and not slot["etymology"]:
            slot["etymology"] = text[:240]
    return grouped


def homonyms_from_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """En az iki numaralı etimolojisi olan başlıklar (anlamı olan etimolojiler)."""
    out = []
    for word, etyms in group_etymologies(records).items():
        etyms = {n: e for n, e in etyms.items() if e["glosses"]}
        if len(etyms) < 2:
            continue
        out.append({
            "word": word,
            "etymologies": [{"number": n, **etyms[n]} for n in sorted(etyms)],
            "distinguishable": is_distinguishable(list(etyms.values())),
        })
    return sorted(out, key=lambda h: h["word"])


def is_distinguishable(etyms: Sequence[dict[str, Any]]) -> bool:
    """Köken imzası (Proto-Türkçe kök ya da verici) en az iki etimolojide var
    ve farklı: başlık kökü ancak o zaman bir etimolojiye atanabilir."""
    from engine.evaluation.metrics import normalize_proto

    signatures = set()
    roots_seen: list[set[str]] = []
    for e in etyms:
        if e.get("proto_roots"):
            roots = {normalize_proto(r.rstrip("-")) for r in e["proto_roots"]}
            # Kökleri örtüşen iki etimoloji (yüz: ikisi de *yǖŕ) kökle ayrılamaz.
            if any(roots & seen for seen in roots_seen):
                continue
            roots_seen.append(roots)
            signatures.add(("kök", tuple(sorted(roots))))
        # Türk dili içi diriltme (`anık` < Eski Uygurca) de vericili bir
        # etimolojidir; imzası alıntınınkiyle aynı kurulur (örneklem, köken
        # sınıfı ayrılmadan önceki hâliyle aynı kalır).
        elif e.get("origin") in _DONOR_ORIGINS and e.get("donor_lang"):
            signatures.add(("alıntı", e["donor_lang"], e.get("donor_form", "")))
    return len(signatures) >= 2


def iter_dump(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def dump_stats(lang: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = LEXICON_DIR / f"{lang}.jsonl.gz"
    lemma_words: set[str] = set()
    records: list[dict[str, Any]] = []
    for record in iter_dump(path):
        if _is_form_of(record) or not record.get("word"):
            continue
        lemma_words.add(record["word"])
        if record.get("etymology_number") is not None:
            records.append(record)
    homs = homonyms_from_records(records)
    kinds = Counter()
    for h in homs:
        origins = sorted({e["origin"] or "?" for e in h["etymologies"]})
        kinds["+".join(origins)] += 1
    stats = {
        "lemma_headwords": len(lemma_words),
        "homonym_headwords": len(homs),
        "ratio": round(len(homs) / len(lemma_words), 4) if lemma_words else 0.0,
        "distinguishable": sum(h["distinguishable"] for h in homs),
        "etymology_count_hist": dict(Counter(len(h["etymologies"]) for h in homs)),
        "origin_mix": dict(kinds.most_common()),
        "with_proto_root_in_2plus": sum(
            sum(bool(e["proto_roots"]) for e in h["etymologies"]) >= 2 for h in homs),
    }
    return homs, stats


# --- eşleme: anlam ve kök -> etimoloji ----------------------------------------------

_STOP = frozenset(
    "a an the of to in on or and for with by from as at be is are that this which who "
    "something someone one's oneself etc e.g i.e used especially very any some its it "
    "kind type sort being made form not".split()
)
_TOKEN = re.compile(r"[a-z]+")


def gloss_tokens(text: str) -> set[str]:
    out = set()
    for token in _TOKEN.findall((text or "").lower()):
        if len(token) < 3 or token in _STOP:
            continue
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            token = token[:-1]
        out.add(token)
    return out


def assign_gloss(gloss: str, etymologies: Sequence[dict[str, Any]]) -> int | None:
    """Anlamın içerik sözcükleri en çok hangi etimolojinin anlamlarıyla
    örtüşüyor; eşitlik ya da sıfır örtüşme -> ``None``."""
    tokens = gloss_tokens(gloss)
    if not tokens:
        return None
    scores = []
    for e in etymologies:
        vocab = set().union(*(gloss_tokens(g) for g in e["glosses"])) if e["glosses"] else set()
        scores.append((len(tokens & vocab), e["number"]))
    scores.sort(reverse=True)
    if scores[0][0] == 0 or (len(scores) > 1 and scores[1][0] == scores[0][0]):
        return None
    return scores[0][1]


def root_etymologies(form: str, etymologies: Sequence[dict[str, Any]]) -> list[int]:
    """``*``-biçimin kök olarak tuttuğu etimolojiler (numaralar)."""
    from engine.evaluation.metrics import is_acceptable, normalize_proto

    form = (form or "").strip()
    if not form:
        return []
    if not form.startswith("*"):
        form = "*" + form
    # Önce tam eşitlik (normalize_proto): uzunluk farkı ayırıcıdır
    # (düş: *tǖĺ 'dream' / *tüĺ 'noon'); tam eşleşme yoksa is_acceptable.
    exact = [e["number"] for e in etymologies
             if any(normalize_proto(form.rstrip("-")) == normalize_proto(g.rstrip("-")) for g in e["proto_roots"])]
    if exact:
        return exact
    return [e["number"] for e in etymologies
            if any(is_acceptable(form, g.rstrip("-")) for g in e["proto_roots"])]


def assign_root(form: str, etymologies: Sequence[dict[str, Any]]) -> int | str:
    """Tek etimoloji -> numarası; birden çok -> ``belirsiz``; yok -> ``hiçbiri``."""
    hits = root_etymologies(form, etymologies)
    if len(hits) == 1:
        return hits[0]
    return "belirsiz" if hits else "hiçbiri"


def donor_etymologies(claims: Sequence[str], etymologies: Sequence[dict[str, Any]]) -> list[int]:
    """Alıntı hipotezi metinlerinin tuttuğu alıntı etimolojileri. Verici adı
    ya da özgün biçim geçiyorsa tutar; etimolojinin vericisi bilinmiyorsa
    herhangi bir alıntı iddiası tutar."""
    from engine.nlp.borrowing_chain import DONOR_LANGUAGE_NAMES

    if not claims:
        return []
    text = " ".join(claims)
    out = []
    for e in etymologies:
        if e.get("origin") not in _DONOR_ORIGINS:
            continue
        name = DONOR_LANGUAGE_NAMES.get(e.get("donor_lang", ""), "")
        form = e.get("donor_form", "")
        if not e.get("donor_lang") or (name and name.split()[0] in text) or (form and form in text):
            out.append(e["number"])
    return out


# --- motor raporu özeti ---------------------------------------------------------------

_STAR = re.compile(r"\*[^\s,/();?\]\[|'\"]+")
_HOMONYM_TEXT = re.compile(r"eşsesli|eşadlı", re.IGNORECASE)


def summarize(finding: dict[str, Any]) -> dict[str, Any]:
    """Aramadan eşsesli ölçütlerinin okuduğu alanlar (önbelleğe yazılır)."""
    root = finding.get("root") or {}
    nlp = finding.get("nlp_analysis") or {}
    mentions = finding.get("etymology_mentions") or {}
    ranked = nlp.get("ranked_hypotheses") or {}
    proven = nlp.get("proven_hypothesis") or {}
    rec = nlp.get("reconstruction") or {}

    stars: list[str] = []

    def _add(text: Any) -> None:
        for token in _STAR.findall(str(text or "")):
            token = token.rstrip(".-:")
            if len(token) > 2 and token not in stars:
                stars.append(token)

    _add(root.get("proto_turkic"))
    for item in root.get("source_proto_forms") or []:
        _add(item.get("form"))
    _add(root.get("root_note"))
    _add(rec.get("reconstructed_root"))
    # ⚠️ ``reconstruction.alternative_forms`` alınmaz: aynı tanık kümesinin
    # n-en-iyi çözümleridir, ikinci bir etimoloji önerisi değil (ben: *beŋ
    # 'ben' zamirinin alternatif çözümü olarak çıkıyor).
    _add(proven.get("origin_form"))
    borrowed_claims: list[str] = []
    for hyp in ranked.get("hypotheses") or []:
        detail = hyp.get("detail") or {}
        _add(detail.get("attested_root"))
        if hyp.get("kind") == "borrowed" and not hyp.get("rejected"):
            borrowed_claims.append(json.dumps({k: hyp.get(k) for k in ("claim", "supporting", "detail")},
                                              ensure_ascii=False))
    donor = proven.get("donor_language") or ""
    if donor and "Türkçe" not in donor:
        borrowed_claims.append(f"{donor} {proven.get('origin_form') or ''}")

    scan = dict(finding)
    scan["etymology_mentions"] = {k: v for k, v in mentions.items() if k != "homonym_cognates"}
    text_hits = sorted({m.group(0).lower() for m in _HOMONYM_TEXT.finditer(json.dumps(scan, ensure_ascii=False, default=str))})

    return {
        "headline": str(root.get("proto_turkic") or ""),
        "provenance": str(root.get("provenance") or "")[:200],
        "meaning_en": str(root.get("historical_meaning") or ""),
        "meaning": str(root.get("meaning") or "")[:200],
        "verdict_kind": (ranked.get("selected") or {}).get("kind"),
        "stars": stars,
        "borrowed_claims": borrowed_claims,
        "homonym_cognates": [
            {"word": h.get("word"), "meaning": h.get("meaning")} for h in mentions.get("homonym_cognates") or []
        ],
        "homonym_text": text_hits,
        "witnesses": [
            {"lang": w.get("lang_code"), "word": w.get("word"), "meaning": str(w.get("meaning") or "")[:160]}
            for w in finding.get("turkic_languages") or []
        ],
    }


def run_engine(words: Sequence[str], *, ablate_starling: bool, fresh: bool = False) -> dict[str, dict[str, Any]]:
    """Kelime başına özet; HEAD'e bağlı önbellek (``headline_eval`` ile aynı düzen)."""
    from engine.evaluation.headline_eval import _head, build_engine

    config = "starling_yok" if ablate_starling else "yerel"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"homonym_{config}.json"
    head = _head()
    cache: dict[str, Any] = {}
    if path.exists() and not fresh:
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("head") == head:
                cache = stored.get("items", {})
        except (OSError, ValueError):
            cache = {}
    todo = [w for w in dict.fromkeys(words) if w not in cache]
    engine = build_engine(ablate_starling=ablate_starling) if todo else None
    for i, word in enumerate(todo, 1):
        try:
            finding = engine.search(word, save_to_db=False, use_qwen_agent=False, use_cache=False)
            cache[word] = summarize(finding)
        except Exception as exc:  # tek kelime hattı durdurmasın; görünür kalsın
            logger.warning("Arama başarısız: %s", word, exc_info=True)
            cache[word] = {"error": f"{type(exc).__name__}: {exc}"}
        if i % 25 == 0:
            logger.info("%s: %d/%d", config, i, len(todo))
            path.write_text(json.dumps({"head": head, "items": cache}, ensure_ascii=False), encoding="utf-8")
    path.write_text(json.dumps({"head": head, "items": cache}, ensure_ascii=False), encoding="utf-8")
    return {w: cache[w] for w in dict.fromkeys(words)}


# --- puanlama ----------------------------------------------------------------------


def score_word(item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Bir eşsesli için (a), (b), (c) — saf işlev (sentetik veriyle sınanır)."""
    etyms = item["etymologies"]
    k = len(etyms)

    # (a) fark etme
    surfaced: set[int] = set()
    for star in summary.get("stars") or []:
        hits = root_etymologies(star, etyms)
        if len(hits) == 1:  # iki etimolojinin ortak kökü (yüz: *yǖŕ) ayırt etmez
            surfaced.update(hits)
    surfaced.update(donor_etymologies(summary.get("borrowed_claims") or [], etyms))
    warn_cognate = bool(summary.get("homonym_cognates"))
    warn_text = bool(summary.get("homonym_text"))
    multi = len(surfaced) >= 2

    # (b) başlık etimolojisi
    headline = summary.get("headline") or ""
    head_etym: int | str = assign_root(headline, etyms) if headline else "hiçbiri"
    head_basis = "kök"
    if head_etym == "hiçbiri" and summary.get("verdict_kind") == "borrowed":
        # Alıntı başlığı "Verici özgün-biçim"dir: önce özgün biçim, sonra dil adı.
        hits = [e["number"] for e in etyms
                if e.get("origin") in _DONOR_ORIGINS and e.get("donor_form") and e["donor_form"] in headline]
        hits = hits or donor_etymologies([headline] + list(summary.get("borrowed_claims") or []), etyms)
        if hits:
            head_etym = hits[0] if len(hits) == 1 else "belirsiz"
            head_basis = "verici"
    meaning_etym = assign_gloss(summary.get("meaning_en") or "", etyms)
    inconsistent = isinstance(head_etym, int) and meaning_etym is not None and head_etym != meaning_etym

    # (c) tanık karışması
    assigned = [assign_gloss(w.get("meaning") or "", etyms) for w in summary.get("witnesses") or []]
    counts = Counter(a for a in assigned if a is not None)
    reference = head_etym if isinstance(head_etym, int) else meaning_etym
    foreign = sum(n for e, n in counts.items() if reference is not None and e != reference)
    n_assigned = sum(counts.values())
    majority = counts.most_common(1)[0][0] if counts else None
    return {
        "word": item["word"],
        "k": k,
        "a_uyari_tanik": warn_cognate,
        "a_uyari_metin": warn_text,
        "a_cok_koken": multi,
        "a_fark_ediyor": warn_cognate or warn_text or multi,
        "surfaced_etymologies": sorted(surfaced),
        "b_headline": headline,
        "b_headline_etym": head_etym,
        "b_headline_basis": head_basis,
        "b_meaning_etym": meaning_etym,
        "b_tutarsiz_baslik": inconsistent,
        "c_witnesses": len(assigned),
        "c_assigned": n_assigned,
        "c_by_etym": {str(e): n for e, n in sorted(counts.items())},
        "c_reference_etym": reference,
        "c_foreign": foreign,
        "c_karisik": len(counts) >= 2,
        "c_baslik_tanik_catismasi": (
            isinstance(head_etym, int) and majority is not None and majority != head_etym
        ),
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from engine.evaluation.headline_eval import wilson

    def rate(key: str, subset: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        subset = rows if subset is None else subset
        hits = sum(bool(r[key]) for r in subset)
        return {"n": len(subset), "k": hits,
                "oran": round(hits / len(subset), 4) if subset else 0.0, "ci95": wilson(hits, len(subset))}

    head_dist = Counter(
        f"E{r['b_headline_etym']}" if isinstance(r["b_headline_etym"], int) else r["b_headline_etym"] for r in rows
    )
    assigned_head = [r for r in rows if isinstance(r["b_headline_etym"], int)]
    both = [r for r in assigned_head if r["b_meaning_etym"] is not None]
    with_witness = [r for r in rows if r["c_assigned"] > 0]
    with_ref = [r for r in with_witness if r["c_reference_etym"] is not None]
    return {
        "n": len(rows),
        "a": {k: rate(k) for k in ("a_fark_ediyor", "a_uyari_tanik", "a_uyari_metin", "a_cok_koken")},
        "b": {
            "baslik_etimolojisi": dict(head_dist.most_common()),
            "baslik_E1_payi": round(head_dist.get("E1", 0) / len(assigned_head), 4) if assigned_head else 0.0,
            "sans_tabani_E1": round(sum(1 / r["k"] for r in assigned_head) / len(assigned_head), 4)
            if assigned_head else 0.0,
            "tutarsiz_baslik": rate("b_tutarsiz_baslik", both),
        },
        "c": {
            "tanik_atanan_payi": round(
                sum(r["c_assigned"] for r in rows) / max(1, sum(r["c_witnesses"] for r in rows)), 4),
            "karisik": rate("c_karisik", with_witness),
            "yabanci_tanik_payi": round(
                sum(r["c_foreign"] for r in with_ref) / max(1, sum(r["c_assigned"] for r in with_ref)), 4),
            "baslik_tanik_catismasi": rate(
                "c_baslik_tanik_catismasi", [r for r in with_witness if isinstance(r["b_headline_etym"], int)]),
        },
    }


# --- örneklem ve koşu --------------------------------------------------------------

_QUERYABLE = set("abcçdefgğhıijklmnoöprsştuüvyzâîû")


def engine_sample(homonyms: Sequence[dict[str, Any]], n: int = MAX_ENGINE_WORDS, seed: int = SEED) -> list[dict[str, Any]]:
    """Ayırt edilebilir, aranabilir başlıklar; ANCHORS her zaman dahil."""
    pool = [h for h in homonyms if h["distinguishable"] and len(h["word"]) >= 2
            and not set(h["word"]) - _QUERYABLE]
    anchors = [h for h in homonyms if h["word"] in ANCHORS and not set(h["word"]) - _QUERYABLE]
    rest = [h for h in pool if h["word"] not in ANCHORS]
    random.Random(seed).shuffle(rest)
    chosen = anchors + rest[: max(0, n - len(anchors))]
    return sorted(chosen, key=lambda h: h["word"])


def run(*, sample: int = MAX_ENGINE_WORDS, fresh: bool = False, engine: bool = True) -> dict[str, Any]:
    from engine.evaluation.headline_eval import _head

    tr_homs, tr_stats = dump_stats("tr")
    _, ota_stats = dump_stats("ota")
    payload: dict[str, Any] = {
        "_schema": "homonym_eval/v1",
        "commit": _head(),
        "circularity": CIRCULARITY,
        "source": "kaikki İngilizce Wiktionary dökümü (data/lexicons/{tr,ota}.jsonl.gz), Etymology N bölümleri",
        "dump": {"tr": tr_stats, "ota": ota_stats},
        "metrics": {
            "a": "fark_ediyor = uyari_tanik (homonym_cognates) | uyari_metin ('eşsesli'/'eşadlı') | "
                 "cok_koken (rapordaki *-kökler/alıntı iddiaları >=2 etimolojiye atanıyor)",
            "b": "başlık kökü -> etimoloji (is_acceptable); tutarsiz_baslik = kökün ve gösterilen "
                 "İngilizce anlamın etimolojileri farklı",
            "c": "tanık anlamı -> etimoloji (içerik sözcüğü örtüşmesi); karisik = tanıklar >=2 "
                 "etimolojiye yayılıyor; yabancı = başvuru (başlık, yoksa anlam) etimolojisi dışı",
        },
    }
    if not engine:
        return payload
    items = engine_sample(tr_homs, sample)
    words = [h["word"] for h in items]
    payload["sample"] = {"n": len(items), "seed": SEED, "anchors": [w for w in ANCHORS if w in words]}
    payload["configs"] = {}
    for config, ablate in (("yerel", False), ("starling_yok", True)):
        runs = run_engine(words, ablate_starling=ablate, fresh=fresh)
        rows = [score_word(h, runs[h["word"]]) for h in items if "error" not in runs[h["word"]]]
        payload["configs"][config] = {"summary": aggregate(rows), "errors": sum("error" in runs[w] for w in words),
                                      "items": rows}
    payload["gold_items"] = items
    return payload


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR

    parser = argparse.ArgumentParser(description="Eşsesli ayrımı (kaikki Etymology N)")
    parser.add_argument("--sample", type=int, default=MAX_ENGINE_WORDS)
    parser.add_argument("--fresh", action="store_true", help="önbelleği yok say")
    parser.add_argument("--no-engine", action="store_true", help="yalnız döküm istatistiği")
    args = parser.parse_args()
    payload = run(sample=min(args.sample, MAX_ENGINE_WORDS), fresh=args.fresh, engine=not args.no_engine)
    print(f"\n=== eşsesli ayrımı · commit {payload['commit']} ===")
    print(f"⚠️ {payload['circularity']}")
    for lang, st in payload["dump"].items():
        print(f"{lang}: {st['homonym_headwords']} eşsesli / {st['lemma_headwords']} lema başlık "
              f"({st['ratio']:.2%}); köken imzasıyla ayırt edilebilir {st['distinguishable']}")
    for config, block in (payload.get("configs") or {}).items():
        s = block["summary"]
        a, b, c = s["a"], s["b"], s["c"]
        print(f"\n[{config}] n={s['n']} (hata {block['errors']})")
        print(f"  (a) fark ediyor {a['a_fark_ediyor']['oran']:.3f} {a['a_fark_ediyor']['ci95']}  "
              f"tanık uyarısı {a['a_uyari_tanik']['oran']:.3f}  metin {a['a_uyari_metin']['oran']:.3f}  "
              f"çok köken {a['a_cok_koken']['oran']:.3f}")
        print(f"  (b) başlık → {b['baslik_etimolojisi']}  E1 payı {b['baslik_E1_payi']:.3f} "
              f"(şans {b['sans_tabani_E1']:.3f})  tutarsız başlık {b['tutarsiz_baslik']['k']}/"
              f"{b['tutarsiz_baslik']['n']}")
        print(f"  (c) tanık atanan pay {c['tanik_atanan_payi']:.3f}  karışık {c['karisik']['k']}/{c['karisik']['n']}  "
              f"yabancı tanık payı {c['yabanci_tanik_payi']:.3f}  başlık-tanık çatışması "
              f"{c['baslik_tanik_catismasi']['k']}/{c['baslik_tanik_catismasi']['n']}")
    out = EVAL_DIR / "homonym.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
