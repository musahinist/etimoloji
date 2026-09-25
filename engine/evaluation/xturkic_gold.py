"""
Türk dilleri arası BAĞIMSIZ alıntı altını — ``make xturkic-gold`` (plan X1).

Neden: Türkçe alıntı ölçümü döngüseldir (Wiktionary tr etiketi TDK+Nişanyan
ile %96,1 aynı) ve tek bağımsız ölçüt WOLD/Saha'dır (n=769). Bu modül kaikki
**en** dökümlerinin etimoloji şablonlarından yedi Türk dili (+ Tuvaca'nın
Moğolca katmanı) için etiketli bir küme kurar, etimona göre böler ve test
bölümünü mühürler.

Etiket (plan, bağlayıcı):

``alıntı``
    Zincirin İLK halkası ``bor|bor+|lbor|slbor`` şablonudur ve vericisi Türk
    ailesi dışıdır (makro sınıf ru/ar/fa/mn/diğer). Türk vericili miras
    şablonu (``inh``) yoktur.
``miras``
    ≥1 ``inh``; zincirin bütün halkaları Türk; alıntı şablonu yok.

Dışlananlar (her biri sayılır, ``stats.json``): Türk içi alıntı, yalnız
``der``, öyküntü (``cal``…), belirsiz (``unc``…), çelişen (aynı sözcükte iki
etiket, ya da hem alıntı hem Türk mirası şablonu), çok sözcüklü / <3 harf,
özel ad, ``form_of``, anlamsız, anlamında köken ipucu (K3).

⚠️ Plandan iki ekleme (gerekçe raporda): (1) etimoloji metninde çekince
sözcüğü (``possibly``, ``alternatively``…) taşıyan madde dışlanır — Q4
kesinliği içindir; (2) alıntı kökü + Türk eki melez türetmeler (alıntı
şablonu + yapım şablonu) dışlanır.

Bölme: grup = etimon (alıntıda verici makro sınıfı + verici biçim, mirasta
Proto-Türkçe biçim) → aynı etimonun farklı dillerdeki yansımaları aynı
bölümdedir; diller arası sızıntı yoktur. Tuz ``xturkic-borrowing-gold-v1``,
ayar %40 · R1 %20 · R2 %20 · test %20. Önce bölme, sonra bölüm içi dengeleme
(dil×sınıf ≤250·oran; Rusça payı ≤%50; Moğolca etiketlilerin hepsi).

⚠️ ``test.frozen.jsonl`` geliştirme boyunca AÇILMAZ; yalnız dalga sonunda.
Bu modül onu yazar ve mühürler, OKUMAZ.

Kullanım::

    python -m engine.evaluation.xturkic_gold build
    python -m engine.evaluation.xturkic_gold audit     # Q4 kanıt kartları (yalnız ayar)
    python -m engine.evaluation.xturkic_gold quality   # Q1–Q3
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import GOLD_DIR, LEXICON_DIR, PROJECT_ROOT
from engine.db.lexicon_index import (
    TURKIC_FAMILY_CODES,
    _etymology_text,
    _first_gloss,
    _romanised_comparison,
    comparison_for,
    parse_tree_template,
)
from engine.utils.orthography import to_comparison_form
from engine.utils.transliteration import transliterate_to_latin

XTURKIC_DIR = GOLD_DIR / "xturkic"
WORK_DIR = PROJECT_ROOT / "data" / "cache" / "work" / "xtr"

#: Bölme tuzu — değiştirilirse bütün bölme ve mühür geçersiz olur.
SPLIT_SALT = "xturkic-borrowing-gold-v1"
SPLIT_RATIOS: dict[str, float] = {"tune": 0.40, "r1": 0.20, "r2": 0.20, "test": 0.20}
SPLIT_FILES = {"tune": "tune.jsonl", "r1": "r1.jsonl", "r2": "r2.jsonl", "test": "test.frozen.jsonl"}

#: Dil × sınıf üst sınırı (bütün bölümler toplamı); bölüm payı oranla.
CAP_PER_LANG_CLASS = 250
#: Alıntılarda Rusça payı üst sınırı (dil × bölüm içinde).
MAX_RUSSIAN_SHARE = 0.5

LANGUAGES = ("kk", "ky", "tt", "ba", "uz", "ug", "tk", "tyv")
#: Q4 elle denetimi geçemeyen diller — bölümler AÇILMADAN çıkarılır (plan).
#: Etimon bölmesi ve dengeleme dil başına bağımsız olduğundan çıkarma öteki
#: dillerin maddelerini değiştirmez.
EXCLUDED_AFTER_Q4: dict[str, str] = {
    "tyv": (
        "Q4 (LLM ön-etiketi): tyv/alıntı 16/17 doğru, Wilson alt 0,730 < 0,80; "
        "ayar bölümünde genişletilecek başka tyv alıntısı yok"
    ),
}

#: Tuvaca yalnız Moğolca katmanıyla girer (plan).
MONGOLIC_ONLY = frozenset({"tyv"})
#: Verici kümeleri (ön-kayıtlı). ``donors_for`` bu diller için None döndürür;
#: ölçüm kümeyi açık verir.
KIPCHAK = frozenset({"kk", "ky", "tt", "ba", "tyv"})
DONOR_SETS: dict[str, tuple[str, ...]] = {
    **{lang: ("ru", "ar", "fa", "mn") for lang in KIPCHAK},
    **{lang: ("ar", "fa", "ru") for lang in ("uz", "ug", "tk")},
}

BORROWING = frozenset({"bor", "bor+", "lbor", "slbor"})
INHERITANCE = frozenset({"inh", "inh+"})
DERIVATION = frozenset({"der", "der+", "uder"})
#: Plan dışı alıntı türleri, öyküntü, belirsizlik — madde dışlanır.
EXCLUDING = {
    "öyküntü": frozenset({"cal", "calque", "clq", "pcal", "sl", "semantic loan", "psm", "pseudo-loan"}),
    "belirsiz": frozenset({"unc", "unk", "unknown", "uncertain", "onom", "onomatopoeic"}),
    "başka_alıntı_türü": frozenset({"ubor", "obor", "abor", "unadapted borrowing", "adapted borrowing"}),
}
FORMATION = frozenset({
    "af", "affix", "suf", "suffix", "pre", "prefix", "com", "compound", "com+",
    "surf", "surface analysis", "con", "confix", "inf", "infix", "blend",
})

#: Çekince sözcükleri (plana ek; Q4 kesinliği için).
HEDGE = re.compile(
    r"\b(possibl[ey]|perhaps|probabl[ey]|alternativel?y?|uncertain|unclear|unknown|"
    r"disputed|doubtful|obscure|may be|might be|or else|either)\b",
    re.IGNORECASE,
)

#: K3: anlamında köken ipucu taşıyan maddeler dışlanır (motor anlamı okur).
GLOSS_HINT = re.compile(
    r"\b(russian|soviet|ussr|arab\w*|persian|iran\w*|mongol\w*|islam\w*|muslim|"
    r"qur'?an\w*|koran\w*|christian\w*|orthodox|buddhis\w*|lama\w*|chinese|china|"
    r"europe\w*|english|french|german\w*|latin|greek|turkic|turkish|tatar\w*|"
    r"kazakh\w*|kyrgyz\w*|uzbek\w*|uyghur\w*|turkmen\w*|bashkir\w*|tuvan\w*|"
    r"transcription|spelling|abbreviation|initialism|acronym|letter)\b",
    re.IGNORECASE,
)

MONGOLIC = frozenset({
    "mn", "khk", "xal", "bua", "bxr", "cmg", "xng", "xwo", "mvf", "xgn-pro",
    "xgn", "mjg", "dta", "sce", "peh", "yuy", "mhj", "xgn-cen", "xgn-mid",
})
PERSIAN = frozenset({"fa", "fa-cls", "fa-ira", "prs", "tg", "fa-afg"})
ARABIC_PREFIX = ("ar", "acm", "acw", "aeb", "apc", "apd", "arq", "ary", "arz", "ayp", "afb", "ajp")


#: Şablonlarda görülen, ``TURKIC_FAMILY_CODES``te olmayan Türk kodları
#: (Wiktionary kodları; ``khk`` burada Halha MOĞOLCASIDIR, Türk değil).
TURKIC_EXTRA = frozenset({
    "oui", "xqa", "qwm", "kjh", "sty", "kmz", "jct", "aib", "xpc", "zkz", "xbo", "ybe",
})


def is_turkic(code: str) -> bool:
    code = (code or "").strip()
    return code in TURKIC_FAMILY_CODES or code in TURKIC_EXTRA or code.startswith("trk")


def donor_macro(code: str) -> str:
    code = (code or "").strip()
    if code == "ru":
        return "ru"
    if code in PERSIAN:
        return "fa"
    if code == "ar" or code in ARABIC_PREFIX:
        return "ar"
    if code in MONGOLIC:
        return "mn"
    return "diğer"


def _strip_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # Arapça hareke ve tatvil (birleşik değil, Mn sınıfı dışı kalanlar)
    kept = re.sub(r"[ً-ٰٟـ]", "", kept)
    return unicodedata.normalize("NFC", kept).casefold()


def etymon_key(macro: str, form: str) -> str:
    """Bölme grubu anahtarı: sesbilgisel olmayan işaretler atılır."""
    bare = re.sub(r"[\s*\-‐()\[\].,'\"ʼʹ]", "", _strip_marks(form))
    return f"{macro}:{bare}"


@dataclass
class Label:
    label: str | None  # 'alıntı' | 'miras' | None
    reason: str
    donor_lang: str = ""
    donor_macro: str = ""
    donor_form: str = ""
    etymon: str = ""
    chain: list[tuple[str, str, str]] = field(default_factory=list)


def _steps(record: dict[str, Any]) -> tuple[list[tuple[str, str, str]], set[str]]:
    """``(ilişki, dil, biçim)`` halkaları metin sırasıyla + görülen şablon adları."""
    steps: list[tuple[str, str, str]] = []
    names: set[str] = set()
    for template in record.get("etymology_templates") or []:
        name = str(template.get("name", "")).strip().lower()
        names.add(name)
        if name in ("etymon", "ety"):
            for rel, lang, form in parse_tree_template(template):
                rel = rel.lower()
                if rel in ("bor", "lbor", "slbor", "inh", "der"):
                    steps.append((rel, lang, form))
                else:
                    names.add(rel)
            continue
        if name in BORROWING or name in INHERITANCE or name in DERIVATION:
            args = template.get("args") or {}
            donor = str(args.get("2", "") or "").strip()
            form = str(args.get("3", "") or "").strip()
            if not form:
                form = str(args.get("tr", "") or "").strip()
            if donor:
                steps.append((name, donor, form))
    return steps, names


def label_record(record: dict[str, Any], lang: str) -> Label:
    """Tek kaikki kaydının etiketi — plan tanımı (ilk eşleşen dışlama gerekçesi)."""
    steps, names = _steps(record)
    for reason, templates in EXCLUDING.items():
        if names & templates:
            return Label(None, reason, chain=steps)
    if not steps:
        return Label(None, "etimoloji_şablonu_yok")
    if HEDGE.search(_etymology_text(record) or ""):
        return Label(None, "çekince_metni", chain=steps)

    first_rel, first_lang, first_form = steps[0]
    has_bor = any(rel in BORROWING or rel in ("bor", "lbor", "slbor") for rel, _, _ in steps)
    turkic_inh = any(rel in INHERITANCE or rel == "inh" for rel, code, _ in steps if is_turkic(code))
    any_inh = any(rel in INHERITANCE or rel == "inh" for rel, _, _ in steps)
    all_turkic = all(is_turkic(code) for _, code, _ in steps)

    if first_rel in BORROWING:
        if is_turkic(first_lang):
            return Label(None, "türk_içi_alıntı", chain=steps)
        if turkic_inh:
            return Label(None, "çelişen_şablon", chain=steps)
        if names & FORMATION:
            return Label(None, "melez_türetme", chain=steps)
        macro = donor_macro(first_lang)
        if lang in MONGOLIC_ONLY and macro != "mn":
            return Label(None, "tyv_moğolca_dışı", chain=steps)
        key_form = first_form or "?" + comparison_for(str(record.get("word", "")), lang)
        return Label(
            "alıntı", "", first_lang, macro, first_form, etymon_key(macro, key_form), steps
        )
    if has_bor:
        # İlk halka miras/türetme ama zincirde alıntı var (uz ← chg ← fa gibi).
        return Label(None, "aracılı_alıntı" if all_turkic is False else "türk_içi_alıntı", chain=steps)
    if not any_inh:
        return Label(None, "yalnız_der", chain=steps)
    if not all_turkic:
        return Label(None, "aile_dışı_der", chain=steps)
    proto = [f for _, code, f in steps if code == "trk-pro" and f]
    deepest = proto[0] if proto else next((f"{code}:{f}" for _, code, f in reversed(steps) if f), "")
    if not deepest:
        deepest = "?" + comparison_for(str(record.get("word", "")), lang)
    return Label("miras", "", "", "", "", etymon_key("trk", deepest), steps)


def record_form(record: dict[str, Any], lang: str) -> str:
    """İndeksle AYNI karşılaştırma biçimi (``iter_entries`` kuralı)."""
    word = str(record.get("word", "")).strip()
    comparison = comparison_for(word, lang)
    if not comparison:
        comparison = to_comparison_form(transliterate_to_latin(word))
    romanised = _romanised_comparison(record)
    if len(romanised) >= len(comparison):
        comparison = romanised
    return comparison


def _is_form_of(record: dict[str, Any]) -> bool:
    senses = record.get("senses") or []
    if not senses:
        return True
    return all(
        s.get("form_of") or s.get("alt_of")
        or {"form-of", "alt-of", "romanization"} & set(s.get("tags") or [])
        for s in senses
    )


def item_exclusion(record: dict[str, Any], lang: str, form: str, gloss: str) -> str:
    word = str(record.get("word", "")).strip()
    if _is_form_of(record):
        return "form_of"
    if str(record.get("pos", "")) in ("name", "proper noun"):
        return "özel_ad"
    if word[:1].isupper():
        return "özel_ad"
    if re.search(r"[\s\-‐]", word) or len(form) < 3:
        return "çok_sözcüklü_veya_kısa"
    if str(record.get("pos", "")) in ("suffix", "prefix", "infix", "affix", "character", "symbol", "punct"):
        return "ek_veya_simge"
    if not gloss.strip():
        return "anlamsız"
    if GLOSS_HINT.search(gloss):
        return "anlam_ipucu_K3"
    return ""


def _hash01(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(etymon: str) -> str:
    """``gold.assign_split`` deseni, kendi tuzuyla."""
    position = _hash01(f"{SPLIT_SALT}:{etymon}")
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS.items():
        cumulative += ratio
        if position < cumulative:
            return name
    return "test"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Aday çıkarma ----------------------------------------------------------


def extract_candidates(lang: str, path: Path | None = None) -> tuple[list[dict[str, Any]], Counter]:
    """Bir dilin bütün etiketli adayları (dengeleme öncesi) + dışlama sayıları."""
    path = path or LEXICON_DIR / f"{lang}.jsonl.gz"
    reasons: Counter = Counter()
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            word = str(record.get("word", "")).strip()
            if not word:
                continue
            lab = label_record(record, lang)
            if lab.label is None:
                reasons[lab.reason] += 1
                # Etiketsiz kayıt da eşadlı çelişkisi için tutulur (yalnız
                # alıntı/miras beyanı olanlar çelişki sayılır).
                continue
            form = record_form(record, lang)
            gloss = _first_gloss(record)
            excl = item_exclusion(record, lang, form, gloss)
            if excl:
                reasons[excl] += 1
                continue
            by_word[word].append({
                "lang": lang,
                "word": word,
                "form": form,
                "pos": str(record.get("pos", "")),
                "gloss": gloss,
                "label": lab.label,
                "donor_lang": lab.donor_lang,
                "donor_macro": lab.donor_macro,
                "donor_form": lab.donor_form,
                "etymon": lab.etymon,
                "chain": [list(s) for s in lab.chain[:6]],
                "etymology_text": (_etymology_text(record) or "")[:500],
            })
    items: list[dict[str, Any]] = []
    for word, rows in by_word.items():
        labels = {(r["label"], r["etymon"]) for r in rows}
        if len({lab for lab, _ in labels}) > 1:
            reasons["çelişen_eşadlı"] += len(rows)
            continue
        if len(labels) > 1:
            # Aynı etiket, farklı etimon (iki ayrı alıntı): grup belirsiz.
            reasons["çok_etimonlu"] += len(rows)
            continue
        item = dict(rows[0])
        item["id"] = f"{lang}:{word}"
        item["n_entries"] = len(rows)
        items.append(item)
        reasons["_kabul"] += 1
    return items, reasons


def _query_form(item: dict[str, Any]) -> str:
    """Motora verilecek biçim: sözcük okunamıyorsa (Arap yazısı) karşılaştırma biçimi."""
    return item["word"] if to_comparison_form(item["word"]) else item["form"]


def _hard_flag(item: dict[str, Any]) -> bool | None:
    """Zor alt küme (ön-kayıtlı): ``phonotactic_only``nin miras dediği alıntı."""
    if item["label"] != "alıntı":
        return None
    from engine.nlp.borrowing_detector import BorrowingDetector

    return not BorrowingDetector._phonotactic_signal(_query_form(item), item["lang"]).fired


def balance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bölüm içi dengeleme: dil × sınıf ≤ 250·oran, Rusça ≤ %50, Moğolcanın hepsi."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(item["split"], item["lang"], item["label"])].append(item)
    chosen: list[dict[str, Any]] = []
    for (split, _lang, label), pool in sorted(groups.items()):
        cap = round(CAP_PER_LANG_CLASS * SPLIT_RATIOS[split])
        pool = sorted(pool, key=lambda i: _hash01(f"{SPLIT_SALT}:sample:{i['id']}"))
        if label == "miras":
            chosen.extend(pool[:cap])
            continue
        picked = [i for i in pool if i["donor_macro"] == "mn"]
        ru_max = int(cap * MAX_RUSSIAN_SHARE)
        ru = 0
        for item in pool:
            if len(picked) >= cap:
                break
            if item["donor_macro"] == "mn":
                continue
            if item["donor_macro"] == "ru":
                if ru >= ru_max:
                    continue
                ru += 1
            picked.append(item)
        # Rusça payı son kümede de ≤ %50 olmalı (Rusça dışı az ise küçülür).
        while picked and sum(i["donor_macro"] == "ru" for i in picked) > MAX_RUSSIAN_SHARE * len(picked):
            last_ru = max(j for j, i in enumerate(picked) if i["donor_macro"] == "ru")
            picked.pop(last_ru)
        chosen.extend(picked)
    return chosen


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)


def admitted_languages() -> tuple[str, ...]:
    return tuple(lang for lang in LANGUAGES if lang not in EXCLUDED_AFTER_Q4)


def build(out_dir: Path = XTURKIC_DIR, *, force: bool = False) -> dict[str, Any]:
    all_items: list[dict[str, Any]] = []
    reasons_by_lang: dict[str, dict[str, int]] = {}
    candidates: dict[str, dict[str, int]] = {}
    provenance: dict[str, Any] = {}
    for lang in LANGUAGES:
        if lang in EXCLUDED_AFTER_Q4:
            continue
        path = LEXICON_DIR / f"{lang}.jsonl.gz"
        items, reasons = extract_candidates(lang, path)
        reasons_by_lang[lang] = dict(sorted(reasons.items()))
        candidates[lang] = dict(Counter(i["label"] for i in items))
        meta = json.loads((LEXICON_DIR / f"{lang}.provenance.json").read_text(encoding="utf-8"))
        provenance[lang] = {
            "file": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "url": meta.get("url"),
            "retrieved_at": meta.get("retrieved_at"),
        }
        all_items.extend(items)

    # Diller arası etimon: bir etimon bütün dillerde AYNI bölüme düşer.
    for item in all_items:
        item["split"] = assign_split(item["etymon"])
    chosen = balance(all_items)
    for item in chosen:
        item["query"] = _query_form(item)
        item["hard"] = _hard_flag(item)
        item["donors"] = list(DONOR_SETS[item["lang"]])
    chosen.sort(key=lambda i: (i["split"], i["lang"], i["label"], i["id"]))

    # Sızıntı denetimi: bir etimon iki bölümde olamaz.
    etymon_splits: dict[str, set[str]] = defaultdict(set)
    for item in chosen:
        etymon_splits[item["etymon"]].add(item["split"])
    leaks = sorted(e for e, s in etymon_splits.items() if len(s) > 1)
    if leaks:
        raise SystemExit(f"etimon sızıntısı: {leaks[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        split: _jsonl([i for i in chosen if i["split"] == split]) for split in SPLIT_RATIOS
    }
    checksums = {
        split: hashlib.sha256(text.encode("utf-8")).hexdigest() for split, text in payloads.items()
    }
    seal_path = out_dir / "SEAL.json"
    if seal_path.exists() and not force:
        old = json.loads(seal_path.read_text(encoding="utf-8"))
        if old.get("checksums") != checksums:
            raise SystemExit(
                "SEAL farklı: altın zaten mühürlü ve içerik değişti. Bilerek "
                "yeniden kurmak için --force (önceki ölçümler geçersizleşir)."
            )
    for split, text in payloads.items():
        path = out_dir / SPLIT_FILES[split]
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")

    counts = Counter((i["split"], i["lang"], i["label"]) for i in chosen)
    stats: dict[str, Any] = {
        "_schema": "xturkic-borrowing-gold-stats/v1",
        "total": len(chosen),
        "by_split": dict(Counter(i["split"] for i in chosen)),
        "by_split_lang_label": {
            split: {
                lang: {lab: counts[(split, lang, lab)] for lab in ("alıntı", "miras")}
                for lang in admitted_languages()
            }
            for split in SPLIT_RATIOS
        },
        "by_lang_label": {
            lang: {lab: sum(counts[(s, lang, lab)] for s in SPLIT_RATIOS) for lab in ("alıntı", "miras")}
            for lang in admitted_languages()
        },
        "donor_macro": dict(Counter(i["donor_macro"] for i in chosen if i["label"] == "alıntı")),
        "donor_macro_by_lang": {
            lang: dict(Counter(i["donor_macro"] for i in chosen if i["lang"] == lang and i["label"] == "alıntı"))
            for lang in admitted_languages()
        },
        "hard_borrowed": {
            lang: sum(1 for i in chosen if i["lang"] == lang and i["hard"]) for lang in admitted_languages()
        },
        "candidates_before_balance": candidates,
        "natural_borrowed_rate": {
            lang: round(c.get("alıntı", 0) / max(1, sum(c.values())), 4) for lang, c in candidates.items()
        },
        "exclusions": reasons_by_lang,
        "etymon_groups": len(etymon_splits),
        "excluded_languages": EXCLUDED_AFTER_Q4,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prov = {
        "_schema": "xturkic-borrowing-gold-provenance/v1",
        "source": "kaikki.org en (Wiktionary) — etymology_templates",
        "dumps": provenance,
        "code_commit": _git_head(),
        "salt": SPLIT_SALT,
        "ratios": SPLIT_RATIOS,
        "cap_per_lang_class": CAP_PER_LANG_CLASS,
        "max_russian_share": MAX_RUSSIAN_SHARE,
        "donor_sets": {k: list(v) for k, v in DONOR_SETS.items()},
        "excluded_languages": EXCLUDED_AFTER_Q4,
        "rules": {
            "alıntı": "ilk halka bor|bor+|lbor|slbor, verici Türk dışı; Türk vericili inh yok",
            "miras": ">=1 inh, bütün halkalar Türk, alıntı şablonu yok",
            "plan_ekleri": ["çekince_metni dışlanır", "melez_türetme dışlanır"],
        },
    }
    prov_text = json.dumps(prov, ensure_ascii=False, indent=2) + "\n"
    prov_path = out_dir / "provenance.json"
    # Yeniden kurulumda yalnız commit alanı değiştiyse dosya korunur.
    if prov_path.exists():
        old = json.loads(prov_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "code_commit"} == {
            k: v for k, v in json.loads(prov_text).items() if k != "code_commit"
        }:
            prov_text = prov_path.read_text(encoding="utf-8")
    prov_path.write_text(prov_text, encoding="utf-8")

    sealed_at = datetime.now(UTC).isoformat(timespec="seconds")
    if seal_path.exists():
        old = json.loads(seal_path.read_text(encoding="utf-8"))
        if old.get("checksums") == checksums:
            sealed_at = old.get("sealed_at", sealed_at)
    seal = {
        "_schema": "xturkic-borrowing-gold-seal/v1",
        "salt": SPLIT_SALT,
        "ratios": SPLIT_RATIOS,
        "counts": dict(Counter(i["split"] for i in chosen)),
        "files": SPLIT_FILES,
        "checksums": checksums,
        "sealed_at": sealed_at,
        "note": (
            "test.frozen.jsonl geliştirme boyunca açılmaz (yalnız dalga sonunda "
            "--final-report). R1/R2 yalnız commit edilmiş PREREG_<ad>.md ile, "
            "birer kez açılır."
        ),
    }
    seal_text = json.dumps(seal, ensure_ascii=False, indent=2) + "\n"
    if not seal_path.exists() or seal_path.read_text(encoding="utf-8") != seal_text:
        seal_path.write_text(seal_text, encoding="utf-8")
    return stats


def verify_seal(out_dir: Path = XTURKIC_DIR) -> dict[str, bool]:
    """Bölüm dosyalarının sha256'sı mühürle aynı mı? (test içeriği okunmaz, yalnız baytlar)"""
    seal = json.loads((out_dir / "SEAL.json").read_text(encoding="utf-8"))
    return {
        split: file_sha256(out_dir / SPLIT_FILES[split]) == seal["checksums"][split]
        for split in SPLIT_RATIOS
    }


def load_split(split: str, out_dir: Path = XTURKIC_DIR) -> list[dict[str, Any]]:
    """Bir bölümü okur. ``test`` burada OKUNAMAZ."""
    if split == "test":
        raise PermissionError("test.frozen.jsonl geliştirme boyunca açılmaz")
    if split == "r3":
        return load_r3(out_dir)
    if split not in SPLIT_FILES:
        raise ValueError(split)
    path = out_dir / SPLIT_FILES[split]
    seal = json.loads((out_dir / "SEAL.json").read_text(encoding="utf-8"))
    if file_sha256(path) != seal["checksums"][split]:
        raise RuntimeError(f"{path.name} mühürle uyuşmuyor")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- Q4: elle denetim ------------------------------------------------------

AUDIT_PER_CELL = 20


def audit_sample(tune: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dil × sınıf başına 20 madde, YALNIZ ayar bölümünden (R1/R2/test görülmez)."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in tune:
        cells[(item["lang"], item["label"])].append(item)
    out = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda i: _hash01(f"{SPLIT_SALT}:audit:{i['id']}"))
        out.extend(pool[:AUDIT_PER_CELL])
    return out


def evidence_card(item: dict[str, Any]) -> str:
    chain = " ; ".join(f"{r} {code} {f}" for r, code, f in item["chain"])
    return (
        f"[{item['id']}] ({item['form']}, {item['pos']}) '{item['gloss'][:80]}'\n"
        f"   ETİKET {item['label']} {item['donor_lang']}:{item['donor_form']}\n"
        f"   şablon: {chain[:200]}\n"
        f"   metin: {item['etymology_text'][:300]}"
    )


def write_audit(out: Path = WORK_DIR / "audit_cards.txt") -> Path:
    tune = load_split("tune")
    sample = audit_sample(tune)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(evidence_card(i) for i in sample) + "\n", encoding="utf-8")
    with (out.parent / "audit_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "lang", "label", "donor_lang", "word", "form", "gloss"])
        for i in sample:
            writer.writerow([i["id"], i["lang"], i["label"], i["donor_lang"], i["word"], i["form"], i["gloss"]])
    return out


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    from engine.evaluation.headline_eval import wilson as _w

    return _w(successes, n, z)


def audit_verdicts(verdict_csv: Path = WORK_DIR / "audit_verdicts.csv") -> dict[str, Any]:
    """Q4 hükümleri (id, verdict ∈ {doğru, yanlış, belirsiz}) → dil×sınıf kesinlik + Wilson."""
    rows = list(csv.DictReader(verdict_csv.open(encoding="utf-8")))
    cells: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        lang = row["id"].split(":", 1)[0]
        cells[f"{lang}/{row['label']}"][row["verdict"]] += 1
    report: dict[str, Any] = {}
    for cell, c in sorted(cells.items()):
        n = c["doğru"] + c["yanlış"] + c["belirsiz"]
        ok = c["doğru"]
        low, high = wilson(ok, n)
        report[cell] = {
            "n": n, "doğru": ok, "yanlış": c["yanlış"], "belirsiz": c["belirsiz"],
            "precision": round(ok / n, 4) if n else None,
            "wilson95": [round(low, 4), round(high, 4)],
            # Kabul: kesinlik ≥0,90 VE Wilson alt ≥0,80 (belirsiz = yanlış sayılır).
            "accepted": bool(n) and ok / n >= 0.90 and low >= 0.80,
        }
    return report


# --- Q1–Q3 -----------------------------------------------------------------


def _skeleton(text: str) -> str:
    """Kaba karşılaştırma iskeleti (Starling çevriyazısı ~ Kiril karşılaştırma)."""
    table = str.maketrans({
        "ž": "j", "ǯ": "j", "č": "ç", "š": "ş", "ɨ": "ı", "ä": "e", "ɔ": "o",
        "ŋ": "n", "ɣ": "g", "ġ": "g", "q": "k", "χ": "h", "x": "h", "ʷ": "",
        "ə": "e", "ǟ": "e", "ȫ": "ö", "ǖ": "ü", "ō": "o", "ā": "a", "ū": "u", "ī": "i", "ē": "e",
        "ʌ": "a", "ʁ": "g", "c": "j", "y": "j", "w": "v", "ı": "i", "ö": "o", "ü": "u",
    })
    base = _strip_marks(text).translate(table)
    base = to_comparison_form(base).translate(table)
    return re.sub(r"[^a-zçğışjh]", "", base)


def q2_starling(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Q2: Starling turcet çapraz denetimi — altın biçim o dilin turcet yansımasıyla eşleşiyor mu?"""
    from engine.db.starling import FIELD_LANGUAGES, _reflex_tokens, load_turcet

    by_lang: dict[str, set[str]] = defaultdict(set)
    code_field = {v: k for k, v in FIELD_LANGUAGES.items()}
    for etym in load_turcet():
        for lang in LANGUAGES:
            text = etym.reflexes.get(code_field.get(lang, ""), "")
            for form, _, _ in _reflex_tokens(text):
                key = _skeleton(form.strip("-"))
                if len(key) >= 3:
                    by_lang[lang].add(key)
    out: dict[str, Any] = {}
    for lang in LANGUAGES:
        if not by_lang[lang]:
            continue
        cell: dict[str, Any] = {}
        for lab in ("alıntı", "miras"):
            pool = [i for i in items if i["lang"] == lang and i["label"] == lab]
            hits = [i for i in pool if _skeleton(i["form"]) in by_lang[lang]]
            cell[lab] = {"n": len(pool), "turcet_match": len(hits),
                         "rate": round(len(hits) / len(pool), 4) if pool else None,
                         "examples": [i["id"] for i in hits[:8]] if lab == "alıntı" else []}
        out[lang] = cell
    return out


_RU_TURKIC = re.compile(r"(пратюрк|др\.-тюрк|древнетюрк|общетюрк|тюркск|прото-тюрк|пра-тюрк|ст\.-тат|кыпч)", re.I)
_RU_FOREIGN = re.compile(
    r"(русск|рус\.|араб|перс|монг|лат\.|латин|нем\.|франц|фр\.|англ|греч|итал|польск|кит\.|китайск|тадж|иран|санскр)",
    re.I,
)


def ru_edition_label(record: dict[str, Any]) -> str | None:
    """Rusça sürümün kaba etiketi: ilk köken ipucu Türk mü, yabancı mı?"""
    text = _etymology_text(record)
    if not text or "??" in text.split(".")[0]:
        return None
    head = text[:160]
    t = _RU_TURKIC.search(head)
    f = _RU_FOREIGN.search(head)
    if t and (not f or t.start() < f.start()):
        return "miras"
    if f:
        return "alıntı"
    return None


def q3_ru_kappa(candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lang in ("tt", "ba"):
        path = LEXICON_DIR / "ru_edition" / f"{lang}.jsonl.gz"
        if not path.exists():
            continue
        ru: dict[str, set[str]] = defaultdict(set)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                lab = ru_edition_label(record)
                if lab:
                    ru[str(record.get("word", "")).strip()].add(lab)
        pairs = [
            (i["label"], next(iter(ru[i["word"]])))
            for i in candidates.get(lang, [])
            if len(ru.get(i["word"], ())) == 1
        ]
        n = len(pairs)
        if not n:
            out[lang] = {"n": 0}
            continue
        agree = sum(a == b for a, b in pairs) / n
        pa = sum(a == "alıntı" for a, _ in pairs) / n
        pb = sum(b == "alıntı" for _, b in pairs) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        confusion = Counter(f"en={a}/ru={b}" for a, b in pairs)
        out[lang] = {"n": n, "agreement": round(agree, 4), "kappa": round(kappa, 4),
                     "confusion": dict(confusion)}
    return out


def q1_sakha_wold() -> dict[str, Any]:
    """Q1: aynı etiketleyici sah dökümüne uygulanır, WOLD Saha ile karşılaştırılır."""
    from engine.evaluation.borrowing_eval import load_wold_cases

    items, _ = extract_candidates("sah")
    ours = {to_comparison_form(i["word"]): i["label"] for i in items}
    pairs = []
    for case in load_wold_cases(with_witnesses=False):
        key = to_comparison_form(case.word)
        if key in ours:
            pairs.append((ours[key], "alıntı" if case.is_borrowed else "miras"))
    n = len(pairs)
    if not n:
        return {"n": 0}
    agree = sum(a == b for a, b in pairs)
    ours_b = [b for a, b in pairs if a == "alıntı"]
    ours_i = [b for a, b in pairs if a == "miras"]
    return {
        "n": n,
        "agreement": round(agree / n, 4),
        "borrowed_precision_vs_wold": round(sum(b == "alıntı" for b in ours_b) / len(ours_b), 4) if ours_b else None,
        "inherited_precision_vs_wold": round(sum(b == "miras" for b in ours_i) / len(ours_i), 4) if ours_i else None,
        "n_borrowed": len(ours_b), "n_inherited": len(ours_i),
        "note": "plan ölçümü (index.db etiketi): 466 madde %94,4; bu satır AYNI etiketleyiciyle",
    }


def quality(out: Path = WORK_DIR / "quality.json") -> dict[str, Any]:
    tune = load_split("tune")
    candidates = {lang: extract_candidates(lang)[0] for lang in ("tt", "ba")}
    report = {
        "Q1_sah_wold": q1_sakha_wold(),
        "Q1_tr_note": "tr↔TDK+Nişanyan %96,1 (plan; Türkçe altın yeniden okunmadı)",
        "Q2_starling_tune": q2_starling(tune),
        "Q3_ru_edition_kappa_candidates": q3_ru_kappa(candidates),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# --- R3 (plan X5): daha büyük, dokunulmamış rapor bölümü --------------------
#
# X1 altını (tune/R1/R2/test) DEĞİŞMEZ; R3 ayrı dosya + ayrı mühürdür
# (``r3.jsonl``, ``SEAL_r3.json``, ``stats_r3.json``). İki kaynak:
#
# (a) X1'de kullanılmayan Türk dilleri: kaikki en dökümü olanlar (az, crh,
#     gag, kum, nog, alt). kaa: kaikki en dökümü yok (404); krc: yalnız Rusça
#     sürüm (``gap/``) — plan gereği altın değil. sah: WOLD (Q1'de kullanılır).
# (b) X1 dillerinde dengeleme sınırı (250) yüzünden seçilmemiş maddeler.
#
# Etimon sızıntısı yok: X1 kurulumu bellekte YENİDEN hesaplanır (dosyalar
# okunmaz; yeniden hesaplanan bölüm içeriklerinin sha256'sı SEAL.json ile
# birebir aynı olmalı) ve herhangi bir X1 bölümünde (test DAHİL) bulunan
# her etimon grubu R3'ten dışlanır. R3'ün kendi içinde diller arası aynı
# etimon zaten tek bölümdedir.
#
# Melez türetme süzgeci (X1 kullanıcı denetimi izi; ug نۇقسانسىز) R3'te
# güçlendirilir — X1 bölümlerine UYGULANMAZ (mühür bozulmasın):
#   * miras + yapım şablonu (af/suf/surf/com…) bileşeni aynı dökümde
#     yabancı kökenli bir sözcükse (zincirinde Türk dışı halka, ``root``
#     şablonu Türk dışı ya da metinde yabancı dil) -> ``melez_kök``;
#   * miras + etimoloji metninin akraba/karşılaştırma cümleleri DIŞINDA
#     Türk dışı dil adı (Çağatay aracılı "inherited" alıntılar: uz harorat,
#     ug مۇسۇلمان) -> ``miras_metninde_yabancı_dil``;
#   * miras + ``root``/``der`` şablonunda Türk dışı dil -> ``miras_yabancı_kök``.

R3_NEW_LANGUAGES = ("az", "crh", "gag", "kum", "nog", "alt")
R3_UNAVAILABLE = {
    "kaa": "kaikki en dökümü yok (Karakalpak, 404; 2026-09-25)",
    "krc": "yalnız Rusça sürüm (data/lexicons/gap/krc) — plan: ru sürümü altın değil",
    "sah": "WOLD ile örtüşüyor (Q1'de kullanılır)",
}
#: Yeni dillerin verici kümeleri (ön-kayıtlı, X1 deseni): Oğuz -> ar, fa, ru;
#: Kıpçak + Altay -> ru, ar, fa, mn.
R3_DONOR_SETS: dict[str, tuple[str, ...]] = {
    **DONOR_SETS,
    **{lang: ("ar", "fa", "ru") for lang in ("az", "crh", "gag")},
    **{lang: ("ru", "ar", "fa", "mn") for lang in ("kum", "nog", "alt")},
}
R3_FILE = "r3.jsonl"
R3_SEAL = "SEAL_r3.json"
R3_STATS = "stats_r3.json"
#: R3 dil × sınıf üst sınırı. MDE hesabından (MDE_r3.json) sonra seçildi,
#: R3 hiç okunmadan; sınıf başına en az ``R3_MIN_CELL`` madde yoksa hücre
#: alınmaz (Q4 denetlenemez).
R3_CAP_PER_LANG_CLASS = 350
R3_MIN_CELL = 20
#: Q4'ü geçemeyen R3 hücreleri (dil/sınıf) — mühürden ÖNCE çıkarılır.
R3_EXCLUDED_CELLS: dict[str, str] = {
    "alt/alıntı": (
        "Q4 (LLM ön-etiketi): 18/20, Wilson alt 0,699; havuzda 27 madde — hepsi doğru olsa "
        "25/27, Wilson alt 0,766 < 0,80 (data/cache/work/xtr/x5/Q4_r3.json)"
    ),
}
#: Q4'te (alt безгек) görülen, ``HEDGE``in kaçırdığı çekince: "Or from …",
#: "or an early borrowing …". Yalnız R3'te uygulanır.
R3_HEDGE_EXTRA = re.compile(r"(?:^|[.\n]\s*)Or\s+(?:from|a|an|perhaps|via)\b|\bor an? (?:early |late )?borrowing\b")

_FOREIGN_TEXT = re.compile(
    r"\b(Arabic|Persian|Farsi|Iranian|Russian|Mongol\w*|Chinese|Sanskrit|Sogdian|Tocharian|"
    r"Greek|Latin|French|German|Tibetan|Hindi|Urdu|Italian|English|Armenian|Georgian|Slavic|"
    r"Khotanese|Aramaic|Hebrew|Syriac|Polish|Ukrainian)\b"
)
_COGNATE_SENTENCE = re.compile(r"^\s*(cognates?|compare|related|see also|see|cf\.?|doublet)\b", re.IGNORECASE)


def _non_cognate_foreign(text: str) -> str:
    """Akraba/karşılaştırma cümleleri dışında geçen ilk Türk dışı dil adı."""
    for sentence in re.split(r"(?<=[.;\n])\s+", text or ""):
        if _COGNATE_SENTENCE.match(sentence):
            continue
        # "Cognate with …" satırından sonraki liste satırları da atlanır.
        match = _FOREIGN_TEXT.search(sentence)
        if match and not re.search(r"\bcognate", sentence, re.IGNORECASE):
            return match.group(0)
    return ""


def _foreign_templates(record: dict[str, Any]) -> bool:
    for template in record.get("etymology_templates") or []:
        name = str(template.get("name", "")).strip().lower()
        if name in ("root", "der", "der+", "uder", "bor", "bor+", "lbor", "slbor", "inh", "inh+"):
            code = str((template.get("args") or {}).get("2", "") or "").strip()
            if code and not is_turkic(code):
                return True
    return False


def _formation_components(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for template in record.get("etymology_templates") or []:
        if str(template.get("name", "")).strip().lower() not in FORMATION:
            continue
        for key, value in (template.get("args") or {}).items():
            if key.isdigit() and int(key) >= 2:
                comp = str(value or "").split("<")[0].strip().strip("-‐ـ").strip()
                if comp:
                    out.append(comp)
    return out


def foreign_origin_words(lang: str, path: Path | None = None) -> set[str]:
    """Dökümde yabancı kökenli sayılan sözcükler (melez kök denetimi için)."""
    path = path or LEXICON_DIR / f"{lang}.jsonl.gz"
    out: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            word = str(record.get("word", "")).strip()
            steps, _ = _steps(record)
            if (
                any(not is_turkic(code) for _, code, _ in steps)
                or _foreign_templates(record)
                or _non_cognate_foreign(_etymology_text(record))
            ):
                out.add(word)
    return out


def hybrid_reason(record: dict[str, Any], label: str, foreign_words: set[str]) -> str:
    """Güçlendirilmiş melez türetme süzgeci (yalnız R3; bkz. bölüm başı)."""
    if R3_HEDGE_EXTRA.search(_etymology_text(record) or ""):
        return "çekince_or"
    if label != "miras":
        return ""
    if _foreign_templates(record):
        return "miras_yabancı_kök"
    if _non_cognate_foreign(_etymology_text(record)):
        return "miras_metninde_yabancı_dil"
    word = str(record.get("word", "")).strip()
    if any(comp in foreign_words and comp != word for comp in _formation_components(record)):
        return "melez_kök"
    return ""


def extract_candidates_r3(lang: str) -> tuple[list[dict[str, Any]], Counter]:
    """``extract_candidates`` + güçlendirilmiş melez süzgeci (yalnız R3)."""
    path = LEXICON_DIR / f"{lang}.jsonl.gz"
    items, reasons = extract_candidates(lang, path)
    foreign = foreign_origin_words(lang, path)
    hybrid: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            word = str(record.get("word", "")).strip()
            lab = label_record(record, lang)
            if lab.label is None:
                continue
            why = hybrid_reason(record, lab.label, foreign)
            if why:
                hybrid.setdefault(word, why)
    kept = []
    for item in items:
        why = hybrid.get(item["word"])
        if why:
            reasons[why] += 1
            reasons["_kabul"] -= 1
            continue
        kept.append(item)
    return kept, reasons


def x1_used_etymons(out_dir: Path = XTURKIC_DIR) -> tuple[set[str], set[str]]:
    """X1 bölümlerinin (test DAHİL) etimon grupları ve madde kimlikleri.

    Bölüm dosyaları OKUNMAZ: ``build`` seçimi bellekte yeniden hesaplanır ve
    içeriklerin sha256'sı ``SEAL.json`` ile karşılaştırılır (yeniden üretim
    kanıtı). Test maddelerinin yalnız etimon anahtarı kullanılır.
    """
    all_items: list[dict[str, Any]] = []
    for lang in LANGUAGES:
        if lang in EXCLUDED_AFTER_Q4:
            continue
        all_items.extend(extract_candidates(lang)[0])
    for item in all_items:
        item["split"] = assign_split(item["etymon"])
    chosen = balance(all_items)
    for item in chosen:
        item["query"] = _query_form(item)
        item["hard"] = _hard_flag(item)
        item["donors"] = list(DONOR_SETS[item["lang"]])
    chosen.sort(key=lambda i: (i["split"], i["lang"], i["label"], i["id"]))
    seal = json.loads((out_dir / "SEAL.json").read_text(encoding="utf-8"))
    for split in SPLIT_RATIOS:
        text = _jsonl([i for i in chosen if i["split"] == split])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != seal["checksums"][split]:
            raise SystemExit(f"X1 yeniden hesaplanamadı ({split} sha256 mühürle aynı değil)")
    return {i["etymon"] for i in chosen}, {i["id"] for i in chosen}


def balance_r3(items: list[dict[str, Any]], cap: int = R3_CAP_PER_LANG_CLASS) -> list[dict[str, Any]]:
    """R3 dengelemesi: dil × sınıf ≤ cap, Rusça ≤ %50, Moğolcanın hepsi (X1 kuralı)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(item["lang"], item["label"])].append(item)
    chosen: list[dict[str, Any]] = []
    for (_lang, label), pool in sorted(groups.items()):
        pool = sorted(pool, key=lambda i: _hash01(f"{SPLIT_SALT}:r3sample:{i['id']}"))
        if label == "miras":
            picked = pool[:cap]
        else:
            picked = [i for i in pool if i["donor_macro"] == "mn"][:cap]
            ru_max = int(cap * MAX_RUSSIAN_SHARE)
            ru = 0
            for item in pool:
                if len(picked) >= cap:
                    break
                if item["donor_macro"] == "mn":
                    continue
                if item["donor_macro"] == "ru":
                    if ru >= ru_max:
                        continue
                    ru += 1
                picked.append(item)
            while picked and sum(i["donor_macro"] == "ru" for i in picked) > MAX_RUSSIAN_SHARE * len(picked):
                last_ru = max(j for j, i in enumerate(picked) if i["donor_macro"] == "ru")
                picked.pop(last_ru)
        if len(picked) < R3_MIN_CELL:
            continue
        chosen.extend(picked)
    return chosen


def r3_pool() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """R3 aday havuzu (dengeleme öncesi) + dışlama sayıları."""
    used_etymons, used_ids = x1_used_etymons()
    pool: list[dict[str, Any]] = []
    report: dict[str, Any] = {"exclusions": {}, "candidates": {}, "etymon_leak_excluded": {}}
    for lang in (*admitted_languages(), *R3_NEW_LANGUAGES):
        items, reasons = extract_candidates_r3(lang)
        leak = Counter()
        for item in items:
            if item["id"] in used_ids:
                continue
            if item["etymon"] in used_etymons:
                leak[item["label"]] += 1
                continue
            item["source"] = "yeni_dil" if lang in R3_NEW_LANGUAGES else "x1_artık"
            pool.append(item)
        report["exclusions"][lang] = dict(sorted(reasons.items()))
        report["etymon_leak_excluded"][lang] = dict(leak)
        report["candidates"][lang] = dict(Counter(i["label"] for i in pool if i["lang"] == lang))
    return pool, report


def load_r3_dataset_from_pool(pool: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    chosen = [i for i in balance_r3(pool, cap) if f"{i['lang']}/{i['label']}" not in R3_EXCLUDED_CELLS]
    for item in chosen:
        item["split"] = "r3"
        item["query"] = _query_form(item)
        item["hard"] = _hard_flag(item)
        item["donors"] = list(R3_DONOR_SETS[item["lang"]])
    chosen.sort(key=lambda i: (i["lang"], i["label"], i["id"]))
    return chosen


def r3_languages() -> tuple[str, ...]:
    """R3'teki diller (mühürlü istatistikten)."""
    stats = json.loads((XTURKIC_DIR / R3_STATS).read_text(encoding="utf-8"))
    return tuple(stats["languages"])


def build_r3(out_dir: Path = XTURKIC_DIR, *, cap: int = R3_CAP_PER_LANG_CLASS, dry: bool = False,
             force: bool = False) -> dict[str, Any]:
    pool, report = r3_pool()
    chosen = load_r3_dataset_from_pool(pool, cap)
    counts = Counter((i["lang"], i["label"]) for i in chosen)
    langs = sorted({i["lang"] for i in chosen})
    stats: dict[str, Any] = {
        "_schema": "xturkic-borrowing-gold-r3-stats/v1",
        "total": len(chosen),
        "languages": langs,
        "cap_per_lang_class": cap,
        "by_lang_label": {lang: {lab: counts[(lang, lab)] for lab in ("alıntı", "miras")} for lang in langs},
        "by_source": dict(Counter(i["source"] for i in chosen)),
        "donor_macro": dict(Counter(i["donor_macro"] for i in chosen if i["label"] == "alıntı")),
        "donor_macro_by_lang": {
            lang: dict(Counter(i["donor_macro"] for i in chosen if i["lang"] == lang and i["label"] == "alıntı"))
            for lang in langs
        },
        "hard_borrowed": {lang: sum(1 for i in chosen if i["lang"] == lang and i["hard"]) for lang in langs},
        "etymon_groups": len({i["etymon"] for i in chosen}),
        "pool_before_balance": report["candidates"],
        # Doğal oran: dilin BÜTÜN etiketli adayları (X1 dilleri için X1 stats ile aynı tanım).
        "natural_borrowed_rate": {},
        "exclusions": report["exclusions"],
        "etymon_leak_excluded": report["etymon_leak_excluded"],
        "excluded_cells": R3_EXCLUDED_CELLS,
        "unavailable_languages": R3_UNAVAILABLE,
        "donor_sets": {lang: list(R3_DONOR_SETS[lang]) for lang in langs},
    }
    x1_stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    for lang in langs:
        if lang in x1_stats["natural_borrowed_rate"]:
            stats["natural_borrowed_rate"][lang] = x1_stats["natural_borrowed_rate"][lang]
        else:
            c = Counter(i["label"] for i in extract_candidates(lang)[0])
            stats["natural_borrowed_rate"][lang] = round(c.get("alıntı", 0) / max(1, sum(c.values())), 4)
    if dry:
        return stats
    text = _jsonl(chosen)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seal_path = out_dir / R3_SEAL
    if seal_path.exists() and not force:
        old = json.loads(seal_path.read_text(encoding="utf-8"))
        if old.get("checksum") != checksum:
            raise SystemExit("SEAL_r3 farklı: R3 zaten mühürlü ve içerik değişti (--force bilerek)")
    path = out_dir / R3_FILE
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    (out_dir / R3_STATS).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sealed_at = datetime.now(UTC).isoformat(timespec="seconds")
    if seal_path.exists():
        old = json.loads(seal_path.read_text(encoding="utf-8"))
        if old.get("checksum") == checksum:
            sealed_at = old.get("sealed_at", sealed_at)
    seal = {
        "_schema": "xturkic-borrowing-gold-r3-seal/v1",
        "file": R3_FILE,
        "count": len(chosen),
        "checksum": checksum,
        "x1_seal_checksums": json.loads((out_dir / "SEAL.json").read_text(encoding="utf-8"))["checksums"],
        "sources": {
            "x1_languages": list(admitted_languages()),
            "new_languages": [lang for lang in R3_NEW_LANGUAGES if lang in langs],
            "dumps": {
                lang: file_sha256(LEXICON_DIR / f"{lang}.jsonl.gz") for lang in langs
            },
        },
        "code_commit": _git_head(),
        "sealed_at": sealed_at,
        "note": (
            "R3 yalnız commit edilmiş PREREG_<ad>.md ile BİR KEZ açılır. X1 bölümlerinde "
            "(test dahil) bulunan hiçbir etimon grubu R3'te yok (bellekte yeniden hesaplandı)."
        ),
    }
    seal_text = json.dumps(seal, ensure_ascii=False, indent=2) + "\n"
    if not seal_path.exists() or seal_path.read_text(encoding="utf-8") != seal_text:
        seal_path.write_text(seal_text, encoding="utf-8")
    return stats


def load_r3(out_dir: Path = XTURKIC_DIR) -> list[dict[str, Any]]:
    path = out_dir / R3_FILE
    seal = json.loads((out_dir / R3_SEAL).read_text(encoding="utf-8"))
    if file_sha256(path) != seal["checksum"]:
        raise RuntimeError("r3.jsonl mühürle uyuşmuyor")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def r3_audit_sample(pool: list[dict[str, Any]], langs: tuple[str, ...] = R3_NEW_LANGUAGES,
                    per_cell: int = AUDIT_PER_CELL) -> list[dict[str, Any]]:
    """Q4 (R3, yeni diller): dil × sınıf başına ``per_cell`` madde.

    ⚠️ Örneklem R3 HAVUZUNDAN (dengeleme öncesi) çekilir, motor çıktısı
    görülmeden; yalnız etiket kalitesi içindir (kanıt kartı = altın etiket +
    şablon + metin). R3'ün değerlendirme açılışı (motor kararları) değildir.
    """
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        if item["lang"] in langs:
            cells[(item["lang"], item["label"])].append(item)
    out = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda i: _hash01(f"{SPLIT_SALT}:r3audit:{i['id']}"))
        out.extend(ordered[:per_cell])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Türk dilleri arası bağımsız alıntı altını")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    sub.add_parser("audit")
    sub.add_parser("quality")
    sub.add_parser("verify")
    b3 = sub.add_parser("build-r3", help="X5: R3 bölümü (ayrı dosya + ayrı mühür)")
    b3.add_argument("--cap", type=int, default=R3_CAP_PER_LANG_CLASS)
    b3.add_argument("--dry", action="store_true")
    b3.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build-r3":
        stats = build_r3(cap=args.cap, dry=args.dry, force=args.force)
        print(json.dumps({k: stats[k] for k in ("total", "by_lang_label", "by_source", "donor_macro",
                                                 "etymon_groups", "pool_before_balance")},
                         ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "build":
        stats = build(force=args.force)
        print(json.dumps({k: stats[k] for k in ("total", "by_split", "by_lang_label", "donor_macro", "hard_borrowed")},
                         ensure_ascii=False, indent=1))
        print("mühür:", verify_seal())
    elif args.cmd == "audit":
        print(write_audit())
    elif args.cmd == "quality":
        print(json.dumps(quality(), ensure_ascii=False, indent=1))
    elif args.cmd == "verify":
        print(verify_seal())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
