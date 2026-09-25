"""
Türk dilleri arası alıntı değerlendirmesi — ``make eval-xborrowing`` (plan X1).

Altın: ``engine/evaluation/xturkic_gold.py`` (kaikki en şablonları, etimona
göre bölünmüş, test mühürlü). ``borrowing_eval.py`` DEĞİŞMEZ; parçaları
buradan içe aktarılır.

**Ana ölçüm: KÖR indeks + zincir kapalı.** Altın etiketi Wiktionary
şablonlarından gelir; motorun ``zincir_kanıtı`` sinyali aynı etiketi sözlük
indeksinden okur. Ölçüm ``ETY_LEXICON_INDEX`` köken sütunları boşaltılmış
kopyayı (``scripts/build_blind_index.py``) gösterirken yapılır.

İki aşama:

``capture``
    Her madde için bütün sinyaller BİR KEZ hesaplanır ve önbelleğe yazılır
    (tanıklar, kural sinyalleri, verici yakınlığı + verici etiketi, tam
    indeksle zincir — yalnız K1 için). Ağır kısım budur.
``replay``
    Önbellekten; fonotaktik dizilim modeli ve birleştirici BELLEKTE eğitilir
    (K7: ``phonotactic_lm.save`` / ``borrowing_combiner.save`` çağrılmaz —
    çağrılırsa hata verir). Ayar bölümünde etimona göre gruplanmış 5 katlı
    çapraz uydurma: her maddenin kararı onu görmemiş modelden gelir.

Sistemler: ``always_*``, ``phonotactic_only``, ``phonotactic_model_only``,
``donor_proximity_only``, ``engine_trained`` (bellek içi, zincir kapalı),
``engine_prod`` (üretim birleştiricisi — WOLD/Saha'da eğitildi: ALAN DIŞI).

⚠️ ``--split r1|r2`` yalnız commit edilmiş bir ``--prereg`` dosyasıyla ve
bir kez açılır (kilit ``data/cache/work/xtr/.opened_<split>_<ad>``). ``test``
yalnız ``--final-report`` ile.

Kullanım::

    ETY_LEXICON_INDEX=data/cache/work/xtr/index_blind.db \\
        python -m engine.evaluation.xborrowing_eval capture --split tune
    python -m engine.evaluation.xborrowing_eval replay --split tune
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.evaluation.borrowing_eval import PRF, find_witnesses
from engine.evaluation.xturkic_gold import (
    SPLIT_SALT,
    WORK_DIR,
    XTURKIC_DIR,
    _hash01,
    admitted_languages,
    donor_macro,
    load_split,
)
from engine.utils.orthography import to_comparison_form

FULL_INDEX = PROJECT_ROOT / "data" / "lexicons" / "index.db"
BLIND_INDEX = WORK_DIR / "index_blind.db"
FOLDS = 5
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260925

SYSTEMS = (
    "always_inherited",
    "always_borrowed",
    "phonotactic_only",
    "phonotactic_model_only",
    "donor_proximity_only",
    "engine_trained",
    "engine_prod",
)


def cache_path(split: str, tag: str = "") -> Path:
    return WORK_DIR / f"signals_{split}{('_' + tag) if tag else ''}.jsonl"


# --- K7: bellek içi eğitim bekçisi ------------------------------------------


def forbid_model_writes() -> None:
    """Üretim model dosyalarına yazmayı yasaklar (K7)."""
    from engine.nlp import borrowing_combiner, phonotactic_lm

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("K7: xborrowing_eval model dosyası YAZMAZ (bellek içi eğitim)")

    phonotactic_lm.save = _refuse  # type: ignore[assignment]
    borrowing_combiner.save = _refuse  # type: ignore[assignment]


# --- capture -----------------------------------------------------------------


def _assert_blind() -> None:
    from engine.db import lexicon_index

    path = Path(lexicon_index.INDEX_PATH)
    if path.resolve() != BLIND_INDEX.resolve():
        raise SystemExit(
            f"ETY_LEXICON_INDEX kör indeksi göstermiyor ({path}); ana ölçüm kör "
            f"indeksle yapılır: ETY_LEXICON_INDEX={BLIND_INDEX}"
        )
    import sqlite3

    with sqlite3.connect(path) as connection:
        info = dict(connection.execute("SELECT key, value FROM build_info").fetchall())
        left = connection.execute("SELECT COUNT(*) FROM entries WHERE origin IS NOT NULL").fetchone()[0]
    if info.get("blind") != "1" or left:
        raise SystemExit(f"kör indeks değil: blind={info.get('blind')} origin dolu={left}")


def capture(split: str, *, limit: int = 0, tag: str = "") -> Path:
    """Sinyalleri hesaplar ve önbelleğe yazar (yarıda kalırsa kaldığı yerden sürer)."""
    _assert_blind()
    forbid_model_writes()
    from engine.db.lexicon_index import LexiconIndex
    from engine.nlp.borrowing_detector import BorrowingDetector
    from engine.nlp.donor_proximity import attribute_donor, nearest_donor

    items = load_split(split)
    if limit:
        items = items[:limit]
    out = cache_path(split, tag)
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    blind = BorrowingDetector()
    full = BorrowingDetector(index=LexiconIndex(FULL_INDEX))
    started = time.time()
    with out.open("a", encoding="utf-8") as handle:
        for n, item in enumerate(items):
            if item["id"] in done:
                continue
            query, lang, sense = item["query"], item["lang"], item["gloss"]
            donors = list(item["donors"])
            witnesses = find_witnesses(query, sense=sense, source_lang=lang)
            entries = [{"lang_code": c, "word": w} for c, w in witnesses]
            verdict = blind.detect(query, entries, lang=lang, sense=sense, donors=donors)
            signals = {s.name: (s.strength if s.fired else 0.0) for s in verdict.signals}
            chain_full, _, chain_donor = full._chain_signal(query, lang)
            comparison = to_comparison_form(query)
            match = nearest_donor(comparison, sense, languages=donors)
            attribution = None
            if match is not None and signals.get("verici_yakınlığı", 0.0) > 0:
                attribution = attribute_donor(comparison, sense, languages=donors)
            row = {
                "id": item["id"],
                "lang": lang,
                "y": item["label"] == "alıntı",
                "etymon": item["etymon"],
                "donor_macro": item["donor_macro"],
                "hard": item["hard"],
                "query": query,
                "form": item["form"],
                "witnesses": [list(w) for w in witnesses],
                "signals": signals,
                "phonotactic_rule": bool(signals.get("fonotaktik_ihlal", 0.0) > 0),
                "chain_full": chain_full.strength if chain_full.fired else 0.0,
                "chain_full_donor": chain_donor,
                "dp": None if match is None else {
                    "lang": match.lang_code, "word": match.word, "comparison": match.comparison,
                    "distance": round(match.distance, 4), "pct": match.chance_percentile,
                    "close": match.is_close,
                },
                "attributed": None if attribution is None else attribution.lang_code,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 100 == 0:
                print(f"{n}/{len(items)} {time.time() - started:.0f}s", flush=True)
    return out


def load_cache(split: str, tag: str = "") -> list[dict[str, Any]]:
    path = cache_path(split, tag)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    admitted = set(admitted_languages())
    return [r for r in rows if r["lang"] in admitted]


# --- replay ------------------------------------------------------------------


def _fold(etymon: str, k: int = FOLDS) -> int:
    return int(_hash01(f"{SPLIT_SALT}:fold:{etymon}") * k)


def _half(etymon: str) -> int:
    return int(_hash01(f"{SPLIT_SALT}:half:{etymon}") * 2)


def _fit_lms(rows: list[dict[str, Any]]):
    from engine.nlp.phonotactic_lm import fit

    models = {}
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    for lang, pool in by_lang.items():
        try:
            models[lang] = fit([(r["query"], r["y"]) for r in pool], language=lang, trained_on=f"xtr/{lang}/mem")
        except ValueError:
            continue
    return models


def _features(row: dict[str, Any], lms: dict[str, Any], *, chain: str, dp_binary: bool = False) -> dict[str, float]:
    feats = dict(row["signals"])
    model = lms.get(row["lang"])
    feats["fonotaktik_model"] = model.strength(row["query"]) if model is not None else 0.0
    if chain == "off":
        feats["zincir_kanıtı"] = 0.0
    elif chain == "full":
        feats["zincir_kanıtı"] = row["chain_full"]
    # chain == "blind": kör indeksin zinciri (K2: hep 0 olmalı)
    if dp_binary:
        feats["verici_yakınlığı"] = 1.0 if (row["dp"] or {}).get("close") else 0.0
    return feats


def _train_predict(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> dict[str, dict[str, bool]]:
    """Ayar protokolü: LM eğitim yarısının A parçasında, birleştirici B parçasında."""
    from engine.nlp.borrowing_combiner import fit

    part_a = [r for r in train if _half(r["etymon"]) == 0]
    part_b = [r for r in train if _half(r["etymon"]) == 1]
    lms = _fit_lms(part_a)
    variants = {
        "engine_trained": dict(chain="off"),
        "engine_trained_chain_full": dict(chain="full"),     # K1
        "engine_trained_chain_blind": dict(chain="blind"),   # K2
        "engine_trained_dp_binary": dict(chain="off", dp_binary=True),  # MDE vekili
    }
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for name, kw in variants.items():
        samples = [(_features(r, lms, **kw), r["y"]) for r in part_b]
        combiner = fit(samples, trained_on=f"xtr/tune/mem/{name}")
        for r in test:
            out[name][r["id"]] = combiner.predict(_features(r, lms, **kw))
        if name == "engine_trained":
            out["_combiner"] = combiner.as_dict()  # type: ignore[assignment]
    for r in test:
        model = lms.get(r["lang"])
        out["phonotactic_model_only"][r["id"]] = model is not None and model.predict(r["query"])
    return out


def _prod_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    """Üretim birleştiricisi (WOLD/Saha) — ALAN DIŞI; fonotaktik model diskteki (bu diller için yok)."""
    from engine.nlp.borrowing_combiner import load
    from engine.nlp.phonotactic_lm import load as load_lm

    combiner = load()
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    if combiner is None:
        return out
    lms = {lang: load_lm(lang) for lang in {r["lang"] for r in rows}}
    lms = {k: v for k, v in lms.items() if v is not None}
    for r in rows:
        out["engine_prod"][r["id"]] = combiner.predict(_features(r, lms, chain="off"))
        out["engine_prod_chain_full"][r["id"]] = combiner.predict(_features(r, lms, chain="full"))
    out["_meta"] = {"trained_on": combiner.trained_on, "n": combiner.n, "lms_on_disk": sorted(lms)}  # type: ignore[assignment]
    return out


def crossfit(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, bool]], list[dict[str, Any]]]:
    """Ayar bölümünde kat dışı kararlar."""
    preds: dict[str, dict[str, bool]] = defaultdict(dict)
    combiners = []
    for k in range(FOLDS):
        train = [r for r in rows if _fold(r["etymon"]) != k]
        test = [r for r in rows if _fold(r["etymon"]) == k]
        fold = _train_predict(train, test)
        combiners.append(fold.pop("_combiner"))  # type: ignore[arg-type]
        for name, p in fold.items():
            preds[name].update(p)
    for r in rows:
        preds["always_inherited"][r["id"]] = False
        preds["always_borrowed"][r["id"]] = True
        preds["phonotactic_only"][r["id"]] = r["phonotactic_rule"]
        preds["donor_proximity_only"][r["id"]] = bool((r["dp"] or {}).get("close"))
    prod = _prod_predictions(rows)
    meta = prod.pop("_meta", None)
    preds.update(prod)
    return preds, combiners + ([{"engine_prod": meta}] if meta else [])


# --- ölçüler -------------------------------------------------------------------


def prf(rows: list[dict[str, Any]], pred: dict[str, bool]) -> PRF:
    out = PRF()
    for r in rows:
        p, y = pred[r["id"]], r["y"]
        out.per_item.append(p == y)
        if p and y:
            out.tp += 1
        elif p and not y:
            out.fp += 1
        elif not p and y:
            out.fn += 1
        else:
            out.tn += 1
    return out


def weighted_accuracy(rows: list[dict[str, Any]], pred: dict[str, bool], natural: dict[str, float]) -> float:
    """Dil başına doğal alıntı oranıyla ağırlıklı doğruluk (diller eşit ağırlık)."""
    values = []
    for lang in sorted({r["lang"] for r in rows}):
        pos = [r for r in rows if r["lang"] == lang and r["y"]]
        neg = [r for r in rows if r["lang"] == lang and not r["y"]]
        if not pos or not neg:
            continue
        tpr = sum(pred[r["id"]] for r in pos) / len(pos)
        tnr = sum(not pred[r["id"]] for r in neg) / len(neg)
        pi = natural.get(lang, 0.5)
        values.append(pi * tpr + (1 - pi) * tnr)
    return sum(values) / len(values) if values else 0.0


def _f(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def cluster_bootstrap(
    rows: list[dict[str, Any]], a: dict[str, bool], b: dict[str, bool] | None = None,
    *, iterations: int = BOOTSTRAP,
) -> dict[str, Any]:
    """Etimona göre KÜMELENMİŞ bootstrap: F (ve verilirse F farkı a−b)."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        clusters[r["etymon"]].append(r)
    keys = sorted(clusters)

    def counts(pred: dict[str, bool], members: list[dict[str, Any]]) -> tuple[int, int, int]:
        tp = sum(1 for r in members if pred[r["id"]] and r["y"])
        fp = sum(1 for r in members if pred[r["id"]] and not r["y"])
        fn = sum(1 for r in members if not pred[r["id"]] and r["y"])
        return tp, fp, fn

    per_a = {k: counts(a, clusters[k]) for k in keys}
    per_b = {k: counts(b, clusters[k]) for k in keys} if b is not None else None
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(iterations):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        ta = [sum(per_a[k][j] for k in pick) for j in range(3)]
        value = _f(*ta)
        if per_b is not None:
            tb = [sum(per_b[k][j] for k in pick) for j in range(3)]
            value -= _f(*tb)
        samples.append(value)
    samples.sort()
    observed = _f(*[sum(per_a[k][j] for k in keys) for j in range(3)])
    if per_b is not None:
        observed -= _f(*[sum(per_b[k][j] for k in keys) for j in range(3)])
    mean = sum(samples) / len(samples)
    se = math.sqrt(sum((s - mean) ** 2 for s in samples) / (len(samples) - 1))
    return {
        "value": round(observed, 4),
        "ci95": [round(samples[int(0.025 * iterations)], 4), round(samples[int(0.975 * iterations)], 4)],
        "se": round(se, 4),
        "clusters": len(keys),
    }


def donor_identification(rows: list[dict[str, Any]], pred: dict[str, bool]) -> dict[str, Any]:
    """Verici tanıma: motorun alıntı dediği, vericisi kümede olan alıntılarda etiket doğru mu?"""
    pool = [
        r for r in rows
        if r["y"] and pred[r["id"]] and r["attributed"] and r["donor_macro"] in ("ru", "ar", "fa", "mn")
    ]
    ok = sum(donor_macro(r["attributed"]) == r["donor_macro"] for r in pool)
    confusion = Counter(f"{r['donor_macro']}->{donor_macro(r['attributed'])}" for r in pool)
    return {"n": len(pool), "accuracy": round(ok / len(pool), 4) if pool else None,
            "confusion": dict(confusion.most_common())}


def breakdowns(rows: list[dict[str, Any]], pred: dict[str, bool]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_lang": {}, "recall_by_donor": {}, "easy_hard": {}}
    for lang in sorted({r["lang"] for r in rows}):
        out["by_lang"][lang] = prf([r for r in rows if r["lang"] == lang], pred).as_dict()
    for macro in ("ru", "ar", "fa", "mn", "diğer"):
        pool = [r for r in rows if r["y"] and r["donor_macro"] == macro]
        if pool:
            out["recall_by_donor"][macro] = {"n": len(pool), "recall": round(sum(pred[r["id"]] for r in pool) / len(pool), 4)}
    for name, flag in (("kolay", False), ("zor", True)):
        pool = [r for r in rows if r["y"] and bool(r["hard"]) == flag]
        if pool:
            out["easy_hard"][name] = {"n": len(pool), "recall": round(sum(pred[r["id"]] for r in pool) / len(pool), 4)}
    inherited = [r for r in rows if not r["y"]]
    out["easy_hard"]["miras_özgüllük"] = {
        "n": len(inherited),
        "specificity": round(sum(not pred[r["id"]] for r in inherited) / len(inherited), 4) if inherited else None,
    }
    return out


# --- K tanıları ------------------------------------------------------------


def circularity(rows: list[dict[str, Any]], preds: dict[str, dict[str, bool]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    from engine.evaluation.negative_controls import ALL_BATTERIES
    from engine.nlp.loanword_classifier import KNOWN_REVERSED_LOAN_DIRECTION

    stats = json.loads((XTURKIC_DIR / "stats.json").read_text(encoding="utf-8"))
    # K1 — zincir açık (tam indeks) / kapalı
    fired = [r for r in rows if r["chain_full"] > 0]
    k1 = {
        "chain_full_fired": len(fired),
        "chain_full_fired_rate": round(len(fired) / len(rows), 4),
        "chain_full_fired_on_borrowed": sum(r["y"] for r in fired),
        "chain_full_precision": round(sum(r["y"] for r in fired) / len(fired), 4) if fired else None,
        "chain_full_recall": round(sum(r["y"] for r in fired) / max(1, sum(r["y"] for r in rows)), 4),
        "F_engine_trained_chain_full_minus_off": cluster_bootstrap(
            rows, preds["engine_trained_chain_full"], preds["engine_trained"]),
        "F_engine_prod_chain_full_minus_off": cluster_bootstrap(
            rows, preds["engine_prod_chain_full"], preds["engine_prod"]) if "engine_prod" in preds else None,
    }
    # K2 — kör indekste zincir hiç ateşlenmemeli; açık/kapalı kararlar birebir
    blind_fired = sum(1 for r in rows if r["signals"].get("zincir_kanıtı", 0) > 0)
    same = sum(preds["engine_trained_chain_blind"][r["id"]] == preds["engine_trained"][r["id"]] for r in rows)
    k2 = {"blind_chain_fired": blind_fired, "on_off_identical": same, "n": len(rows),
          "identical_rate": round(same / len(rows), 4), "pass": blind_fired == 0 and same == len(rows)}
    # K3 — anlam ipucu dışlaması (kurulumda)
    k3 = {lang: ex.get("anlam_ipucu_K3", 0) for lang, ex in stats["exclusions"].items()}
    # K4 — bilinen ters yön / negatif kontrol / tohum örtüşmesi
    control_words = {to_comparison_form(i.word) for items in ALL_BATTERIES.values() for i in items if getattr(i, "word", "")}
    reversed_forms = {to_comparison_form(w) for w in KNOWN_REVERSED_LOAN_DIRECTION}
    by_id = {g["id"]: g for g in gold}
    rueckwanderer = [
        r for r in rows if r["y"] and "turkic" in by_id[r["id"]]["etymology_text"].lower()
        and ("ultimately" in by_id[r["id"]]["etymology_text"].lower()
             or "from proto-turkic" in by_id[r["id"]]["etymology_text"].lower()
             or "from ottoman" in by_id[r["id"]]["etymology_text"].lower())
    ]
    k4 = {
        "known_reversed_overlap": sorted(r["id"] for r in rows if r["form"] in reversed_forms),
        "negative_control_overlap": sorted(r["id"] for r in rows if r["form"] in control_words),
        "geri_dönen_alıntı_n": len(rueckwanderer),
        "geri_dönen_alıntı_ids": sorted(r["id"] for r in rueckwanderer)[:40],
        "geri_dönen_alıntı_recall_engine_trained": round(
            sum(preds["engine_trained"][r["id"]] for r in rueckwanderer) / len(rueckwanderer), 4) if rueckwanderer else None,
        "seed_overlap_savelyev": _seed_overlap(rows),
    }
    # K5 — donors.db'de ters yön (Türkçeden alınmış verici maddesi)
    k5 = _donor_reverse(rows)
    # K6 — kolay/zor ayrı: breakdowns içinde
    k6 = {
        "hard_borrowed": sum(1 for r in rows if r["y"] and r["hard"]),
        "easy_borrowed": sum(1 for r in rows if r["y"] and r["hard"] is False),
    }
    return {"K1": k1, "K2": k2, "K3_excluded_gloss_hint": k3, "K4": k4, "K5": k5, "K6": k6,
            "K7": "bileşenler bellekte eğitildi; save() çağrıları hata verecek biçimde kapatıldı; "
                  "engine_prod = üretim birleştiricisi (WOLD/Saha), ALAN DIŞI"}


_SAVELYEV_LANG = {"kk": "Kazakh", "ky": "Kirghiz", "tt": "Tatar", "ba": "Bashkir", "uz": "Uzbek", "ug": "Uighur", "tk": "Turkmen", "tyv": "Tuvan"}


def _seed_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Denklik tabloları savelyevturkic'ten öğrenildi: altın biçimi o veride var mı?"""
    import csv

    from engine.config import CLDF_DIR
    from engine.evaluation.xturkic_gold import _skeleton

    path = CLDF_DIR / "savelyevturkic" / "forms.csv"
    if not path.exists():
        return {"available": False}
    forms: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            forms[row["Language_ID"]].add(_skeleton(row.get("Form") or ""))
    out = {}
    for lab, flag in (("alıntı", True), ("miras", False)):
        pool = [r for r in rows if r["y"] == flag]
        hit = [r for r in pool if _skeleton(r["form"]) in forms.get(_SAVELYEV_LANG.get(r["lang"], ""), set())]
        out[lab] = {"n": len(pool), "in_savelyev": len(hit)}
    return out


def _donor_reverse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import sqlite3

    from engine.config import LEXICON_DIR

    path = LEXICON_DIR / "donors" / "donors.db"
    if not path.exists():
        return {"available": False}
    out: dict[str, Any] = {}
    with sqlite3.connect(path) as connection:
        for lab, flag in (("alıntı", True), ("miras", False)):
            pool = [r for r in rows if r["y"] == flag]
            hit = 0
            for r in pool:
                langs = ["ru", "ar", "fa", "mn"]
                q = connection.execute(
                    f"SELECT COUNT(*) FROM donor_entries WHERE from_turkic = 1 AND comparison = ? "
                    f"AND lang_code IN ({','.join('?' * len(langs))})", [r["form"], *langs]).fetchone()[0]
                hit += bool(q)
            dp_fired = [r for r in pool if r["signals"].get("verici_yakınlığı", 0) > 0]
            out[lab] = {"n": len(pool), "donor_entry_from_turkic_same_form": hit,
                        "dp_fired": len(dp_fired)}
    return out


# --- MDE -------------------------------------------------------------------


def mde(rows: list[dict[str, Any]], preds: dict[str, dict[str, bool]], *, target_n: int) -> dict[str, Any]:
    """R1/R2 için en küçük saptanabilir etki (α=0,05 iki yönlü, güç 0,80).

    SE ayar bölümünde (kat dışı kararlar) etimona göre kümelenmiş bootstrapla
    ölçülür ve ``sqrt(n_ayar / n_hedef)`` ile ölçeklenir. İki yakın sistem
    çifti vekil olarak kullanılır — İş 2/3 adayları verici yakınlığı gücünü
    değiştirir.
    """
    z = 1.959964 + 0.841621
    scale = math.sqrt(len(rows) / target_n)
    out: dict[str, Any] = {"n_tune": len(rows), "n_target": target_n, "z_sum": round(z, 4)}
    pairs = {
        "engine_trained_vs_dp_binary": ("engine_trained", "engine_trained_dp_binary"),
        "engine_trained_vs_donor_proximity_only": ("engine_trained", "donor_proximity_only"),
        "engine_trained_vs_phonotactic_model_only": ("engine_trained", "phonotactic_model_only"),
    }
    for name, (a, b) in pairs.items():
        boot = cluster_bootstrap(rows, preds[a], preds[b])
        discord = sum(preds[a][r["id"]] != preds[b][r["id"]] for r in rows) / len(rows)
        out[name] = {
            "tune_delta_F": boot["value"], "tune_se": boot["se"],
            "discordance": round(discord, 4),
            "mde_F": round(z * boot["se"] * scale, 4),
            # McNemar yaklaşımı: doğruluk farkı için SE ≈ sqrt(uyumsuzluk / n)
            "mde_accuracy": round(z * math.sqrt(discord / target_n), 4),
        }
    single = cluster_bootstrap(rows, preds["engine_trained"])
    out["engine_trained_F_se_at_target"] = round(single["se"] * scale, 4)
    out["note"] = (
        "İş 3 beklenen etkisi ~0,014 F. mde_F bundan büyükse ön-kayıtta "
        "'güç yetersiz' önceden yazılır."
    )
    return out


# --- rapor -------------------------------------------------------------------


def report_tune(tag: str = "") -> dict[str, Any]:
    forbid_model_writes()
    rows = load_cache("tune", tag)
    gold = load_split("tune")
    gold_ids = {g["id"] for g in gold if g["lang"] in set(admitted_languages())}
    missing = gold_ids - {r["id"] for r in rows}
    if missing:
        raise SystemExit(f"önbellek eksik: {len(missing)} madde (önce capture)")
    stats = json.loads((XTURKIC_DIR / "stats.json").read_text(encoding="utf-8"))
    natural = stats["natural_borrowed_rate"]
    preds, combiners = crossfit(rows)
    systems: dict[str, Any] = {}
    for name in list(SYSTEMS) + ["engine_trained_chain_full", "engine_prod_chain_full", "engine_trained_dp_binary"]:
        if name not in preds:
            continue
        p = prf(rows, preds[name])
        systems[name] = {
            **p.as_dict(),
            "weighted_accuracy_natural": round(weighted_accuracy(rows, preds[name], natural), 4),
            "F_ci95_cluster": cluster_bootstrap(rows, preds[name])["ci95"],
        }
    comparisons = {
        f"engine_trained_minus_{b}": cluster_bootstrap(rows, preds["engine_trained"], preds[b])
        for b in ("phonotactic_only", "phonotactic_model_only", "donor_proximity_only", "engine_prod", "always_borrowed")
        if b in preds
    }
    payload = {
        "_schema": "xturkic-borrowing-eval/v1",
        "split": "tune",
        "protocol": (
            "kör indeks + zincir kapalı (ana); ayar bölümünde etimona göre 5 katlı çapraz "
            "uydurma; her katta LM eğitim yarısının A parçasında, birleştirici B parçasında"
        ),
        "n": len(rows),
        "n_borrowed": sum(r["y"] for r in rows),
        "languages": sorted({r["lang"] for r in rows}),
        "systems": systems,
        "comparisons_F_cluster": comparisons,
        "breakdowns": {n: breakdowns(rows, preds[n]) for n in ("engine_trained", "engine_prod", "donor_proximity_only", "phonotactic_only") if n in preds},
        "donor_identification": {n: donor_identification(rows, preds[n]) for n in ("engine_trained", "donor_proximity_only") if n in preds},
        "circularity": circularity(rows, preds, gold),
        "combiners_per_fold": combiners,
        "code_commit": _git_head(),
        "index": str(BLIND_INDEX.relative_to(PROJECT_ROOT)),
    }
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / f"xborrowing_tune{('_' + tag) if tag else ''}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    seal = json.loads((XTURKIC_DIR / "SEAL.json").read_text(encoding="utf-8"))
    target = min(seal["counts"]["r1"], seal["counts"]["r2"])
    mde_payload = mde(rows, preds, target_n=target)
    (WORK_DIR / "MDE.json").write_text(json.dumps(mde_payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _print(payload, mde_payload)
    return payload


def _print(payload: dict[str, Any], mde_payload: dict[str, Any]) -> None:
    print(f"\n=== Türk dilleri arası alıntı — AYAR (n={payload['n']}, alıntı {payload['n_borrowed']}) ===")
    print("kör indeks + zincir kapalı · kat dışı kararlar")
    print(f"{'sistem':28} {'F':>7} {'GA95':>17} {'kes.':>6} {'duy.':>6} {'doğ.':>6} {'ağ.doğ':>7}")
    for name, s in sorted(payload["systems"].items(), key=lambda kv: -kv[1]["fscore"]):
        ci = s["F_ci95_cluster"]
        print(f"{name:28} {s['fscore']:7.4f} [{ci[0]:.3f},{ci[1]:.3f}] {s['precision']:6.3f} {s['recall']:6.3f} "
              f"{s['accuracy']:6.3f} {s['weighted_accuracy_natural']:7.4f}")
    k = payload["circularity"]
    print(f"\nK1 zincir(tam) ateşlenme {k['K1']['chain_full_fired_rate']:.3f}, kesinlik {k['K1']['chain_full_precision']}; "
          f"ΔF(açık−kapalı) {k['K1']['F_engine_trained_chain_full_minus_off']}")
    print(f"K2 kör zincir ateşlenme {k['K2']['blind_chain_fired']}, açık/kapalı aynı {k['K2']['identical_rate']:.4f} -> {'GEÇTİ' if k['K2']['pass'] else 'KALDI'}")
    print(f"MDE (R hedef n={mde_payload['n_target']}): " + ", ".join(
        f"{n}: F {v['mde_F']}" for n, v in mde_payload.items() if isinstance(v, dict)))


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# --- R1/R2 (ön kayıtlı, bir kez) -------------------------------------------------


def _guard_open(split: str, prereg: str | None, final_report: bool) -> Path:
    if split == "test" and not final_report:
        raise SystemExit("test yalnız --final-report ile (dalga sonu)")
    if not prereg:
        raise SystemExit(f"--split {split} için --prereg zorunlu (commit edilmiş PREREG_<ad>.md)")
    path = Path(prereg)
    if not path.exists():
        raise SystemExit(f"ön-kayıt yok: {path}")
    status = subprocess.run(["git", "status", "--porcelain", "--", str(path)], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=PROJECT_ROOT,
                             capture_output=True, text=True).returncode == 0
    if status or not tracked:
        raise SystemExit(f"ön-kayıt commit edilmemiş: {path}")
    name = path.stem.removeprefix("PREREG_") or "x"
    lock = WORK_DIR / f".opened_{split}_{name}"
    if lock.exists():
        raise SystemExit(f"{split} bu ön-kayıtla zaten açıldı: {lock}")
    return lock


def report_split(split: str, prereg: str | None, final_report: bool, tag: str = "") -> None:
    """Ayar bölümünün TAMAMINDA eğit, ``split``te bir kez ölç (henüz koşulmadı)."""
    lock = _guard_open(split, prereg, final_report)
    forbid_model_writes()
    tune = load_cache("tune", tag)
    rows = load_cache(split, tag)
    preds = _train_predict(tune, rows)
    preds.pop("_combiner", None)
    for r in rows:
        preds["always_inherited"][r["id"]] = False
        preds["always_borrowed"][r["id"]] = True
        preds["phonotactic_only"][r["id"]] = r["phonotactic_rule"]
        preds["donor_proximity_only"][r["id"]] = bool((r["dp"] or {}).get("close"))
    prod = _prod_predictions(rows)
    prod.pop("_meta", None)
    preds.update(prod)
    lock.write_text(json.dumps({"prereg": prereg, "commit": _git_head(), "at": time.time()}), encoding="utf-8")
    stats = json.loads((XTURKIC_DIR / "stats.json").read_text(encoding="utf-8"))
    payload = {
        "split": split,
        "prereg": prereg,
        "systems": {n: {**prf(rows, p).as_dict(),
                        "weighted_accuracy_natural": round(weighted_accuracy(rows, p, stats["natural_borrowed_rate"]), 4),
                        "F_ci95_cluster": cluster_bootstrap(rows, p)["ci95"]}
                    for n, p in preds.items()},
        "breakdowns": {n: breakdowns(rows, preds[n]) for n in ("engine_trained", "engine_prod") if n in preds},
    }
    out = WORK_DIR / f"xborrowing_{split}_{Path(prereg or 'x').stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Türk dilleri arası alıntı değerlendirmesi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--split", default="tune")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--tag", default="")
    c.add_argument("--prereg")
    c.add_argument("--final-report", action="store_true")
    r = sub.add_parser("replay")
    r.add_argument("--split", default="tune")
    r.add_argument("--tag", default="")
    r.add_argument("--prereg")
    r.add_argument("--final-report", action="store_true")
    args = ap.parse_args()
    if args.split not in ("tune", "r1", "r2", "test"):
        raise SystemExit(f"bilinmeyen bölüm {args.split}")
    if args.cmd == "capture":
        if args.split != "tune":
            _guard_open(args.split, args.prereg, args.final_report)
        print(capture(args.split, limit=args.limit, tag=args.tag))
    elif args.split == "tune":
        report_tune(args.tag)
    else:
        report_split(args.split, args.prereg, args.final_report, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
