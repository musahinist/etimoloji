"""
Başlık kökü doğruluğu — ``make eval-headline``.

Kullanıcının İLK gördüğü satır ``search(...)['root']['proto_turkic']``dir.
Rekonstrüksiyon ölçümleri (``make eval``, ``make eval-cv``) motorun
karşılaştırmalı yöntemini ölçer; başlık ise beş ayrı atama noktasından
(fetcher, tanıksız kanıtlayıcı, karşılaştırmalı yöntem, donör, sıralayıcı,
kaynak kökü) beslenir ve **hiç ölçülmemişti**.

Ölçüm düzeni
------------
Arama hattı yalnız YEREL kaynaklarla koşar (tohum/indeks + indirilmiş
Starling); Nişanyan, TDK ve EtimolojiTürkçe ağ kaynaklarıdır ve dışarıda
kalır. İki yapılandırma:

* ``yerel``: yerel portföyün tamamı (Starling dahil).
* ``starling_yok``: ``StarlingFetcher`` portföyden ÇIKARILMIŞ.

İki başvuru:

1. **Starling turcet** (``engine.db.starling.lookup_turkish``) — Türkçe
   biçimin Starling kök(ler)i. Kelime örneklemi de Starling'in TRK
   alanından çekilir (sabit tohum) — Türkçe altın küme (TDK+Nişanyan)
   **kullanılmaz**.
2. **savelyevturkic dev bölümü** — Türkçe tanığı olan dev maddeleri
   (Savelyev & Robbeets 2020 rekonstrüksiyonları). Test bölümü açılmaz.

⚠️ Döngüsellik durumu (her alt küme JSON'da ``circularity`` alanında):

* ``starling/yerel/tümü`` — **DÖNGÜSEL.** Motor Starling'i başlık yedeği
  olarak gösterir; Starling'den gelen başlık tanımı gereği doğrudur.
* ``starling/yerel/starling_dışı`` — başlık damgası (provenance) Starling
  anmayan maddeler. Döngü kırılır ama örneklem seçilimlidir (Starling'in
  devreye girmediği, yani başka kaynağın kök verdiği maddeler).
* ``starling/starling_yok`` — Starling motor tarafından hiç okunmaz.
  **Kısmen bağımsız:** yerel sözlük indeksindeki Wiktionary Proto-Türkçe
  biçimleri büyük ölçüde EDAL/Starostin soyundandır (bkz. ``gold.py`` 1.
  madde); uyumun bir kısmı bu ortak soydan gelir.
* ``savelyev_dev/*`` — motorun örüntü tabloları yalnız TRAIN bölümünden
  öğrenilir; dev bu tablolara girmez. Wiktionary ile kısmi ortak kaynak
  (Clauson/EDAL) yine olasıdır. En bağımsız başvurudur ama küçüktür.

Taban çizgisi: ``*<sorgu kelimesi>`` (hiçbir şey bilmeyen sistemin
göstereceği başlık).
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from engine.config import PROJECT_ROOT
from engine.evaluation.metrics import best_match, normalize_proto, normalized_edit_distance
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SEED = 20260924
#: Starling TRK alanından örneklenecek kelime sayısı.
STARLING_SAMPLE = 250
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "eval_engine_runs"

_PROTO_TOKEN = re.compile(r"\*[^\s,/();?]+")
#: savelyev çevriyazısı -> Türkiye Türkçesi imlası (yalnız Türkçe tanık için).
_SAVELYEV_TO_TR = str.maketrans({"ï": "ı", "š": "ş", "č": "ç", "ǰ": "c", "ž": "j", "ğ": "ğ"})


# --- istatistik ---------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Oran için Wilson %95 aralığı (n küçükken normal yaklaşımdan iyi)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def bootstrap_mean_ci(
    values: Sequence[float], *, iterations: int = 2000, seed: int = SEED
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iterations))
    return (round(means[int(0.025 * iterations)], 4), round(means[int(0.975 * iterations) - 1], 4))


# --- başvurular ---------------------------------------------------------------


def starling_candidates(word: str) -> list[str]:
    """Kelimenin Starling kökleri; eşsesli kayıtların HEPSİ aday sayılır."""
    from engine.db.starling import lookup_turkish

    out: list[str] = []
    for etym in lookup_turkish(word):
        for token in _PROTO_TOKEN.findall(etym.proto):
            token = token.rstrip("-")
            if len(token) > 2 and token not in out:
                out.append(token)
    return out


_TR_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyz")


def is_turkish_word(word: str) -> bool:
    """Starling TRK alanında kısaltma (``adv``) ve ağız çevriyazısı (``arqun``)
    da geçer. Türk alfabesi dışı harf taşımayan VE yerel sözlük indeksinde
    Türkçe maddesi olan biçim kabul edilir (yalnız varlık denetimi)."""
    if not word or set(word) - _TR_LETTERS:
        return False
    from engine.db.lexicon_index import LexiconIndex

    index = _index_cache.setdefault("tr", LexiconIndex())
    if not getattr(index, "exists", False):
        return True
    return bool(index.lookup(word, languages=["tr"], limit=1))


_index_cache: dict[str, Any] = {}


def starling_sample(n: int = STARLING_SAMPLE, seed: int = SEED, *, predicate: Callable | None = None) -> list[str]:
    """Starling TRK alanındaki Türkçe biçimlerden sabit tohumlu örneklem.

    Fiil anahtarı ``uç-`` sorgu biçimi ``uç`` olur (motor mastarsız kökü arar).
    """
    from engine.db.starling import turkish_lookup

    table = turkish_lookup()
    words: dict[str, None] = {}
    for key in sorted(table):
        if predicate is not None and not predicate(key, table[key]):
            continue
        word = key.rstrip("-")
        if len(word) >= 2:
            words.setdefault(word, None)
    pool = [w for w in words if is_turkish_word(w)]
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n])


def savelyev_dev_turkish() -> list[tuple[str, tuple[str, ...]]]:
    """``(Türkçe biçim, altın adaylar)`` — YALNIZ dev bölümü."""
    from engine.evaluation.gold import GoldStandard

    out: list[tuple[str, tuple[str, ...]]] = []
    for item in GoldStandard.build().split("dev"):
        form = item.witnesses.get("Turkish", "")
        if not form:
            continue
        word = form.translate(_SAVELYEV_TO_TR).strip("-").split()[0].lower()
        out.append((word, item.gold_candidates))
    return out


# --- motor koşusu ---------------------------------------------------------------


def _head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "?"


def build_engine(*, ablate_starling: bool):
    """Yalnız yerel kaynaklı arama motoru; istenirse Starling'siz."""
    # ⚠️ `AcademicTurkologyFetcher` tohum kaynağıdır ama her aramada TDK
    # terim ucuna da gider; uç kaldırıldı (JSON değil HTML döner, fetcher
    # onu zaten atar), yani sonuca katkısı yoktur, önbellekte olmayan her
    # kelimede ~19 sn zaman aşımı bekletir. Ölçüm ağsızdır: çağrı susturulur.
    import engine.fetchers.academic_turkology as academic
    from engine.fetchers.starling import StarlingFetcher
    from engine.search_engine import SearchEngine, default_fetchers

    academic.http_get = lambda *args, **kwargs: None
    fetchers = [
        f for f in default_fetchers()
        if getattr(f, "is_seed_source", False) or getattr(f, "is_local", False)
    ]
    if ablate_starling:
        fetchers = [f for f in fetchers if not isinstance(f, StarlingFetcher)]
    return SearchEngine(fetchers=fetchers)


def summarize_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Aramadan yalnız ölçülen alanlar: başlık, damga, A-HVP 2. aşama yılı."""
    nlp = finding.get("nlp_analysis") or {}
    root = finding.get("root") or {}

    def _stage2(report: dict[str, Any] | None) -> Any:
        return (((report or {}).get("stage_breakdown") or {}).get("stage2_time_lock") or {}).get("attestation_year")

    year = _stage2((nlp.get("proven_hypothesis") or {}).get("validation_report"))
    unattested = nlp.get("unattested_word_reconstruction") or {}
    if year is None:
        year = _stage2(unattested.get("validation_report"))
    record = (unattested.get("attestation") or {}).get("first_attestation_record")
    # Dönem düzeyindeki tanık (Wilkens "9.-14. yy") nokta yıl değildir;
    # eval-chronology iki kapsamı ayrı sayar.
    precision = None
    for report_ in ((nlp.get("proven_hypothesis") or {}).get("validation_report"),
                    unattested.get("validation_report")):
        stage2 = ((report_ or {}).get("stage_breakdown") or {}).get("stage2_time_lock") or {}
        if stage2.get("attestation_year") is not None:
            precision = stage2.get("attestation_precision") or "point"
            break
    # A-HVP rozeti (``eval-badge``): hipotez ve raporu.
    hypothesis = nlp.get("proven_hypothesis") or {}
    report = hypothesis.get("validation_report") or {}
    ranked = nlp.get("ranked_hypotheses") or finding.get("ranked_hypotheses") or {}
    selected = ranked.get("selected") or {}
    return {
        "headline": str(root.get("proto_turkic") or ""),
        "provenance": str(root.get("provenance") or ""),
        "attestation_year": int(year) if year is not None else None,
        "attestation_record": record,
        "attestation_precision": precision,
        "badge": report.get("status_code"),
        "verdict_badge": (nlp.get("verdict_badge") or {}).get("code"),
        "verdict_badge_score": (nlp.get("verdict_badge") or {}).get("score"),
        "badge_score": report.get("final_confidence_score"),
        "hypothesis_donor": hypothesis.get("donor_language"),
        "hypothesis_form": hypothesis.get("origin_form"),
        "hypothesis_kind": hypothesis.get("evidence_kind"),
        "verdict_kind": selected.get("kind"),
        "verdict_conflicts": list(ranked.get("conflicts") or []),
    }


#: Önbellekte bu alan yoksa madde eski bir özet biçimindedir; yeniden koşar.
SUMMARY_KEYS = ("badge", "verdict_badge")


def run_engine(words: Iterable[str], *, ablate_starling: bool, fresh: bool = False) -> dict[str, dict[str, Any]]:
    """Kelime başına özet. Sonuçlar ``data/cache`` altında HEAD'e bağlı
    önbelleğe yazılır (commit değişince yeniden koşar; kirli ağaç için
    ``fresh=True``)."""
    config = "starling_yok" if ablate_starling else "yerel"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{config}.json"
    head = _head()
    cache: dict[str, Any] = {}
    if path.exists() and not fresh:
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("head") == head:
                cache = stored.get("items", {})
        except (OSError, ValueError):
            cache = {}
    engine = None
    out: dict[str, dict[str, Any]] = {}
    todo = [w for w in dict.fromkeys(words)
            if w not in cache or any(k not in cache[w] for k in SUMMARY_KEYS)]
    for i, word in enumerate(todo, 1):
        if engine is None:
            engine = build_engine(ablate_starling=ablate_starling)
        try:
            finding = engine.search(word, save_to_db=False, use_qwen_agent=False, use_cache=False)
            cache[word] = summarize_finding(finding)
        except Exception as exc:  # tek kelime hattı durdurmasın; görünür kalsın
            logger.warning("Arama başarısız: %s", word, exc_info=True)
            cache[word] = {"headline": "", "provenance": "", "attestation_year": None,
                           "attestation_record": None, "badge": None, "verdict_badge": None,
                           "error": f"{type(exc).__name__}: {exc}"}
        if i % 25 == 0:
            logger.info("%s: %d/%d", config, i, len(todo))
            path.write_text(json.dumps({"head": head, "items": cache}, ensure_ascii=False), encoding="utf-8")
    path.write_text(json.dumps({"head": head, "items": cache}, ensure_ascii=False), encoding="utf-8")
    for word in dict.fromkeys(words):
        out[word] = cache[word]
    return out


# --- puanlama -------------------------------------------------------------------


def from_starling(provenance: str) -> bool:
    return "starling" in provenance.lower()


def score_item(predicted: str, candidates: Sequence[str]) -> dict[str, Any]:
    """Tam / kabul edilebilir / NED (en yakın adaya göre)."""
    shown = predicted.strip()
    if not shown.startswith("*"):
        shown = "*" + shown
    chosen, exact, acceptable = best_match(shown, candidates)
    ned = min(
        normalized_edit_distance(normalize_proto(shown), normalize_proto(c)) for c in candidates
    )
    return {"exact": exact, "acceptable": acceptable, "ned": round(ned, 4), "matched": chosen}


def score_set(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    exact = sum(r["exact"] for r in rows)
    acceptable = sum(r["acceptable"] for r in rows)
    neds = [r["ned"] for r in rows]
    return {
        "n": n,
        "exact": round(exact / n, 4) if n else 0.0,
        "exact_ci95": wilson(exact, n),
        "acceptable": round(acceptable / n, 4) if n else 0.0,
        "acceptable_ci95": wilson(acceptable, n),
        "mean_ned": round(sum(neds) / n, 4) if n else 0.0,
        "mean_ned_ci95": bootstrap_mean_ci(neds),
    }


def evaluate_subset(
    items: Sequence[tuple[str, Sequence[str]]],
    runs: dict[str, dict[str, Any]],
    *,
    keep: Callable[[dict[str, Any]], bool] = lambda r: True,
) -> dict[str, Any]:
    engine_rows, baseline_rows, detail = [], [], []
    for word, candidates in items:
        run = runs.get(word) or {}
        if not candidates or not keep(run):
            continue
        headline = run.get("headline") or word
        scored = score_item(headline, candidates)
        engine_rows.append(scored)
        baseline_rows.append(score_item(word, candidates))
        detail.append({"word": word, "headline": headline, "reference": list(candidates),
                       "provenance": run.get("provenance", "")[:160], **scored})
    return {
        "engine": score_set(engine_rows),
        "baseline_identity": score_set(baseline_rows),
        "headline_is_starred": round(
            sum(d["headline"].startswith("*") for d in detail) / len(detail), 4) if detail else 0.0,
        "items": detail,
    }


CIRCULARITY = {
    "starling/yerel/tümü": "DÖNGÜSEL — motor Starling kökünü başlık yedeği olarak gösterir.",
    "starling/yerel/starling_dışı": (
        "Döngü damga ile kırıldı (başlık Starling'den gelmedi); seçilimli örneklem."
    ),
    "starling/starling_yok": (
        "Starling motorda hiç okunmuyor. Kısmen bağımsız: yerel indeksteki Wiktionary "
        "Proto-Türkçe biçimleri büyük ölçüde EDAL/Starostin soyundan."
    ),
    "savelyev_dev/yerel": (
        "Dev bölümü motorun örüntü tablolarına girmez (yalnız TRAIN). Başlık Starling "
        "yedeğinden de gelebilir; Starling ≠ Savelyev, ama ortak Clauson/EDAL kaynakçası."
    ),
    "savelyev_dev/starling_yok": "En bağımsız düzen; n küçük.",
}


def run(*, sample: int = STARLING_SAMPLE, fresh: bool = False) -> dict[str, Any]:
    words = starling_sample(sample)
    starling_items = [(w, starling_candidates(w)) for w in words]
    savelyev_items = savelyev_dev_turkish()
    all_words = [w for w, _ in starling_items] + [w for w, _ in savelyev_items]

    local = run_engine(all_words, ablate_starling=False, fresh=fresh)
    ablated = run_engine(all_words, ablate_starling=True, fresh=fresh)

    subsets = {
        "starling/yerel/tümü": evaluate_subset(starling_items, local),
        "starling/yerel/starling_dışı": evaluate_subset(
            starling_items, local, keep=lambda r: not from_starling(r.get("provenance", ""))),
        "starling/starling_yok": evaluate_subset(starling_items, ablated),
        "savelyev_dev/yerel": evaluate_subset(savelyev_items, local),
        "savelyev_dev/starling_yok": evaluate_subset(savelyev_items, ablated),
    }
    starling_share = sum(from_starling(local[w].get("provenance", "")) for w in words) / len(words)
    return {
        "_schema": "headline_eval/v1",
        "commit": _head(),
        "fetchers": "yalnız yerel (tohum/indeks + indirilmiş Starling); ağ kaynakları kapalı",
        "sample": {"starling_words": len(words), "savelyev_dev_turkish": len(savelyev_items), "seed": SEED},
        "headline_from_starling_share": round(starling_share, 4),
        "metric": "normalize_proto ile tam eşleşme, is_acceptable, NED (en yakın adaya)",
        "circularity": CIRCULARITY,
        "subsets": subsets,
    }


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=STARLING_SAMPLE)
    parser.add_argument("--fresh", action="store_true", help="önbelleği yok say")
    args = parser.parse_args()
    payload = run(sample=args.sample, fresh=args.fresh)
    print(f"\n=== başlık kökü doğruluğu · commit {payload['commit']} ===")
    print(f"başlığın Starling'den geldiği pay (yerel): {payload['headline_from_starling_share']:.3f}")
    for name, sub in payload["subsets"].items():
        e, b = sub["engine"], sub["baseline_identity"]
        print(
            f"{name:32} n={e['n']:3}  tam {e['exact']:.3f} {e['exact_ci95']}  "
            f"kabul {e['acceptable']:.3f}  NED {e['mean_ned']:.3f} {e['mean_ned_ci95']}  "
            f"| taban tam {b['exact']:.3f} NED {b['mean_ned']:.3f}"
        )
        print(f"{'':32} ⚠️ {payload['circularity'][name]}")
    out = EVAL_DIR / "headline.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
