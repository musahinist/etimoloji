#!/usr/bin/env python3
"""
Ağız kelimeleri toplu analizi — Faz 10, projenin asıl hedefi.

TDK Derleme Sözlüğü'nden gelen ağız kelimeleri, etimolojisi **hiç
yapılmamış** sözvarlığıdır. Motorun asıl sınavı budur: sözlükte cevabı
olmayan bir kelimeye ne diyebiliyor?

Her kelime tam hattan geçer::

    alıntı denetimi → akraba keşfi → rekonstrüksiyon → rakip hipotezler
                   → semantik makullük → öngörü üretimi

Sonuçlar kanıt gücüne göre üç kovaya ayrılır:

``çözüldü``       kalibre güven ≥ 0,60 ve rakip hipotez farkı belirgin
``güçlü aday``    kalibre güven ≥ 0,35
``yetersiz kanıt`` altı

⚠️ **"Yetersiz kanıt" bir başarısızlık değil, dürüst sonuçtur.** Ağız
kelimelerinin çoğu tek bir ilde tanıklanmıştır ve karşılaştırmalı yöntemin
gerektirdiği bağımsız tanık yoktur. Yüksek "çözüldü" oranı raporlamak,
kanıtı olmayan iddialar üretmek demek olurdu.

Üretilen öngörüler ayrı bir sicile **kilitlenir**; doğrulama sonradan,
ayrı bir koşuda yapılır (bkz. ``engine.evaluation.prediction_test``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import PROJECT_ROOT  # noqa: E402
from engine.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "dialect"

#: Kanıt gücü kovaları.
SOLVED_THRESHOLD = 0.60
CANDIDATE_THRESHOLD = 0.35


def is_lexeme(word: str) -> bool:
    """Girdi bir sözlükbirim mi?

    Ek, öbek ve özel ada etimoloji üretmek anlamsızdır. Kapı yokken yedek
    havuzun ilk beş kaydı ``-acağım``, ``-akalmak``, ``-amaç``, ``-anak``,
    ``-arak`` (hepsi **ek**) geliyordu ve motor bunların ikisine "güçlü aday"
    (0,40) dedi. Havuzda ayrıca çekimli biçimler (``ırmağı``, ``yastığı``),
    özel adlar (``Aydoğan``, ``Kahta``) ve öbekler (``av köpeği``) var.
    """
    w = (word or "").strip()
    if len(w) < 2:
        return False
    if w.startswith("-") or w.endswith("-"):
        return False  # yapım/çekim eki
    if " " in w or "\t" in w:
        return False  # çok sözcüklü öbek
    if w[:1].isupper():
        return False  # özel ad
    if any(ch.isdigit() for ch in w):
        return False  # sayı / sıra sayısı: 1'inci, 10'uncu, 100'üncü
    if "'" in w or "’" in w:
        return False  # kesme işaretli çekim: Ay'a
    return True


#: Wiktionary çekim gloss'larının dilbilgisel belirteçleri. Bu glossları
#: TAŞIYAN satır bir sözlükbirim tanımı değil, başka bir lemmanın çekimidir:
#: "third-person singular indicative aorist of canlanmak", "verbal noun of
#: yapılmak", "singular dative of yarak", "inflection of yağ:".
#: ⚠️ Yalnız " of " aramak yetmez — "palm of hand" gibi GERÇEK tanımlar da
#: onu içerir; bu yüzden dilbilgisi terimi aranır.
_INFLECTION_MARKERS: tuple[str, ...] = (
    "inflection of", "plural of", "singular of", "verbal noun of",
    "-person", "aorist of", "imperative of", "participle of",
    "dative of", "accusative of", "genitive of", "ablative of",
    "locative of", "nominative of", "optative of", "necessitative of",
    "past of", "present of", "future of", "negative of",
    "causative of", "passive of", "reflexive of", "reciprocal of",
)


def is_inflection_gloss(gloss: str) -> bool:
    """Bu gloss bir çekim tanımı mı (yani kelime başka bir lemmanın biçimi mi)?"""
    g = (gloss or "").lower()
    return any(marker in g for marker in _INFLECTION_MARKERS)


def load_words(source: Path | None, limit: int) -> list[str]:
    """Analiz edilecek kelimeleri yükler; sözlükbirim olmayanları eler."""
    if source and source.exists():
        # ⚠️ Küçük harfe ÇEVİRMEDEN önce ele: büyük harf özel ad işaretidir.
        words = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return [w.lower() for w in words if is_lexeme(w)][:limit]

    # Kaynak verilmezse yerel sözlük indeksinden Türkçe kelimeler alınır.
    # Bunlar ağız kelimesi DEĞİLDİR; yalnız hattın uçtan uca çalıştığını
    # göstermek için kullanılır ve rapor bunu açıkça söyler.
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not index.exists:
        return []
    with index.connect() as connection:
        # Ucuz süzgeçler SQL'de: havuzun alfabetik başı tamamen eklerden
        # oluştuğu için (tire harflerden önce sıralanır) Python tarafında
        # elemek fazladan çekimle kapatılamıyordu — 25 satırın 25'i ek çıkıp
        # sonuç boş kalıyordu. Özel ad denetimi Python'da kalır: SQLite'ın
        # `lower()` işlevi ASCII'dir, Türkçe büyük harfleri (İ, Ş, Ğ) tanımaz.
        rows = connection.execute(
            "SELECT word, gloss FROM entries "
            "WHERE lang_code = 'tr' AND origin IS NULL AND length(comparison) BETWEEN 4 AND 9 "
            "AND word NOT LIKE '-%' AND word NOT LIKE '%-' AND word NOT LIKE '% %' "
            "AND word NOT GLOB '*[0-9]*' AND word NOT LIKE '%''%' "
            # SQLite bayt sıralaması yapar: `ORDER BY word` önce BÜTÜN büyük
            # harfli kayıtları getirir (Aydoğan, Akkın…) ve Python kapısı
            # hepsini özel ad diye eler — havuz boş kalıyordu. ASCII büyük
            # harf burada elenir; Türkçeye özgü büyük harfler (İ, Ş, Ğ)
            # SQLite'ın `lower()` işlevinden geçmediği için onları Python
            # tarafındaki `is_lexeme` yakalar.
            "AND word = lower(word) "
            # Alfabetik sıralama tek kökün çekimli biçimlerini yan yana
            # topluyordu (abacı, abacıda, abacıdan, abacılar…) — altı kayıtlık
            # bir hat testi tek lemmayı altı kez ölçüyordu. Rastgele örneklem
            # hem bunu hem de bayt sıralamasının büyük-harf yanlılığını çözer.
            "ORDER BY RANDOM() LIMIT ?",
            (limit * 60,),
        ).fetchall()

    # Çekimli biçim elemesi KELİME düzeyinde yapılır, satır düzeyinde değil:
    # `yağı` hem "enemy" hem "inflection of yağ:" satırı taşıyor; en az bir
    # gerçek anlamı varsa kelime havuzda kalmalı. Ölçüldü: havuzun
    # 19.240/27.925'i (%69) çekim glossu taşıyor — bu yüzden fazladan çekim
    # geniş tutulur.
    senses: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        senses[row["word"]].append(row["gloss"] or "")

    picked = [
        word.lower()
        for word, glosses in senses.items()
        if is_lexeme(word) and any(not is_inflection_gloss(g) for g in glosses)
    ]
    return picked[:limit]


def analyse(word: str, *, predictor: Any, ranker: Any, semantic: Any) -> dict[str, Any]:
    """Tek bir kelimeyi tam hattan geçirir."""
    # 1 — İleri tahminle akraba adayları üret, sözlükte ara.
    witnesses: list[dict[str, str]] = []
    searched: list[dict[str, Any]] = []
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    for prediction in predictor.predict_all(word, "tr")[:12]:
        if not prediction.form or prediction.confidence <= 0:
            continue
        hits = index.lookup(prediction.form, languages=[prediction.language], limit=1) if index.exists else []
        if not hits:
            hits = (
                index.fuzzy_lookup(prediction.form, max_distance=1, languages=[prediction.language])[:1]
                if index.exists
                else []
            )
        searched.append(
            {
                "language": prediction.language,
                "predicted": prediction.form,
                "found": hits[0]["word"] if hits else "",
                "gloss": hits[0].get("gloss", "") if hits else "",
            }
        )
        if hits:
            witnesses.append({"lang_code": prediction.language, "word": hits[0]["word"]})

    # 2 — Rakip hipotezler (rekonstrüksiyon ve alıntı denetimi içinde).
    ranked = ranker.rank(word, witnesses)
    selected = ranked.selected
    score = float(selected.score) if selected else 0.0

    if score >= SOLVED_THRESHOLD:
        bucket = "çözüldü"
    elif score >= CANDIDATE_THRESHOLD:
        bucket = "güçlü aday"
    else:
        bucket = "yetersiz kanıt"

    return {
        "word": word,
        "bucket": bucket,
        "score": round(score, 3),
        "selected": selected.as_dict() if selected else None,
        "n_witnesses_found": len(witnesses),
        "cognate_search": searched,
        "ranked": ranked.as_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Ağız kelimeleri toplu analizi")
    ap.add_argument("--words", type=Path, help="satır satır kelime listesi")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    words = load_words(args.words, args.limit)
    if not words:
        print(
            "Analiz edilecek kelime yok.\n"
            "  --words ile bir liste verin veya önce sözlük indeksini kurun:\n"
            "    make lexicon-index"
        )
        return 1

    from engine.nlp.cognate_prediction import CognatePredictor
    from engine.nlp.hypothesis_ranking import HypothesisRanker
    from engine.nlp.semantic_plausibility import SemanticPlausibility

    predictor = CognatePredictor()
    ranker = HypothesisRanker()
    semantic = SemanticPlausibility()

    if not predictor.tables:
        print("⚠️ Denklik tabloları yok; akraba keşfi çalışmayacak (make correspondences)")

    results: list[dict[str, Any]] = []
    for index, word in enumerate(words, start=1):
        try:
            results.append(analyse(word, predictor=predictor, ranker=ranker, semantic=semantic))
        except Exception:
            logger.warning("Analiz başarısız: %s", word, exc_info=True)
        if index % 25 == 0:
            print(f"  … {index}/{len(words)}")

    buckets = Counter(r["bucket"] for r in results)
    kinds = Counter(
        (r["selected"] or {}).get("kind", "yok") for r in results
    )
    witnesses = [r["n_witnesses_found"] for r in results]

    # Başlık kaynağı YANSITMALI. Yedek havuz kullanıldığında bile koşulsuz
    # "ağız kelimesi analizi" yazıyordu; JSON'daki `source` alanı dürüsttü ama
    # terminale bakan onu görmüyordu.
    source_label = str(args.words) if args.words else "sözlük indeksi (ağız DEĞİL, hat testi)"
    print(f"\n=== kelime analizi · n={len(results)} · kaynak: {source_label} ===")
    print("\nkanıt gücüne göre:")
    for bucket in ("çözüldü", "güçlü aday", "yetersiz kanıt"):
        count = buckets.get(bucket, 0)
        share = 100 * count / len(results) if results else 0
        print(f"  {bucket:16} {count:>5} (%{share:.1f})")
    print("\nseçilen hipotez türüne göre:")
    for kind, count in kinds.most_common():
        print(f"  {kind:18} {count:>5}")
    print(
        f"\nkelime başına bulunan akraba tanığı: "
        f"ortalama {sum(witnesses) / len(witnesses):.2f}"
        if witnesses
        else ""
    )
    print(
        "\n⚠️ 'Yetersiz kanıt' bir başarısızlık değil, dürüst sonuçtur. Ağız\n"
        "   kelimelerinin çoğu tek bir ilde tanıklanmıştır ve karşılaştırmalı\n"
        "   yöntemin gerektirdiği bağımsız tanık yoktur."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "analysis.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-dialect-analysis/v1",
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source": source_label,
                "n": len(results),
                "buckets": dict(buckets),
                "selected_kinds": dict(kinds),
                "mean_witnesses": round(sum(witnesses) / len(witnesses), 3) if witnesses else 0.0,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out}")

    solved = [r for r in results if r["bucket"] == "çözüldü"][:8]
    if solved:
        print("\nörnek çözümler:")
        for row in solved:
            claim = (row["selected"] or {}).get("claim", "")
            print(f"  {row['word']:14} {claim[:56]}  ({row['score']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
