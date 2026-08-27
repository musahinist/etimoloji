"""
Alıntı tespiti değerlendirmesi — ``make eval-borrowing``.

⚠️ **Döngüsellik uyarısı, planın en kritik ikinci maddesi.** Motorun en güçlü
sinyali (``zincir_kanıtı``) sözlük indeksindeki Wiktionary etiketini okur.
Aynı etikete karşı ölçüm yapmak, sistemin kendi girdisini "doğrulaması"
olurdu ve hiçbir şey ifade etmezdi.

Bu yüzden iki ayrı ölçüm yapılır:

**1. WOLD — birincil, bağımsız ölçüt.**
    World Loanword Database uzman derlemesidir ve Wiktionary'den bağımsızdır.
    41 dilinden **yalnız biri Türki**: Sakha (Yakutça). Az ama gerçek.

**2. Ablasyon — zincir sinyali KAPALI.**
    Wiktionary etiketine karşı ölçülür ama motorun o etiketi okuyan sinyali
    devre dışı bırakılır. Ölçülen şey şudur: *fonotaktik + ses kanunu +
    değişimsiz yayılım* sinyalleri, etiketi hiç görmeden alıntıyı bulabiliyor
    mu? Asıl bilimsel soru budur.

Taban çizgileri:

``always_inherited``
    Her şeye "miras" de. Alıntı oranı düşükse yüksek doğruluk alır; bu yüzden
    raporlanması zorunludur.
``always_borrowed``
    Her şeye "alıntı" de.
``phonotactic_only``
    Yalnız fonotaktik ihlal — Miller ve ark. 2020'nin tek dilli taban
    çizgisinin muadili (yayınlanan F1 ≈ 0,55).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from engine.config import CLDF_DIR
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

#: Miller ve ark. 2020, tek dilli fonotaktik alıntı tespiti.
REFERENCE_MONOLINGUAL_F1 = 0.55

#: List & Forkel 2022, aileler arası ``seabor``.
REFERENCE_CROSSFAMILY_F = 0.87


@dataclass(frozen=True)
class BorrowingCase:
    """Etiketli tek bir kelime."""

    word: str
    lang_code: str
    is_borrowed: bool
    donor: str = ""
    source: str = ""
    witnesses: tuple[tuple[str, str], ...] = ()


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_item: list[bool] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def fscore(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "fscore": round(self.fscore, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


# --- Tanık bulma ------------------------------------------------------------


def find_witnesses(
    word: str,
    *,
    source_lang: str = "tr",
    max_languages: int = 12,
) -> tuple[tuple[str, str], ...]:
    """Bir kelime için akraba tanıklarını bulur.

    ⚠️ **Bu adım eksikti ve ölçümü geçersiz kılıyordu.** Dört alıntı
    sinyalinden ikisi tanık gerektirir (``_sound_law_signal``:
    ``len(witnesses) < 2``, ``_uniformity_signal``: ``len(forms) < 3``).
    ``witnesses`` alanı hiç doldurulmadığı için o iki sinyal ablasyon
    boyunca **yapısal olarak devre dışıydı** ve sonuç "yalnız fonotaktik"
    ile birebir aynı çıkıyordu. Ölçüm "bu sinyaller katkı sağlamıyor"
    demiyordu; "hiç çalıştırılmadı" diyordu.

    Doğrulandı — tanıklar doldurulunca (n=150): ``ses_kanunu_ihlali`` 0 → 15
    kez (%10,0), ``değişimsiz_yayılım`` 0 → 6 kez (%4,0).

    Akış ``scripts/analyse_dialect_words.py``dekiyle aynıdır: ileri tahminle
    aday üret, sözlükte tam ve bulanık ara. Kelime başına ortalama 2,6 tanık.
    """
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not index.exists:
        return ()

    predictor = _shared_predictor_for_witnesses()
    found: list[tuple[str, str]] = []
    for prediction in predictor.predict_all(word, source_lang)[:max_languages]:
        if not prediction.form or prediction.confidence <= 0:
            continue
        hits = index.lookup(prediction.form, languages=[prediction.language], limit=1)
        if not hits:
            hits = index.fuzzy_lookup(
                prediction.form, max_distance=1, languages=[prediction.language]
            )[:1]
        if hits:
            found.append((prediction.language, hits[0]["word"]))
    return tuple(found)


_WITNESS_PREDICTOR: Any = None


def _shared_predictor_for_witnesses():
    global _WITNESS_PREDICTOR
    if _WITNESS_PREDICTOR is None:
        from engine.nlp.cognate_prediction import CognatePredictor

        _WITNESS_PREDICTOR = CognatePredictor()
    return _WITNESS_PREDICTOR


# --- Altın standartlar ------------------------------------------------------


def load_wold_cases(
    *, language: str = "Sakha", with_witnesses: bool = True
) -> list[BorrowingCase]:
    """WOLD'dan uzman etiketli alıntı/miras kayıtları.

    WOLD'un ``Borrowed`` alanı beş kademelidir; burada "kesinlikle alıntı"
    ve "muhtemelen alıntı" alıntı sayılır, "kesinlikle miras" miras sayılır,
    kararsız kademe **dışarıda bırakılır** — belirsiz maddeyi ölçüme sokmak
    hem sistemi hem ölçütü haksız yere cezalandırır.
    """
    directory = CLDF_DIR / "wold"
    forms_path = directory / "forms.csv"
    if not forms_path.exists():
        logger.info("WOLD indirilmemiş: python scripts/download_cldf.py wold")
        return []

    borrowed_scores = {"1. clearly borrowed", "2. probably borrowed"}
    inherited_scores = {"5. no evidence for borrowing", "4. very little evidence for borrowing"}

    cases: list[BorrowingCase] = []
    with forms_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Language_ID") != language:
                continue
            label = (row.get("Borrowed") or "").strip().lower()
            word = (row.get("Form") or row.get("Value") or "").strip()
            if not word or not to_comparison_form(word):
                continue
            if label in borrowed_scores:
                is_borrowed = True
            elif label in inherited_scores:
                is_borrowed = False
            else:
                continue
            cases.append(
                BorrowingCase(
                    word=word,
                    lang_code="sah",
                    is_borrowed=is_borrowed,
                    donor=(row.get("Borrowed_base") or "").strip(),
                    source="wold",
                )
            )
    return _attach_witnesses(cases, source_lang="sah") if with_witnesses else cases


def load_wiktionary_cases(
    *, lang: str = "tr", limit: int = 4000, with_witnesses: bool = True
) -> list[BorrowingCase]:
    """Sözlük indeksinden etiketli kayıtlar — **yalnız ablasyon ölçümü için**."""
    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not index.exists:
        return []
    with index.connect() as connection:
        rows = connection.execute(
            "SELECT word, origin, donor_lang FROM entries "
            "WHERE lang_code = ? AND origin IS NOT NULL AND length(comparison) >= 3 "
            "ORDER BY word LIMIT ?",
            (lang, limit),
        ).fetchall()
    cases = [
        BorrowingCase(
            word=row["word"],
            lang_code=lang,
            is_borrowed=row["origin"] == "alıntı",
            donor=row["donor_lang"] or "",
            source="wiktionary",
        )
        for row in rows
    ]
    return _attach_witnesses(cases, source_lang=lang) if with_witnesses else cases


def _attach_witnesses(
    cases: list[BorrowingCase], *, source_lang: str
) -> list[BorrowingCase]:
    """Her maddeye akraba tanıklarını iliştirir.

    ``BorrowingCase`` dondurulmuş bir veri sınıfı olduğu için yeniden
    kurulur. Tanık bulunamayan madde ELENMEZ — tanıksız da olsa fonotaktik
    ve zincir sinyalleri çalışır; elemek ölçümü kolaylaştırırdı.
    """
    out: list[BorrowingCase] = []
    for case in cases:
        witnesses = find_witnesses(case.word, source_lang=source_lang)
        out.append(
            BorrowingCase(
                word=case.word,
                lang_code=case.lang_code,
                is_borrowed=case.is_borrowed,
                donor=case.donor,
                source=case.source,
                witnesses=witnesses,
            )
        )
    counts = [len(c.witnesses) for c in out]
    logger.info(
        "Tanıklar iliştirildi: %d madde, kelime başına ort. %.2f tanık",
        len(out),
        sum(counts) / len(counts) if counts else 0.0,
    )
    return out


# --- Sistemler --------------------------------------------------------------


def always_inherited(case: BorrowingCase) -> bool:
    return False


def always_borrowed(case: BorrowingCase) -> bool:
    return True


def score_of(case: BorrowingCase, *, use_chain: bool) -> float:
    """Bir kelimenin alıntı skoru. ``use_chain=False`` ise zincir sinyali kapalı."""
    from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

    detector = _shared_detector()
    entries = [{"lang_code": c, "word": w} for c, w in case.witnesses]
    verdict = detector.detect(case.word, entries, lang=case.lang_code)
    if use_chain:
        return verdict.score
    # ⚠️ ABLASYON: zincir sinyali sözlük etiketini okur; ölçüm o etikete
    # karşı yapılıyorsa döngüseldir. Sinyal skordan çıkarılır ve kalan
    # ağırlıkların toplamına göre yeniden ölçeklenir.
    remaining = sum(w for name, w in SIGNAL_WEIGHTS.items() if name != "zincir_kanıtı")
    if not remaining:
        return 0.0
    score = sum(
        SIGNAL_WEIGHTS[s.name] * s.strength
        for s in verdict.signals
        if s.fired and s.name != "zincir_kanıtı"
    )
    return score / remaining


_DETECTOR: Any = None


def _shared_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from engine.nlp.borrowing_detector import BorrowingDetector

        _DETECTOR = BorrowingDetector()
    return _DETECTOR


def tune_threshold(cases: list[BorrowingCase], *, use_chain: bool) -> tuple[float, float]:
    """Eşiği **ayar bölümünde** arar.

    ⚠️ Ablasyonda varsayılan eşik kullanılamaz: o eşik zincir sinyali AÇIKKEN
    anlamlıdır. Sinyal çıkarılınca skor dağılımı tümden değişir ve aynı eşik
    hiçbir zaman aşılmaz — ölçüm "motor sıfır aldı" der ama bu bir bulgu
    değil, ölçüm hatasıdır.
    """
    scores = [(score_of(c, use_chain=use_chain), c.is_borrowed) for c in cases]
    best = (0.5, 0.0)
    for step in range(1, 20):
        threshold = step / 20
        prf = PRF()
        for score, is_borrowed in scores:
            predicted = score >= threshold
            if predicted and is_borrowed:
                prf.tp += 1
            elif predicted and not is_borrowed:
                prf.fp += 1
            elif not predicted and is_borrowed:
                prf.fn += 1
            else:
                prf.tn += 1
        if prf.fscore > best[1]:
            best = (threshold, prf.fscore)
    return best


def _detector(use_chain: bool, threshold: float):
    def run(case: BorrowingCase) -> bool:
        return score_of(case, use_chain=use_chain) >= threshold

    return run


def phonotactic_only(case: BorrowingCase) -> bool:
    """Miller ve ark. 2020 muadili: yalnız fonotaktik ihlal."""
    from engine.nlp.borrowing_detector import BorrowingDetector

    return BorrowingDetector._phonotactic_signal(case.word).fired


def score_system(system: Callable[[BorrowingCase], bool], cases: list[BorrowingCase]) -> PRF:
    result = PRF()
    for case in cases:
        predicted = system(case)
        correct = predicted == case.is_borrowed
        result.per_item.append(correct)
        if predicted and case.is_borrowed:
            result.tp += 1
        elif predicted and not case.is_borrowed:
            result.fp += 1
        elif not predicted and case.is_borrowed:
            result.fn += 1
        else:
            result.tn += 1
    return result


def evaluate(
    cases: list[BorrowingCase], *, use_chain: bool
) -> tuple[dict[str, PRF], float]:
    """Eşik AYAR yarısında seçilir, sonuç RAPOR yarısında verilir."""
    tune_set = [c for i, c in enumerate(cases) if i % 2 == 0]
    report_set = [c for i, c in enumerate(cases) if i % 2 == 1]
    threshold, _ = tune_threshold(tune_set, use_chain=use_chain)

    systems: dict[str, Callable[[BorrowingCase], bool]] = {
        "always_inherited": always_inherited,
        "always_borrowed": always_borrowed,
        "phonotactic_only": phonotactic_only,
        "engine": _detector(use_chain, threshold),
    }
    return {name: score_system(fn, report_set) for name, fn in systems.items()}, threshold


def _print_block(title: str, note: str, cases: list[BorrowingCase], scores: dict[str, PRF]) -> None:
    borrowed = sum(1 for c in cases if c.is_borrowed)
    print(f"\n=== {title} ===")
    print(f"{note}")
    print(f"n={len(cases)} · alıntı {borrowed} (%{100 * borrowed / len(cases):.1f}) · miras {len(cases) - borrowed}")
    print(f"\n{'sistem':20} {'F':>7} {'kesinlik':>9} {'duyarlılık':>11} {'doğruluk':>9}")
    print("-" * 60)
    for name, prf in sorted(scores.items(), key=lambda kv: -kv[1].fscore):
        marker = " *" if name == "engine" else ""
        print(
            f"{name + marker:20} {prf.fscore:>7.4f} {prf.precision:>9.4f} "
            f"{prf.recall:>11.4f} {prf.accuracy:>9.4f}"
        )


def _print_ablation_verdict(comparisons: list[dict[str, Any]]) -> None:
    """Motorun yalnız fonotaktiğe karşı üstünlüğü kanıtlandı mı?

    ⚠️ Bu ablasyon bir kez **ölçüm hatası yüzünden** anlamsız çıktı: tanıklar
    hiç doldurulmadığı için dört sinyalden ikisi devre dışıydı ve sonuç
    fonotaktikle birebir aynıydı. O sonuç "sinyaller katkısız" diye
    raporlanmıştı; doğrusu "sinyaller hiç çalıştırılmadı"ydı.
    """
    for row in comparisons:
        low, high = row["ci95"]
        significant = row["significant_after_fdr"]
        if not significant:
            verdict = "anlamlı DEĞİL — güven aralığı sıfırı içeriyor"
        elif row["difference"] > 0:
            verdict = "ANLAMLI ve POZİTİF — bağımsız sinyaller katkı sağlıyor"
        else:
            # ⚠️ Anlamlı ama NEGATİF fark, "katkı yok"tan kötüdür: sinyaller
            # ölçülebilir biçimde ZARAR veriyor demektir. Yönü gözden kaçıran
            # bir hüküm metni bunu "başarı" diye raporlardı.
            verdict = "ANLAMLI ama NEGATİF — bağımsız sinyaller ZARAR veriyor"
        print(
            f"\n  ablasyon hükmü: motor vs yalnız fonotaktik "
            f"fark {row['difference']:+.4f} [{low:+.4f}, {high:+.4f}] "
            f"p={row['permutation_p']:.4f}\n  -> {verdict}"
        )


def main() -> int:
    import argparse

    from engine.evaluation.report import EVAL_DIR
    from engine.evaluation.significance import compare_systems

    ap = argparse.ArgumentParser(description="Alıntı tespiti değerlendirmesi")
    ap.add_argument("--wiktionary-limit", type=int, default=3000)
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "_schema": "turkic-etymology-borrowing-eval/v1",
        "reference_monolingual_f1": REFERENCE_MONOLINGUAL_F1,
        "reference_crossfamily_f": REFERENCE_CROSSFAMILY_F,
    }

    wold = load_wold_cases()
    if wold:
        scores, wold_threshold = evaluate(wold, use_chain=True)
        _print_block(
            "BİRİNCİL ÖLÇÜT — WOLD (uzman, Wiktionary'den bağımsız)",
            "Sakha (Yakutça); WOLD'un 41 dilinden tek Türki olanı.",
            [c for i, c in enumerate(wold) if i % 2 == 1],
            scores,
        )
        print(f"(eşik ayar yarısında seçildi: {wold_threshold:.2f})")
        payload["wold"] = {
            "language": "sah",
            "n": len(wold) // 2,
            "tuned_threshold": wold_threshold,
            "systems": {k: v.as_dict() for k, v in scores.items()},
            "significance": compare_systems(
                {k: v.per_item for k, v in scores.items()}, reference="always_inherited"
            ),
            # ⚠️ ASIL ABLASYON SORUSU: motor, yalnız fonotaktikten iyi mi?
            # `always_inherited`e karşı karşılaştırma bunu cevaplamaz.
            "vs_phonotactic": compare_systems(
                {
                    "engine": scores["engine"].per_item,
                    "phonotactic_only": scores["phonotactic_only"].per_item,
                },
                reference="phonotactic_only",
            ),
        }
        _print_ablation_verdict(payload["wold"]["vs_phonotactic"])
    else:
        print("WOLD yok — birincil ölçüt atlandı (python scripts/download_cldf.py wold)")

    wiktionary = load_wiktionary_cases(limit=args.wiktionary_limit)
    if wiktionary:
        scores, ablation_threshold = evaluate(wiktionary, use_chain=False)
        _print_block(
            "ABLASYON — Wiktionary etiketi, zincir sinyali KAPALI",
            (
                "⚠️ Motorun zincir sinyali bu etiketi okur; açık bırakmak ölçümü\n"
                "döngüsel yapardı. Ölçülen: fonotaktik + ses kanunu + değişimsiz\n"
                "yayılım sinyalleri etiketi HİÇ GÖRMEDEN alıntıyı bulabiliyor mu?"
            ),
            [c for i, c in enumerate(wiktionary) if i % 2 == 1],
            scores,
        )
        print(f"(eşik ayar yarısında seçildi: {ablation_threshold:.2f})")
        payload["wiktionary_ablation"] = {
            "language": "tr",
            "n": len(wiktionary) // 2,
            "tuned_threshold": ablation_threshold,
            "chain_signal": "disabled",
            "systems": {k: v.as_dict() for k, v in scores.items()},
            "significance": compare_systems(
                {k: v.per_item for k, v in scores.items()}, reference="always_inherited"
            ),
            "vs_phonotactic": compare_systems(
                {
                    "engine": scores["engine"].per_item,
                    "phonotactic_only": scores["phonotactic_only"].per_item,
                },
                reference="phonotactic_only",
            ),
        }
        _print_ablation_verdict(payload["wiktionary_ablation"]["vs_phonotactic"])

    print(
        f"\nreferanslar: tek dilli fonotaktik F1 ≈ {REFERENCE_MONOLINGUAL_F1} "
        f"(Miller ve ark. 2020) · aileler arası seabor F ≈ {REFERENCE_CROSSFAMILY_F} "
        f"(List & Forkel 2022)"
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "borrowing.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
