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
from dataclasses import dataclass, field, replace
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
    #: Kavram adı (Concepticon glossu). Verici yakınlığı sinyali **anlam
    #: kısıtlıdır** (sabor'un yayınlanmış kurulumu); kavram olmadan arama
    #: 440.910 maddelik Rusça sözlüğe yayılır ve şans benzerliğine açılır.
    sense: str = ""


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

    parameters_path = directory / "parameters.csv"
    senses: dict[str, str] = {}
    if parameters_path.exists():
        with parameters_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                senses[row["ID"]] = (
                    row.get("Concepticon_Gloss") or row.get("Name") or ""
                ).strip()

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
                    sense=senses.get(row.get("Parameter_ID") or "", ""),
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
            # ⚠️ ``gloss`` da alınır: verici yakınlığı sinyali ANLAM
            # KISITLIDIR ve anlamsız madde o sinyali hiç ateşlemez.
            # İlk sürümde alınmıyordu ve sinyal Türkçe ablasyonu boyunca
            # F=0,0000 veriyordu — "katkısız" değil, "hiç çalışmadı".
            "SELECT word, origin, donor_lang, gloss FROM entries "
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
            sense=row["gloss"] or "",
        )
        for row in rows
    ]
    return _attach_witnesses(cases, source_lang=lang) if with_witnesses else cases


def _attach_witnesses(
    cases: list[BorrowingCase], *, source_lang: str
) -> list[BorrowingCase]:
    """Her maddeye akraba tanıklarını iliştirir.

    Tanık bulunamayan madde ELENMEZ — tanıksız da olsa fonotaktik ve zincir
    sinyalleri çalışır; elemek ölçümü kolaylaştırırdı.

    ⚠️ ``dataclasses.replace`` kullanılır, alan alan yeniden kurulmaz.
    Elle kurulan sürüm ``sense`` alanı eklendiğinde onu **sessizce
    düşürüyordu** ve verici yakınlığı sinyali anlamsız kaldığı için
    ablasyon boyunca F=0,0000 veriyordu. Bu, ``witnesses`` alanının daha
    önce aynı yolla düşmesiyle birebir aynı hatadır; ``replace`` o hata
    sınıfına yapısal olarak kapalıdır.
    """
    out = [
        replace(case, witnesses=find_witnesses(case.word, source_lang=source_lang))
        for case in cases
    ]
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


#: Hangi verici dillere bakılacak? Ölçüt Sakha olduğu için WOLD'da ölçülen
#: gerçek kaynak dağılımı kullanılır: Rusça 284 · Moğolca 253 · Evenkice 19.
SAKHA_DONORS = ["ru", "mn", "evn"]

#: Türkçenin tarihsel vericileri. Sakha'dan bambaşka bir kümedir; aynı
#: listeyi iki dile birden vermek her iki ölçümü de bozar.
TURKISH_DONORS = ["ar", "fa", "el", "hy", "fr", "it"]


def donors_for(lang_code: str) -> list[str] | None:
    """Bu dil için hangi verici sözlüklerine bakılacak?

    ⚠️ Verici kümesi dile göre değişir ve bu **ölçümü belirler**: havuz
    büyüdükçe şans benzerliği artar. Sakha'ya Fransızca sözlüğü açmak
    yalnız gürültü ekler.
    """
    if lang_code == "sah":
        return SAKHA_DONORS
    if lang_code in ("tr", "ota"):
        return TURKISH_DONORS
    return None


def score_of(
    case: BorrowingCase, *, use_chain: bool, drop: tuple[str, ...] = ()
) -> float:
    """Bir kelimenin alıntı skoru.

    :param use_chain: ``False`` ise zincir sinyali skordan çıkarılır.
    :param drop: skordan çıkarılacak başka sinyal adları (ablasyon).

    ⚠️ Sinyal çıkarıldığında skor **kalan ağırlıkların toplamına** yeniden
    ölçeklenir. Ölçeklenmezse skor dağılımı çöker ve varsayılan eşik hiçbir
    zaman aşılmaz; ölçüm "motor sıfır aldı" der ama bu bulgu değil ölçüm
    hatasıdır.
    """
    from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

    detector = _shared_detector()
    entries = [{"lang_code": c, "word": w} for c, w in case.witnesses]
    verdict = detector.detect(
        case.word,
        entries,
        lang=case.lang_code,
        sense=case.sense,
        donors=donors_for(case.lang_code),
    )
    removed = set(drop) | ({"zincir_kanıtı"} if not use_chain else set())
    if not removed:
        return verdict.score
    remaining = sum(w for name, w in SIGNAL_WEIGHTS.items() if name not in removed)
    if not remaining:
        return 0.0
    score = sum(
        SIGNAL_WEIGHTS[s.name] * s.strength
        for s in verdict.signals
        if s.fired and s.name not in removed
    )
    return score / remaining


def signal_strengths(case: BorrowingCase, *, use_chain: bool) -> dict[str, float]:
    """Maddenin sinyal güçleri — eğitilmiş birleştiricinin girdisi.

    ``use_chain=False`` ise zincir sinyali **sıfırlanır**, çıkarılmaz:
    modelin girdi boyutu sabit kalmalı.
    """
    detector = _shared_detector()
    entries = [{"lang_code": c, "word": w} for c, w in case.witnesses]
    verdict = detector.detect(
        case.word,
        entries,
        lang=case.lang_code,
        sense=case.sense,
        donors=donors_for(case.lang_code),
    )
    out = {s.name: (s.strength if s.fired else 0.0) for s in verdict.signals}
    if not use_chain:
        out["zincir_kanıtı"] = 0.0
    return out


def train_combiner(
    cases: list[BorrowingCase], *, use_chain: bool, trained_on: str, objective: str = "fscore"
):
    """Birleştiriciyi **ayar yarısında** eğitir."""
    from engine.nlp.borrowing_combiner import fit

    samples = [
        (signal_strengths(c, use_chain=use_chain), c.is_borrowed) for c in cases
    ]
    return fit(samples, trained_on=trained_on, objective=objective)


def train_phonotactic_lm(cases: list[BorrowingCase], *, language: str, trained_on: str):
    """Fonotaktik dizilim modelini **ayar yarısında** eğitir ve kaydeder.

    ⚠️ Model diske yazılır çünkü ``BorrowingDetector`` onu oradan okur.
    Eğitim ayar yarısındadır; rapor yarısı hiç görülmez.
    """
    from engine.nlp.phonotactic_lm import fit, save

    samples = [(c.word, c.is_borrowed) for c in cases]
    classifier = fit(samples, language=language, trained_on=trained_on)
    save(classifier)
    return classifier


def phonotactic_model_only(case: BorrowingCase) -> bool:
    """Yalnız eğitilmiş dizilim modeli — PyBor'un (Miller ve ark. 2020)
    yayınlanmış kurulumu. WOLD 41 dil ortalaması F1 0,59-0,61."""
    from engine.nlp.phonotactic_lm import load

    classifier = load(case.lang_code)
    return classifier is not None and classifier.predict(case.word)


def donor_proximity_only(case: BorrowingCase) -> bool:
    """Yalnız verici yakınlığı — sabor'un (Miller & List 2023) tek sinyali.

    Bu, motorun geçmesi gereken **gerçek** taban çizgidir: yayınlanmış F1
    0,806'dır ve dört sinyalli motorumuz WOLD/Sakha'da 0,385 alıyordu.
    """
    from engine.nlp.donor_proximity import nearest_donor

    match = nearest_donor(
        to_comparison_form(case.word), case.sense, languages=donors_for(case.lang_code)
    )
    return match is not None and match.is_close


_DETECTOR: Any = None


def _shared_detector():
    global _DETECTOR
    if _DETECTOR is None:
        from engine.nlp.borrowing_detector import BorrowingDetector

        _DETECTOR = BorrowingDetector()
    return _DETECTOR


def tune_threshold(
    cases: list[BorrowingCase], *, use_chain: bool, drop: tuple[str, ...] = ()
) -> tuple[float, float]:
    """Eşiği **ayar bölümünde** arar.

    ⚠️ Ablasyonda varsayılan eşik kullanılamaz: o eşik zincir sinyali AÇIKKEN
    anlamlıdır. Sinyal çıkarılınca skor dağılımı tümden değişir ve aynı eşik
    hiçbir zaman aşılmaz — ölçüm "motor sıfır aldı" der ama bu bir bulgu
    değil, ölçüm hatasıdır.
    """
    scores = [(score_of(c, use_chain=use_chain, drop=drop), c.is_borrowed) for c in cases]
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


def signal_ablation(
    cases: list[BorrowingCase],
    *,
    use_chain: bool,
    tune_set: list[BorrowingCase] | None = None,
) -> dict[str, PRF]:
    """Her sinyali TEK TEK çıkarıp motoru yeniden ölçer.

    ⚠️ "Sinyal katkı sağlıyor mu?" sorusunun tek dürüst cevabı budur.
    Motoru trivial bir taban çizgiyle karşılaştırmak sinyallerin **hangisinin**
    işe yaradığını söylemez; bir sinyal ötekinin zararını kapatıyor olabilir.

    Her ablasyonda eşik **yeniden ayarlanır**: sinyal çıkınca skor dağılımı
    değişir, eski eşik anlamını yitirir.
    """
    from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

    # ⚠️ Eşik, tam motorunkiyle **aynı** ayar kümesinde seçilmeli. Dizilim
    # modeli devreye girince ayar yarısı ikiye bölündü; ablasyon o bölünmeyi
    # görmezse daha çok veriyle daha iyi bir eşik bulur ve "sinyali
    # çıkarmak motoru iyileştirdi" gibi sahte bir sonuç üretir.
    if tune_set is None:
        tune_set = [c for i, c in enumerate(cases) if i % 2 == 0]
    report_set = [c for i, c in enumerate(cases) if i % 2 == 1]
    out: dict[str, PRF] = {}
    for name in SIGNAL_WEIGHTS:
        if not use_chain and name == "zincir_kanıtı":
            continue
        drop = (name,)
        threshold, _ = tune_threshold(tune_set, use_chain=use_chain, drop=drop)

        def run(case: BorrowingCase, _d=drop, _t=threshold) -> bool:
            return score_of(case, use_chain=use_chain, drop=_d) >= _t

        out[f"-{name}"] = score_system(run, report_set)
    return out


def evaluate(
    cases: list[BorrowingCase], *, use_chain: bool, trained_on: str = ""
) -> tuple[dict[str, PRF], float, Any, list[BorrowingCase]]:
    """Eşik AYAR yarısında seçilir, sonuç RAPOR yarısında verilir."""
    tune_set = [c for i, c in enumerate(cases) if i % 2 == 0]
    report_set = [c for i, c in enumerate(cases) if i % 2 == 1]

    # ⚠️ **YIĞIN SIZINTISI.** Dizilim modeli eğitildiği veride kendi
    # eğitim örneklerini EZBERLER. Modeli tüm ayar yarısında eğitip eşiği
    # de aynı yarıda ayarlamak, eşiğe modelin gerçekte sahip olmadığı bir
    # ayırt etme gücünü varsaydırır. Ölçüldü::
    #
    #     ayar yarısı (model burada eğitildi)  sınıf ayrımı 1,9197
    #     rapor yarısı (hiç görülmedi)         sınıf ayrımı 0,6278
    #                                          -> 3,1 kat şişkin
    #
    # Sonuç: motorun F'si 0,6461'den 0,5868'e DÜŞTÜ ve ablasyon "verici
    # yakınlığı zararlı" gibi saçma bir sonuç verdi.
    #
    # Düzeltme: ayar yarısı ikiye bölünür. Model YALNIZ ilk parçada
    # eğitilir; eşik ikinci parçada, modelin hiç görmediği veride ayarlanır.
    language = tune_set[0].lang_code if tune_set else ""
    if language:
        model_set = [c for i, c in enumerate(tune_set) if i % 2 == 0]
        threshold_set = [c for i, c in enumerate(tune_set) if i % 2 == 1]
        train_phonotactic_lm(model_set, language=language, trained_on=trained_on)
        from engine.nlp.borrowing_detector import BorrowingDetector

        BorrowingDetector.reset_combiner_cache()
        tune_set = threshold_set

    threshold, _ = tune_threshold(tune_set, use_chain=use_chain)

    systems: dict[str, Callable[[BorrowingCase], bool]] = {
        "always_inherited": always_inherited,
        "always_borrowed": always_borrowed,
        "phonotactic_only": phonotactic_only,
        "donor_proximity_only": donor_proximity_only,
        "phonotactic_model_only": phonotactic_model_only,
        "engine": _detector(use_chain, threshold),
    }

    # ⚠️ Eğitilmiş birleştirici AYAR yarısında eğitilir, RAPOR yarısında
    # ölçülür. Aynı veride hem eğitip hem ölçmek ölçümü yok sayar.
    combiner = train_combiner(tune_set, use_chain=use_chain, trained_on=trained_on)
    systems["engine_trained"] = lambda case: combiner.predict(
        signal_strengths(case, use_chain=use_chain)
    )
    # ⚠️ Aynı model, yalnız eşik hedefi farklı. F ile doğruluk aynı anda
    # alınamaz; ikisi de raporlanır ki seçim gizlenmesin.
    accurate = train_combiner(
        tune_set, use_chain=use_chain, trained_on=trained_on, objective="accuracy"
    )
    systems["engine_trained_acc"] = lambda case: accurate.predict(
        signal_strengths(case, use_chain=use_chain)
    )
    scores = {name: score_system(fn, report_set) for name, fn in systems.items()}
    return scores, threshold, combiner, tune_set


def _print_block(title: str, note: str, cases: list[BorrowingCase], scores: dict[str, PRF]) -> None:
    borrowed = sum(1 for c in cases if c.is_borrowed)
    print(f"\n=== {title} ===")
    print(f"{note}")
    print(f"n={len(cases)} · alıntı {borrowed} (%{100 * borrowed / len(cases):.1f}) · miras {len(cases) - borrowed}")
    print(f"\n{'sistem':20} {'F':>7} {'kesinlik':>9} {'duyarlılık':>11} {'doğruluk':>9}")
    print("-" * 60)
    trivial = scores.get("always_borrowed")
    degenerate: list[str] = []
    for name, prf in sorted(scores.items(), key=lambda kv: -kv[1].fscore):
        marker = " *" if name == "engine" else ""
        print(
            f"{name + marker:20} {prf.fscore:>7.4f} {prf.precision:>9.4f} "
            f"{prf.recall:>11.4f} {prf.accuracy:>9.4f}"
        )
        if (
            trivial is not None
            and name not in ("always_borrowed", "always_inherited")
            and prf.per_item == trivial.per_item
        ):
            degenerate.append(name)
    if degenerate:
        # ⚠️ F skoru dengesiz sınıfta "hepsine alıntı de" diyerek en yükseğe
        # çıkabilir. O skor bir başarı değil, ölçünün bilinen patolojisidir;
        # tabloya bakıp geçen biri bunu "motor trivialle başa baş" sanır.
        print(
            f"\n  ⚠️ ÇÖKMÜŞ SİSTEM: {', '.join(degenerate)} — "
            f"kararları `always_borrowed` ile BİREBİR aynı.\n"
            f"     Alıntı oranı %{100 * borrowed / len(cases):.1f} olduğu için F'yi "
            f"en yükselten eşik 'hepsine alıntı de'dir.\n"
            f"     Bu bir başarı değil, F ölçüsünün dengesiz sınıftaki bilinen "
            f"patolojisidir."
        )


def _print_signal_ablation(full: PRF, ablated: dict[str, PRF]) -> None:
    """Sinyal sinyal katkı tablosu — düşüş ne kadarsa katkı o kadardır."""
    if not ablated:
        return
    print(f"\n  sinyal katkısı (çıkarınca F ne kadar düşüyor?) — tam motor F={full.fscore:.4f}")
    rows = sorted(ablated.items(), key=lambda kv: kv[1].fscore)
    for name, prf in rows:
        delta = full.fscore - prf.fscore
        mark = "katkılı" if delta > 0 else ("nötr" if delta == 0 else "ZARARLI")
        print(f"    {name:24} F={prf.fscore:.4f}  katkı {delta:+.4f}  {mark}")


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
    from engine.nlp.borrowing_combiner import save

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
        scores, wold_threshold, wold_combiner, wold_tune = evaluate(
            wold, use_chain=True, trained_on="wold/sah/tune"
        )
        _print_block(
            "BİRİNCİL ÖLÇÜT — WOLD (uzman, Wiktionary'den bağımsız)",
            "Sakha (Yakutça); WOLD'un 41 dilinden tek Türki olanı.",
            [c for i, c in enumerate(wold) if i % 2 == 1],
            scores,
        )
        print(f"(eşik ayar yarısında seçildi: {wold_threshold:.2f})")
        print(f"  eğitilmiş birleştirici: {wold_combiner.explain()}")
        save(wold_combiner)
        payload["wold"] = {
            "language": "sah",
            "n": len(wold) // 2,
            "tuned_threshold": wold_threshold,
            "combiner": wold_combiner.as_dict(),
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
            # ⚠️ ASIL SORU BU: hangi sinyal katkı sağlıyor?
            "vs_donor_proximity": compare_systems(
                {
                    "engine": scores["engine"].per_item,
                    "engine_trained": scores["engine_trained"].per_item,
                    "donor_proximity_only": scores["donor_proximity_only"].per_item,
                },
                reference="donor_proximity_only",
            ),
        }
        wold_signals = signal_ablation(wold, use_chain=True, tune_set=wold_tune)
        payload["wold"]["signal_ablation"] = {
            k: v.as_dict() for k, v in wold_signals.items()
        }
        payload["wold"]["signal_significance"] = compare_systems(
            {"engine": scores["engine"].per_item,
             **{k: v.per_item for k, v in wold_signals.items()}},
            reference="engine",
        )
        _print_signal_ablation(scores["engine"], wold_signals)
        _print_ablation_verdict(payload["wold"]["vs_phonotactic"])
    else:
        print("WOLD yok — birincil ölçüt atlandı (python scripts/download_cldf.py wold)")

    wiktionary = load_wiktionary_cases(limit=args.wiktionary_limit)
    if wiktionary:
        scores, ablation_threshold, wiki_combiner, _ = evaluate(
            wiktionary, use_chain=False, trained_on="wiktionary/tr/tune"
        )
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
