"""
N-best üretimi ve yeniden sıralama (Faz D5).

⚠️ Bu modül **ölçülmüş bir tavanı** hedefler. Motorun sütun sütun verdiği
top-1 karar dev bölümünde %28,4 tam doğruluk veriyordu; aynı sütunların
2. ve 3. seçeneklerinden kombinatorik olarak üretilen adaylar arasında
doğru cevap **%43,2** oranında bulunuyordu (ort. 32 aday). Yani doğru cevap
zaten üretiliyor ama tepede değil.

Lu, Wang & Mortensen (LREC-COLING 2024) aynı örüntüyü dört veri setinde
ölçüp yeniden sıralamayla **+0,9…+3,1 puan** kazanıyor (P2D: ata biçim
adayından kız dilleri üretip gerçek tanıklarla karşılaştırma).

⚠️ Onların kurulumu **denetimlidir** (sinir ağı üreteç + sinir ağı
sıralayıcı). Bizimki kural/sayım tabanlı bir üretecin üstüne kuruluyor ve
bu ayarda yayınlanmış bir doğrulaması yok.

## ⚠️ SONUÇ: yeniden sıralama KAZANÇ VERMİYOR — kapalı

Ölçüldü (dev, n=83, sütun başına 3 aday, ort. 11 aday)::

    sıralama ölçütü                 tam doğruluk
    konsensüs (= top-1, mevcut)        0,4337
    yalnız P2D üretim uyumu            0,3614
    konsensüs + 0,2·P2D                0,4217
    konsensüs + 0,5·P2D                0,4337
    konsensüs + 0,8·P2D                0,3855
    N-best oracle (tavan)              0,5060

Hiçbir karışım konsensüsü **geçmiyor**. Doğru cevap adayların içinde
(oracle 0,506, top-1'in 7 puan üstünde) ama elimizdeki sıralayıcı onu öne
çıkaramıyor.

Kök neden büyük olasılıkla üreteç zayıflığı: ``pt -> X`` denklikleri yalnız
~237 rekonstrüksiyonlu eğitim kümesinden öğreniliyor; P2D'nin yayınlanmış
kazançları sinir ağı üreteçlerle elde edilmiş.

**Karar:** adaylar üretiliyor ve kullanıcıya **rakip hipotez** olarak
gösteriliyor (bu kendi başına değerli), ama seçilen biçim değişmiyor.
:func:`rerank` deney ve ölçüm içindir.

⚠️ İlk sürümde üretim uyumu **jenerik refleksle** hesaplanıyordu
(``to_expected_reflex``) ve daha da kötüydü: tek başına 0,3133. Gerçek P2D
için ata dil, denklik tablolarına **sözde dil** olarak katıldı
(``cognate_prediction.PROTO_CODE``) ve her tanık dilin kendi biçmi
üretiliyor; bu 0,3133'ten 0,3614'e çıkardı ama yine yetmedi.

## Sıralama ölçütü

Her aday ata biçim için üç şey ölçülür:

1. **Üretim uyumu** — adaydan Ortak Türkçe'de beklenen refleks üretilip
   (``to_expected_reflex``) gerçek tanıklara olan ortalama uzaklığı.
   P2D'nin çekirdek fikri budur.
2. **Sütun konsensüsü** — adayın seçtiği seslerin sütun puanlarının çarpımı.
3. **Ata biçim makullüğü** — ``proto_plausibility``; uydurma bir dizi
   tanıklara uyabilir ama Türkçe olmayabilir.

⚠️ Üçü de **aynı yönde** değil: konsensüs zaten top-1'i seçen ölçüttür, tek
başına yeniden sıralama yapmaz. Kazanç varsa üretim uyumundan gelir.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.cognate_prediction import PROTO_CODE
from engine.nlp.proto_phonology import ColumnDecision, proto_plausibility
from engine.utils.orthography import to_comparison_form, to_expected_reflex

logger = get_logger(__name__)

#: Sütun başına en çok kaç aday taşınır?
#:
#: ⚠️ Sayı kombinatoriktir: 5 sütun × 3 aday = 243 birleşim. Ölçülen oracle
#: (%43,2) sütun başına 2-3 adayla elde edilmişti.
MAX_PER_COLUMN = 3

#: Toplam aday sayısı tavanı.
MAX_CANDIDATES = 64

#: Bir alternatifin taşınması için gereken asgari göreli puan.
#:
#: ⚠️ Eşiksiz taşımak, tek bir dilin tek gözlemini "aday" mertebesine
#: çıkarır ve aday havuzunu gürültüyle doldurur. Ölçüldü (dev, n=83)::
#:
#:     eşik   ort. aday   oracle
#:     0,15      3,9      0,4578
#:     0,05     11,0      0,5060
#:     0,00     17,6      0,5060
#:
#: 0,05'ten sonra oracle artmıyor; havuzu büyütmek yalnız gürültü ekliyor.
MIN_ALTERNATIVE_SCORE = 0.05

#: Üretim uyumunun sıralamadaki ağırlığı (λ).
GENERATION_WEIGHT = 0.6

#: Ata biçim makullüğünün ağırlığı.
PLAUSIBILITY_WEIGHT = 0.2


@dataclass(frozen=True)
class Candidate:
    """Tek bir ata biçim adayı ve puan dökümü."""

    form: str
    consensus: float
    generation: float
    plausibility: float

    @property
    def score(self) -> float:
        return (
            (1.0 - GENERATION_WEIGHT - PLAUSIBILITY_WEIGHT) * self.consensus
            + GENERATION_WEIGHT * self.generation
            + PLAUSIBILITY_WEIGHT * self.plausibility
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "form": f"*{self.form}",
            "score": round(self.score, 4),
            "consensus": round(self.consensus, 4),
            "generation_fit": round(self.generation, 4),
            "plausibility": round(self.plausibility, 4),
        }


def _normalised_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


@lru_cache(maxsize=1)
def _proto_predictor() -> Any:
    """``pt -> X`` denklikleriyle kız dil üreteci — yoksa ``None``.

    ⚠️ P2D'nin çekirdeği budur: ata biçim adayından **her tanık dilin
    kendi** biçmi üretilir. Jenerik "Ortak Türkçe refleksi" yetmez.
    """
    from engine.nlp.cognate_prediction import CognatePredictor, load_proto_tables

    tables = load_proto_tables()
    if not tables:
        logger.info("Ata dil denklik tablosu yok; P2D yeniden sıralaması devre dışı")
        return None
    return CognatePredictor(tables)


def reset_cache() -> None:
    _proto_predictor.cache_clear()


def generation_fit(proto_form: str, witnesses: dict[str, str]) -> float:
    """Adaydan üretilen kız biçimler tanıklara ne kadar uyuyor? ``[0, 1]``.

    ⚠️ **İlk sürüm jenerik refleks kullanıyordu ve ZARAR VERİYORDU.**
    ``to_expected_reflex`` ata biçmi tek bir "Ortak Türkçe" biçme çevirir ve
    o biçim bütün tanıklarla karşılaştırılırdı; bu, "ortalamaya benzeyen"
    adayı ödüllendirir — yani konsensüsün yaptığını daha gürültülü yapar.
    Ölçüldü (dev, n=83): tek başına sıralayıcı olarak **0,3133**, konsensüs
    ise 0,4337.

    Şimdi her tanık dil için **kendi** biçmi üretiliyor
    (``pt -> tr``, ``pt -> cv``…), P2D'nin yayınlanmış kurulumu gibi.
    Tablo yoksa jenerik yola düşülür ama o yol ölçülmüş biçimde zayıftır.
    """
    if not proto_form or not witnesses:
        return 0.0
    predictor = _proto_predictor()
    distances: list[float] = []
    if predictor is not None:
        for language, form in witnesses.items():
            target = to_comparison_form(form)
            if not target:
                continue
            prediction = predictor.predict(proto_form, PROTO_CODE, language)
            if prediction.form and prediction.confidence > 0:
                distances.append(_normalised_distance(prediction.form, target))
    if not distances:
        expected = to_expected_reflex(proto_form)
        distances = [
            _normalised_distance(expected, to_comparison_form(form))
            for form in witnesses.values()
            if to_comparison_form(form)
        ]
    if not distances:
        return 0.0
    return 1.0 - sum(distances) / len(distances)


def generate(
    decisions: list[ColumnDecision],
    *,
    max_per_column: int = MAX_PER_COLUMN,
    max_candidates: int = MAX_CANDIDATES,
) -> list[tuple[str, float]]:
    """Sütun kararlarından N-best ata biçim adayları üretir.

    :returns: ``(biçim, konsensüs)`` çiftleri, konsensüse göre sıralı.
        İlk eleman her zaman top-1 karardır.
    """
    options: list[list[tuple[str, float]]] = []
    for decision in decisions:
        alternatives = [
            (sound, score)
            for sound, score in (decision.alternatives or ((decision.sound, 1.0),))
            if sound and score >= MIN_ALTERNATIVE_SCORE
        ][:max_per_column]
        if not alternatives:
            alternatives = [(decision.sound, 1.0)] if decision.sound else []
        # ⚠️ Top-1 kararı her zaman ilk sırada olmalı: yeniden sıralama
        # başarısız olduğunda sistem eski davranışına dönebilmeli.
        if decision.sound and alternatives[0][0] != decision.sound:
            alternatives = [(decision.sound, 1.0)] + [
                a for a in alternatives if a[0] != decision.sound
            ]
        if alternatives:
            options.append(alternatives)

    if not options:
        return []

    candidates: list[tuple[str, float]] = []
    for combination in product(*options):
        form = "".join(sound for sound, _ in combination)
        consensus = 1.0
        for _, score in combination:
            consensus *= score
        candidates.append((form, consensus))
        if len(candidates) >= max_candidates:
            break
    candidates.sort(key=lambda pair: -pair[1])
    return candidates


def rerank(
    decisions: list[ColumnDecision],
    witnesses: dict[str, str],
    *,
    max_per_column: int = MAX_PER_COLUMN,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """N-best adayları üretip üretim uyumuna göre yeniden sıralar."""
    raw = generate(
        decisions, max_per_column=max_per_column, max_candidates=max_candidates
    )
    if not raw:
        return []
    scored = [
        Candidate(
            form=form,
            consensus=consensus,
            generation=generation_fit(form, witnesses),
            plausibility=proto_plausibility(form)[0],
        )
        for form, consensus in raw
    ]
    scored.sort(key=lambda c: (-c.score, c.form))
    return scored
