"""
Eğitilmiş fonotaktik dil modeli — PyBor'un (Miller ve ark. 2020) yaklaşımı.

Mevcut ``fonotaktik_ihlal`` sinyali **elle yazılmış kurallara** dayanıyor
(ünlü uyumu, yasak söz başı sesler) ve WOLD/Sakha'da tek başına
**F 0,215** alıyor. PyBor'un yayınlanmış WOLD ortalaması **F1 0,59-0,61**.

Fark yöntemde: PyBor kural yazmaz, **iki ayrı dil modeli eğitir** — biri
miras kelimelerden, biri alıntılardan — ve kelimeyi hangisinin daha iyi
açıkladığına bakar. Böylece "Türkçede ünlü uyumu vardır" gibi kaba bir
kuralın kaçırdığı ince dizilim farkları (``pençe`` vs ``pencere``) yakalanır.

Uygulama: **karakter 3-gram Markov modeli**, Witten-Bell yumuşatmalı.
LSTM sürümü yayında biraz daha iyi (F1 0,61 vs 0,59) ama bağımlılık
gerektiriyor; Markov sürümü repoda ve deterministik.

⚠️ **Eğitim ve ölçüm ayrı yarılarda.** Model AYAR yarısında eğitilir.
Aynı veride hem eğitip hem ölçmek, ölçümü yok sayar.

⚠️ Model **dile özgüdür**. Sakha'da eğitilmiş bir model Türkçeye
uygulanamaz: fonotaktik dilden dile değişir, zaten ölçtüğü şey odur.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

MODEL_DIR = PROJECT_ROOT / "data" / "models"

#: Bağlam uzunluğu. PyBor 3-gram kullanıyor (yani 2 karakterlik bağlam).
ORDER = 3

#: Sınır işareti. Söz başı ve söz sonu dizilimleri ayırt edici olduğu için
#: (``ostuol``daki öntüreme ünlüsü gibi) sınırlar modele girer.
BOUNDARY = "#"


@dataclass
class MarkovModel:
    """Karakter n-gram modeli, Witten-Bell yumuşatmalı.

    ⚠️ Yumuşatma şart: eğitimde görülmemiş tek bir üçlü, yumuşatmasız
    modelde olasılığı **sıfıra** indirir ve kelime hangi sınıfa ait olursa
    olsun elenir.
    """

    order: int = ORDER
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    vocabulary: set[str] = field(default_factory=set)

    def _grams(self, word: str) -> list[tuple[str, str]]:
        padded = BOUNDARY * (self.order - 1) + word + BOUNDARY
        return [
            (padded[i : i + self.order - 1], padded[i + self.order - 1])
            for i in range(len(padded) - self.order + 1)
        ]

    def observe(self, word: str) -> None:
        for context, character in self._grams(word):
            bucket = self.counts.setdefault(context, {})
            bucket[character] = bucket.get(character, 0) + 1
            self.vocabulary.add(character)

    def log_probability(self, word: str) -> float:
        """Kelimenin ortalama karakter başına log olasılığı.

        Kelime uzunluğuna bölünür: uzun kelimeler her modelde daha düşük
        toplam olasılık alır, bölmezsek karşılaştırma uzunluğa kayardı.
        """
        grams = self._grams(word)
        if not grams:
            return -math.inf
        size = max(len(self.vocabulary), 1)
        total = 0.0
        for context, character in grams:
            bucket = self.counts.get(context, {})
            observed = sum(bucket.values())
            distinct = len(bucket)
            # Witten-Bell: görülmemiş olaya ayrılan kütle, farklı devam
            # sayısıyla orantılıdır.
            if observed:
                seen = bucket.get(character, 0)
                lambda_ = observed / (observed + distinct)
                probability = lambda_ * (seen / observed) + (1 - lambda_) / size
            else:
                probability = 1.0 / size
            total += math.log(max(probability, 1e-12))
        return total / len(grams)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "counts": {k: v for k, v in sorted(self.counts.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarkovModel:
        model = cls(order=int(data.get("order", ORDER)))
        for context, bucket in (data.get("counts") or {}).items():
            model.counts[context] = {k: int(v) for k, v in bucket.items()}
            model.vocabulary.update(bucket)
        return model


@dataclass
class PhonotacticClassifier:
    """İki modelli alıntı sınıflandırıcısı (PyBor).

    Karar, iki modelin log olasılık **farkıdır**: kelime alıntı modelinde
    miras modelinden daha olası ise alıntı adayıdır.
    """

    inherited: MarkovModel = field(default_factory=MarkovModel)
    borrowed: MarkovModel = field(default_factory=MarkovModel)
    threshold: float = 0.0
    language: str = ""
    trained_on: str = ""
    n_inherited: int = 0
    n_borrowed: int = 0
    trained_at: str = ""

    @property
    def is_trained(self) -> bool:
        return self.n_inherited > 0 and self.n_borrowed > 0

    def score(self, word: str) -> float:
        """``log P(alıntı) - log P(miras)``. Pozitif ise alıntı yönünde."""
        form = to_comparison_form(word)
        if not form or not self.is_trained:
            return 0.0
        return self.borrowed.log_probability(form) - self.inherited.log_probability(form)

    def predict(self, word: str) -> bool:
        return self.score(word) >= self.threshold

    def strength(self, word: str) -> float:
        """Skoru ``[0, 1]`` aralığında sinyal gücüne çevirir."""
        score = self.score(word)
        if score <= self.threshold:
            return 0.0
        # Eşikten 0,5 nat uzaklıkta tam güç: ölçülen skorların çoğu bu
        # aralıkta yayılıyor.
        return min(1.0, (score - self.threshold) / 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "_schema": "turkic-etymology-phonotactic-lm/v1",
            "language": self.language,
            "trained_on": self.trained_on,
            "trained_at": self.trained_at,
            "threshold": round(self.threshold, 6),
            "n_inherited": self.n_inherited,
            "n_borrowed": self.n_borrowed,
            "inherited": self.inherited.as_dict(),
            "borrowed": self.borrowed.as_dict(),
        }


def model_path(language: str) -> Path:
    return MODEL_DIR / f"phonotactic_{language}.json"


def fit(
    samples: list[tuple[str, bool]], *, language: str, trained_on: str
) -> PhonotacticClassifier:
    """İki modeli eğitir ve eşiği aynı veride F'ye göre seçer.

    ⚠️ ``samples`` **ayar yarısı** olmalıdır. Eşik de burada seçilir; rapor
    yarısı hiç görülmez.
    """
    classifier = PhonotacticClassifier(language=language, trained_on=trained_on)
    for word, borrowed in samples:
        form = to_comparison_form(word)
        if not form:
            continue
        if borrowed:
            classifier.borrowed.observe(form)
            classifier.n_borrowed += 1
        else:
            classifier.inherited.observe(form)
            classifier.n_inherited += 1
    if not classifier.is_trained:
        raise ValueError("her iki sınıftan da örnek gerekiyor")

    scored = [(classifier.score(word), borrowed) for word, borrowed in samples]
    classifier.threshold = _best_threshold(scored)
    classifier.trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    return classifier


def _best_threshold(scored: list[tuple[float, bool]]) -> float:
    """F skorunu en yükselten log-oran eşiği."""
    if not scored:
        return 0.0
    candidates = sorted({round(s, 3) for s, _ in scored})
    best_threshold, best_f = 0.0, -1.0
    for threshold in candidates:
        tp = sum(1 for s, y in scored if s >= threshold and y)
        fp = sum(1 for s, y in scored if s >= threshold and not y)
        fn = sum(1 for s, y in scored if s < threshold and y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f > best_f:
            best_threshold, best_f = threshold, f
    return best_threshold


def save(classifier: PhonotacticClassifier, path: Path | None = None) -> Path:
    target = path or model_path(classifier.language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(classifier.as_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return target


def load(language: str, path: Path | None = None) -> PhonotacticClassifier | None:
    """Dile özgü modeli yükler; yoksa ``None``.

    ⚠️ Başka bir dilin modeline **dönülmez**: fonotaktik dilden dile
    değişir ve zaten ölçtüğü şey odur.
    """
    source = path or model_path(language)
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("fonotaktik model okunamadı: %s", source)
        return None
    if data.get("language") != language:
        logger.warning(
            "fonotaktik model BAŞKA dil için eğitilmiş (%s != %s); kullanılmıyor",
            data.get("language"),
            language,
        )
        return None
    return PhonotacticClassifier(
        inherited=MarkovModel.from_dict(data.get("inherited") or {}),
        borrowed=MarkovModel.from_dict(data.get("borrowed") or {}),
        threshold=float(data.get("threshold", 0.0)),
        language=language,
        trained_on=str(data.get("trained_on", "")),
        n_inherited=int(data.get("n_inherited", 0)),
        n_borrowed=int(data.get("n_borrowed", 0)),
        trained_at=str(data.get("trained_at", "")),
    )
