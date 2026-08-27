"""
Alıntı sinyallerini birleştiren **eğitilmiş** katman.

⚠️ Bu modül bir ölçümün doğrudan sonucudur. El ile konmuş ağırlıklı toplam
(``SIGNAL_WEIGHTS`` + sabit eşik) WOLD/Sakha'da en güçlü sinyalin kararını
**bozuyordu**::

    madde başına doğruluk (n=769)
    yalnız verici yakınlığı   0,7334
    beş sinyalli motor        0,7035
    fark -0,0299, %95 GA [-0,0494, -0,0104], p=0,004  -> ANLAMLI

Yani dört sinyal ekledikçe sistem, tek sinyalden **anlamlı biçimde
kötüleşiyordu**. Sebebi el ağırlıklarında görünüyor: ``ses_kanunu_ihlali``
ve ``değişimsiz_yayılım`` toplamın %20'sini alıyordu ama ablasyonda
katkıları sırasıyla +0,0002 ve **-0,0058**'di.

Ayrıca aritmetik bir kusur vardı: zincir sinyali yokken
``0,20 + 0,10 = 0,30 < 0,45`` — o iki sinyal tek başlarına **hiçbir kararı
değiştiremiyordu**.

Çözüm: ağırlıkları veriden öğren. Lojistik regresyon seçildi çünkü
katsayıları **okunabilir** — hangi sinyalin ne kadar ağırlık aldığı
doğrudan raporlanabilir; bu proje kara kutu kabul etmiyor.

⚠️ **Eğitim ve ölçüm ayrı yarılarda.** Model AYAR yarısında eğitilir, sayı
RAPOR yarısında verilir. Aynı veride hem eğitip hem ölçmek, ölçümü yok
sayar.

⚠️ Model yoksa el ağırlıklarına dönülür ama bu **ilan edilir**
(``trained=False``), sessizce yapılmaz.

sklearn kullanılmaz (bağımlılık yok); optimizasyon bu dosyada, deterministik.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "borrowing_combiner.json"

#: Sinyal sırası **sabittir**: model dosyasındaki katsayılar bu sıraya göre
#: yazılır. Sıra değişirse eski model yanlış sinyale ağırlık verir; bu yüzden
#: model dosyası kendi sırasını da saklar ve yükleme sırasında doğrulanır.
SIGNAL_ORDER: tuple[str, ...] = (
    "zincir_kanıtı",
    "verici_yakınlığı",
    "ses_kanunu_ihlali",
    "fonotaktik_ihlal",
    "değişimsiz_yayılım",
)

#: L2 düzenlileştirme. n≈770'te beş katsayı için küçük bir değer yeter;
#: amaç tek bir sinyalin katsayısının patlamasını engellemek.
L2_PENALTY = 0.01

LEARNING_RATE = 0.5
ITERATIONS = 3000


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass
class BorrowingCombiner:
    """Sinyal güçlerinden alıntı olasılığı üreten lojistik model."""

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    trained_on: str = ""
    n: int = 0
    trained_at: str = ""
    #: Eğitim yarısında seçilen karar eşiği.
    threshold: float = 0.5
    #: Eşiğin hangi ölçüye göre seçildiği (``fscore`` | ``accuracy``).
    #: Sonucu belirler ve gizlenemez — bkz. :func:`_best_threshold`.
    objective: str = "fscore"

    @property
    def is_trained(self) -> bool:
        return bool(self.weights)

    def features(self, signals: dict[str, float]) -> list[float]:
        return [float(signals.get(name, 0.0)) for name in SIGNAL_ORDER]

    def probability(self, signals: dict[str, float]) -> float:
        if not self.is_trained:
            return 0.0
        z = self.bias + sum(
            self.weights.get(name, 0.0) * float(signals.get(name, 0.0))
            for name in SIGNAL_ORDER
        )
        return _sigmoid(z)

    def predict(self, signals: dict[str, float]) -> bool:
        return self.probability(signals) >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "_schema": "turkic-etymology-borrowing-combiner/v1",
            "signal_order": list(SIGNAL_ORDER),
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "bias": round(self.bias, 6),
            "threshold": round(self.threshold, 4),
            "objective": self.objective,
            "trained_on": self.trained_on,
            "n": self.n,
            "trained_at": self.trained_at,
        }

    def explain(self) -> str:
        """Katsayıları okunur biçimde döndürür — kara kutu kabul edilmiyor."""
        if not self.is_trained:
            return "eğitilmemiş (el ağırlıkları kullanılıyor)"
        rows = sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))
        body = " · ".join(f"{name} {value:+.3f}" for name, value in rows)
        return f"sabit {self.bias:+.3f} · {body}"


def fit(
    samples: list[tuple[dict[str, float], bool]],
    *,
    trained_on: str,
    objective: str = "fscore",
    l2: float = L2_PENALTY,
    iterations: int = ITERATIONS,
    learning_rate: float = LEARNING_RATE,
) -> BorrowingCombiner:
    """Lojistik regresyonu tam-toplu gradyan inişiyle eğitir.

    Deterministiktir: rastgele başlangıç yok, karıştırma yok. Aynı veri aynı
    katsayıları verir — ölçümün tekrarlanabilirliği bunu gerektiriyor.
    """
    if not samples:
        raise ValueError("eğitim örneği yok")

    model = BorrowingCombiner(trained_on=trained_on, n=len(samples))
    weights = [0.0] * len(SIGNAL_ORDER)
    bias = 0.0
    rows = [(model.features(signals), 1.0 if label else 0.0) for signals, label in samples]
    size = len(rows)

    for _ in range(iterations):
        gradient = [0.0] * len(SIGNAL_ORDER)
        bias_gradient = 0.0
        for features, target in rows:
            prediction = _sigmoid(bias + sum(w * x for w, x in zip(weights, features, strict=True)))
            error = prediction - target
            bias_gradient += error
            for index, value in enumerate(features):
                gradient[index] += error * value
        bias -= learning_rate * bias_gradient / size
        for index in range(len(weights)):
            weights[index] -= learning_rate * (gradient[index] / size + l2 * weights[index])

    model.weights = dict(zip(SIGNAL_ORDER, weights, strict=True))
    model.bias = bias
    model.trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    model.threshold = _best_threshold(model, samples, objective=objective)
    model.objective = objective
    return model


def _best_threshold(
    model: BorrowingCombiner,
    samples: list[tuple[dict[str, float], bool]],
    *,
    objective: str = "fscore",
) -> float:
    """Eğitim yarısında hedef ölçüyü en yükselten eşiği seçer.

    ⚠️ 0,5 varsayılanı sınıf dengesizliğinde yanlıştır: WOLD/Sakha'da
    alıntılar %30, yani 0,5 eşiği duyarlılığı bastırır.

    ⚠️ **Hedef ölçü seçimi sonucu belirler ve gizlenemez.** Ölçüldü
    (WOLD/Sakha rapor yarısı, aynı model, yalnız eşik farklı)::

        F hedefli         eşik 0,33   F 0,5982   doğruluk 0,7100
        doğruluk hedefli  eşik 0,43   F 0,2714   doğruluk 0,7347

    İkisi aynı anda alınamaz; hangi eşiğin seçildiği modelle birlikte
    saklanır.
    """
    scored = [(model.probability(signals), label) for signals, label in samples]
    best_threshold, best_score = 0.5, -1.0
    for step in range(1, 100):
        threshold = step / 100
        tp = sum(1 for p, y in scored if p >= threshold and y)
        fp = sum(1 for p, y in scored if p >= threshold and not y)
        fn = sum(1 for p, y in scored if p < threshold and y)
        tn = len(scored) - tp - fp - fn
        if objective == "accuracy":
            value = (tp + tn) / len(scored) if scored else 0.0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if value > best_score:
            best_threshold, best_score = threshold, value
    return best_threshold


def save(model: BorrowingCombiner, path: Path | None = None) -> Path:
    target = Path(path) if path else MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load(path: Path | None = None) -> BorrowingCombiner | None:
    """Kaydedilmiş modeli yükler; yoksa veya bozuksa ``None``.

    ⚠️ Sinyal sırası doğrulanır. Sıra değişmişse model **yüklenmez**: eski
    katsayıları yeni sıraya uygulamak, her sinyale başkasının ağırlığını
    vermek olurdu ve hiçbir hata mesajı üretmezdi.
    """
    source = Path(path) if path else MODEL_PATH
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("alıntı birleştirici modeli okunamadı: %s", source)
        return None
    if tuple(data.get("signal_order") or ()) != SIGNAL_ORDER:
        logger.warning(
            "alıntı birleştirici modeli ESKİ sinyal sırasıyla eğitilmiş; kullanılmıyor"
        )
        return None
    return BorrowingCombiner(
        weights={k: float(v) for k, v in (data.get("weights") or {}).items()},
        bias=float(data.get("bias", 0.0)),
        trained_on=str(data.get("trained_on", "")),
        n=int(data.get("n", 0)),
        trained_at=str(data.get("trained_at", "")),
        threshold=float(data.get("threshold", 0.5)),
        objective=str(data.get("objective", "fscore")),
    )
