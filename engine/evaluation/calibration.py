"""
Güven kalibrasyonu — "%89 eminim" derken gerçekten %89 haklı mı?

Ölçüldü: motor ortalama **0,894** güven verirken gerçek doğruluğu **0,454**
idi — sistematik **+0,44 aşırı güven**. Kalibre edilmemiş bir skor
kullanıcıyı yanıltır ve seçici tahmin (çekimser kalma) imkânsızlaşır.

Raporlanan büyüklükler:

``ECE``
    Expected Calibration Error — güven ile gerçek doğruluk arasındaki
    ortalama sapma. Kutulara ayırıp her kutuda ``|ortalama güven − doğruluk|``
    hesaplanır, kutu büyüklüğüyle ağırlıklanır.
``Brier``
    Ortalama kare hata. ECE'den farkı: hem kalibrasyonu hem ayırt ediciliği
    birlikte cezalandırır.
``AUC``
    Skorun doğru/yanlış ayırma gücü. Kalibrasyon kötü olsa da AUC yüksekse
    skor **sıralama** olarak işe yarar ve kalibre edilerek kurtarılabilir.
``risk-coverage``
    Eşik yükseldikçe kaç maddede cevap veriliyor (coverage) ve o maddelerde
    hata oranı ne (risk). Seçici tahminin temel eğrisi.

⚠️ **İç içe çapraz doğrulama zorunludur.** Ağırlık öğrenme, kalibrasyon ve
ECE ölçümü üç ayrı katmandır; üçünü aynı veride yapmak ölçümü geçersiz kılar.
:func:`cross_validated_calibration` dış katmanı kurar.

⚠️ n≈100 civarında ECE kutu tahminleri **oynaktır**; bu yüzden güven aralığı
olmadan ECE raporlanmaz (:func:`bootstrap_ci`).

İzotonik regresyon bu büyüklükte aşırı uyar; bu yüzden Platt ölçekleme ile
karşılaştırılıp veriye göre seçilir.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Kalibrasyon ve bootstrap için sabit tohum — yeniden üretilebilirlik şartı.
CALIBRATION_SEED = 20260827

#: Varsayılan kutu sayısı. Az örnekte az kutu: 10 kutu × 100 örnek = kutu
#: başına 10 madde, zaten sınırda.
DEFAULT_BINS = 10


@dataclass
class ReliabilityBin:
    """Güvenilirlik diyagramının tek bir kutusu."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)


@dataclass
class CalibrationReport:
    """Bir skor kümesinin kalibrasyon künyesi."""

    n: int
    ece: float
    brier: float
    auc: float
    mean_confidence: float
    accuracy: float
    bins: list[ReliabilityBin] = field(default_factory=list)
    ece_ci: tuple[float, float] | None = None

    @property
    def overconfidence(self) -> float:
        """Pozitifse motor kendini olduğundan iyi sanıyor."""
        return self.mean_confidence - self.accuracy

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "ECE": round(self.ece, 4),
            "ECE_CI": [round(v, 4) for v in self.ece_ci] if self.ece_ci else None,
            "Brier": round(self.brier, 4),
            "AUC": round(self.auc, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "overconfidence": round(self.overconfidence, 4),
        }


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = DEFAULT_BINS,
) -> tuple[float, list[ReliabilityBin]]:
    """ECE ve güvenilirlik diyagramı kutuları."""
    if not confidences:
        return 0.0, []
    total = len(confidences)
    edges = [i / bins for i in range(bins + 1)]
    report: list[ReliabilityBin] = []
    ece = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        members = [
            (c, y)
            for c, y in zip(confidences, correct, strict=True)
            if (low <= c < high) or (i == bins - 1 and c == high)
        ]
        if not members:
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        acc = sum(1 for _, y in members if y) / len(members)
        report.append(ReliabilityBin(low, high, len(members), mean_conf, acc))
        ece += (len(members) / total) * abs(mean_conf - acc)
    return ece, report


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Ortalama kare hata — kalibrasyon ve ayırt ediciliği birlikte ölçer."""
    if not confidences:
        return 0.0
    return sum(
        (c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correct, strict=True)
    ) / len(confidences)


def auc_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """ROC eğrisi altındaki alan — Mann-Whitney U ile, bağlar yarım sayılır.

    0,5 = rastgele. Kalibrasyon kötü olsa bile AUC yüksekse skor **sıralama**
    olarak değerlidir ve kalibre edilerek kurtarılabilir.
    """
    positives = [c for c, y in zip(confidences, correct, strict=True) if y]
    negatives = [c for c, y in zip(confidences, correct, strict=True) if not y]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def bootstrap_ci(
    values: Sequence[float],
    correct: Sequence[bool],
    statistic: Callable[[Sequence[float], Sequence[bool]], float],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = CALIBRATION_SEED,
) -> tuple[float, float]:
    """Madde-bazlı bootstrap güven aralığı.

    Aralıksız tek sayı raporlanmaz: n≈100'de ECE kutu tahminleri çok oynaktır
    ve iki sistem arasındaki fark kolayca gürültü olabilir.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    indices = range(size)
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rng.choice(indices) for _ in indices]
        samples.append(statistic([values[i] for i in picked], [correct[i] for i in picked]))
    samples.sort()
    low = samples[int(alpha / 2 * iterations)]
    high = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (low, high)


def evaluate(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = DEFAULT_BINS,
    with_ci: bool = True,
) -> CalibrationReport:
    """Bir skor kümesinin tam kalibrasyon künyesi."""
    ece, reliability = expected_calibration_error(confidences, correct, bins=bins)
    report = CalibrationReport(
        n=len(confidences),
        ece=ece,
        brier=brier_score(confidences, correct),
        auc=auc_score(confidences, correct),
        mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        accuracy=sum(1 for y in correct if y) / len(correct) if correct else 0.0,
        bins=reliability,
    )
    if with_ci and len(confidences) >= 20:
        report.ece_ci = bootstrap_ci(
            confidences,
            correct,
            lambda c, y: expected_calibration_error(c, y, bins=bins)[0],
        )
    return report


# --- Kalibratörler ---------------------------------------------------------


class IsotonicCalibrator:
    """İzotonik regresyon (PAVA) — monoton, parametresiz.

    ⚠️ Küçük örneklemde **aşırı uyar**: eğitim verisindeki her dalgalanmayı
    ezberler. Bu yüzden :class:`PlattCalibrator` ile karşılaştırılıp veriye
    göre seçilir.
    """

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> IsotonicCalibrator:
        pairs = sorted(zip(confidences, correct, strict=True))
        if not pairs:
            return self
        # Pool Adjacent Violators
        blocks: list[list[float]] = []  # [toplam, adet, x_ortalama]
        for x, y in pairs:
            blocks.append([1.0 if y else 0.0, 1.0, x])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                last = blocks.pop()
                blocks[-1][0] += last[0]
                blocks[-1][1] += last[1]
                blocks[-1][2] = (blocks[-1][2] + last[2]) / 2
        self._x = [b[2] for b in blocks]
        self._y = [b[0] / b[1] for b in blocks]
        return self

    def predict(self, confidence: float) -> float:
        if not self._x:
            return confidence
        if confidence <= self._x[0]:
            return self._y[0]
        if confidence >= self._x[-1]:
            return self._y[-1]
        for i in range(1, len(self._x)):
            if confidence <= self._x[i]:
                span = self._x[i] - self._x[i - 1]
                if span <= 0:
                    return self._y[i]
                ratio = (confidence - self._x[i - 1]) / span
                return self._y[i - 1] + ratio * (self._y[i] - self._y[i - 1])
        return self._y[-1]


class PlattCalibrator:
    """Platt ölçekleme — tek boyutlu lojistik regresyon.

    İki parametresi olduğu için küçük örneklemde izotonikten daha kararlıdır.
    """

    def __init__(self, iterations: int = 200, learning_rate: float = 0.5) -> None:
        self.a = 1.0
        self.b = 0.0
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, confidences: Sequence[float], correct: Sequence[bool]) -> PlattCalibrator:
        if not confidences:
            return self
        targets = [1.0 if y else 0.0 for y in correct]
        n = len(confidences)
        for _ in range(self.iterations):
            grad_a = grad_b = 0.0
            for x, t in zip(confidences, targets, strict=True):
                p = self.predict(x)
                error = p - t
                grad_a += error * x
                grad_b += error
            self.a -= self.learning_rate * grad_a / n
            self.b -= self.learning_rate * grad_b / n
        return self

    def predict(self, confidence: float) -> float:
        z = self.a * confidence + self.b
        # Taşma korumalı sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


CALIBRATORS: dict[str, Callable[[], IsotonicCalibrator | PlattCalibrator]] = {
    "isotonic": IsotonicCalibrator,
    "platt": PlattCalibrator,
}


def cross_validated_calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    folds: int = 5,
    method: str = "isotonic",
    seed: int = CALIBRATION_SEED,
) -> list[float]:
    """Çapraz doğrulamalı kalibre skorlar.

    Her madde, **kendisinin katılmadığı** katmanlarda öğrenilen kalibratörle
    dönüştürülür. Aynı veride hem kalibre edip hem ölçmek, ECE'yi yapay
    olarak sıfıra yaklaştırırdı.
    """
    n = len(confidences)
    if n < folds * 2:
        logger.warning("Kalibrasyon için örneklem çok küçük (n=%d); ham skor döndürülüyor", n)
        return list(confidences)

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    assignment = {index: position % folds for position, index in enumerate(order)}

    out = [0.0] * n
    for fold in range(folds):
        train_idx = [i for i in range(n) if assignment[i] != fold]
        test_idx = [i for i in range(n) if assignment[i] == fold]
        calibrator = CALIBRATORS[method]()
        calibrator.fit([confidences[i] for i in train_idx], [correct[i] for i in train_idx])
        for i in test_idx:
            out[i] = calibrator.predict(confidences[i])
    return out


def risk_coverage(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    steps: int = 20,
) -> list[dict[str, float]]:
    """Risk-coverage eğrisi — seçici tahminin temeli.

    Eşik yükseldikçe motor daha az maddede cevap verir (coverage düşer) ama
    verdiği cevapların hata oranı da düşmelidir (risk düşer). Düşmüyorsa
    skorun sıralama değeri yoktur.
    """
    if not confidences:
        return []
    pairs = sorted(zip(confidences, correct, strict=True), reverse=True)
    total = len(pairs)
    curve: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        take = max(1, round(total * step / steps))
        subset = pairs[:take]
        errors = sum(1 for _, y in subset if not y)
        curve.append(
            {
                "coverage": round(take / total, 4),
                "risk": round(errors / take, 4),
                "threshold": round(subset[-1][0], 4),
            }
        )
    return curve


def main() -> int:
    import argparse
    import json

    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation.gold import GoldStandard
    from engine.evaluation.harness import comparative_reconstructor, run
    from engine.evaluation.report import EVAL_DIR

    ap = argparse.ArgumentParser(description="Güven kalibrasyonu ölçümü")
    ap.add_argument("--split", default="dev", choices=("train", "dev", "all"))
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    args = ap.parse_args()

    gold = GoldStandard.build(args.dataset)
    items = gold.items if args.split == "all" else gold.split(args.split)
    mapping = build_mapping(CldfWordlist.load(args.dataset))
    outcome = run(comparative_reconstructor(), items, mapping=mapping, split=args.split)

    raw = outcome.confidences
    correct = outcome.correctness
    if len(raw) < 20:
        print(f"Kalibrasyon için yeterli veri yok (n={len(raw)})")
        return 1

    payload: dict[str, object] = {"split": args.split, "n": len(raw)}
    print(f"\n=== kalibrasyon · {args.dataset}/{args.split} · n={len(raw)} ===\n")
    print(f"{'skor':22} {'ECE':>8} {'GA':>18} {'Brier':>8} {'AUC':>7} {'ort.güven':>10} {'doğruluk':>9}")

    variants: dict[str, Sequence[float]] = {"ham": raw}
    for method in CALIBRATORS:
        variants[f"kalibre ({method})"] = cross_validated_calibration(
            raw, correct, method=method
        )

    for name, scores in variants.items():
        report = evaluate(scores, correct, bins=args.bins)
        data = report.as_dict()
        payload[name] = data
        ci = data["ECE_CI"]
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        print(
            f"{name:22} {data['ECE']:>8.4f} {ci_text:>18} {data['Brier']:>8.4f} "
            f"{data['AUC']:>7.4f} {data['mean_confidence']:>10.4f} {data['accuracy']:>9.4f}"
        )

    print("\nrisk-coverage (ham skor):")
    for point in risk_coverage(raw, correct, steps=5):
        print(
            f"   kapsam {point['coverage']:.2f}  hata {point['risk']:.3f}  "
            f"eşik {point['threshold']:.3f}"
        )

    payload["risk_coverage"] = risk_coverage(raw, correct)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "calibration.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
