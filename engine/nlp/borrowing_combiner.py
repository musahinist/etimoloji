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

⚠️ **Saha'da neden yalnız verici yakınlığı + zincir çalışıyor — ölçüldü,
olumsuz sonuç (2026-09-24).** Analiz yalnız AYAR yarısında (WOLD n=770,
Türkçe n=350; tekrarlı ÇD), karar ön-kayıtlı, rapor yarısına bir kez bakıldı.

Saha ayar yarısında sinyal kapsamı ve ayırt ediciliği::

    ses_kanunu_ihlali   20 ateşleme, 19'u mirasta   maddelerin %88'inde tanık yok;
                                                    lang != tr için beklenti TÜRKÇE
                                                    refleks, Saha biçimiyle kıyaslanıyor
    değişimsiz_yayılım  12 ateşleme, 10'u mirasta
    fonotaktik_ihlal    miras %15 / alıntı %15-18   AUC ~0,50 (Saha h- < *s- düzenli;
                                                    "söz başı *h- yok" kuralı yanlış ateşler)
    fonotaktik_model    AUC Rusça 0,73 · Moğolca 0,67
    verici_yakınlığı    AUC Rusça 0,89 · Moğolca 0,60 (Moğolca havuzu tek başına 0,53)
    zincir_kanıtı       miras %1 / alıntı %24       tek başına ek katkı veren sinyal

Hata sınıfları (kat dışı): yalnız verici yakınlığı Moğolca alıntıların 63/81'ini,
Rusçanın 29/119'unu kaçırıyor; Moğolca alıntılar ne fonotaktikle ne verici
havuzuyla ayrılıyor. Rusça harf/küme göstergeleri (f, v, ts, ş, söz başı ünsüz
kümesi) WOLD'un uyumlanmış biçimlerinde ~%1 — kullanılamaz. L2 ızgarası
(0-0,3) fark yaratmıyor. ``verici_yakınlığı`` rampası (SCA 0,35-0,60, şans
denetimsiz) karşı-kanıt: rampada alıntı oranı Saha 0,19 (taban 0,28),
Türkçe 0,25 (taban 0,60).

Ön-kayıtlı aday P1: verici yakınlığı ikili (güç = 1) + tekrarlı ÇD ile
(5 kat x 3) doğruluk ölçütlü sinyal altkümesi seçimi. Ölçüt: Saha doğruluk
GA'sı verici yakınlığına karşı sıfırı dışlasın VE Türkçe F düşmesin::

                              Saha doğ. farkı (verici'ye)   Saha F   Türkçe F
    üretim (bu dosya)         -0,0065 [-0,026, +0,013]      0,6554   0,8873
    P1 (seçti: verici+zincir) +0,0039 [-0,009, +0,017]      0,6517   0,8612
    yalnız ikilileştirme      (karar adayı değil)           0,6707   0,8765

İki ölçüt de TUTMADI (ayar yarısında beklenen +0,019 rapor yarısında
+0,004'e indi; Türkçede seçim 175 maddede LM'yi eledi). Üretim değişmedi.
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
    "fonotaktik_model",
    "ses_kanunu_ihlali",
    "fonotaktik_ihlal",
    "değişimsiz_yayılım",
)

#: L2 düzenlileştirme. n≈770'te beş katsayı için küçük bir değer yeter;
#: amaç tek bir sinyalin katsayısının patlamasını engellemek.
L2_PENALTY = 0.01

LEARNING_RATE = 0.5
ITERATIONS = 3000
#: Gradyan adımı bu değerin altına inince eğitim durur (deterministik).
TOLERANCE = 1e-7

#: İç çapraz doğrulamada denenen L1 cezaları. 0 dahil: düzenlileştirmenin
#: işe yaramadığı da bir sonuçtur ve seçilebilmelidir.
L1_GRID: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1)

#: İç kat sayısı. Ayar kümesi küçük (WOLD'da ~190, Türkçede ~87 madde);
#: daha çok kat, katları anlamsız biçimde küçültür.
INNER_FOLDS = 5

#: "Daha basit model" toleransı: iç ÇD skoru en iyiden bu kadar düşük olan
#: ama DAHA AZ sinyal kullanan yapılandırma tercih edilir. Basitlik bir
#: kazançtır; eşit iş gören sinyal ek bir kırılganlık kaynağıdır.
SIMPLICITY_TOLERANCE = 0.01


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
    #: L1 cezası (iç ÇD ile seçildiyse o değer).
    l1: float = 0.0
    #: Modelin kullanmasına izin verilen sinyaller; ötekilerin katsayısı 0.
    active: tuple[str, ...] = SIGNAL_ORDER
    #: İç çapraz doğrulama kaydı — seçimin nasıl yapıldığı gizlenemez.
    selection: dict[str, Any] = field(default_factory=dict)

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
            "l1": self.l1,
            "active_signals": list(self.active),
            "selection": self.selection,
            "trained_on": self.trained_on,
            "n": self.n,
            "trained_at": self.trained_at,
        }

    def explain(self) -> str:
        """Katsayıları okunur biçimde döndürür — kara kutu kabul edilmiyor."""
        if not self.is_trained:
            return "eğitilmemiş (el ağırlıkları kullanılıyor)"
        rows = sorted(
            ((k, v) for k, v in self.weights.items() if v), key=lambda kv: -abs(kv[1])
        )
        body = " · ".join(f"{name} {value:+.3f}" for name, value in rows)
        zeroed = [name for name, value in self.weights.items() if not value]
        tail = f" · sıfır: {', '.join(zeroed)}" if zeroed else ""
        return f"sabit {self.bias:+.3f} · {body}{tail}"


def fit(
    samples: list[tuple[dict[str, float], bool]],
    *,
    trained_on: str,
    objective: str = "fscore",
    l1: float = 0.0,
    l2: float = L2_PENALTY,
    active: tuple[str, ...] | None = None,
    iterations: int = ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    threshold: float | None = None,
) -> BorrowingCombiner:
    """Lojistik regresyonu tam-toplu (proksimal) gradyan inişiyle eğitir.

    :param l1: L1 cezası. Her adımdan sonra yumuşak eşikleme uygulanır
        (ISTA); katkısız sinyalin katsayısı **tam sıfıra** iner, yani L1
        sinyal seçimi de yapar.
    :param active: kullanılacak sinyaller; ötekilerin katsayısı 0'da
        sabit kalır (ablasyon / geriye doğru eleme).
    :param threshold: verilirse eşik eğitim verisinde **aranmaz** — iç
        çapraz doğrulamada, modelin görmediği katlarda seçilmiş eşik budur.

    Deterministiktir: rastgele başlangıç yok, karıştırma yok. Aynı veri aynı
    katsayıları verir — ölçümün tekrarlanabilirliği bunu gerektiriyor.
    """
    if not samples:
        raise ValueError("eğitim örneği yok")

    allowed = tuple(active) if active is not None else SIGNAL_ORDER
    unknown = set(allowed) - set(SIGNAL_ORDER)
    if unknown:
        raise ValueError(f"bilinmeyen sinyal: {sorted(unknown)}")
    mask = [name in allowed for name in SIGNAL_ORDER]

    model = BorrowingCombiner(trained_on=trained_on, n=len(samples))
    rows = [(model.features(signals), 1.0 if label else 0.0) for signals, label in samples]
    weights, bias = _optimise(
        rows, mask, l1=l1, l2=l2, iterations=iterations, learning_rate=learning_rate
    )

    model.weights = dict(zip(SIGNAL_ORDER, weights, strict=True))
    model.bias = bias
    model.l1 = l1
    model.active = tuple(name for name in SIGNAL_ORDER if name in allowed)
    model.trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    model.threshold = (
        threshold
        if threshold is not None
        else _best_threshold(model, samples, objective=objective)
    )
    model.objective = objective
    return model


def _optimise(
    rows: list[tuple[list[float], float]],
    mask: list[bool],
    *,
    l1: float,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], float]:
    """Proksimal gradyan inişi: L2 gradyanda, L1 yumuşak eşiklemede."""
    size = len(rows)
    dims = len(mask)
    weights = [0.0] * dims
    bias = 0.0
    shrink = learning_rate * l1
    for _ in range(iterations):
        gradient = [0.0] * dims
        bias_gradient = 0.0
        for features, target in rows:
            prediction = _sigmoid(bias + sum(w * x for w, x in zip(weights, features, strict=True)))
            error = prediction - target
            bias_gradient += error
            for index, value in enumerate(features):
                gradient[index] += error * value
        step = learning_rate * bias_gradient / size
        bias -= step
        largest = abs(step)
        for index in range(dims):
            if not mask[index]:
                continue
            old = weights[index]
            value = old - learning_rate * (gradient[index] / size + l2 * old)
            if shrink:
                value = math.copysign(max(abs(value) - shrink, 0.0), value)
            weights[index] = value
            largest = max(largest, abs(value - old))
        if largest < TOLERANCE:
            break
    return weights, bias


def _stratified_folds(samples: list[tuple[dict[str, float], bool]], folds: int) -> list[int]:
    """Deterministik, **tabakalı** kat ataması.

    ⚠️ Düz ``sıra mod kat`` ataması, etiketler sırayla dönüşümlüyse (ör. her
    üçüncü madde alıntı) bir katı yalnız alıntılarla doldurur; o katı
    dışarıda bırakan model hiç alıntı görmez ve iç ÇD skoru 0 çıkar.
    Burada her sınıf kendi içinde sırayla katlara dağıtılır.
    """
    counters = {True: 0, False: 0}
    out: list[int] = []
    for _, label in samples:
        key = bool(label)
        out.append(counters[key] % folds)
        counters[key] += 1
    return out


def _score(scored: list[tuple[float, bool]], threshold: float, objective: str) -> float:
    tp = sum(1 for p, y in scored if p >= threshold and y)
    fp = sum(1 for p, y in scored if p >= threshold and not y)
    fn = sum(1 for p, y in scored if p < threshold and y)
    tn = len(scored) - tp - fp - fn
    if objective == "accuracy":
        return (tp + tn) / len(scored) if scored else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _threshold_on(scored: list[tuple[float, bool]], objective: str) -> tuple[float, float]:
    best_threshold, best_score = 0.5, -1.0
    for step in range(1, 100):
        threshold = step / 100
        value = _score(scored, threshold, objective)
        if value > best_score:
            best_threshold, best_score = threshold, value
    return best_threshold, best_score


def _out_of_fold(
    samples: list[tuple[dict[str, float], bool]],
    *,
    l1: float,
    active: tuple[str, ...],
    folds: int,
    iterations: int,
) -> list[tuple[float, bool]]:
    """Her maddenin olasılığı, o maddeyi GÖRMEYEN modelden."""
    scored: list[tuple[float, bool]] = []
    assignment = _stratified_folds(samples, folds)
    for fold in range(folds):
        train = [s for s, f in zip(samples, assignment, strict=True) if f != fold]
        held = [s for s, f in zip(samples, assignment, strict=True) if f == fold]
        if not train or not held:
            continue
        model = fit(
            train, trained_on="iç-kat", l1=l1, active=active,
            iterations=iterations, threshold=0.5,
        )
        scored.extend((model.probability(signals), label) for signals, label in held)
    return scored


def fit_nested(
    samples: list[tuple[dict[str, float], bool]],
    *,
    trained_on: str,
    objective: str = "fscore",
    l1_grid: tuple[float, ...] = L1_GRID,
    folds: int = INNER_FOLDS,
    candidates: tuple[str, ...] | None = None,
    eliminate: bool = True,
    tolerance: float = SIMPLICITY_TOLERANCE,
    iterations: int = ITERATIONS,
) -> BorrowingCombiner:
    """L1 cezasını, sinyal kümesini ve eşiği **iç çapraz doğrulamayla** seçer.

    ⚠️ Yalnız verilen ``samples`` (AYAR yarısı) kullanılır. Rapor yarısı bu
    fonksiyona hiç girmez; seçim rapor yarısına bakarak yapılsaydı, ölçülen
    fark seçimin kendisini ölçerdi.

    Yöntem:

    1. Her (sinyal kümesi, L1) için ayar kümesi ``folds`` kata bölünür; her
       maddenin olasılığı o maddeyi görmemiş modelden alınır (kat dışı).
    2. Eşik bu kat dışı olasılıklarda seçilir — eğitim verisinde seçilen eşik
       iyimserdir, çünkü model o maddeleri ezberlemiştir.
    3. Geriye doğru eleme: bir sinyali çıkarmak kat dışı skoru
       ``tolerance``'tan fazla düşürmüyorsa sinyal çıkarılır. **Daha az
       sinyalle aynı skor bir kazançtır.**
    4. Seçilen yapılandırma tüm ayar kümesinde yeniden eğitilir; eşik kat
       dışı eşiktir.

    ⚠️ **Ölçüldü, üretime alınmadı.** Rapor yarısında (2026-09-24)::

                          tek-geçişli L2       iç içe ÇD
        WOLD/Sakha   F    0,6554 (6 sinyal)    0,6380 (3 sinyal, L1=0)
        Türkçe altın F    0,8873 (5 sinyal)    0,8455 (2 sinyal, L1=0,003)

    Ayar kümeleri küçük (WOLD ~190, Türkçe ~87 madde); iç katlarda eleme
    ve eşik gürültüye göre seçiliyor. L1 WOLD'da hiç seçilmedi (ceza 0).
    """
    if not samples:
        raise ValueError("eğitim örneği yok")
    pool = tuple(candidates) if candidates is not None else SIGNAL_ORDER
    # Hiç ateşlenmeyen sinyal (ör. Türkçede kapalı zincir) aday değildir.
    pool = tuple(
        name for name in pool
        if any(float(signals.get(name, 0.0)) for signals, _ in samples)
    )

    trail: list[dict[str, Any]] = []
    cache: dict[tuple[tuple[str, ...], float], tuple[float, float]] = {}

    def evaluate(active: tuple[str, ...]) -> tuple[float, float, float]:
        best = (-1.0, 0.0, 0.5)
        for l1 in l1_grid:
            key = (active, l1)
            if key not in cache:
                scored = _out_of_fold(
                    samples, l1=l1, active=active, folds=folds, iterations=iterations
                )
                cache[key] = _threshold_on(scored, objective)[::-1]
            score, threshold = cache[key]
            # Eşitlikte büyük ceza kazanır: daha sade model.
            if score >= best[0]:
                best = (score, l1, threshold)
        return best

    current = pool
    score, l1, threshold = evaluate(current)
    trail.append({"signals": list(current), "cv_score": round(score, 4), "l1": l1})
    best_score = score
    while eliminate and len(current) > 1:
        options = []
        for name in current:
            reduced = tuple(n for n in current if n != name)
            option = evaluate(reduced)
            options.append((option[0], name, reduced, option))
        options.sort(key=lambda item: (-item[0], SIGNAL_ORDER.index(item[1])))
        top_score, dropped, reduced, option = options[0]
        if top_score < best_score - tolerance:
            break
        current = reduced
        score, l1, threshold = option
        best_score = max(best_score, score)
        trail.append(
            {"dropped": dropped, "signals": list(current),
             "cv_score": round(score, 4), "l1": l1}
        )

    model = fit(
        samples, trained_on=trained_on, objective=objective, l1=l1,
        active=current, iterations=iterations, threshold=threshold,
    )
    model.selection = {
        "method": f"iç {folds} katlı ÇD (yalnız ayar yarısı)",
        "objective": objective,
        "l1_grid": list(l1_grid),
        "tolerance": tolerance,
        "cv_score": round(score, 4),
        "trail": trail,
    }
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
    return _threshold_on(scored, objective)[0]


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
        l1=float(data.get("l1", 0.0)),
        active=tuple(data.get("active_signals") or SIGNAL_ORDER),
        selection=dict(data.get("selection") or {}),
    )
