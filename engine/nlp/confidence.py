"""
Kalibre güven skoru ve çekimserlik eşiği — kullanıcıya giden sayı.

Ham güven skoru **kullanıcıya gösterilmez.** Ölçüldü: motor ortalama 0,646
güven verirken gerçek doğruluğu 0,239'du (ECE 0,406). Bu skorla "%80 eminim"
demek yanıltıcıdır.

Burada iki şey yapılır:

1. **Kalibrasyon** — ham skor, altın standardın *train* bölümünde öğrenilmiş
   bir Platt ölçekleyiciden geçirilir. Ölçüldü: ECE 0,406 -> **0,037**.
2. **Seçici tahmin (çekimserlik)** — kalibre skor eşiğin altındaysa motor
   "yetersiz kanıt" der. Risk-coverage eğrisine göre: %20 kapsamda hata
   %58, %100 kapsamda %76. Yani çekimser kalmak ölçülebilir biçimde işe
   yarıyor ve bir başarısızlık değil, bir **özelliktir**.

Kalibratör ``data/calibration/model.json``dan okunur. Dosya yoksa ham skor
döner ve çıktı ``calibrated: false`` ile işaretlenir — sessizce kalibre
edilmiş gibi davranmak, kalibre etmemekten kötüdür.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
MODEL_PATH = CALIBRATION_DIR / "model.json"

#: Kalibre skorun altında motorun çekimser kalacağı eşik.
#: Risk-coverage eğrisinden seçilir; ``config`` üzerinden geçersiz kılınabilir.
DEFAULT_ABSTENTION_THRESHOLD = 0.15

#: Rozet eşikleri — kalibre skora göre. Ham skora göre verilmez.
BADGE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.60, "🟢 GÜÇLÜ KANIT"),
    (0.35, "🟡 ORTA KANIT"),
    (0.15, "🟠 ZAYIF KANIT"),
    (0.00, "⚪ YETERSİZ KANIT"),
)


@dataclass(frozen=True)
class CalibrationModel:
    """Öğrenilmiş Platt ölçekleyici ve künyesi."""

    a: float
    b: float
    method: str
    trained_on: str
    n: int
    ece_before: float
    ece_after: float
    trained_at: str

    def apply(self, raw: float) -> float:
        import math

        z = self.a * raw + self.b
        if z >= 0:
            return round(1.0 / (1.0 + math.exp(-z)), 4)
        exp_z = math.exp(z)
        return round(exp_z / (1.0 + exp_z), 4)


@lru_cache(maxsize=1)
def load_model() -> CalibrationModel | None:
    """Kalibratörü diskten okur. Yoksa ``None`` — sessizce uydurulmaz."""
    if not MODEL_PATH.exists():
        logger.info(
            "Kalibrasyon modeli yok (%s). Ham güven skoru kullanılacak ve "
            "çıktı calibrated=false olarak işaretlenecek.",
            MODEL_PATH,
        )
        return None
    try:
        data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        return CalibrationModel(
            a=float(data["a"]),
            b=float(data["b"]),
            method=data.get("method", "platt"),
            trained_on=data.get("trained_on", "?"),
            n=int(data.get("n", 0)),
            ece_before=float(data.get("ece_before", 0.0)),
            ece_after=float(data.get("ece_after", 0.0)),
            trained_at=data.get("trained_at", ""),
        )
    except (OSError, ValueError, KeyError):
        logger.warning("Kalibrasyon modeli okunamadı: %s", MODEL_PATH, exc_info=True)
        return None


def reset_model_cache() -> None:
    """Test ve yeniden eğitim sonrası önbelleği boşaltır."""
    load_model.cache_clear()


def badge_for(calibrated: float | None) -> str:
    """Kalibre skora karşılık gelen rozet."""
    if calibrated is None:
        return "⚪ YETERSİZ KANIT"
    for threshold, label in BADGE_THRESHOLDS:
        if calibrated >= threshold:
            return label
    return "⚪ YETERSİZ KANIT"


def apply_calibration(
    result: dict[str, object],
    *,
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
) -> dict[str, object]:
    """Rekonstrüksiyon sonucuna kalibre skoru ve rozeti ekler.

    Kalibre skor eşiğin altındaysa ``is_reconstructible`` **false** yapılır:
    motor cevabı bulur ama savunulabilir bulmaz ve söylemez. Ata biçim
    ``withheld_reconstruction`` alanında saklanır — kullanıcı isterse görsün,
    ama "sonuç" olarak sunulmasın.
    """
    raw = result.get("confidence")
    if not isinstance(raw, (int, float)):
        result["calibrated_confidence"] = None
        result["calibrated"] = False
        result["confidence_badge"] = badge_for(None)
        return result

    model = load_model()
    calibrated = model.apply(float(raw)) if model else float(raw)
    result["calibrated_confidence"] = calibrated
    result["calibrated"] = model is not None
    result["confidence_badge"] = badge_for(calibrated)
    if model is not None:
        result["calibration_note"] = (
            f"{model.trained_on} üzerinde {model.method} ile kalibre edildi "
            f"(n={model.n}, ECE {model.ece_before:.3f} -> {model.ece_after:.3f})."
        )
    else:
        result["calibration_note"] = (
            "Kalibrasyon modeli yok; gösterilen skor HAM skordur ve "
            "sistematik olarak yüksek olabilir."
        )

    if calibrated < abstention_threshold and result.get("is_reconstructible"):
        result["withheld_reconstruction"] = result.get("reconstructed_root")
        result["reconstructed_root"] = ""
        result["is_reconstructible"] = False
        result["abstained"] = True
        result["reconstruction_notes"] = (
            f"Kanıt yetersiz: kalibre güven {calibrated:.2f} < {abstention_threshold:.2f}. "
            f"Motorun ürettiği aday `withheld_reconstruction` alanındadır ama sonuç "
            f"olarak sunulmamaktadır."
        )
    return result


def train_and_save(
    raw_scores: Sequence[float],
    correct: Sequence[bool],
    *,
    trained_on: str,
    method: str = "platt",
) -> CalibrationModel:
    """Kalibratörü eğitir ve diske yazar.

    ⚠️ Yalnız **train** bölümüyle çağrılmalıdır. dev veya test ile eğitmek
    ölçümü geçersiz kılar.
    """
    from engine.evaluation.calibration import (
        PlattCalibrator,
        cross_validated_calibration,
        expected_calibration_error,
    )

    if method != "platt":
        raise ValueError("Diske yalnız Platt parametreleri yazılabilir (a, b).")

    fitted = PlattCalibrator().fit(raw_scores, correct)
    before, _ = expected_calibration_error(raw_scores, correct)
    # Kendi eğitim verisinde ölçmek ECE'yi yapay olarak düşürür; çapraz
    # doğrulamalı skorla ölçülür.
    cv_scores = cross_validated_calibration(raw_scores, correct, method="platt")
    after, _ = expected_calibration_error(cv_scores, correct)

    model = CalibrationModel(
        a=fitted.a,
        b=fitted.b,
        method="platt",
        trained_on=trained_on,
        n=len(raw_scores),
        ece_before=before,
        ece_after=after,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-calibration/v1",
                "a": model.a,
                "b": model.b,
                "method": model.method,
                "trained_on": model.trained_on,
                "n": model.n,
                "ece_before": round(model.ece_before, 4),
                "ece_after": round(model.ece_after, 4),
                "trained_at": model.trained_at,
                "note": (
                    "Yalnız altın standardın TRAIN bölümüyle eğitilmiştir. "
                    "dev/test ile eğitmek ölçümü geçersiz kılar."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    reset_model_cache()
    logger.info(
        "Kalibrasyon modeli yazıldı: %s (ECE %.3f -> %.3f)", MODEL_PATH, before, after
    )
    return model


def main() -> int:
    import argparse

    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation.gold import GoldStandard
    from engine.evaluation.harness import comparative_reconstructor, run

    ap = argparse.ArgumentParser(description="Kalibratörü TRAIN bölümünde eğit")
    ap.add_argument("--dataset", default="savelyevturkic")
    args = ap.parse_args()

    gold = GoldStandard.build(args.dataset)
    train_items = gold.split("train")
    mapping = build_mapping(CldfWordlist.load(args.dataset))
    outcome = run(comparative_reconstructor(), train_items, mapping=mapping, split="train")

    if len(outcome.confidences) < 30:
        print(f"Eğitim için yeterli veri yok (n={len(outcome.confidences)})")
        return 1

    model = train_and_save(
        outcome.confidences,
        outcome.correctness,
        trained_on=f"{args.dataset}/train",
    )
    print(f"Kalibratör eğitildi: a={model.a:.4f} b={model.b:.4f}")
    print(f"ECE {model.ece_before:.4f} -> {model.ece_after:.4f} (n={model.n})")
    print(f"Yazıldı: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
