"""
Öngörü testi — yanlışlanabilir bilimsel iddia.

Bu faz olmadan "motor yeni etimoloji buldu" iddiası ciddiye alınmaz.
Protokol Bodt & List (2022, *Diachronica*) ve Blum ve ark. (2024,
*Scientific Reports*) çalışmalarından alınmıştır: tahminler **önceden kayıt
altına alınır**, sonra bağımsız olarak doğrulanır.

Döngü::

    1. ÜRET     motor bir hipotez kurunca "bu kelimenin Başkurtçada X
                olması gerekir" biçiminde öngörü çıkarır
    2. KİLİTLE  öngörüler değiştirilemez biçimde kaydedilir (içerik özeti)
    3. DOĞRULA  AYRI bir koşuda sözlük indeksinde aranır
    4. SKORLA   hipotezin sicili güncellenir

⚠️ **Kendi reponuzdaki zaman damgası ön-kayıt DEĞİLDİR.** Commit geçmişi
yeniden yazılabilir; hakem buna güvenmez. Gerçek ön-kayıt üçüncü tarafta
(OSF) yapılır ve :attr:`PredictionRegistry.external_doi` alanına yazılır.
Bu alan boşken üretilen rapor "ön-kayıtlı" diye sunulamaz ve modül bunu
çıktıda açıkça söyler.

⚠️ **Çapraz doğrulama zorunlu.** Öngörüyü üretirken kullanılan diller ile
doğrulanan dil ayrı olmalıdır; aksi hâlde motor kendi girdisini "doğrulamış"
olur.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PREDICTIONS_DIR
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Doğrulama sonucu: tahmin sözlükte bulundu mu?
HIT = "tuttu"
NEAR = "yakın"
MISS = "tutmadı"
UNTESTABLE = "denetlenemedi"

#: Sözlükte bulunsa bile **bulgu sayılmayan** eşleşmeler.
#:
#: ⚠️ Ölçüldü: ``em`` için üretilen ``a`` tahmini Azerbaycan Türkçesinde
#: "A — alfabenin ilk harfi" maddesiyle eşleşiyordu. Tek harflik biçimler,
#: harf adları ve kısaltmalar sözlükte vardır ama sözcük değildir; bunları
#: "tahmin tuttu" saymak isabet oranını yapay olarak şişirir.
MIN_MEANINGFUL_LENGTH = 2
SPURIOUS_GLOSS_MARKERS = (
    "letter of the",
    "harfi",
    "abbreviation",
    "kısaltma",
    "symbol for",
    "romanization of",
    "cyrillic spelling of",
    "latin spelling of",
    "alternative spelling of",
    "obsolete spelling of",
)


def is_spurious_match(form: str, gloss: str) -> bool:
    """Eşleşme gerçek bir sözcük mü, sözlük artefaktı mı?"""
    if len(form.strip()) < MIN_MEANINGFUL_LENGTH:
        return True
    lowered = gloss.lower()
    return any(marker in lowered for marker in SPURIOUS_GLOSS_MARKERS)


@dataclass(frozen=True)
class Prediction:
    """Tek bir yanlışlanabilir öngörü."""

    hypothesis: str
    source_word: str
    source_languages: tuple[str, ...]
    target_language: str
    predicted_form: str
    confidence: float
    rationale: str

    def digest(self) -> str:
        """Öngörünün içerik özeti — sonradan değiştirilirse tutmaz."""
        payload = "|".join(
            [
                self.hypothesis,
                self.source_word,
                ",".join(sorted(self.source_languages)),
                self.target_language,
                self.predicted_form,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_languages"] = list(self.source_languages)
        data["digest"] = self.digest()
        return data


@dataclass
class Verification:
    """Bir öngörünün doğrulama sonucu."""

    digest: str
    outcome: str
    found_form: str = ""
    edit_distance: int | None = None
    gloss: str = ""

    @property
    def is_hit(self) -> bool:
        return self.outcome == HIT

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRegistry:
    """Kilitlenmiş öngörü kümesi."""

    name: str
    predictions: list[Prediction] = field(default_factory=list)
    created_at: str = ""
    external_doi: str = ""
    engine_version: str = ""

    @property
    def is_preregistered(self) -> bool:
        """Üçüncü tarafta kayıtlı mı? Yerel zaman damgası yeterli DEĞİLDİR."""
        return bool(self.external_doi.strip())

    def seal(self) -> str:
        """Bütün öngörülerin birleşik özeti."""
        combined = "".join(sorted(p.digest() for p in self.predictions))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def path(self) -> Path:
        return PREDICTIONS_DIR / f"{self.name}.locked.json"

    def save(self) -> Path:
        """Sicili **değiştirilemez** biçimde yazar; varsa üzerine yazmaz."""
        target = self.path()
        if target.exists():
            raise FileExistsError(
                f"{target} zaten var. Kilitlenmiş bir sicilin üzerine yazmak, "
                f"ön-kaydı geçersiz kılar. Yeni bir ad kullanın."
            )
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.created_at = self.created_at or datetime.now(UTC).isoformat(timespec="seconds")
        target.write_text(
            json.dumps(
                {
                    "_schema": "turkic-etymology-predictions/v1",
                    "name": self.name,
                    "created_at": self.created_at,
                    "engine_version": self.engine_version,
                    "external_doi": self.external_doi,
                    "preregistered": self.is_preregistered,
                    "seal": self.seal(),
                    "n_predictions": len(self.predictions),
                    "note": (
                        "Bu dosya KİLİTLİDİR. Doğrulama sonuçları ayrı bir dosyaya "
                        "yazılır. Yerel zaman damgası ön-kayıt yerine geçmez; "
                        "gerçek ön-kayıt için external_doi doldurulmalıdır."
                    ),
                    "predictions": [p.as_dict() for p in self.predictions],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        logger.info("Öngörü sicili kilitlendi: %s (%d öngörü)", target, len(self.predictions))
        return target

    @classmethod
    def load(cls, name: str) -> PredictionRegistry:
        path = PREDICTIONS_DIR / f"{name}.locked.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        registry = cls(
            name=data["name"],
            created_at=data.get("created_at", ""),
            external_doi=data.get("external_doi", ""),
            engine_version=data.get("engine_version", ""),
            predictions=[
                Prediction(
                    hypothesis=p["hypothesis"],
                    source_word=p["source_word"],
                    source_languages=tuple(p["source_languages"]),
                    target_language=p["target_language"],
                    predicted_form=p["predicted_form"],
                    confidence=p["confidence"],
                    rationale=p["rationale"],
                )
                for p in data["predictions"]
            ],
        )
        if registry.seal() != data.get("seal"):
            raise ValueError(
                f"{path} mührü tutmuyor: dosya kilitlendikten sonra DEĞİŞTİRİLMİŞ. "
                f"Bu sicille yapılan hiçbir doğrulama geçerli değildir."
            )
        return registry


def generate_predictions(
    word: str,
    entries: Sequence[dict[str, Any]],
    *,
    targets: Sequence[str],
    hypothesis: str = "",
) -> list[Prediction]:
    """Bir rekonstrüksiyondan yanlışlanabilir öngörüler üretir.

    ⚠️ **Çapraz doğrulama:** öngörünün üretiminde kullanılan diller,
    doğrulanacak dilden ayrıdır. Hedef dilin tanığı girdiden çıkarılır;
    aksi hâlde motor kendi girdisini "doğrulamış" olur.
    """
    from engine.nlp.cognate_prediction import CognatePredictor
    from engine.nlp.comparative_reconstruction import ComparativeReconstructor

    predictor = CognatePredictor()
    reconstructor = ComparativeReconstructor()
    out: list[Prediction] = []

    for target in targets:
        # Hedef dilin tanığı ÇIKARILIR — döngüsellik önlemi.
        held_out = [e for e in entries if e.get("lang_code") != target]
        if len(held_out) < 2:
            continue
        reconstruction = reconstructor.reconstruct(word, held_out)
        if not reconstruction.get("is_reconstructible"):
            continue

        source_languages = tuple(sorted(reconstruction.get("witness_languages", [])))
        prediction = predictor.predict(word, "tr", target)
        if not prediction.form:
            continue

        out.append(
            Prediction(
                hypothesis=hypothesis or str(reconstruction["reconstructed_root"]),
                source_word=word,
                source_languages=source_languages,
                target_language=target,
                predicted_form=prediction.form,
                confidence=float(reconstruction.get("calibrated_confidence") or 0.0),
                rationale=(
                    f"{reconstruction['reconstructed_root']} "
                    f"[*{reconstruction.get('proto_level', '?')}] "
                    f"{len(source_languages)} tanıktan türetildi; "
                    f"{target} refleksi öğrenilmiş denkliklerle tahmin edildi"
                ),
            )
        )
    return out


def verify(registry: PredictionRegistry, *, max_distance: int = 1) -> list[Verification]:
    """Kilitli sicili sözlük indeksine karşı doğrular.

    Bu **ayrı bir koşudur**: üretim ve doğrulama aynı adımda yapılırsa
    öngörü testinin anlamı kalmaz.
    """
    from engine.db.lexicon_index import LexiconIndex
    from engine.utils.orthography import to_comparison_form

    index = LexiconIndex()
    if not index.exists:
        logger.warning("Sözlük indeksi yok; doğrulama yapılamıyor")
        return [Verification(p.digest(), UNTESTABLE) for p in registry.predictions]

    results: list[Verification] = []
    for prediction in registry.predictions:
        target = to_comparison_form(prediction.predicted_form)
        exact = [
            row
            for row in index.lookup(target, languages=[prediction.target_language], limit=5)
            if not is_spurious_match(row["word"], row.get("gloss", ""))
        ]
        if exact:
            results.append(
                Verification(
                    prediction.digest(), HIT, exact[0]["word"], 0, exact[0].get("gloss", "")
                )
            )
            continue
        near = [
            row
            for row in index.fuzzy_lookup(
                target, max_distance=max_distance, languages=[prediction.target_language]
            )
            if not is_spurious_match(row["word"], row.get("gloss", ""))
        ]
        if near:
            best = near[0]
            results.append(
                Verification(
                    prediction.digest(),
                    NEAR,
                    best["word"],
                    best["edit_distance"],
                    best.get("gloss", ""),
                )
            )
            continue
        results.append(Verification(prediction.digest(), MISS))
    return results


def score_verifications(verifications: Sequence[Verification]) -> dict[str, Any]:
    """Öngörü isabetini özetler."""
    total = len(verifications)
    if not total:
        return {"n": 0}
    counts = {outcome: 0 for outcome in (HIT, NEAR, MISS, UNTESTABLE)}
    for verification in verifications:
        counts[verification.outcome] = counts.get(verification.outcome, 0) + 1
    testable = total - counts[UNTESTABLE]
    return {
        "n": total,
        "testable": testable,
        **counts,
        "hit_rate": round(counts[HIT] / testable, 4) if testable else 0.0,
        "hit_or_near_rate": round((counts[HIT] + counts[NEAR]) / testable, 4) if testable else 0.0,
        "reference": (
            "Bodt & List 2022: ~%70. ⚠️ Doğrudan karşılaştırılamaz: onlarınki "
            "uzman seçimli, belirli boşluklara yöneltilmiş ve saha çalışmasıyla "
            "doğrulanmış tahminlerdi; burada altın standardın tamamı taranıyor."
        ),
    }


def chance_baseline(
    registry: PredictionRegistry, *, iterations: int = 500
) -> dict[str, Any]:
    """Bu isabet oranı şans eseri olabilir mi? (Kessler 2001)

    Motor 100 binden fazla kayıtlık bir indekste arıyor. Kısa bir biçim salt
    şansla eşleşme bulur. Null model: **aynı uzunlukta rastgele** biçimler
    aynı ölçütle aranırsa kaç tanesi tutar?

    Bu kontrol olmadan "tahminlerimizin %12'si tuttu" cümlesi bir bulgu
    değildir.
    """
    import random

    from engine.db.lexicon_index import LexiconIndex

    index = LexiconIndex()
    if not index.exists:
        return {"available": False}

    rng = random.Random(20260827)
    alphabet = "abcdefgıiklmnoöprstuüyz"
    vowels = "aeıioöuü"
    null_hits: list[int] = []
    for _ in range(iterations):
        hits = 0
        for prediction in registry.predictions:
            length = max(2, len(prediction.predicted_form))
            # Fonotaktik olarak makul rastgele biçim: ünlü/ünsüz dönüşümlü.
            decoy = "".join(
                rng.choice(vowels) if i % 2 else rng.choice(alphabet) for i in range(length)
            )
            if index.lookup(decoy, languages=[prediction.target_language], limit=1):
                hits += 1
        null_hits.append(hits)

    observed = sum(1 for p in registry.predictions if p)  # yer tutucu, çağıran doldurur
    del observed
    expected = sum(null_hits) / len(null_hits)
    return {
        "available": True,
        "chance_expected_hits": round(expected, 2),
        "chance_hit_rate": round(expected / len(registry.predictions), 4)
        if registry.predictions
        else 0.0,
        "iterations": iterations,
        "note": (
            "Aynı uzunlukta, fonotaktik olarak makul RASTGELE biçimlerin aynı "
            "indekste bulunma oranı. Gözlenen isabet bunun belirgin üstünde "
            "değilse bulgu sayılmaz (Kessler 2001)."
        ),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Öngörü testi: üret-kilitle-doğrula")
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="öngörü üret ve kilitle")
    gen.add_argument("--name", required=True, help="sicil adı")
    gen.add_argument("--split", default="dev")
    gen.add_argument("--limit", type=int, default=200)
    gen.add_argument("--doi", default="", help="OSF ön-kayıt DOI'si")

    ver = sub.add_parser("verify", help="kilitli sicili doğrula")
    ver.add_argument("--name", required=True)

    args = ap.parse_args()

    if args.command == "generate":
        from engine.db.cldf_wordlist import CldfWordlist
        from engine.db.language_mapping import build_mapping
        from engine.evaluation.gold import GoldStandard
        from engine.evaluation.harness import _anchor_for, _witnesses_for

        gold = GoldStandard.build()
        items = gold.split(args.split)
        mapping = build_mapping(CldfWordlist.load("savelyevturkic"))

        registry = PredictionRegistry(name=args.name, external_doi=args.doi)
        for item in items:
            if len(registry.predictions) >= args.limit:
                break
            witnesses = _witnesses_for(item, mapping)
            anchor, anchor_lang = _anchor_for(witnesses)
            if not anchor:
                continue
            targets = sorted({w["lang_code"] for w in witnesses} - {anchor_lang})
            registry.predictions.extend(
                generate_predictions(anchor, witnesses, targets=targets[:3])
            )

        path = registry.save()
        print(f"{len(registry.predictions)} öngörü kilitlendi -> {path}")
        print(f"mühür: {registry.seal()[:32]}…")
        if not registry.is_preregistered:
            print(
                "\n⚠️ Bu sicil ÜÇÜNCÜ TARAFTA kayıtlı değil. Yerel zaman damgası\n"
                "   ön-kayıt yerine geçmez; sonuçlar 'ön-kayıtlı çalışma' olarak\n"
                "   sunulamaz. OSF kaydı yapıp --doi ile yeniden üretin."
            )
        return 0

    registry = PredictionRegistry.load(args.name)
    verifications = verify(registry)
    summary = score_verifications(verifications)

    chance = chance_baseline(registry)
    summary["chance_control"] = chance

    print(f"\n=== öngörü doğrulaması · {args.name} ===")
    print(f"ön-kayıtlı: {'EVET ' + registry.external_doi if registry.is_preregistered else 'HAYIR'}")
    for key, value in summary.items():
        if key in ("reference", "chance_control"):
            continue
        print(f"  {key:18} {value}")
    if chance.get("available"):
        print(
            f"  {'ŞANS TABAN ÇİZGİSİ':18} {chance['chance_hit_rate']:.4f} "
            f"({chance['chance_expected_hits']:.1f} eşleşme beklenir)"
        )
        lift = summary["hit_rate"] / chance["chance_hit_rate"] if chance["chance_hit_rate"] else float("inf")
        print(f"  {'şansın kaç katı':18} {lift:.1f}x")
    print(f"\n  {summary['reference']}")

    out = PREDICTIONS_DIR / f"{args.name}.verified.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-verification/v1",
                "registry": args.name,
                "registry_seal": registry.seal(),
                "preregistered": registry.is_preregistered,
                "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "summary": summary,
                "verifications": [v.as_dict() for v in verifications],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out}")

    print("\nörnek tutmalar:")
    by_digest = {p.digest(): p for p in registry.predictions}
    shown = 0
    for verification in verifications:
        if verification.outcome != HIT or shown >= 8:
            continue
        prediction = by_digest[verification.digest]
        print(
            f"  {prediction.source_word:12} -> {prediction.target_language:5} "
            f"{prediction.predicted_form:14} bulundu: {verification.found_form:14} "
            f"({verification.gloss[:32]})"
        )
        shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
