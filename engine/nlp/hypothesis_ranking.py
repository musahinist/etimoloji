"""
Rakip hipotez sıralaması ve **karşıtsal red gerekçesi**.

Mevcut sistemler tek bir cevap verir. Bu modül bütün makul kökenleri yan yana
kurar, hepsini aynı kanıtlarla puanlar ve **reddedilenleri gerekçesiyle
birlikte çıktıda tutar**::

    kitap
      1. ALINTI — Arapça              0,60   ✓ seçildi
      2. MİRAS — Proto-Türkçe         0,10   ✗ reddedildi
         neden: sözlükte Arapça alıntı olarak tanıklanmış; miras olsaydı
         söz başı /k/ Oğuz ötümlüleşmesinden geçmiş olurdu
      3. MODERN TÜRETME               0,00   ✗ reddedildi
         neden: 13. yüzyıldan önce tanıklanmış

**Literatür durumu.** N-best aday üretimi yapılmış ama gizlidir: Lu ve ark.
(2024) refleks tahminiyle adayları yeniden sıralar, kullanıcıya tek çıktı
gider. Belirsizliğin kullanıcıya sunumu tek örnekte var — List ve ark.
(2023), ``*[p a|i t]`` pipe gösterimi — ama o **ses konumu** düzeyindedir,
bütün-hipotez düzeyinde değil ve red gerekçesi üretmez. Blum ve ark. (2024)
formel hipotez karşılaştırması yapar ama yalnız ikilidir (H0/H1).

Red gerekçesinin teknik adı **karşıtsal açıklama**dır (contrastive
explanation: "neden P, Q değil?"). Şablon planlama literatüründe olgundur
(Krarup ve ark., *JAIR*: bir seçeneğin neden plana girmediğini, girseydi
hangi özelliklerin geçerli olacağını kullanarak açıklamak) ama tarihsel
dilbilime taşınmamıştır.

⚠️ Gerekçenin **içeriği** sembolik katmandan gelir; LLM yalnız metni
akıcılaştırabilir, kararı vermez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Hipotez türleri ve okunabilir adları.
HYPOTHESIS_KINDS: dict[str, str] = {
    "inherited": "MİRAS — Proto-Türkçe'den",
    "borrowed": "ALINTI",
    "derived": "TÜRETME — Türkçe kökten",
    "modern_coinage": "MODERN TÜRETME",
    "unknown": "KÖKENİ BELİRSİZ",
}


@dataclass
class Hypothesis:
    """Tek bir köken önerisi, kanıtları ve varsa red gerekçesi."""

    kind: str
    claim: str
    score: float
    supporting: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)
    rejected_because: str = ""
    counterfactual: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return HYPOTHESIS_KINDS.get(self.kind, self.kind)

    @property
    def is_rejected(self) -> bool:
        return bool(self.rejected_because)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "claim": self.claim,
            "score": round(self.score, 3),
            "supporting": self.supporting,
            "against": self.against,
            "rejected": self.is_rejected,
            "rejected_because": self.rejected_because,
            "counterfactual": self.counterfactual,
            "detail": self.detail,
        }


@dataclass
class RankedHypotheses:
    """Sıralanmış hipotezler; reddedilenler **silinmez**."""

    word: str
    hypotheses: list[Hypothesis] = field(default_factory=list)

    @property
    def selected(self) -> Hypothesis | None:
        alive = [h for h in self.hypotheses if not h.is_rejected]
        return alive[0] if alive else None

    @property
    def margin(self) -> float:
        """Birinci ile ikinci arasındaki fark.

        Küçük fark, kararın **kırılgan** olduğunu söyler: kanıt biraz
        değişse sıralama değişirdi. Bu bilgi kullanıcıdan saklanmaz.
        """
        scores = sorted((h.score for h in self.hypotheses), reverse=True)
        return round(scores[0] - scores[1], 3) if len(scores) >= 2 else 0.0

    @property
    def is_contested(self) -> bool:
        return 0 < self.margin < 0.15

    def explain(self) -> str:
        lines = [f"{self.word}"]
        for index, hypothesis in enumerate(
            sorted(self.hypotheses, key=lambda h: -h.score), start=1
        ):
            mark = "✗ reddedildi" if hypothesis.is_rejected else "✓ seçildi" if index == 1 else ""
            lines.append(f"  {index}. {hypothesis.label:28} {hypothesis.score:.2f}   {mark}")
            for evidence in hypothesis.supporting[:3]:
                lines.append(f"       + {evidence}")
            if hypothesis.rejected_because:
                lines.append(f"       neden: {hypothesis.rejected_because}")
            if hypothesis.counterfactual:
                lines.append(f"       doğru olsaydı: {hypothesis.counterfactual}")
        if self.is_contested:
            lines.append(
                f"  ⚠️ karar kırılgan: ilk iki hipotez arasındaki fark yalnız {self.margin:.2f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "selected": self.selected.as_dict() if self.selected else None,
            "margin": self.margin,
            "is_contested": self.is_contested,
            "hypotheses": [
                h.as_dict() for h in sorted(self.hypotheses, key=lambda h: -h.score)
            ],
            "explanation": self.explain(),
        }


class HypothesisRanker:
    """Rakip kökenleri kurar, puanlar ve reddedilenleri gerekçelendirir."""

    def __init__(self, reconstructor: Any = None, borrowing_detector: Any = None):
        self._reconstructor = reconstructor
        self._borrowing = borrowing_detector

    @property
    def reconstructor(self) -> Any:
        if self._reconstructor is None:
            from engine.nlp.comparative_reconstruction import ComparativeReconstructor

            self._reconstructor = ComparativeReconstructor()
        return self._reconstructor

    @property
    def borrowing(self) -> Any:
        if self._borrowing is None:
            from engine.nlp.borrowing_detector import BorrowingDetector

            self._borrowing = BorrowingDetector()
        return self._borrowing

    def rank(
        self,
        word: str,
        entries: list[dict[str, Any]] | None = None,
        *,
        attested_before: int | None = None,
    ) -> RankedHypotheses:
        """Bütün makul kökenleri kurar ve sıralar.

        :param attested_before: kelimenin bilinen en eski tanıklama yılı;
            modern türetme hipotezini elemek için kullanılır.
        """
        entries = entries or []
        borrowing = self.borrowing.detect(word, entries)
        reconstruction = self.reconstructor.reconstruct(word, entries, check_borrowing=False)

        hypotheses = [
            self._borrowed_hypothesis(borrowing),
            self._inherited_hypothesis(reconstruction, borrowing),
            self._modern_hypothesis(word, attested_before, borrowing),
        ]
        hypotheses = [h for h in hypotheses if h is not None]

        if not any(h.score > 0.1 for h in hypotheses):
            hypotheses.append(
                Hypothesis(
                    kind="unknown",
                    claim="Kökeni belirlenemedi",
                    score=0.15,
                    supporting=["hiçbir hipotez için yeterli kanıt yok"],
                )
            )

        ranked = RankedHypotheses(word=word, hypotheses=hypotheses)
        self._write_rejections(ranked)
        return ranked

    # -- tek tek hipotezler -------------------------------------------------

    @staticmethod
    def _borrowed_hypothesis(borrowing: Any) -> Hypothesis:
        from engine.nlp.borrowing_chain import language_name

        donor = language_name(borrowing.donor_language) if borrowing.donor_language else "?"
        supporting = [s.explanation for s in borrowing.signals if s.fired]
        against = [s.explanation for s in borrowing.signals if not s.fired]
        return Hypothesis(
            kind="borrowed",
            claim=f"ALINTI — {donor}" if borrowing.donor_language else "ALINTI",
            score=borrowing.score,
            supporting=supporting,
            against=against,
            detail={
                "chain": borrowing.chain,
                "donor": borrowing.donor_language,
                "expected_if_inherited": borrowing.expected_if_inherited,
            },
        )

    @staticmethod
    def _inherited_hypothesis(reconstruction: dict[str, Any], borrowing: Any) -> Hypothesis:
        if not reconstruction.get("is_reconstructible"):
            return Hypothesis(
                kind="inherited",
                claim="MİRAS — ata biçim türetilemedi",
                score=0.05,
                against=[str(reconstruction.get("reconstruction_notes", ""))[:160]],
            )
        root = str(reconstruction["reconstructed_root"])
        level = reconstruction.get("proto_level", "?")
        supporting = [
            f"{reconstruction['witness_count']} dil tanığı, "
            f"{reconstruction['branch_count']} Türki kol",
            f"sütun uyumu {reconstruction['column_agreement']}",
        ]
        supporting += list(reconstruction.get("applied_correspondences", []))[:2]
        return Hypothesis(
            kind="inherited",
            claim=f"MİRAS — {root} [*{level}]",
            score=float(reconstruction.get("calibrated_confidence") or 0.0),
            supporting=supporting,
            detail={
                "reconstructed_root": root,
                "proto_level": level,
                "witness_languages": reconstruction.get("witness_languages", []),
            },
        )

    @staticmethod
    def _modern_hypothesis(
        word: str, attested_before: int | None, borrowing: Any
    ) -> Hypothesis | None:
        """Dil Devrimi sonrası türetilmiş olabilir mi?"""
        try:
            from engine.nlp.neologism_detector import NeologismDetector

            # ``detect`` tespit YOKSA ``None`` döner — bu bir hata değil,
            # "modern türetme göstergesi bulunamadı" demektir.
            detection = NeologismDetector().detect(word) or {}
        except Exception:
            logger.warning("Neolojizm denetimi başarısız: %s", word, exc_info=True)
            return None

        is_neologism = bool(detection)
        score = float(detection.get("confidence", 0.6)) if is_neologism else 0.05
        supporting = []
        if is_neologism:
            supporting.append(
                str(
                    detection.get("reason")
                    or detection.get("explanation")
                    or detection.get("pattern")
                    or "Cumhuriyet dönemi türetme kalıbı"
                )
            )
        return Hypothesis(
            kind="modern_coinage",
            claim="MODERN TÜRETME — Dil Devrimi sonrası",
            score=score,
            supporting=supporting,
            detail={"attested_before": attested_before, **detection},
        )

    # -- karşıtsal red ------------------------------------------------------

    @staticmethod
    def _write_rejections(ranked: RankedHypotheses) -> None:
        """Reddedilen her hipoteze **neden reddedildiğini** yazar.

        Karşıtsal açıklama iki parça taşır: hangi kanıt onu eledi, ve o
        hipotez doğru olsaydı ne beklenirdi. İkincisi olmadan gerekçe
        yanlışlanabilir değildir.
        """
        ordered = sorted(ranked.hypotheses, key=lambda h: -h.score)
        if not ordered:
            return
        winner = ordered[0]
        for hypothesis in ordered[1:]:
            if hypothesis.score >= winner.score - 1e-9:
                continue
            hypothesis.rejected_because = _rejection_reason(hypothesis, winner)
            hypothesis.counterfactual = _counterfactual_for(hypothesis, winner)


def _rejection_reason(rejected: Hypothesis, winner: Hypothesis) -> str:
    """Somut, kanıta dayalı red gerekçesi.

    "Skoru düşüktü" bir gerekçe değildir; hangi kanıtın onu elediği
    söylenmelidir. Karşıtsal açıklamanın birinci parçası budur.
    """
    # Miras hipotezi, ALINTI kanıtıyla elendiyse o kanıt gösterilir.
    if rejected.kind == "inherited" and winner.kind == "borrowed":
        evidence = winner.supporting[0] if winner.supporting else ""
        if evidence:
            return evidence
    # Alıntı hipotezi elendiyse, alıntı sinyallerinin neden ateşlenmediği.
    if rejected.kind == "borrowed" and rejected.against:
        return rejected.against[0]
    if rejected.against:
        return rejected.against[0]
    gap = winner.score - rejected.score
    return f"kanıt gücü {winner.label} hipotezinin {gap:.2f} gerisinde"


def _counterfactual_for(rejected: Hypothesis, winner: Hypothesis) -> str:
    """"Bu hipotez doğru olsaydı ne beklerdik?" — yanlışlanabilir kısım."""
    if rejected.kind == "inherited":
        expected = winner.detail.get("expected_if_inherited") or ""
        if expected:
            return f"Türkçe biçim {expected!r} olurdu, düzenli ses kanunları uygulanırdı"
        return (
            "akraba dillerde düzenli ses karşılıkları görülür, "
            "söz başı ötümlüleşme izleri bulunurdu"
        )
    if rejected.kind == "borrowed":
        return (
            "verici dilde uygun anlam ve biçimde bir kaynak kelime "
            "tanıklanır, uyarlama kuralları tutarlı olurdu"
        )
    if rejected.kind == "modern_coinage":
        return "kelime 20. yüzyıldan önce hiçbir kaynakta tanıklanmazdı"
    return ""


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Rakip hipotez sıralaması")
    ap.add_argument("words", nargs="*", default=[])
    args = ap.parse_args()

    ranker = HypothesisRanker()
    samples = {
        "kitap": [("kk", "kitap"), ("tt", "kitap"), ("uz", "kitob")],
        "göz": [("tr", "göz"), ("kk", "көз"), ("cv", "куҫ"), ("tt", "күз")],
        "deniz": [("tr", "deniz"), ("kk", "теңіз"), ("cv", "тинӗс"), ("tk", "deňiz")],
        "bilgisayar": [],
    }
    words = args.words or list(samples)
    for word in words:
        entries = [
            {"lang_code": code, "word": form} for code, form in samples.get(word, [])
        ]
        print(ranker.rank(word, entries).explain())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
