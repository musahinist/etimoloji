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
    #: Veri olmadığı için ÇALIŞAMAYAN sinyaller. Karşı kanıt değildir:
    #: "yeterli tanık yok" alıntıya karşı bir bulgu değil, bulgu yokluğudur
    #: (üstelik Türki akrabası olmayan kelime alıntıyla uyumludur).
    not_evaluated: list[str] = field(default_factory=list)
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
            "not_evaluated": self.not_evaluated,
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
    #: Aynı türden iki doğrudan tanıklığın çeliştiği durumlar (kaynakta
    #: tanıklı miras kök × sözlükte alıntı kaydı). Hüküm skordan çıkar;
    #: çelişki kullanıcıdan saklanmaz.
    conflicts: list[str] = field(default_factory=list)

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
        for conflict in self.conflicts:
            lines.append(f"  ⚠️ kaynaklar çelişiyor: {conflict}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "selected": self.selected.as_dict() if self.selected else None,
            "margin": self.margin,
            "is_contested": self.is_contested,
            "conflicts": list(self.conflicts),
            "hypotheses": [
                h.as_dict() for h in sorted(self.hypotheses, key=lambda h: -h.score)
            ],
            "explanation": self.explain(),
        }


#: `borrowing_detector._donor_signal` adı ve "karşılık yok" açıklaması.
DONOR_PROXIMITY_SIGNAL = "verici_yakınlığı"
DONOR_PROXIMITY_MISS = "verici sözlüğünde yakın karşılık yok"


def _attested_donor_form(chain: list[str] | None) -> str:
    """Zincirde tanıklı verici biçim varsa onu ("Arapça عَسْكَر") döndürür."""
    return str(chain[-1]) if chain and len(chain) >= 2 else ""


def _donor_proximity_moot(attested: str) -> str:
    """Verici yakınlığı ıskası, tanıklı verici biçim varken karşı kanıt DEĞİLDİR.

    ⚠️ Yakınlık ölçütü verici sözlüğünde AYNI ANLAMLI ve fonetik olarak yakın
    bir madde arar; doğrudan verici biçmi sınamaz. Tanıklı verici biçim
    (akü ← Fransızca accumulateur) dururken "verici sözlüğünde yakın karşılık
    yok" demek aynı raporda kendi köken zinciriyle çelişiyordu (denetim: 105
    kelimeden 42'si).
    """
    return (
        f"verici sözlüğü yakınlık araması eşleşme bulmadı; tanıklı verici biçim "
        f"({attested}) bu ölçütle sınanmadı"
    )


#: Birleştirici alıntıyı reddedince (p < eşik) alıntı skoru bu tavanın altında
#: kalır: "belirsiz" tabanı (``any(score > 0,1)``) hiçbir zaman geçilmez.
REJECTED_LOAN_CEILING = 0.10
#: Birleştirici alıntıyı reddedince miras hipotezinin tabanı: "belirsiz"in
#: (0,15) üstü, her gerçek kanıtın (rekonstrüksiyon güveni, tanıklı kök 0,5,
#: modern türetme) altında ya da onunla birleşir (``max``).
LOAN_REJECTED_INHERITED_FLOOR = 0.20


def _combiner_verdict(borrowing: Any) -> tuple[float, float] | None:
    """Eğitilmiş birleştiricinin ``(olasılık, eşik)``i; model yoksa ``None``."""
    probability = getattr(borrowing, "trained_probability", None)
    threshold = getattr(borrowing, "_trained_threshold", None)
    if isinstance(probability, float) and isinstance(threshold, float) and 0.0 < threshold < 1.0:
        return probability, threshold
    return None


def _trained_loan_score(borrowing: Any, *, direct_record: bool = False) -> float | None:
    """Alıntı hipotezinin skoru, EĞİTİLMİŞ birleştiricinin kararından.

    ⚠️ Eskiden skor elle ağırlıklı toplamdı (``SIGNAL_WEIGHTS``; ör. 0,32 ×
    verici yakınlığı gücü) ve 0,1'i aşan her toplam "belirsiz"i geçip ALINTI
    seçtiriyordu. Birleştirici (``borrowing_combiner``) detektörün ölçülmüş
    karar yoludur ama sıralayıcı onu kullanmıyordu: arama yolunda verici
    yakınlığı açılınca rampa mirasları ALINTI'ya çevirdi (150 kelimede uyum
    110 → 100, c0e8dd7). Ölçüm: ``data/cache/work/ranker/PREREG.md``.

    * p ≥ eşik: eşik ``BORROWING_THRESHOLD``'a, 1 → 1'e doğrusal eşlenir.
    * p < eşik: birleştirici alıntıyı reddediyor; skor
      ``REJECTED_LOAN_CEILING`` × p/eşik (sürekli eşleme denendi: kanıtsız
      kelimede p≈0,12 "belirsiz"i geçip ALINTI seçtiriyordu).
    * Doğrudan sözlük tanıklığı (alıntı kaydı ya da kaynağın alıntı adımı)
      karar verici kalır: skor en az ``SIGNAL_WEIGHTS["zincir_kanıtı"]`` —
      tanıklı miras kökle eşit ağırlık (``_with_attested_root``). Saha'da
      eğitilen model Türkçe sözlük kaydını tek başına alıntı saymıyor
      (p=0,31 < 0,39); kural b39e36f'deki simetriyi korur.

    Model yoksa ``None``: çağıran el ağırlıklı toplama döner.
    """
    from engine.nlp.borrowing_detector import BORROWING_THRESHOLD, SIGNAL_WEIGHTS

    verdict = _combiner_verdict(borrowing)
    if verdict is None:
        return None
    probability, threshold = verdict
    if probability < threshold:
        score = REJECTED_LOAN_CEILING * probability / threshold
    else:
        score = BORROWING_THRESHOLD + (1 - BORROWING_THRESHOLD) * (probability - threshold) / (1 - threshold)
    direct_record = direct_record or any(
        s.name == "zincir_kanıtı" and s.fired for s in getattr(borrowing, "signals", None) or []
    )
    if direct_record:
        score = max(score, SIGNAL_WEIGHTS["zincir_kanıtı"])
    return round(score, 3)


def _with_loan_rejection(hypothesis: Hypothesis, borrowing: Any) -> Hypothesis:
    """Birleştirici alıntıyı reddediyorsa (p < eşik) bu, miras yönünde zayıf kanıttır.

    Türkçe altının train bölümünde (438 kelime) sıralayıcı rekonstrüksiyonu
    kurulamayan mirasların 83/179'unda "KÖKENİ BELİRSİZ" diyordu; detektörün
    kendi hükmü ise "miras adayı"/"belirsiz"di. Taban düşüktür: tanıklı kök,
    rekonstrüksiyon güveni ya da modern türetme varsa onlar belirler.
    """
    verdict = _combiner_verdict(borrowing)
    if verdict is None or verdict[0] >= verdict[1] or hypothesis.score >= LOAN_REJECTED_INHERITED_FLOOR:
        return hypothesis
    probability, threshold = verdict
    return Hypothesis(
        kind=hypothesis.kind,
        claim=hypothesis.claim,
        score=LOAN_REJECTED_INHERITED_FLOOR,
        supporting=[
            f"alıntı birleştiricisi alıntıyı reddediyor (olasılık {probability:.2f} < eşik {threshold:.2f})",
            *hypothesis.supporting,
        ],
        against=hypothesis.against,
        not_evaluated=hypothesis.not_evaluated,
        rejected_because=hypothesis.rejected_because,
        counterfactual=hypothesis.counterfactual,
        detail={**hypothesis.detail, "loan_probability": round(probability, 4)},
    )


def _score_or(trained: float | None, fallback: float) -> float:
    return fallback if trained is None else trained


def _direct_loan_record(borrowed: Hypothesis, borrowing: Any) -> str:
    """Alıntı hipotezinin doğrudan sözlük tanıklığı (zincir kanıtı ya da
    kaynağın alıntı adımı) varsa açıklaması; yoksa ``""``."""
    if borrowed.detail.get("source_chain"):
        return borrowed.supporting[0] if borrowed.supporting else "kaynağın alıntı zinciri"
    for signal in getattr(borrowing, "signals", None) or []:
        if signal.name == "zincir_kanıtı" and signal.fired:
            return signal.explanation
    return ""


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
        attested_root: str = "",
        attested_root_source: str = "",
    ) -> RankedHypotheses:
        """Bütün makul kökenleri kurar ve sıralar.

        :param attested_before: kelimenin bilinen en eski tanıklama yılı;
            modern türetme hipotezini elemek için kullanılır.
        :param attested_root: kaynağın kelimenin KENDİSİ için verdiği
            Proto-Türkçe kök (sorgunun kendi miras kaydı ya da kendi Starling
            kökü; A-HVP'nin sınadığıyla aynı, bkz.
            ``iterative_hypothesis_engine._attested_proto_root``).
        """
        entries = entries or []
        borrowing = self.borrowing.detect(word, entries)
        reconstruction = self.reconstructor.reconstruct(word, entries, check_borrowing=False)

        borrowed = self._borrowed_hypothesis(borrowing)
        if not borrowing.donor_language:
            borrowed = self._with_source_loan(borrowed, borrowing, entries)
        inherited = self._inherited_hypothesis(reconstruction, borrowing)
        conflicts: list[str] = []
        if attested_root:
            inherited = self._with_attested_root(inherited, attested_root, attested_root_source)
            loan_record = _direct_loan_record(borrowed, borrowing)
            if loan_record:
                conflicts.append(
                    f"{attested_root_source or 'kaynak'} miras kök veriyor ({attested_root}), "
                    f"ama {loan_record}"
                )
        inherited = _with_loan_rejection(inherited, borrowing)
        hypotheses = [
            borrowed,
            inherited,
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

        ranked = RankedHypotheses(word=word, hypotheses=hypotheses, conflicts=conflicts)
        self._write_rejections(ranked)
        return ranked

    # -- tek tek hipotezler -------------------------------------------------

    @staticmethod
    def _borrowed_hypothesis(borrowing: Any) -> Hypothesis:
        from engine.nlp.borrowing_chain import language_name

        donor = language_name(borrowing.donor_language) if borrowing.donor_language else "?"
        supporting = [s.explanation for s in borrowing.signals if s.fired]
        attested_donor = _attested_donor_form(borrowing.chain)
        against = [s.explanation for s in borrowing.signals
                   if not s.fired and not s.evidence.get("no_data")
                   and not (attested_donor and s.name == DONOR_PROXIMITY_SIGNAL)]
        not_evaluated = [s.explanation for s in borrowing.signals
                         if not s.fired and s.evidence.get("no_data")]
        if attested_donor and any(
            s.name == DONOR_PROXIMITY_SIGNAL and not s.fired for s in borrowing.signals
        ):
            not_evaluated.append(_donor_proximity_moot(attested_donor))
        return Hypothesis(
            kind="borrowed",
            claim=f"ALINTI — {donor}" if borrowing.donor_language else "ALINTI",
            score=_score_or(_trained_loan_score(borrowing), borrowing.score),
            supporting=supporting,
            against=against,
            not_evaluated=not_evaluated,
            detail={
                "chain": borrowing.chain,
                "donor": borrowing.donor_language,
                "expected_if_inherited": borrowing.expected_if_inherited,
            },
        )

    @staticmethod
    def _with_source_loan(hypothesis: Hypothesis, borrowing: Any, entries: list[dict[str, Any]]) -> Hypothesis:
        """Kaynağın açık alıntı zincirini, sözlük alıntı kaydıyla AYNI ağırlıkla ekler.

        Sözlük indeksindeki alıntı kaydı skoru en az ``SIGNAL_WEIGHTS["zincir_kanıtı"]``
        yapar (model yoksa bu kadar katar); kaynağın "Alıntı" adımı aynı türden
        doğrudan tanıklamadır.
        """
        from engine.nlp.borrowing_chain import source_loan_step
        from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

        step = source_loan_step(entries)
        if step is None:
            return hypothesis
        donor_name = str(step.get("lang_name") or "?")
        evidence = f"kaynağın alıntı zinciri: {donor_name} {step.get('word')} ({step.get('source') or 'kaynak'})"
        return Hypothesis(
            kind="borrowed",
            claim=f"ALINTI — {donor_name}",
            score=_score_or(
                _trained_loan_score(borrowing, direct_record=True),
                round(min(1.0, borrowing.score + SIGNAL_WEIGHTS["zincir_kanıtı"]), 3),
            ),
            supporting=[evidence, *hypothesis.supporting],
            against=[
                a for a in hypothesis.against
                if "alıntı kaydı yok" not in a and a != DONOR_PROXIMITY_MISS
            ],
            not_evaluated=[
                *hypothesis.not_evaluated,
                *([_donor_proximity_moot(f"{donor_name} {step.get('word')}")]
                  if DONOR_PROXIMITY_MISS in hypothesis.against else []),
            ],
            detail={
                **hypothesis.detail,
                "chain": [f"Türkçe {borrowing.word}", f"{donor_name} {step.get('word')}"],
                "donor": donor_name,
                "source_chain": True,
            },
        )

    @staticmethod
    def _inherited_hypothesis(reconstruction: dict[str, Any], borrowing: Any) -> Hypothesis:
        # ⚠️ `anchor_fallback` bir REKONSTRÜKSİYON DEĞİLDİR: motor sorgu
        # biçmini aday olarak döndürmüştür çünkü ölçümde cevapsızlık mümkün
        # olan en kötü değeri alır. Onu miras hipotezi için kanıt saymak,
        # yapılmamış bir işi kanıt göstermek olurdu.
        if reconstruction.get("unattested_ban"):
            # Tanıksız kök yasağı: tanıklar birbiriyle uyumlu ama HİÇBİRİ
            # sözlükte yok. Aday ata biçim `detail`de görünür, iddia edilmez.
            return Hypothesis(
                kind="inherited",
                claim="MİRAS — tanıksız; ata biçim iddia edilmiyor",
                score=0.05,
                against=[
                    "tanık biçimlerinden hiçbiri sözlük indeksinde yok; "
                    "tanıkların yalnız birbiriyle uyumu kök iddiasına yetmez"
                ],
                detail={
                    "withheld_reconstruction": reconstruction.get("withheld_reconstruction", ""),
                    "confidence_badge": reconstruction.get("confidence_badge", ""),
                },
            )
        if reconstruction.get("method") == "anchor_fallback":
            return Hypothesis(
                kind="inherited",
                claim="MİRAS — karşılaştırmalı yöntem uygulanamadı",
                score=0.05,
                against=["akraba tanığı yok; ata biçim türetilemedi"],
            )
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
    def _with_attested_root(hypothesis: Hypothesis, root: str, source: str) -> Hypothesis:
        """Kaynakta tanıklı miras kök, miras hipotezinin DOĞRUDAN kanıtıdır.

        ⚠️ Eskiden miras hipotezi yalnız motorun kendi rekonstrüksiyonundan
        puanlanıyordu: Starling `katır` için *KAtɨr verirken tek tanıkla
        rekonstrüksiyon kurulamıyor, miras 0,05 alıyor ve en zayıf alıntı
        sinyali bile kazanıyordu. Ağırlık, sözlük alıntı kaydıyla
        (``SIGNAL_WEIGHTS["zincir_kanıtı"]``) AYNIDIR: ikisi de sözlükçünün
        kelimenin kendisi için verdiği doğrudan hükümdür. Eşitlikte alıntı
        önde kalır (liste sırası); çelişki ``conflicts``e yazılır.
        """
        from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

        evidence = f"kaynakta tanıklı miras kök: {root} ({source or 'kaynak'})"
        # Rekonstrüksiyonun kurulamaması ("akraba tanığı yok") kök tanıklıyken
        # karşı kanıt değil, yöntemin uygulanamamasıdır.
        moved = [] if hypothesis.score > 0.05 else list(hypothesis.against)
        return Hypothesis(
            kind="inherited",
            claim=hypothesis.claim if hypothesis.score > 0.05 else f"MİRAS — {root} (tanıklı kök)",
            score=round(max(hypothesis.score, SIGNAL_WEIGHTS["zincir_kanıtı"]), 3),
            supporting=[evidence, *hypothesis.supporting],
            against=[a for a in hypothesis.against if a not in moved],
            not_evaluated=[*hypothesis.not_evaluated, *moved],
            detail={**hypothesis.detail, "attested_root": root, "attested_root_source": source},
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
