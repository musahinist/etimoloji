"""
Açıklamalı alıntı tespiti — "neden alıntı?" sorusuna cevap veren katman.

Mevcut sistemler (seabor, PyBor) **ikili sınıflandırıcıdır**: "alıntı" veya
"miras" der, gerekçe vermez. Buradaki fark, her kararın adı konmuş
kanıtlara dayanması ve **miras olsaydı beklenen biçmin** hesaplanmasıdır::

    kitap  →  ALINTI
      · Türki dillerde biçim neredeyse aynı (ç=0,95): miras kelimeler
        düzenli ses farkları gösterir, bu göstermiyor
      · söz başı /k/ + söz içi /t/ Oğuz ötümlüleşmesinden geçmemiş;
        miras olsaydı beklenen Türkçe biçim: *gidap
      · zincir: Türkçe kitap ← Osmanlıca كتاب ← Arapça كِتَاب

Ölçülmüş gerekçe: negatif kontrol bataryasında alıntı tuzakları (kitap,
duvar, çorap, sabun, pencere, çay) **%100 yanlış-pozitif** veriyordu — motor
hepsini rekonstrükte edilebilir sayıyordu.

Dört bağımsız sinyal kullanılır; hiçbiri tek başına karar vermez:

============================  =============================================
``zincir_kanıtı``             sözlükte tanıklanmış verici dil ve yol
``fonotaktik_ihlal``          Proto-Türkçe'de bulunmayan ses veya dizim
``ses_kanunu_ihlali``         beklenen refleks tutmuyor (özgün katkı)
``değişimsiz_yayılım``        bütün dillerde neredeyse aynı biçim
``verici_yakınlığı``          verici dil sözlüğünde aynı kavramın karşılığı
                              fonetik olarak neredeyse aynı (sabor)
============================  =============================================
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.donor_proximity import nearest_donor, proximity_strength
from engine.nlp.proto_phonology import PROHIBITED_INITIALS
from engine.utils.orthography import to_comparison_form
from engine.utils.phonotactics import VOWELS, has_vowel_harmony

logger = get_logger(__name__)

#: Ağırlıklar. Zincir kanıtı en güçlüsüdür çünkü doğrudan tanıklamadır;
#: ötekiler dolaylı göstergedir.
#:
#: ``verici_yakınlığı`` alanın **ölçülmüş en güçlü tek sinyalidir**
#: (Miller & List 2023, ``sabor``: F1 0,806, kesinlik 0,931). Zincir
#: kanıtından sonra en yüksek ağırlığı alır; ondan düşük olmasının sebebi
#: zincirin DOĞRUDAN tanıklama, bunun ise çıkarım olmasıdır.
SIGNAL_WEIGHTS: dict[str, float] = {
    "zincir_kanıtı": 0.40,
    "verici_yakınlığı": 0.25,
    "ses_kanunu_ihlali": 0.15,
    "fonotaktik_ihlal": 0.15,
    "değişimsiz_yayılım": 0.05,
}

#: Bu eşiğin üstünde kelime **alıntı olarak raporlanır**.
BORROWING_THRESHOLD = 0.45

#: Bu eşiğin üstünde miras rekonstrüksiyonu **hiç yapılmaz**.
#:
#: ⚠️ İki ayrı eşik gerekiyor çünkü zincir sinyali tek başına EŞADLILARA
#: takılabiliyor. Ölçüldü: tek eşikle altın standarttaki 400 maddenin 26'sı
#: yanlışlıkla engellendi (`bil`, `ben`, `bär`, `dal` — hepsi miras, ama
#: sözlükte aynı yazılışta bir alıntı madde de var). Rekonstrüksiyon
#: doğruluğu %22,3'ten %20,0'a düşüyordu.
#:
#: Engelleme yalnız zincir kanıtı BAŞKA bir sinyalle desteklendiğinde
#: yapılır. Arada kalan kelimeler "alıntı" diye raporlanır ama motor yine de
#: miras hipotezini kurar ve kullanıcı iki okumayı da görür.
BLOCK_THRESHOLD = 0.55

#: Bu benzerlik oranının üstündeki yayılım şüphelidir. Miras kelimeler
#: bin yılda düzenli ses farkları biriktirir; birikmemişse yayılım yenidir.
UNIFORMITY_SUSPICION = 0.85


@dataclass
class Signal:
    """Tek bir kanıt kalemi."""

    name: str
    fired: bool
    strength: float
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.name,
            "fired": self.fired,
            "strength": round(self.strength, 3),
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class BorrowingVerdict:
    """Bir kelime için alıntı kararı ve tam gerekçesi."""

    word: str
    score: float
    signals: list[Signal] = field(default_factory=list)
    expected_if_inherited: str = ""
    chain: list[str] = field(default_factory=list)
    donor_language: str = ""

    @property
    def is_borrowed(self) -> bool:
        return self.score >= BORROWING_THRESHOLD

    @property
    def blocks_inherited_reconstruction(self) -> bool:
        """Kanıt, miras rekonstrüksiyonunu hiç denememeyi haklı çıkarıyor mu?"""
        return self.score >= BLOCK_THRESHOLD

    @property
    def verdict(self) -> str:
        if self.score >= BORROWING_THRESHOLD:
            return "alıntı"
        if self.score >= BORROWING_THRESHOLD / 2:
            return "belirsiz"
        return "miras adayı"

    def explain(self) -> str:
        """İnsan-okunur gerekçe — çıktının asıl değeri budur."""
        lines = [f"{self.word} → {self.verdict.upper()} (skor {self.score:.2f})"]
        for signal in self.signals:
            if signal.fired:
                lines.append(f"  · {signal.explanation}")
        if self.expected_if_inherited:
            lines.append(
                f"  · miras olsaydı beklenen Türkçe biçim: {self.expected_if_inherited}"
            )
        if self.chain:
            lines.append(f"  · zincir: {' ← '.join(self.chain)}")
        if not any(s.fired for s in self.signals):
            lines.append("  · alıntı göstergesi bulunamadı")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "verdict": self.verdict,
            "is_borrowed": self.is_borrowed,
            "score": round(self.score, 3),
            "donor_language": self.donor_language,
            "chain": self.chain,
            "expected_if_inherited": self.expected_if_inherited,
            "signals": [s.as_dict() for s in self.signals],
            "explanation": self.explain(),
        }


class BorrowingDetector:
    """Dört sinyalli, gerekçeli alıntı tespiti."""

    def __init__(self, index: Any = None, predictor: Any = None):
        self._index = index
        self._predictor = predictor

    @property
    def index(self) -> Any:
        if self._index is None:
            from engine.db.lexicon_index import LexiconIndex

            self._index = LexiconIndex()
        return self._index

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            from engine.nlp.cognate_prediction import CognatePredictor

            self._predictor = CognatePredictor()
        return self._predictor

    # -- sinyaller ----------------------------------------------------------

    def _chain_signal(self, word: str, lang: str) -> tuple[Signal, list[str], str]:
        """Sözlükte tanıklanmış alıntı kaydı var mı?

        En güçlü sinyal, çünkü dolaylı gösterge değil doğrudan tanıklamadır.
        Zincir çok halkalı olabilir: Türkçe ← Osmanlıca ← Arapça.
        """
        from engine.nlp.borrowing_chain import language_name

        if not getattr(self.index, "exists", False):
            return (
                Signal("zincir_kanıtı", False, 0.0, "sözlük indeksi yok"),
                [],
                "",
            )
        rows = self.index.lookup(word, languages=[lang], limit=10)
        borrowed = [r for r in rows if r.get("origin") == "alıntı"]
        inherited = [r for r in rows if r.get("origin") == "miras"]

        # ⚠️ EŞADLILIK. Aynı yazılışta hem miras hem alıntı madde olabilir:
        # Türkçe `su` (miras, "water") ile Fransızca kökenli bir `su` maddesi
        # sözlükte yan yanadır. Yalnız "alıntı kaydı var mı" diye bakmak,
        # miras kelimeyi alıntı ilan eder. Kanıt gücü, alıntı kayıtlarının
        # ORANIYLA ölçülür; miras kayıt varsa sinyal zayıflar.
        if borrowed and inherited:
            share = len(borrowed) / (len(borrowed) + len(inherited))
            if share < 0.6:
                return (
                    Signal(
                        "zincir_kanıtı",
                        False,
                        share,
                        (
                            f"eşadlılık: sözlükte {len(inherited)} miras, "
                            f"{len(borrowed)} alıntı kaydı var — kanıt belirsiz"
                        ),
                        {"borrowed_entries": len(borrowed), "inherited_entries": len(inherited)},
                    ),
                    [],
                    "",
                )
        if not borrowed:
            return (
                Signal(
                    "zincir_kanıtı",
                    False,
                    0.0,
                    "sözlükte alıntı kaydı yok",
                    {"checked_entries": len(rows)},
                ),
                [],
                "",
            )
        row = borrowed[0]
        donor = str(row.get("donor_lang") or "")
        chain = [
            f"{language_name(lang)} {row['word']}",
            f"{language_name(donor)} {row.get('donor_form') or '?'}",
        ]
        return (
            Signal(
                "zincir_kanıtı",
                True,
                1.0,
                f"sözlükte alıntı olarak tanıklanmış: verici dil {language_name(donor)}",
                {"donor_lang": donor, "donor_form": row.get("donor_form", "")},
            ),
            chain,
            donor,
        )

    @staticmethod
    def _phonotactic_signal(word: str) -> Signal:
        """Proto-Türkçe'de bulunmayan ses veya dizim var mı?"""
        form = to_comparison_form(word)
        violations: list[str] = []
        if not form:
            return Signal("fonotaktik_ihlal", False, 0.0, "biçim çözümlenemedi")

        if form[0] in PROHIBITED_INITIALS:
            violations.append(f"Proto-Türkçe'de söz başı *{form[0]}- bulunmaz")
        if len(form) >= 2 and form[0] not in VOWELS and form[1] not in VOWELS:
            violations.append("söz başı ünsüz kümesi — Türkçede bulunmaz")
        vowels = [ch for ch in form if ch in VOWELS]
        if len(vowels) >= 2 and not has_vowel_harmony(form):
            violations.append("ünlü uyumu ihlali")

        strength = min(1.0, len(violations) / 2)
        return Signal(
            "fonotaktik_ihlal",
            bool(violations),
            strength,
            "; ".join(violations) if violations else "fonotaktik olarak Türkçeye uygun",
            {"violations": violations},
        )

    def _sound_law_signal(self, word: str, witnesses: dict[str, str]) -> tuple[Signal, str]:
        """**Özgün katkı:** miras olsaydı beklenen biçim tutuyor mu?

        Miras bir kelime, akraba dillerdeki biçimleriyle **düzenli** ses
        denklikleri gösterir. Öğrenilmiş denklik tablolarıyla her tanıktan
        Türkçe biçim tahmin edilir; tahminler gerçek biçme uymuyorsa kelime
        ses kanunlarının işlediği dönemde dilde yoktu demektir.
        """
        actual = to_comparison_form(word)
        if not actual or len(witnesses) < 2:
            return (
                Signal("ses_kanunu_ihlali", False, 0.0, "yeterli tanık yok"),
                "",
            )

        expectations: list[str] = []
        for source_lang, source_form in sorted(witnesses.items()):
            if source_lang == "tr":
                continue
            prediction = self.predictor.predict(source_form, source_lang, "tr")
            if prediction.form and prediction.confidence > 0:
                expectations.append(prediction.form)
        if not expectations:
            return (
                Signal("ses_kanunu_ihlali", False, 0.0, "denklik tablosu tahmin üretmedi"),
                "",
            )

        expected, votes = Counter(expectations).most_common(1)[0]
        agreement = sum(1 for e in expectations if e == actual) / len(expectations)

        # ⚠️ Sinyal ancak beklenti KENDİ İÇİNDE tutarlıysa anlamlıdır.
        # Tek bir gürültülü tahmin ("Çuvaşça şıv -> Türkçe şa") miras bir
        # kelimeyi alıntı ilan edebilir. Üç koşul birden aranır:
        #   1. en az üç tanıktan tahmin üretilmiş olmalı
        #   2. tahminlerin çoğunluğu AYNI biçimde birleşmeli
        #   3. o ortak beklenti gerçek biçimden farklı olmalı
        consensus = votes / len(expectations)
        fired = (
            len(expectations) >= 3
            and consensus >= 0.5
            and agreement < 0.34
            and expected != actual
        )
        return (
            Signal(
                "ses_kanunu_ihlali",
                fired,
                1.0 - agreement,
                (
                    f"akraba biçimlerden beklenen Türkçe refleks {expected!r}, "
                    f"gerçek biçim {actual!r} — ses kanunları işlememiş"
                    if fired
                    else f"beklenen refleks tutuyor (uyum {agreement:.2f})"
                ),
                {"expected": expected, "actual": actual, "agreement": round(agreement, 3)},
            ),
            expected if fired else "",
        )

    @staticmethod
    def _uniformity_signal(witnesses: dict[str, str]) -> Signal:
        """Bütün dillerde neredeyse aynı biçim — yeni yayılım göstergesi.

        Miras kelimeler bin yılda düzenli ses farkları biriktirir
        (``göz ~ көз ~ küz ~ куҫ``). Farklar birikmemişse yayılım yenidir
        ve büyük olasılıkla ortak bir verici dilden gelmiştir
        (``kitap ~ kitap ~ kitap ~ kitob``).
        """
        forms = [to_comparison_form(f) for f in witnesses.values() if f]
        forms = [f for f in forms if f]
        if len(forms) < 3:
            return Signal("değişimsiz_yayılım", False, 0.0, "yeterli tanık yok")

        from engine.evaluation.metrics import normalized_edit_distance

        pairs = [
            1.0 - normalized_edit_distance(a, b)
            for i, a in enumerate(forms)
            for b in forms[i + 1 :]
        ]
        similarity = sum(pairs) / len(pairs) if pairs else 0.0
        fired = similarity >= UNIFORMITY_SUSPICION
        return Signal(
            "değişimsiz_yayılım",
            fired,
            max(0.0, (similarity - UNIFORMITY_SUSPICION) / (1 - UNIFORMITY_SUSPICION))
            if fired
            else 0.0,
            (
                f"Türki dillerde biçim neredeyse aynı (benzerlik {similarity:.2f}): "
                f"miras kelimeler düzenli ses farkları gösterir, bu göstermiyor"
                if fired
                else f"diller arası düzenli farklar var (benzerlik {similarity:.2f})"
            ),
            {"similarity": round(similarity, 3), "n_forms": len(forms)},
        )

    # -- karar --------------------------------------------------------------

    @staticmethod
    def _donor_signal(word: str, sense: str, donors: list[str] | None) -> Signal:
        """Verici dil sözlüğüne fonetik yakınlık (sabor, Miller & List 2023).

        ⚠️ Mesafe **SCA**'dır, düz Levenshtein değil: Sakha Rusça ``stol``u
        ``ostuol`` yapar; düz uzaklık 0,50 verip eşiğin üstünde kalır, SCA
        0,216 verir.

        ⚠️ Arama **anlam kısıtlıdır**. Kısıtsız arama 440.910 maddelik Rusça
        sözlüğe yayılır ve şans benzerliğine açılır.
        """
        comparison = to_comparison_form(word)
        match = nearest_donor(comparison, sense, languages=donors)
        strength = proximity_strength(match)
        if match is None or strength <= 0.0:
            return Signal(
                "verici_yakınlığı",
                False,
                0.0,
                "verici sözlüğünde yakın karşılık yok",
            )
        return Signal(
            "verici_yakınlığı",
            True,
            strength,
            f"verici sözlüğünde aynı kavramın karşılığı fonetik olarak yakın: "
            f"{match.describe()}",
            match.as_dict(),
        )

    def detect(
        self,
        word: str,
        entries: list[dict[str, Any]] | None = None,
        *,
        lang: str = "tr",
        sense: str = "",
        donors: list[str] | None = None,
    ) -> BorrowingVerdict:
        """Bir kelimenin alıntı olup olmadığına gerekçeli karar verir.

        :param sense: kelimenin anlamı. Verici yakınlığı sinyali bunsuz
            çalışmaz (anlam kısıtı yayınlanmış kurulumun parçasıdır).
        :param donors: bakılacak verici dil kodları. ``None`` ise hepsi.
        """
        witnesses = {
            e["lang_code"]: e.get("word", "")
            for e in (entries or [])
            if e.get("lang_code") and e.get("word")
        }
        witnesses.setdefault(lang, word)

        chain_signal, chain, donor = self._chain_signal(word, lang)
        phonotactic = self._phonotactic_signal(word)
        sound_law, expected = self._sound_law_signal(word, witnesses)
        uniformity = self._uniformity_signal(witnesses)
        donor_proximity = self._donor_signal(word, sense, donors)

        signals = [chain_signal, phonotactic, sound_law, uniformity, donor_proximity]
        score = sum(
            SIGNAL_WEIGHTS[signal.name] * signal.strength for signal in signals if signal.fired
        )
        return BorrowingVerdict(
            word=word,
            score=round(score, 3),
            signals=signals,
            expected_if_inherited=expected,
            chain=chain,
            donor_language=donor,
        )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Açıklamalı alıntı tespiti")
    ap.add_argument("words", nargs="*", default=[])
    ap.add_argument("--controls", action="store_true", help="negatif kontrol bataryasını koş")
    args = ap.parse_args()

    detector = BorrowingDetector()

    if args.controls:
        from engine.evaluation.negative_controls import ALL_BATTERIES

        for name, items in ALL_BATTERIES.items():
            print(f"\n--- {name}")
            for item in items:
                entries = [{"lang_code": c, "word": w} for c, w in item.witnesses]
                verdict = detector.detect(item.query, entries)
                if name == "alinti_tuzagi":
                    expected = verdict.is_borrowed
                elif name == "eşadlı":
                    # Eşadlıda doğru cevap KESİN KARAR DEĞİL, belirsizliktir.
                    expected = not verdict.blocks_inherited_reconstruction
                else:
                    expected = not verdict.is_borrowed
                mark = "OK " if expected else "!! "
                print(f"  {mark}{verdict.word:10} {verdict.verdict:12} {verdict.score:.2f}")
        return 0

    for word in args.words or ["kitap", "göz", "çorap", "deniz", "sabun", "yol"]:
        print(detector.detect(word).explain())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
