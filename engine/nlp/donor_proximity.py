"""
Verici dil yakınlığı — alıntı tespitinin ölçülmüş en güçlü tek sinyali.

Miller & List (2023, EACL, ``sabor``) ölçtü: bir kelimenin verici dil
sözlüğündeki **aynı kavramın** karşılığına SCA mesafesi tek başına
F1 **0,806**, kesinlik **0,931** veriyor. Bizim motorumuz WOLD/Sakha'da
F 0,385 — "her şeye alıntı de" diyen trivial sistemin (0,464) altında.

Sezgi doğrudan gözlenebilir::

    Sakha ostuol   ~ Rusça стол   (stol)   SCA 0,216   "masa"
    Sakha muora    ~ Rusça море   (more)   SCA 0,067   "deniz"
    Sakha mıla     ~ Rusça мыло   (mılo)   SCA 0,033   "sabun"
    Sakha bagana   ~ Rusça çuçka           SCA 0,630   (ilgisiz)

⚠️ **Mesafe SCA'dır, düz Levenshtein değil.** Sakha Rusça ``stol``u
``ostuol`` yapar (öntüreme ünlü + ikizünlü); düz düzenlenme uzaklığı 3/6 =
0,50 verir ve eşiğin üstünde kalır. SCA ses sınıflarıyla çalıştığı için
aynı çift 0,216'dır. LingPy yoksa modül **devre dışı kalır** — düz
Levenshtein'a düşmek, ölçülmemiş bir sinyali ölçülmüş gibi sunmak olurdu.

⚠️ **Anlam kısıtı yayınlanmış kurulumun parçasıdır.** Kısıtsız arama şans
benzerliğine açıktır: 440.910 maddelik Rusça sözlükte kısa bir biçme
benzeyen bir şey her zaman bulunur. Kısıtın bedeli de ölçülmüştür — sabor'da
kaçan alıntıların **%45'i** tam bu kısıttan gelir.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Bu SCA mesafesinin altındaki en yakın verici maddesi **alıntı kanıtıdır**.
#:
#: Ayarlanabilir; ölçüm ``evaluation/borrowing_eval.py`` içinde
#: ayar yarısında yapılır (test yarısı görülmez).
DONOR_DISTANCE_THRESHOLD = 0.35

#: Sinyalin gücü bu mesafede sıfıra iner.
DONOR_DISTANCE_CEILING = 0.60


@dataclass(frozen=True)
class DonorMatch:
    """Verici sözlüğündeki en yakın madde."""

    lang_code: str
    word: str
    comparison: str
    gloss: str
    distance: float
    sense_constrained: bool

    @property
    def is_close(self) -> bool:
        return self.distance <= DONOR_DISTANCE_THRESHOLD

    def describe(self) -> str:
        kısıt = "anlam kısıtlı" if self.sense_constrained else "kısıtsız"
        return (
            f"{self.lang_code} {self.word} ({self.comparison}) "
            f"SCA {self.distance:.3f} [{kısıt}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "donor_lang": self.lang_code,
            "donor_word": self.word,
            "donor_gloss": self.gloss,
            "sca_distance": round(self.distance, 4),
            "sense_constrained": self.sense_constrained,
        }


@lru_cache(maxsize=1)
def _pairwise() -> Any:
    """LingPy'nin ``Pairwise``ı — yoksa ``None``.

    ⚠️ Düz Levenshtein'a **düşülmez**. Yayınlanmış sayı (F1 0,806) SCA ile
    ölçülmüştür; başka bir mesafeyle o sayıyı iddia etmek olurdu.
    """
    try:
        from lingpy import Pairwise
    except ImportError:
        logger.info("LingPy yok — verici yakınlığı sinyali devre dışı")
        return None
    return Pairwise


@lru_cache(maxsize=1)
def _index() -> Any:
    from engine.db.donor_index import DonorIndex

    return DonorIndex()


def reset_cache() -> None:
    _pairwise.cache_clear()
    _index.cache_clear()
    # ⚠️ Mesafe önbelleği de temizlenmeli: LingPy'siz koşuyu ölçerken
    # önbellekte duran LingPy'li sonuç geri dönerdi.
    sca_distance.cache_clear()


@lru_cache(maxsize=200000)
def sca_distance(a: str, b: str) -> float:
    """İki biçim arasındaki SCA (ses sınıfı tabanlı) mesafesi.

    LingPy yoksa ``1.0`` döner — yani "kanıt yok", yanlış bir kanıt değil.
    """
    pairwise = _pairwise()
    if pairwise is None or not a or not b:
        return 1.0
    try:
        analysis = pairwise(a, b)
        analysis.align(distance=True, model="sca")
        return float(analysis.alignments[0][2])
    except Exception:  # LingPy bilinmeyen sesle çökebiliyor
        logger.debug("SCA hesaplanamadı: %r ~ %r", a, b, exc_info=True)
        return 1.0


def nearest_donor(
    comparison: str,
    sense: str = "",
    *,
    languages: list[str] | None = None,
    sense_constrained: bool = True,
    max_candidates: int = 400,
) -> DonorMatch | None:
    """Verici sözlüklerindeki en yakın maddeyi bulur.

    :param sense: kelimenin anlamı. ``sense_constrained`` açıkken adaylar
        **yalnız** anlamı örtüşen verici maddeleridir (sabor'un yayınlanmış
        kurulumu).
    :param sense_constrained: kapatılırsa uzunluk penceresiyle kısıtsız
        aranır. ⚠️ Kısıtsız yol şans benzerliğine açıktır; ablasyon içindir.
    """
    index = _index()
    if _pairwise() is None or not comparison or not getattr(index, "exists", False):
        return None

    if sense_constrained:
        rows = index.by_sense(sense, languages=languages, limit=max_candidates)
    else:
        rows = index.candidates(comparison, languages=languages, limit=max_candidates)
    if not rows:
        return None

    best: DonorMatch | None = None
    for row in rows:
        candidate = row["comparison"]
        if not candidate:
            continue
        distance = sca_distance(comparison, candidate)
        if best is None or distance < best.distance:
            best = DonorMatch(
                lang_code=row["lang_code"],
                word=row["word"],
                comparison=candidate,
                gloss=row["gloss"] or "",
                distance=distance,
                sense_constrained=sense_constrained,
            )
    return best


def proximity_strength(match: DonorMatch | None) -> float:
    """Eşleşmeyi ``[0, 1]`` aralığında bir sinyal gücüne çevirir.

    Eşiğin altındaki mesafe tam güç, tavanın üstü sıfır; arası doğrusal.
    Keskin bir eşik yerine yumuşak geçiş kullanılır: 0,349 ile 0,351'in
    kararı ters çevirmesi için bir sebep yok.
    """
    if match is None:
        return 0.0
    if match.distance <= DONOR_DISTANCE_THRESHOLD:
        return 1.0
    if match.distance >= DONOR_DISTANCE_CEILING:
        return 0.0
    span = DONOR_DISTANCE_CEILING - DONOR_DISTANCE_THRESHOLD
    return round((DONOR_DISTANCE_CEILING - match.distance) / span, 4)
