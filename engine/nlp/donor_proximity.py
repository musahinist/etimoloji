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

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

#: Bu SCA mesafesinin altındaki en yakın verici maddesi **alıntı kanıtıdır**.
#:
#: Ayarlanabilir; ölçüm ``evaluation/borrowing_eval.py`` içinde
#: ayar yarısında yapılır (test yarısı görülmez).
DONOR_DISTANCE_THRESHOLD = 0.35

#: Sinyalin gücü bu mesafede sıfıra iner.
DONOR_DISTANCE_CEILING = 0.60

#: Şans benzerliği denetiminde kullanılacak kontrol kelimesi sayısı.
#:
#: ⚠️ Ham mesafe eşiği **verici havuzunun büyüklüğüne bağlıdır** ve bu
#: gizli bir bağımlılıktır. Sakha ölçütünde havuz 3 dil / 448.000 madde;
#: Türkçede 6 dil / 1.600.000 madde. Aynı 0,35 eşiği ikisinde aynı şeyi
#: ölçmez: büyük havuzda rastgele bir kelime bile yakın bir eşleşme bulur.
#: Ölçüldü — Türkçe ``göz`` Fransızca ``Grées``e 0,231 uzaklıkta çıkıyor.
#:
#: Kessler (2001) *The Significance of Word Lists*: gözlenen benzerlik
#: **aynı havuza karşı** kurulmuş bir null modele göre yorumlanmalıdır.
#: Kontrol kelimeleri aynı uzunlukta ve aynı aday havuzuna karşı ölçülür;
#: havuz büyüdükçe null da kayar ve eşik kendini ayarlar.
#: ⚠️ Sayı doğrudan ölçüm süresidir. Türkçe havuzu kavram başına ~400 madde;
#: kontrol başına tam tarama 400 SCA demek. 24 kontrol sorgu başına 0,4 s
#: ediyordu ve tam ölçüm saatler sürüyordu. 12 kontrol + ucuz ön eleme
#: (bkz. :data:`SCA_SHORTLIST`) aynı kararı ~10 kat hızlı veriyor.
CHANCE_CONTROL_COUNT = 12

#: Gözlenen mesafe, kontrol dağılımının bu yüzdeliğinden düşük olmalı.
CHANCE_MAX_PERCENTILE = 0.10

#: SCA hesaplanmadan önce ucuz düzenlenme uzaklığıyla kaç aday elenir?
#:
#: ⚠️ SCA aday başına ~39 µs; havuz 400 maddeyken sorgu başına 16 ms eder ve
#: kontrollerle çarpılınca ölçüm saatlere çıkar. Düz düzenlenme uzaklığı SCA
#: ile güçlü bağıntılıdır: en yakın 40 aday, SCA asgarisini pratikte her
#: zaman içerir. Ön eleme **yalnız hızdır**, karar ölçütü hâlâ SCA'dır.
SCA_SHORTLIST = 40


@dataclass(frozen=True)
class DonorMatch:
    """Verici sözlüğündeki en yakın madde."""

    lang_code: str
    word: str
    comparison: str
    gloss: str
    distance: float
    sense_constrained: bool
    #: Aynı havuza karşı ölçülen kontrol kelimelerinin kaçı bu kadar yakın?
    #: ``None`` ise şans denetimi yapılmadı.
    chance_percentile: float | None = None

    @property
    def beats_chance(self) -> bool:
        """Şans denetimi yapıldıysa geçti mi?"""
        if self.chance_percentile is None:
            return True
        return self.chance_percentile <= CHANCE_MAX_PERCENTILE

    @property
    def is_close(self) -> bool:
        return self.distance <= DONOR_DISTANCE_THRESHOLD and self.beats_chance

    def describe(self) -> str:
        kısıt = "anlam kısıtlı" if self.sense_constrained else "kısıtsız"
        şans = (
            f", şans %{100 * self.chance_percentile:.0f}"
            if self.chance_percentile is not None
            else ""
        )
        return (
            f"{self.lang_code} {self.word} ({self.comparison}) "
            f"SCA {self.distance:.3f} [{kısıt}{şans}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "donor_lang": self.lang_code,
            "donor_word": self.word,
            "donor_gloss": self.gloss,
            "sca_distance": round(self.distance, 4),
            "sense_constrained": self.sense_constrained,
            "chance_percentile": self.chance_percentile,
            "beats_chance": self.beats_chance,
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


#: Kontrol kelimeleri: gerçek Türki biçimlerden, uzunluğa göre gruplanmış.
#:
#: ⚠️ Rastgele harf dizisi kullanılmaz. Türkçe fonotaktiğine uymayan bir
#: dizi verici sözlüğüne de uzak düşer ve null'ı yapay olarak kolaylaştırır;
#: o zaman her gerçek kelime "anlamlı derecede yakın" çıkardı.
_CONTROL_SOURCE = (
    "kelime dünya insan zaman gerek bilgi güzel yürek deniz orman kalın "
    "sıcak yavaş karşı doğru sonra önce açık kapalı derin geniş uzak yakın "
    "beyaz siyah kırmızı sarı yeşil mavi büyük küçük genç yaşlı hızlı ağır "
    "tatlı acı tuzlu ekşi yumuşak sert temiz kirli dolu boş zengin fakir"
)


@lru_cache(maxsize=32)
def _controls(length: int, count: int = CHANCE_CONTROL_COUNT) -> tuple[str, ...]:
    """Verilen uzunlukta kontrol biçimleri.

    Tam uzunlukta yeterli kelime yoksa, mevcut kelimeler o uzunluğa
    **kırpılarak** üretilir: null'ın uzunluğu gözlemle aynı olmalı, yoksa
    mesafeler karşılaştırılabilir değildir.
    """
    words = [to_comparison_form(w) for w in _CONTROL_SOURCE.split()]
    exact = [w for w in words if len(w) == length]
    trimmed = [w[:length] for w in words if len(w) > length]
    pool = list(dict.fromkeys(exact + trimmed))
    return tuple(pool[:count])


def reset_cache() -> None:
    _pairwise.cache_clear()
    _index.cache_clear()
    # ⚠️ Mesafe önbelleği de temizlenmeli: LingPy'siz koşuyu ölçerken
    # önbellekte duran LingPy'li sonuç geri dönerdi.
    sca_distance.cache_clear()
    _controls.cache_clear()
    _control_distances.cache_clear()


def _cheap_distance(a: str, b: str) -> float:
    """Normalize Levenshtein — SCA öncesi ucuz ön eleme için."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def _shortlist(query: str, candidates: list[str], size: int = SCA_SHORTLIST) -> list[str]:
    """Ucuz mesafeye göre en yakın ``size`` adayı seçer."""
    if len(candidates) <= size:
        return candidates
    scored = sorted(candidates, key=lambda c: (_cheap_distance(query, c), c))
    return scored[:size]


def best_sca(query: str, candidates: list[str]) -> tuple[float, str]:
    """Aday havuzundaki en yakın biçim ve SCA mesafesi."""
    best_distance, best_form = 1.0, ""
    for candidate in _shortlist(query, candidates):
        distance = sca_distance(query, candidate)
        if distance < best_distance:
            best_distance, best_form = distance, candidate
    return best_distance, best_form


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
    max_candidates: int = 200,
    chance_control: bool = True,
) -> DonorMatch | None:
    """Verici sözlüklerindeki en yakın maddeyi bulur.

    :param sense: kelimenin anlamı. ``sense_constrained`` açıkken adaylar
        **yalnız** anlamı örtüşen verici maddeleridir (sabor'un yayınlanmış
        kurulumu).
    :param sense_constrained: kapatılırsa uzunluk penceresiyle kısıtsız
        aranır. ⚠️ Kısıtsız yol şans benzerliğine açıktır; ablasyon içindir.
    :param chance_control: aynı havuza karşı kontrol kelimeleriyle şans
        denetimi yapılsın mı? Bkz. :data:`CHANCE_CONTROL_COUNT`.
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

    by_form = {row["comparison"]: row for row in rows if row["comparison"]}
    distance, form = best_sca(comparison, list(by_form))
    best: DonorMatch | None = None
    if form:
        row = by_form[form]
        best = DonorMatch(
            lang_code=row["lang_code"],
            word=row["word"],
            comparison=form,
            gloss=row["gloss"] or "",
            distance=distance,
            sense_constrained=sense_constrained,
        )
    if best is None or not chance_control:
        return best
    # ⚠️ Şans denetimi YALNIZ eşiğin altındaki eşleşmeler için yapılır.
    # Zaten uzak olan bir eşleşme denetimden bağımsız olarak elenir; kontrol
    # hesaplamak ölçüm süresini 24 katına çıkarıp hiçbir kararı değiştirmez.
    if best.distance > DONOR_DISTANCE_THRESHOLD:
        return best

    pool = tuple(sorted(by_form))
    return replace(
        best,
        chance_percentile=_chance_percentile(best.distance, len(comparison), pool),
    )


@lru_cache(maxsize=20000)
def _control_distances(length: int, pool: tuple[str, ...]) -> tuple[float, ...]:
    """Kontrol kelimelerinin bu havuza en yakın mesafeleri.

    Havuza göre önbelleklenir: aynı kavramın birden çok maddesi aynı havuzu
    görür ve kontroller yeniden hesaplanmaz.
    """
    candidates = list(pool)
    return tuple(best_sca(control, candidates)[0] for control in _controls(length))


def _chance_percentile(observed: float, length: int, pool: tuple[str, ...]) -> float | None:
    """Kontrol kelimelerinin kaçı gözlenen kadar yakın? (Kessler 2001)

    ``None`` döner: uygun kontrol bulunamadıysa denetim **yapılmamış**
    sayılır ve sinyal engellenmez — ölçülmemiş bir denetimi geçilmiş gibi
    saymak da geçilmemiş gibi saymak da yanlış olurdu.
    """
    if len(_controls(length)) < 8 or not pool:
        return None
    distances = _control_distances(length, pool)
    return sum(1 for d in distances if d <= observed) / len(distances)


def proximity_strength(match: DonorMatch | None) -> float:
    """Eşleşmeyi ``[0, 1]`` aralığında bir sinyal gücüne çevirir.

    Eşiğin altındaki mesafe tam güç, tavanın üstü sıfır; arası doğrusal.
    Keskin bir eşik yerine yumuşak geçiş kullanılır: 0,349 ile 0,351'in
    kararı ters çevirmesi için bir sebep yok.
    """
    if match is None:
        return 0.0
    if not match.beats_chance:
        return 0.0
    if match.distance <= DONOR_DISTANCE_THRESHOLD:
        return 1.0
    if match.distance >= DONOR_DISTANCE_CEILING:
        return 0.0
    span = DONOR_DISTANCE_CEILING - DONOR_DISTANCE_THRESHOLD
    return round((DONOR_DISTANCE_CEILING - match.distance) / span, 4)
