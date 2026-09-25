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

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger
from engine.utils.edit_distance import normalized_edit_distance
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
    #: Rampa şans denetimi (X4 A2): en yakın maddenin DİLİNİN null'ı
    #: (kontrol kelimelerinin o dilin süzülmüş havuzuna medyan uzaklığı).
    #: ``None`` = denetim yapılmadı.
    ramp_null: float | None = None

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
    _attribution_controls.cache_clear()
    _null_distance.cache_clear()
    _strength_null.cache_clear()
    _control_profile.cache_clear()
    _monget_entries.cache_clear()


#: Normalize Levenshtein — SCA öncesi ucuz ön eleme için.
_cheap_distance = normalized_edit_distance


def _shortlist(query: str, candidates: list[str], size: int = SCA_SHORTLIST) -> list[str]:
    """Ucuz mesafeye göre en yakın ``size`` adayı seçer."""
    if len(candidates) <= size:
        return candidates
    scored = sorted(candidates, key=lambda c: (_cheap_distance(query, c), c))
    return scored[:size]


def best_sca(query: str, candidates: list[str]) -> tuple[float, str]:
    """Aday havuzundaki en yakın biçim ve SCA mesafesi."""
    return _best(query, candidates, sca_distance)


def _best(
    query: str, candidates: list[str], distance_fn: Any, initial: float = 1.0
) -> tuple[float, str]:
    best_distance, best_form = initial, ""
    for candidate in _shortlist(query, candidates):
        distance = distance_fn(query, candidate)
        if distance < best_distance:
            best_distance, best_form = distance, candidate
    return best_distance, best_form


#: Alıntı GÜCÜ adımının ses mesafesi: ``sca`` (üretim), ``mean`` (SCA ile
#: ASJP-PMI'nin ortalaması) ya da ``pmi``. Eşikler SCA ölçeğindedir.
#:
#: ⚠️ ÖLÇÜLDÜ, ön-kayıtlı, OLUMSUZ (2026-09-25). ``mean`` + eşik 0,60 / tavan
#: 0,85 (WOLD ayar yarısında seçildi: yalnız-yakınlık F 0,565 -> 0,585).
#: Rapor yarısı, bir kez::
#:
#:                        SCA (üretim)             mean
#:     WOLD engine_trained F 0,6582                0,6721
#:     WOLD yalnız yakınlık  F 0,6238, doğr. 0,7945  F 0,6339, doğr. 0,8062
#:     doğr. farkı (motor − yakınlık) −0,004 [−0,023, +0,016]  −0,016 [−0,039, +0,008]
#:     Türkçe engine_trained F 0,8873              0,8990
#:
#: Ölçüt 1 (GA sıfırı dışlasın) TUTMADI, ölçüt 2 (Türkçe F düşmesin) tuttu;
#: üretim değişmedi. Yalnız PMI ayar yarısında SCA'nın altında (F 0,515).
#:
#: ⚠️ Bağımsız Türk dilleri arası altında yeniden sınandı (X2, ön-kayıt
#: ``data/cache/work/xtr/PREREG_pmi.md``, R1 bir kez, n=684, kör indeks +
#: zincir kapalı): engine_trained F 0,7654 -> 0,7868, fark +0,021
#: [−0,003, +0,045]; doğruluk +0,025; verici tanıma 0,573 -> 0,563.
#: Birincil ölçüt (GA alt ucu > 0) TUTMADI -> sonuç BELİRSİZ (güç yetersiz:
#: MDE 0,038 önceden yazılmıştı). C1 kapandı; üretim ``sca``.
STRENGTH_DISTANCE = "sca"


def _mean_distance(a: str, b: str) -> float:
    from engine.nlp.pmi_distance import pmi_distance

    return (sca_distance(a, b) + min(1.0, pmi_distance(a, b))) / 2


def _strength_best(query: str, candidates: list[str]) -> tuple[float, str]:
    if STRENGTH_DISTANCE == "mean":
        return _best(query, candidates, _mean_distance)
    if STRENGTH_DISTANCE == "pmi":
        from engine.nlp.pmi_distance import pmi_distance

        return _best(query, candidates, lambda a, b: min(1.0, pmi_distance(a, b)))
    return best_sca(query, candidates)


def label_distance(a: str, b: str) -> float:
    """Etiket adımının mesafesi — bkz. :data:`LABEL_DISTANCE`."""
    if LABEL_DISTANCE == "sca":
        return sca_distance(a, b)
    from engine.nlp.pmi_distance import available, pmi_distance

    if not available():
        return sca_distance(a, b)
    if LABEL_DISTANCE == "pmi":
        return pmi_distance(a, b)
    return (sca_distance(a, b) + pmi_distance(a, b)) / 2


def best_label(query: str, candidates: list[str]) -> tuple[float, str]:
    """Aday havuzundaki en yakın biçim, etiket mesafesiyle."""
    # ⚠️ PMI mesafesi 1'i aşabilir (negatif PMI); başlangıç sonsuz olmalı,
    # yoksa ilgisiz adaylar "aday yok" sayılır ve dil etiketten düşer.
    return _best(query, candidates, label_distance, float("inf") if LABEL_DISTANCE != "sca" else 1.0)


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


def _sense_filter(sense_filter: Any) -> Any:
    """Açık verilmediyse ``ETY_DONOR_SENSE_FILTER`` bayrağı (varsayılan kapalı).

    ``False`` bayraktan bağımsız olarak kapatır. Bkz. :mod:`engine.nlp.sense_match`.
    """
    if sense_filter is False:
        return None
    if sense_filter is not None:
        return sense_filter
    from engine.nlp.sense_match import filter_from_env

    return filter_from_env()


#: Rampa şans denetiminin varsayılanı (``ETY_DONOR_RAMP_CHANCE`` verilmezse).
#: Bkz. ``data/cache/work/xtr/PREREG_x4.md``; sonuç ``donor_index.CLEAN_DEFAULT``
#: notunda (A2: mekanizma tuttu, birincil F ölçütü tutmadı). Varsayılan KAPALI.
RAMP_CHANCE_DEFAULT = False


def ramp_chance_enabled() -> bool:
    """Rampa (eşik–tavan arası) eşleşmesi dilin null'ını geçmeli mi? (X4 A2)

    Null, verici etiketi adımındakiyle aynıdır (:func:`_null_distance`,
    1facc40): kontrol kelimelerinin en yakın maddenin dilinin havuzuna medyan
    uzaklığı. Parametre yok.
    """
    import os

    value = os.environ.get("ETY_DONOR_RAMP_CHANCE", "").strip().lower()
    if value in ("1", "on", "true", "yes"):
        return True
    if value in ("0", "off", "false", "no"):
        return False
    return RAMP_CHANCE_DEFAULT


def nearest_donor(
    comparison: str,
    sense: str = "",
    *,
    languages: list[str] | None = None,
    sense_constrained: bool = True,
    max_candidates: int = 200,
    chance_control: bool = True,
    sense_filter: Any = None,
) -> DonorMatch | None:
    """Verici sözlüklerindeki en yakın maddeyi bulur.

    :param sense: kelimenin anlamı. ``sense_constrained`` açıkken adaylar
        **yalnız** anlamı örtüşen verici maddeleridir (sabor'un yayınlanmış
        kurulumu).
    :param sense_constrained: kapatılırsa uzunluk penceresiyle kısıtsız
        aranır. ⚠️ Kısıtsız yol şans benzerliğine açıktır; ablasyon içindir.
    :param chance_control: aynı havuza karşı kontrol kelimeleriyle şans
        denetimi yapılsın mı? Bkz. :data:`CHANCE_CONTROL_COUNT`.
    :param sense_filter: anlam süzgeci (:class:`engine.nlp.sense_match.SenseFilter`);
        ``None`` = ``ETY_DONOR_SENSE_FILTER`` bayrağı, ``False`` = kapalı.
        Süzgeç ``by_sense`` adaylarına uygulanır; şans denetimi SÜZÜLMÜŞ
        havuzla kurulur.
    """
    index = _index()
    if _pairwise() is None or not comparison or not getattr(index, "exists", False):
        return None

    if sense_constrained:
        rows = index.by_sense(sense, languages=languages, limit=max_candidates)
        active = _sense_filter(sense_filter)
        if active is not None:
            rows = active.filter(sense, rows)
    else:
        rows = index.candidates(comparison, languages=languages, limit=max_candidates)
    if not rows:
        return None

    by_form = {row["comparison"]: row for row in rows if row["comparison"]}
    distance, form = _strength_best(comparison, list(by_form))
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
        if best.distance < DONOR_DISTANCE_CEILING and ramp_chance_enabled():
            own = tuple(sorted(f for f, r in by_form.items() if r["lang_code"] == best.lang_code))
            return replace(best, ramp_null=_ramp_null(len(comparison), own))
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
    return tuple(_strength_best(control, candidates)[0] for control in _controls(length))


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
    if match.ramp_null is not None and match.distance >= match.ramp_null:
        return 0.0  # X4 A2: rampa eşleşmesi dilin null'ından yakın değil
    span = DONOR_DISTANCE_CEILING - DONOR_DISTANCE_THRESHOLD
    return round((DONOR_DISTANCE_CEILING - match.distance) / span, 4)


# ---------------------------------------------------------------------------
# Verici dil ETİKETİ — alıntı gücünden ayrı adım
# ---------------------------------------------------------------------------
#
# ⚠️ Yukarıdaki :func:`nearest_donor` bütün verici dilleri TEK havuzda arar
# ve en yakın maddenin dilini döndürür. Bu "alıntı mı?" için doğru, "kimden?"
# için yanlıştı. Ölçüldü (``make eval-donor``, WOLD Saha, n=440): 166
# Moğolca alıntının 79'u Rusça etiketleniyordu. Sebepler:
#
# * Havuzlar eşit değil: Rusça 440.919 madde, kaikki Moğolcası 6.480 (çoğu
#   çekimli biçim). Hatalı maddelerde kavram başına aday medyanı Rusça 30,
#   Moğolca 1. Büyük havuzda şans eşleşmesi kaçınılmaz (``ʤon`` ~ ``каян``
#   0,20, ``naːr`` ~ ``нары`` 0,15).
# * Mesafe ölçeği diller arasında karşılaştırılamaz: rastgele Türkçe kontrol
#   kelimelerinin havuza medyan uzaklığı Rusçada 0,503, Moğolcada 0,656.
# * Kapsam ve yazı: Saha Yazı Moğolcası biçiminden almıştır (``čakilɣan``),
#   kaikki Halha Kirilini tutar (``цахилгаан``); doğru kelime çoğu kez hiç yok.
#
# Etiket adımı her dilin en yakın maddesini AYRI bulur, o dilin kendi
# null'ını (aynı uzunluktaki kontrol kelimelerinin o havuza medyan uzaklığı)
# düşer ve en küçük farkı seçer; Moğolca için Starling ``monget`` kullanılır.
# Ölçüldü: doğruluk 0,652 -> 0,714 (çift yarı 0,636 -> 0,691, tek 0,668 -> 0,736), Moğolca->Rusça
# 79 -> 45. Parametre seçilmedi (yarı-bölme gereksiz ama yine raporlanır).
#
# ⚠️ Sinyal GÜCÜ bu adımdan etkilenmez: ``monget`` havuza katılınca WOLD
# "alıntı mı?" F'si 0,615'ten 0,584'e düşüyordu. Etiket yalnız sinyal zaten
# ateşlendiğinde hesaplanır.

#: Bu mesafenin üstündeki etiket "verici belirsiz" notuyla gösterilir.
#: Ölçüldü: yalnız bu eşiğin altında etiketlense kapsananda doğruluk 0,855,
#: ama kapsam 0,64'e iner; etiket gizlenmez, belirsizliği ilan edilir.
DONOR_UNCERTAIN_DISTANCE = DONOR_DISTANCE_THRESHOLD

#: Moğolca verici kodu. Etiket adımında kaikki yerine Starling ``monget``.
MONGOLIAN = "mn"

# ⚠️ ÖLÇÜLDÜ, OLUMSUZ (2026-09-25) — aşağıdaki üç bayrak üretimde kapalı.
# Havuz: Moğolca 12.455 monget -> +7.643 (NorthEuraLex khk/bua/xal 3.626 +
# robbeets 4.017), Tunguzca 599 kaikki -> +7.363 (NEL evn/gld 2.118 + robbeets
# 5.245); Rusça alıntı süzgeci 243 NEL biçimi attı. WOLD Saha n=440, motor
# doğruluğu (çift / tek yarı), taban 0,711 (0,686 / 0,736):
#
#     havuz          mesafe  seçim       doğruluk  çift   tek    Mo  Tu
#     yeni (tümü)    SCA     medyan      0,686     0,682  0,691  74  2
#     yeni (yalnız mn) SCA   medyan      0,714     0,691  0,736  85  1
#     yeni (tümü)    PMI     yüzdelik    0,723     0,732  0,714  85  4
#     yeni (tümü)    ort.    medyan      0,709     0,696  0,723  81  5
#     eski           PMI     medyan      0,700     0,691  0,709  73  2
#     çeşit başına   SCA     yüzdelik    0,709     0,705  0,714  87  2
#
# Hiçbiri 0,74'e ve iki yarıda artışa ulaşmadı; en iyi (PMI + yüzdelik) tek
# yarıda düşüyor, McNemar p=0,55. Neden: Tunguz listeleri Moğolca alıntılarla
# dolu (Evenkice Moğolcadan çok almış) — Moğolca->Tunguzca 10 -> 25; yanlış
# etiketlenen Moğolca maddelerin anlamları (GOITER, TEMPLES, IDEA …) ~1.000
# kavramlık listelerde yok. CLICS⁴ komşu kavram genişlemesi (Wientzek: >=0,05,
# ceza 0,1) etikette 0,677'ye düşürdü (Rusça->Tunguzca/Moğolca 28).

#: Etiket adımında Moğolca ve Tunguzca havuzlarına kavram hizalı listeler
#: (NorthEuraLex + robbeetstriangulation) eklensin mi? Bkz.
#: :mod:`engine.db.concept_donors`. Yalnız ETİKET; güç havuzu değişmez.
CONCEPT_DONOR_LABELS = False

#: Etiket seçim ölçütü: ``median`` = mesafe − kontrol medyanı;
#: ``percentile`` = kontrollerin kaçı bu kadar yakın (Kessler 2001), eşitlikte
#: ham mesafe.
ATTRIBUTION_SCORE = "median"

#: Etiket adımının ses mesafesi: ``sca``, ``pmi`` (ASJP-PMI, Jäger 2018;
#: bkz. :mod:`engine.nlp.pmi_distance`) ya da ``mean`` (ikisinin ortalaması).
LABEL_DISTANCE = "sca"

#: Etiket null'ı için kontrol sayısı.
ATTRIBUTION_CONTROL_COUNT = 12


@dataclass(frozen=True)
class DonorAttribution:
    """Verici dil etiketi: dil başına en yakın madde, dilin null'ına göre."""

    lang_code: str
    word: str
    comparison: str
    gloss: str
    distance: float
    #: Kontrol kelimelerinin bu dilin havuzuna medyan uzaklığı.
    null_distance: float
    source: str = "kaikki"
    #: Öbür dillerin (dil, mesafe, null) değerleri — şeffaflık için.
    alternatives: tuple[tuple[str, float, float], ...] = ()

    @property
    def adjusted(self) -> float:
        return self.distance - self.null_distance

    @property
    def uncertain(self) -> bool:
        return self.distance > DONOR_UNCERTAIN_DISTANCE

    def describe(self) -> str:
        source = {
            "starling-monget": ", Starling monget",
            "northeuralex": ", NorthEuraLex",
            "robbeetstriangulation": ", robbeetstriangulation",
        }.get(self.source, "")
        note = " ⚠️ verici belirsiz" if self.uncertain else ""
        return (
            f"{self.lang_code} {self.word} ({self.comparison}) SCA {self.distance:.3f}, "
            f"dil null'ı {self.null_distance:.3f}{source}{note}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "donor_lang": self.lang_code,
            "donor_word": self.word,
            "donor_gloss": self.gloss,
            "donor_distance": round(self.distance, 4),
            "donor_null_distance": round(self.null_distance, 4),
            "donor_source": self.source,
            "donor_uncertain": self.uncertain,
            "donor_alternatives": [
                {"lang": lang, "sca_distance": round(d, 4), "null_distance": round(n, 4)}
                for lang, d, n in self.alternatives
            ],
        }


@lru_cache(maxsize=64)
def _attribution_controls(length: int) -> tuple[str, ...]:
    """Her uzunlukta kontrol biçimleri.

    ⚠️ :func:`_controls` 6 harften uzun sorguda 8'den az kontrol veriyor ve
    şans denetimi o kelimelerde hiç yapılmıyor. Burada eksik kalan uzunluk
    iki gerçek kelime birleştirilip kırpılarak doldurulur: yine gerçek Türkçe
    fonotaktik, yine sorguyla aynı uzunluk.
    """
    words = [to_comparison_form(w) for w in _CONTROL_SOURCE.split()]
    exact = [w for w in words if len(w) == length]
    trimmed = [w[:length] for w in words if len(w) > length]
    joined = [(a + b)[:length] for a, b in zip(words, words[1:]) if len(a + b) >= length]
    # Çok uzun sorgularda komşu çiftler yetmez: öbür çiftlere de bakılır.
    joined += [(a + b)[:length] for a in words for b in words if a != b and len(a + b) >= length]
    return tuple(list(dict.fromkeys(exact + trimmed + joined))[:ATTRIBUTION_CONTROL_COUNT])


@lru_cache(maxsize=20000)
def _control_profile(length: int, pool: tuple[str, ...]) -> tuple[float, ...]:
    """Kontrol kelimelerinin bu havuza en yakın mesafeleri, sıralı."""
    controls = _attribution_controls(length)
    if not controls or not pool:
        return ()
    return tuple(sorted(best_label(control, list(pool))[0] for control in controls))


@lru_cache(maxsize=20000)
def _null_distance(length: int, pool: tuple[str, ...]) -> float:
    """Kontrol kelimelerinin bu havuza medyan en yakın mesafesi."""
    distances = _control_profile(length, pool)
    if not distances:
        return 0.0
    middle = len(distances) // 2
    if len(distances) % 2:
        return distances[middle]
    return (distances[middle - 1] + distances[middle]) / 2


def _ramp_null(length: int, pool: tuple[str, ...]) -> float:
    """Rampa şans denetiminin null'ı, alıntı GÜCÜ mesafesinin ölçeğinde (X5).

    ``sca`` (üretim) ile X4 A2'deki :func:`_null_distance` ile BİREBİR aynıdır
    (etiket mesafesi de SCA). ``STRENGTH_DISTANCE = "mean"`` iken rampa
    0,60–0,85 ortalama-mesafe ölçeğindedir; SCA null'ı ile karşılaştırmak
    ölçek karışıklığı olurdu. Null o zaman aynı kontrol kelimelerinin aynı
    havuza GÜÇ mesafesiyle medyan en yakın uzaklığıdır. Parametre yok;
    ``PREREG_x5.md``de birleşik aday için ölçümden ÖNCE tanımlandı.
    """
    if STRENGTH_DISTANCE == "sca" and LABEL_DISTANCE == "sca":
        return _null_distance(length, pool)
    return _strength_null(length, pool)


@lru_cache(maxsize=20000)
def _strength_null(length: int, pool: tuple[str, ...]) -> float:
    controls = _attribution_controls(length)
    if not controls or not pool:
        return 0.0
    distances = sorted(_strength_best(control, list(pool))[0] for control in controls)
    middle = len(distances) // 2
    if len(distances) % 2:
        return distances[middle]
    return (distances[middle - 1] + distances[middle]) / 2


@lru_cache(maxsize=1)
def _monget_entries() -> tuple[tuple[str, str, str, frozenset[str]], ...]:
    """(karşılaştırma biçimi, biçim, anlam, anlam sözcükleri) — Starling monget."""
    from engine.db.donor_index import MAX_LENGTH, MIN_LENGTH
    from engine.db.starling import load_monget

    out = []
    for entry in load_monget():
        comparison = to_comparison_form(entry.form)
        if not MIN_LENGTH <= len(comparison) <= MAX_LENGTH:
            continue
        tokens = frozenset(t for t in re.split(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ]+", entry.meaning.lower()) if t)
        out.append((comparison, entry.form, entry.meaning, tokens))
    return tuple(out)


def _monget_rows(sense: str) -> list[dict[str, str]]:
    """Anlamı sorguyla örtüşen monget biçimleri (verici indeksiyle aynı kural)."""
    from engine.db.donor_index import sense_tokens_for_match

    # ``DonorIndex.by_sense`` ile aynı anahtar (X4 temizliği açıksa içerik sözcükleri).
    tokens = set(sense_tokens_for_match(sense))
    if not tokens:
        return []
    return [
        {"lang_code": MONGOLIAN, "word": form, "comparison": comparison, "gloss": meaning}
        for comparison, form, meaning, words in _monget_entries()
        if words & tokens
    ]


def _concept_rows(sense: str) -> dict[str, list[dict[str, str]]]:
    """Anlamı sorguyla örtüşen kavram hizalı Moğol/Tunguz biçimleri, havuza göre.

    Bkz. :mod:`engine.db.concept_donors` — NorthEuraLex + robbeetstriangulation.
    Eşleşme kuralı :func:`_monget_rows` ile aynıdır.
    """
    from engine.db.concept_donors import load_concept_donors
    from engine.db.donor_index import sense_tokens_for_match

    # ``DonorIndex.by_sense`` ile aynı anahtar (X4 temizliği açıksa içerik sözcükleri).
    tokens = set(sense_tokens_for_match(sense))
    if not tokens:
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for form, words in load_concept_donors():
        if words & tokens:
            out.setdefault(form.pool, []).append({
                "lang_code": form.pool,
                "word": f"{form.form} ({form.variety})",
                "comparison": form.comparison,
                "gloss": form.gloss,
                "source": form.source,
            })
    return out


def attribute_donor(
    comparison: str,
    sense: str = "",
    *,
    languages: list[str] | None = None,
    max_candidates: int = 200,
    sense_filter: Any = None,
) -> DonorAttribution | None:
    """Alıntı olduğu düşünülen kelimenin verici dilini seçer.

    Her dilin anlam kısıtlı en yakın maddesi ayrı bulunur; seçim ölçütü
    ``mesafe − o dilin null'ı``dır (eşitlikte ham mesafe). Moğolca istendiyse
    ve Starling ``monget`` varsa Moğolca adayları ondan gelir.

    ⚠️ Karar değil etikettir: kelimenin alıntı olup olmadığını bu fonksiyon
    söylemez, :func:`nearest_donor` ve :func:`proximity_strength` söyler.
    """
    index = _index()
    if _pairwise() is None or not comparison or not getattr(index, "exists", False):
        return None
    rows = index.by_sense(sense, languages=languages, limit=max_candidates)
    active = _sense_filter(sense_filter)
    if active is not None:
        rows = active.filter(sense, rows)
    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(row["lang_code"], []).append(row)
    sources = {lang: "kaikki" for lang in groups}
    if languages is None or MONGOLIAN in languages:
        mongolic = _monget_rows(sense)
        if active is not None:
            mongolic = active.filter(sense, mongolic)
        if mongolic or _monget_entries():
            groups.pop(MONGOLIAN, None)
            sources.pop(MONGOLIAN, None)
        if mongolic:
            groups[MONGOLIAN] = mongolic
            sources[MONGOLIAN] = "starling-monget"
    if CONCEPT_DONOR_LABELS:
        for pool, extra in _concept_rows(sense).items():
            if active is not None:
                extra = active.filter(sense, extra)
            if languages is None or pool in languages:
                groups.setdefault(pool, []).extend(extra)

    scored: list[tuple[float, float, str, Any, float]] = []
    for lang, members in groups.items():
        by_form: dict[str, Any] = {}
        for row in members:
            if row["comparison"] and row["comparison"] not in by_form:
                by_form[row["comparison"]] = row
        if not by_form:
            continue
        distance, form = best_label(comparison, list(by_form))
        if not form:
            continue
        pool = tuple(sorted(by_form))
        null = _null_distance(len(comparison), pool)
        if ATTRIBUTION_SCORE == "percentile":
            profile = _control_profile(len(comparison), pool)
            key: Any = (sum(1 for d in profile if d <= distance) / max(len(profile), 1), distance)
        else:
            key = (distance - null, distance)
        scored.append((key, distance, lang, by_form[form], null))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    _, distance, lang, row, null = scored[0]
    return DonorAttribution(
        lang_code=lang,
        word=row["word"],
        comparison=row["comparison"],
        gloss=row["gloss"] or "",
        distance=distance,
        null_distance=null,
        source=(row.get("source") if isinstance(row, dict) else None) or sources.get(lang, "kaikki"),
        alternatives=tuple((item[2], item[1], item[4]) for item in scored[1:]),
    )
