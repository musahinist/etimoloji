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


@lru_cache(maxsize=1)
def _label_index() -> Any:
    """YALNIZ etiket adımının eski dil havuzu (grc, xcl); bkz. :data:`OLD_DONOR_LABELS`."""
    from engine.db.donor_index import LABEL_DB, DonorIndex

    return DonorIndex(LABEL_DB)


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
    _label_index.cache_clear()
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
#:
#: ⚠️ X5 (ön-kayıt ``data/cache/work/xtr/PREREG_x5.md``, R3 bir kez, n=3.706,
#: kör indeks + zincir kapalı): birleşik tek aday (temizlik + rampa şans
#: denetimi + ``mean`` 0,60/0,85 + anlam süzgeci s3:fallback) engine_trained
#: F 0,8397 -> 0,8567, fark +0,017 [+0,006, +0,029] (birincil TUTTU);
#: verici tanıma 0,595 -> 0,623 (tuttu); koruma WOLD 0,6636, Türkçe 0,8883,
#: arama dev 129/135 (tuttu). Mekanizma ölçütü TUTMADI: mirasta rampa
#: 822 -> 984 (mean'in geniş bandı; önceden yazılmıştı) -> KABUL EDİLMEDİ,
#: varsayılanlar KAPALI. Bilgi (karar değil): tek başına ``mean`` +0,019
#: [+0,011, +0,027], A2 +0,007 [+0,000, +0,014] (rampa 822 -> 592),
#: S3 −0,003 [−0,012, +0,006]. Yeni ön kayıt ister.
STRENGTH_DISTANCE = "sca"


def _mean_distance(a: str, b: str) -> float:
    from engine.nlp.pmi_distance import pmi_distance

    return (sca_distance(a, b) + min(1.0, pmi_distance(a, b))) / 2


#: D8 / M8 (Mi ve ark. 2018, PRED): alıcı kelimenin SONUNDAKİ fazlalık
#: cezasız — çekimli/türemiş alıntıda verici biçim alıcının önekiyle
#: eşleşir (kk ``машинасы`` ~ ru ``машина``). Sondan en çok ``PRED_MAX_TRIM``
#: harf kırpılır; kırpılmış gövde vericiden kısa ya da ``PRED_MIN_STEM``den
#: kısa olamaz. Mesafe SCA'dır; kontrol kelimeleri (şans denetimi, null) aynı
#: mesafeyi görür. ``STRENGTH_DISTANCE = "pred"`` ile açılır (deneme).
#: ``derivation``/``root_variants`` Türkçe (Zemberek TR) sözlüğüne bağlı —
#: Türk dilleri arası bölümlere uymadığı için dile bağımsız sondan kırpma.
#:
#: ⚠️ ÖLÇÜLDÜ, ön-kayıtlı, OLUMSUZ (2026-09-26; ``data/cache/work/d8/PREREG.md``,
#: aday C2). Ayar (X1 tune): F 0,7611 -> 0,7622, miras özgüllüğü 0,633 ->
#: 0,594. Bağımsız R4 (n=1.588, bir kez, kör indeks + zincir kapalı):
#: engine_trained F 0,8901 -> 0,8771, fark −0,013 [−0,021, −0,005] (Holm
#: p = 1,0); özgüllük 0,794 -> 0,723 (mirasta YP 49 -> 66); yalnız-yakınlık
#: kesinliği −0,015; verici tanıma 0,773 -> 0,772. Mirasta rampa 146 -> 117
#: (kırpma rampadaki mirası YAKIN'a taşıyor). KABUL EDİLMEDİ: sondan cezasız
#: kırpma, çekimli alıntıyı kurtarmaktan çok miras kelimeye rastlantı eşleşmesi
#: veriyor. Korumalar da düştü: WOLD F 0,6119 < 0,6482, Türkçe 0,8618 <
#: 0,8773 (arama dev 129/135). Üretim ``sca``.
PRED_MAX_TRIM = 4
PRED_MIN_STEM = 3


def pred_distance(query: str, donor: str) -> float:
    best = sca_distance(query, donor)
    for trim in range(1, PRED_MAX_TRIM + 1):
        stem = query[:-trim]
        if len(stem) < max(PRED_MIN_STEM, len(donor)):
            break
        best = min(best, sca_distance(stem, donor))
    return best


def _strength_best(query: str, candidates: list[str]) -> tuple[float, str]:
    if STRENGTH_DISTANCE == "pred":
        return _best(query, candidates, pred_distance)
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
#: X5 (R3): birleşik aday kabul edilmedi — bkz. :data:`STRENGTH_DISTANCE` notu.
#: 2026-09-26: A2'yi birikmiş kanıtla açma denemesi Türkçe koruma eşiğinde
#: düştü (F 0,8735 < 0,8773) -> kapalı; bkz. ``donor_index.CLEAN_DEFAULT``.
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

#: 9g G1 — etiket adımında eski dil havuzları (``donor_index.LABEL_DB``):
#: Eski Yunanca -> Yunanca ailesi, Eski Ermenice -> Ermenice ailesi. Her eski
#: dil AYRI grup (kendi null'ı); seçilirse etiket ailenin kodudur, kaynak
#: ``kaikki-grc``/``kaikki-xcl`` ("Eski Yunanca biçimi"). Yalnız ETİKET; güç
#: havuzu (``donors.db``) değişmez. Ön kayıt ``data/cache/work/donor9g/PREREG.md``.
#:
#: ⚠️ ÖLÇÜLDÜ, ön kayıtlı (6baec31), KORUMADA DÜŞTÜ -> kapalı. Yeni rapor
#: (tr/ota/az/crh Wiktionary, el 150 + hy 78, kör indeks, bir kez): el+hy
#: etiket doğruluğu 0,285 -> 0,439 (McNemar 37/2, Holm p 8,5e-9; Yunanca
#: 36 -> 54/150, Ermenice 29 -> 46/78). Ama Türkçe TDK+Nişanyan train+dev
#: 0,642 -> 0,614 (Arapça -> Yunanca 6 -> 17): ayrı grup Arapça alıntılara
#: rastlantı eşi veriyor. ``merge`` modu (ayar tanısı) da düşürdü (0,621).
#: G2 (``LABEL_FORM_FILTER``) etkisiz: rapor 0/0 uyuşmazlık.
OLD_DONOR_LABELS = False
OLD_DONOR_FAMILY = {"grc": "el", "xcl": "hy"}
#: ``separate``: eski dil ayrı grup, kendi null'ı (ailenin ikinci "bileti");
#: ``merge``: eski dil maddeleri ailenin grubuna katılır, null birleşik havuzdan.
OLD_DONOR_MODE = "separate"

#: 9g G2 — etiket adımında çekim/biçim göndermesi süzgeci: bu dillerin anlamı
#: YALNIZ "plural of X", "inflection of X" gibi dilbilgisi göndermesi olan
#: maddeleri (``donor_index.is_form_of``, X4 (b)) atlanır. Güç yolu ve X4
#: ``CLEAN`` bayrağı değişmez (A2 alıntı gücünde Türkçe korumasını düşürmüştü).
LABEL_FORM_FILTER = False
LABEL_FORM_FILTER_LANGS = frozenset({"el", "hy", "grc", "xcl"})

#: 9j G1' — :data:`OLD_DONOR_LABELS` açıkken eski dil grubu (grc, xcl) YALNIZ
#: en yakın biçimi bu SCA eşiğinin altındaysa seçilebilir (``None`` = sınırsız,
#: 9g G1). Gerekçe (9g SONUÇ): ayrı grup Arapça alıntılara uzak rastlantı eşi
#: veriyordu (Arapça -> Yunanca 6 -> 17). Ön kayıt ``data/cache/work/donor9j/PREREG.md``.
#:
#: ⚠️ ÖLÇÜLDÜ (9j, ön kayıt 3056510), RED: TDK rapor 0,2113 -> 0,2113 (2/2);
#: Türkçe train+dev 0,642 -> 0,625 (Arapça -> Yunanca yine 14).
OLD_DONOR_MAX: float | None = None

#: 9j I1 — etiket adımında İtalyanca (ve Venedikçe/Cenevizce) adayların
#: karşılaştırma biçimi Türkçe sesçil yazıma yaklaştırılır (:func:`italian_phonetic`:
#: ``sci/ce/ci/ge/gi/ch/gh/gli/gn/qu``, çift ünsüz, ``-zione``). Gerekçe: İtalyan
#: imlası sesi Türkçe alıntıdan farklı yazar (``scialuppa`` ~ ``şalopa``);
#: Fransızca havuza dokunulmaz. Yalnız ETİKET.
#:
#: ⚠️ ÖLÇÜLDÜ (9j), RED: yeni TDK rapor (n=710) 0,2113 -> 0,2028 (3/9, Holm 0,44),
#: TETTL 0,255 -> 0,251. Tanı: İtalyanca hatasının ana nedeni anlam araması (Türkçe
#: anlamlı maddelerde aday yok; paylaşılan LIMIT), imla değil.
ITALIAN_ORTHO = False
ITALIAN_LANGS = frozenset({"it", "vec", "lij"})

#: 9j I2 — etiket adımında Venedikçe (vec) ve Cenevizce (lij) yalnız-etiket
#: havuzu (``donor_index.LABEL_DB``); maddeler İtalyanca grubuna KATILIR (null
#: birleşik havuzdan; ayrı "bilet" yok). Seçilirse etiket ``it``, kaynak
#: ``kaikki-vec``/``kaikki-lij`` ("İtalyanca (Venedikçe biçimi)"). Güç havuzu
#: (``donors.db``) değişmez.
#:
#: ⚠️ ÖLÇÜLDÜ (9j), RED: TDK rapor 0,2113 -> 0,2099 (3/4); Türkçe train+dev
#: 0,642 -> 0,601 (Fransızca -> İtalyanca 11 -> 22, Arapça -> İtalyanca 15).
VENETAN_LABELS = False
ITALO_FAMILY = {"vec": "it", "lij": "it"}

#: 9l S1 — etiket adımında Türkçe anlam -> İngilizce köprü: anlam metni İngilizce
#: değilse (Türkçe Vikisözlük tanımı; :func:`engine.db.sense_bridge.is_english_sense`)
#: ve Latin yazılıysa, verici anlam araması kelimenin Vikisözlük ÇEVİRİ karşılıklarıyla
#: (``sense_bridge.english_sense``; tr->en çeviri bölümü, en->tr çeviri tabloları; yedek:
#: aynı biçimli Osmanlıca maddenin İngilizce anlamı) yapılır.
#: Köken alanı ve TDK tanımı kullanılmaz. Yalnız ETİKET; alıntı gücü (``nearest_donor``)
#: aynı anlamla kalır. Ön kayıt ``data/cache/work/donor9l/PREREG.md``. Tablo yoksa
#: (``python -m engine.db.sense_bridge --build ...``) yalnız Osmanlıca yedeği çalışır.
#:
#: ÖLÇÜLDÜ, ön kayıtlı (c2466aa), KABUL. Yeni TDK GTS rapor altını (n=441, önceki tüm
#: altınların dışında, kör indeks, bir kez): 0,329 -> 0,417 (McNemar 39/0, Holm p ~1e-11;
#: Türkçe anlamlı katman 15 -> 54/225, etiketsiz 150 -> 105). Korumalar aynı: Türkçe
#: TDK+Nişanyan train+dev 0,6416 (0 uyuşmazlık), Saha ``eval-donor`` 0,714, xturkic 0,746.
#: 9j TDK raporu (görülmüş, bilgi) 0,211 -> 0,303 (70/5). S2 (İtalyancaya ayrı sorgu) ve
#: S3 (tam eşleşmede ham mesafe) RED (rapor 0/2, 0/1; S2 Türkçe korumayı 0,604'e düşürdü).
SENSE_BRIDGE = True

#: 9l S2 — G2'nin Fransızca için yaptığı ayrı 200'lük anlam sorgusu bu diller için de
#: (paylaşılan havuz aynen kalır; yalnız eklenir). ``()`` = kapalı.
#: ⚠️ ÖLÇÜLDÜ (9l), RED: ``("it",)`` rapor 0,329 -> 0,324 (0/2); Türkçe train+dev
#: 0,642 -> 0,604 (Fransızca -> İtalyanca 11 -> 19, Arapça -> İtalyanca 5 -> 16).
EXTRA_POOL_LANGS: tuple[str, ...] = ()

#: 9l S3 — en yakın biçimi bu SCA eşiğinin altında (tam ya da tama yakın eşleşme)
#: olan diller varsa seçim yalnız onların arasında, ham mesafeyle yapılır (null
#: düzeltmesi tam eşleşmeyi yenemez). ``None`` = kapalı.
#: ⚠️ ÖLÇÜLDÜ (9l), RED: ε=0,05 rapor 0,329 -> 0,327 (0/1); ayar 2/0, xturkic 0,746 -> 0,768
#: (bilgi), 9f raporu 0,438 -> 0,458 (5/0, bilgi) — yeni raporda etkisiz.
EXACT_MATCH_EPS: float | None = None

_LATIN = re.compile(r"^[^\u0370-\u03ff\u0400-\u04ff\u0530-\u058f\u0590-\u06ff]*$")


def bridged_sense(comparison: str, sense: str) -> str:
    """S1: Türkçe anlamlı maddede İngilizce köprü anlamı; yoksa anlam aynen."""
    if not SENSE_BRIDGE or not sense.strip() or not _LATIN.match(sense):
        return sense
    from engine.db.sense_bridge import english_sense, is_english_sense, ottoman_sense

    if is_english_sense(sense):
        return sense
    return english_sense(comparison) or ottoman_sense(comparison) or sense


def _sense_overlap(sense: str, gloss: str) -> int:
    """Sorgu anlamının eşleşme sözcükleri ile verici anlamının ortak İÇERİK sözcüğü sayısı."""
    from engine.db.donor_index import FUNCTION_WORDS, _sense_tokens, sense_tokens_for_match

    shared = set(sense_tokens_for_match(sense)) & set(_sense_tokens(gloss))
    return sum(1 for t in shared if t not in FUNCTION_WORDS)


_IT_VOWEL = "aeiouöüı"


def italian_phonetic(comparison: str) -> str:
    """İtalyanca karşılaştırma biçimi -> Türkçe sesçil yazıma yakın biçim (9j I1).

    ``scialuppa`` -> ``şalupa``, ``ceppo`` -> ``çepo``, ``giranta`` -> ``ciranta``,
    ``chiglia`` -> ``kilya``, ``organizzazione`` -> ``organizazyon``.
    """
    s = re.sub(r"([^" + _IT_VOWEL + r"])\1", r"\1", comparison)
    if s.endswith("zione"):
        s = s[: -len("zione")] + "zyon"
    # "ǰ" = Türkçe c; sonda c'ye döner ("c" -> "k" kuralından korunur).
    rules = (
        (r"sci(?=[aou])", "ş"), (r"sc(?=[ei])", "ş"),
        (r"ci(?=[aou])", "ç"), (r"c(?=[ei])", "ç"),
        (r"gi(?=[aou])", "ǰ"), (r"g(?=[ei])", "ǰ"),
        (r"gli(?=[aeiou])", "ly"), (r"gli", "li"), (r"gn", "ny"),
        (r"ch", "k"), (r"gh", "g"), (r"qu", "kv"), (r"c", "k"),
        (r"h", ""), (r"j", "y"), (r"x", "ks"),
    )
    for pattern, repl in rules:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"([^" + _IT_VOWEL + r"])\1", r"\1", s).replace("ǰ", "c")
    return s or comparison


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
    #: Aracı dil: en yakın biçim bu dildeydi ama o biçim ``lang_code``dan
    #: alıntı (ör. Farsçadaki Arapça alıntı -> ``ar``, ``via="fa"``). Bkz.
    #: :data:`ARABIC_VIA_RULE`.
    via: str = ""
    #: Şans denetimi (9n): kontrol kelimelerinin (:func:`_attribution_controls`) seçilen dilin
    #: havuzuna en yakın mesafelerinden bu eşleşmeninkine eşit ya da küçük olanların payı.
    #: ``0`` = hiçbir rastgele Türkçe biçim o havuzda bu kadar yakın bir şey bulamıyor.
    chance_percentile: float | None = None
    #: Sorgu anlamının eşleşme sözcükleriyle seçilen maddenin anlamı arasında ortak İÇERİK
    #: sözcüğü sayısı (işlev sözcükleri hariç; "the", "one" eşleşmesi sayılmaz).
    sense_overlap: int = 0
    #: Seçilen biçimin ünsüz iskeleti sorgunun iskeletiyle aynı mı (:func:`consonant_skeleton`).
    skeleton_match: bool = False

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
            "kaikki-grc": ", Eski Yunanca biçimi",
            "kaikki-xcl": ", Eski Ermenice biçimi",
            "kaikki-vec": ", İtalyanca (Venedikçe biçimi)",
            "kaikki-lij": ", İtalyanca (Cenevizce biçimi)",
        }.get(self.source, "")
        note = " ⚠️ verici belirsiz" if self.uncertain else ""
        via = ""
        if self.via:
            from engine.nlp.borrowing_chain import language_name

            via = f" — {language_name(self.lang_code)} ({language_name(self.via)} aracılığıyla)"
        return (
            f"{self.lang_code} {self.word} ({self.comparison}) SCA {self.distance:.3f}, "
            f"dil null'ı {self.null_distance:.3f}{source}{note}{via}"
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
            "donor_via": self.via,
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
    sense = bridged_sense(comparison, sense)
    skip = LABEL_FORM_FILTER_LANGS if LABEL_FORM_FILTER else None
    skip_kw = {"skip_form_of": skip} if skip else {}
    shared: set[tuple[str, str, str]] | None = None
    if FRENCH_RULE in ("f1", "f2"):
        rows = index.by_sense(sense, languages=languages, limit=max_candidates, per_language=True, **skip_kw)
    else:
        rows = index.by_sense(sense, languages=languages, limit=max_candidates, **skip_kw)
        shared = {(r["lang_code"], r["word"], r["comparison"]) for r in rows}
        if FRENCH_RULE in ("g1", "g2") and (languages is None or FRENCH in languages):
            seen = {(r["lang_code"], r["word"], r["comparison"]) for r in rows}
            rows = list(rows) + [r for r in index.by_sense(sense, languages=[FRENCH], limit=max_candidates)
                                 if (r["lang_code"], r["word"], r["comparison"]) not in seen]
        for extra_lang in EXTRA_POOL_LANGS:
            if languages is not None and extra_lang not in languages:
                continue
            seen = {(r["lang_code"], r["word"], r["comparison"]) for r in rows}
            rows = list(rows) + [r for r in index.by_sense(sense, languages=[extra_lang], limit=max_candidates)
                                 if (r["lang_code"], r["word"], r["comparison"]) not in seen]
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
    if OLD_DONOR_LABELS and getattr(_label_index(), "exists", False):
        for old, family in OLD_DONOR_FAMILY.items():
            if languages is not None and family not in languages:
                continue
            extra = _label_index().by_sense(sense, languages=[old], limit=max_candidates, **skip_kw)
            if active is not None:
                extra = active.filter(sense, extra)
            if not extra:
                continue
            if OLD_DONOR_MODE == "merge":
                groups.setdefault(family, []).extend(
                    {"lang_code": family, "word": r["word"], "comparison": r["comparison"],
                     "gloss": r["gloss"], "source": f"kaikki-{old}"} for r in extra)
            else:
                groups[old] = list(extra)
                sources[old] = f"kaikki-{old}"
    if VENETAN_LABELS and getattr(_label_index(), "exists", False) and (languages is None or "it" in languages):
        for dialect, family in ITALO_FAMILY.items():
            extra = _label_index().by_sense(sense, languages=[dialect], limit=max_candidates)
            if active is not None:
                extra = active.filter(sense, extra)
            groups.setdefault(family, []).extend(
                {"lang_code": family, "word": r["word"], "comparison": r["comparison"],
                 "gloss": r["gloss"], "source": f"kaikki-{dialect}"} for r in extra)
    if CONCEPT_DONOR_LABELS:
        for pool, extra in _concept_rows(sense).items():
            if active is not None:
                extra = active.filter(sense, extra)
            if languages is None or pool in languages:
                groups.setdefault(pool, []).extend(extra)

    scored: list[tuple[float, float, str, Any, float]] = []
    for lang, members in groups.items():
        by_form: dict[str, Any] = {}
        italian = ITALIAN_ORTHO and lang in ITALIAN_LANGS
        for row in members:
            form_key = italian_phonetic(row["comparison"]) if italian and row["comparison"] else row["comparison"]
            if form_key and form_key not in by_form:
                by_form[form_key] = row
        if not by_form:
            continue
        distance, form = best_label(comparison, list(by_form))
        if not form:
            continue
        if OLD_DONOR_MAX is not None and lang in OLD_DONOR_FAMILY and distance > OLD_DONOR_MAX:
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
    if EXACT_MATCH_EPS is not None and any(item[1] <= EXACT_MATCH_EPS for item in scored):
        exact = sorted((item for item in scored if item[1] <= EXACT_MATCH_EPS),
                       key=lambda item: (item[1], item[0], item[2]))
        scored = exact + sorted((item for item in scored if item[1] > EXACT_MATCH_EPS),
                                key=lambda item: (item[0], item[1], item[2]))
    else:
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
    _, distance, lang, row, null = scored[0]
    raw_lang = lang
    lang = OLD_DONOR_FAMILY.get(lang, lang)
    via = ""
    switched = None
    if ARABIC_VIA_RULE != "off" and (languages is None or ARABIC in languages):
        switched = _arabic_via(comparison, lang, row, groups.get(ARABIC) or [])
        if switched is not None:
            via = lang if lang == PERSIAN else ""
            row, distance = switched
            if row["lang_code"] == ARABIC:
                ar_pool = tuple(sorted({r["comparison"] for r in groups[ARABIC] if r["comparison"]}))
                null = _null_distance(len(comparison), ar_pool)
            lang = ARABIC
    # 9m R2: Arapça ünsüz iskeleti sorguyla eşleşiyorsa Fransızca kuralları (G2 ayrı havuzu,
    # Fransızca aracılı, H1) devreye girmez.
    guard = None
    if switched is None and FRENCH_ARABIC_GUARD and (languages is None or ARABIC in languages):
        guard = _arabic_skeleton_match(comparison, groups.get(ARABIC) or [])
    if guard is not None and lang == FRENCH and shared is not None \
            and (row["lang_code"], row["word"], row["comparison"]) not in shared:
        # Fransızca kazanan yalnız G2'nin ayrı havuzundan geldi -> Arapça iskelet eşi.
        row, distance = guard
        ar_pool = tuple(sorted({r["comparison"] for r in groups[ARABIC] if r["comparison"]}))
        null = _null_distance(len(comparison), ar_pool)
        lang, via = ARABIC, ""
        switched = guard
    # Fransızca aracılı: Arapça kuralı (D1) önce; ateşlediyse dokunulmaz.
    if switched is None and guard is None and FRENCH_RULE in ("f2", "g2") and (languages is None or FRENCH in languages):
        french = _french_via(comparison, lang, row, distance, groups.get(FRENCH) or [])
        if french is not None:
            via = lang
            row, distance = french
            if row["lang_code"] == FRENCH:
                fr_pool = tuple(sorted({r["comparison"] for r in groups[FRENCH] if r["comparison"]}))
                null = _null_distance(len(comparison), fr_pool)
            lang = FRENCH
    # Batı alıntısında Fransızca önceliği (9f): D1 ve G2'den sonra.
    if switched is None and guard is None and lang != FRENCH and WESTERN_RULE != "off" \
            and (languages is None or FRENCH in languages):
        french = _french_prior(comparison, lang, distance, groups.get(FRENCH) or [])
        if french is not None:
            row, distance = french
            fr_pool = tuple(sorted({r["comparison"] for r in groups[FRENCH] if r["comparison"]}))
            null = _null_distance(len(comparison), fr_pool)
            via = ""
            lang = FRENCH
    final_pool = tuple(sorted({r["comparison"] for r in (groups.get(lang) or groups.get(raw_lang) or [])
                               if r["comparison"]}))
    profile = _control_profile(len(comparison), final_pool) if final_pool else ()
    return DonorAttribution(
        chance_percentile=(sum(1 for d in profile if d <= distance) / len(profile)) if profile else None,
        sense_overlap=_sense_overlap(sense, row["gloss"] or ""),
        skeleton_match=consonant_skeleton(row["comparison"] or "") in _query_skeletons(comparison),
        lang_code=lang,
        word=row["word"],
        comparison=row["comparison"],
        gloss=row["gloss"] or "",
        distance=distance,
        null_distance=null,
        source=(row.get("source") if isinstance(row, dict) else None)
        or sources.get(raw_lang if lang == OLD_DONOR_FAMILY.get(raw_lang) else lang, "kaikki"),
        alternatives=tuple((item[2], item[1], item[4]) for item in scored[1:] if item[2] not in (lang, raw_lang)),
        via=via,
    )


# --- Farsça üzerinden Arapça (9d) -------------------------------------------------

ARABIC = "ar"
PERSIAN = "fa"

#: Etiket adımında "Farsça üzerinden Arapça" kuralı: ``off``, ``d1`` (Farsça
#: kazanan aday Arapçadan alıntı işaretli ya da Arapça havuzda aynı yazı
#: iskeletli karşılığı var), ``d2`` (sorgunun ünsüz iskeleti bir Arapça
#: adayınkiyle aynı), ``d3`` (ikisi). Yalnız ETİKET; alıntı gücü değişmez.
#:
#: ÖLÇÜLDÜ, ön kayıtlı (``data/cache/work/donor9d/PREREG.md``), KABUL: Türkçe
#: altın etiket doğruluğu (A2 kapalı; çoğunluk tabanı 0,505)::
#:
#:              TRAIN n=229   DEV n=64 (bir kez)   McNemar DEV   Holm p
#:     off      0,410         0,484                —             —
#:     D1       0,541         0,641                10 / 0        0,004
#:     D2       0,515         0,609                 8 / 0        0,008
#:     D3       0,555         0,656                11 / 0        0,003
#:
#: xturkic verici tanıma (kör önbellek, etiket yeniden oynatıldı): tune
#: 0,550 -> 0,746 (D1); bilgi R1 0,560 -> 0,707, R2 0,548 -> 0,695. Bedel:
#: altını "Farsça" olan, Farsça üzerinden gelmiş Arapça sözcükler (tune fa->ar
#: 29 -> 54). Saha ``make eval-donor`` 0,714 (ar/fa havuzda yok, değişmez).
#: D1 önceden seçilmişti (D3'ün TRAIN farkı +3/229, fa->ar hatası artıyor).
ARABIC_VIA_RULE = "d1"

_HARAKAT = re.compile("[\u064b-\u065f\u0670\u0640\u200c\u200d]")
_SCRIPT_MAP = str.maketrans({"ة": "ت", "ى": "ي", "ی": "ي", "ک": "ك", "أ": "ا", "إ": "ا", "آ": "ا",
                             "ٱ": "ا", "ؤ": "و", "ئ": "ي", "ۀ": "ه", "ء": None, "ا": None})


def script_skeleton(word: str) -> str:
    """Arap yazısı iskeleti: harekesiz, ة=ت, elif/hemze atılmış (Farsça = Arapça yazımı)."""
    return _HARAKAT.sub("", word or "").translate(_SCRIPT_MAP).replace(" ", "")


#: Latin karşılaştırma biçiminde ünsüz sınıfları. Osmanlıca/Türkçe biçim ile
#: Arapça çevriyazı arasında düzenli denklikler: ḍ/ḏ -> d ama Türkçe z
#: (gazap~gadab, zikir~dikr), ṯ -> t ama Türkçe s (servet~tarva), ḵ -> k ama
#: Türkçe h (haber~kabar), söz sonu ötümsüzleşme (gazap~gadab).
_SKELETON_CLASS = {
    **dict.fromkeys("dtsz", "D"), **dict.fromkeys("bp", "B"), **dict.fromkeys("kgğhqx", "K"),
    **dict.fromkeys("cçj", "C"), "v": "V", "w": "V", "ş": "Ş", "f": "F", "l": "L", "r": "R",
    "m": "M", "n": "N", "y": "Y",
}


def consonant_skeleton(comparison: str) -> str:
    """Ünsüz iskeleti (ünlüsüz, sınıflanmış, ikizler tekleşmiş)."""
    out: list[str] = []
    for ch in comparison or "":
        cls = _SKELETON_CLASS.get(ch)
        if cls and (not out or out[-1] != cls):
            out.append(cls)
    return "".join(out)


def _query_skeletons(comparison: str) -> set[str]:
    """Sorgunun iskeletleri; Osmanlıca -et/-at (tā' marbūṭa) sonu atılmış biçim de."""
    out = {consonant_skeleton(comparison)}
    if re.search(r"[aeıiouöü]t$", comparison or ""):
        out.add(consonant_skeleton(comparison[:-1]))
    return {s for s in out if len(s) >= 2}


def persian_arabic_loans() -> frozenset[tuple[str, str]]:
    """Farsça dökümde etimolojisi "from Arabic" olan (madde, anlam) çiftleri."""
    return dump_loans("fa", "from arabic")


@lru_cache(maxsize=8)
def dump_loans(lang: str, phrase: str) -> frozenset[tuple[str, str]]:
    """``lang`` dökümünde etimolojisi ``phrase`` içeren (madde, anlam) çiftleri.

    Anahtar ``donors.db``deki ``(word, gloss)`` ile aynı kurala göre kurulur
    (:func:`engine.db.donor_index._glosses`); eşsesli yerli madde
    işaretlenmez. Döküm yoksa boş küme.
    """
    import gzip
    import json

    from engine.db.donor_index import DONOR_DIR, _glosses

    path = DONOR_DIR / f"{lang}.jsonl.gz"
    if not path.exists():
        path = DONOR_DIR / f"{lang}.jsonl"
        if not path.exists():
            return frozenset()
    opener = gzip.open if path.suffix == ".gz" else open
    out: set[tuple[str, str]] = set()
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = (record.get("etymology_text") or "").lower()
            if phrase in text:
                out.add(((record.get("word") or "").strip(), _glosses(record)))
    return frozenset(out)


def _arabic_via(comparison: str, lang: str, row: Any, arabic: list[Any]) -> tuple[Any, float] | None:
    """Etiket Arapçaya çevrilmeli mi? Evetse (temsilci satır, mesafe)."""
    rule = ARABIC_VIA_RULE
    arabic = [r for r in arabic if r["comparison"]]

    def closest(members: list[Any]) -> tuple[Any, float]:
        best = min(members, key=lambda r: (label_distance(comparison, r["comparison"]), r["comparison"]))
        return best, label_distance(comparison, best["comparison"])

    if lang == ARABIC:
        return None
    if rule in ("d1", "d3") and lang == PERSIAN:
        skeleton = script_skeleton(row["word"])
        same = [r for r in arabic if skeleton and script_skeleton(r["word"]) == skeleton]
        if same:
            return closest(same)
        if (row["word"], row["gloss"] or "") in persian_arabic_loans():
            return closest(arabic) if arabic else (row, label_distance(comparison, row["comparison"]))
    if rule in ("d2", "d3"):
        skeletons = _query_skeletons(comparison)
        same = [r for r in arabic if consonant_skeleton(r["comparison"]) in skeletons]
        if same:
            return closest(same)
    return None


#: 9m R2 — Arapça iskelet koruması: sorgunun ünsüz iskeleti (:func:`_query_skeletons`,
#: D2 mantığı) Arapça havuzdaki bir adayınkiyle aynıysa ve iskelet en az
#: :data:`FRENCH_ARABIC_GUARD_MIN` ünsüzse Fransızca kuralları (G2'nin ayrı havuzundan
#: gelen kazanan, Fransızca aracılı, H1 soneki) devreye girmez. Yalnız ETİKET.
#:
#: ⚠️ ÖLÇÜLDÜ (9m, ön kayıt ``data/cache/work/donor9m/PREREG.md``), RED: doğal oranlı yeni TDK
#: rapor altını (n=600) doğal ağırlıklı doğruluk 0,4648 -> 0,4664 (3/2, Holm p=1,0); Türkçe
#: train+dev Arapça duyarlılığı 0,757 -> 0,770 (0,80'e dönmüyor). Tanı: G2'nin kaybettirdiği
#: Arapça maddelerin tümü taban etiketi yanlış etimonlu şans eşleşmesi; koruma da çoğunlukla
#: şans iskelet eşi buluyor (hasır ~ خسر). G2+H1 doğal dağılımda da olumlu: 0,418 -> 0,465.
FRENCH_ARABIC_GUARD = False
FRENCH_ARABIC_GUARD_MIN = 3
#: "Güçlü" eşleşme: iskelet eşi Arapça aday sorguya en çok bu SCA uzaklığında (``None`` = sınırsız).
FRENCH_ARABIC_GUARD_MAX: float | None = None
#: Arapça dökümde kendisi Batı alıntısı olan aday (``from French/Italian/English``) sayılmaz
#: (``kobalt`` ~ ``كوبالت``: Arapça da Batıdan almış).
FRENCH_ARABIC_GUARD_SKIP_WESTERN = False


def _arabic_skeleton_match(comparison: str, arabic: list[Any]) -> tuple[Any, float] | None:
    """R2: sorguyla aynı ünsüz iskeletli en yakın Arapça aday (yoksa None)."""
    skeletons = {s for s in _query_skeletons(comparison) if len(s) >= FRENCH_ARABIC_GUARD_MIN}
    same = [r for r in arabic if r["comparison"] and consonant_skeleton(r["comparison"]) in skeletons]
    if FRENCH_ARABIC_GUARD_SKIP_WESTERN:
        western = dump_loans(ARABIC, "from french") | dump_loans(ARABIC, "from italian") \
            | dump_loans(ARABIC, "from english")
        same = [r for r in same if (r["word"], r["gloss"] or "") not in western]
    if not same:
        return None
    best = min(same, key=lambda r: (label_distance(comparison, r["comparison"]), r["comparison"]))
    distance = label_distance(comparison, best["comparison"])
    if FRENCH_ARABIC_GUARD_MAX is not None and distance > FRENCH_ARABIC_GUARD_MAX:
        return None
    return best, distance


# --- Fransızca havuzu ve Fransızca aracılı (9e) ---------------------------------

FRENCH = "fr"
#: Kazananı Fransızcaya çevrilebilen diller: bunların uluslararası biçimleri
#: çoğunlukla Fransızca alıntıdır (pozitron, parti, metro).
FRENCH_VIA_FROM = ("fa", "hy", "el")
#: F2 (b) "aynı biçimin yakın karşılığı": Fransızca aday kazanan biçime en çok
#: :data:`FRENCH_NEAR` SCA uzaklığında VE sorguya kazanandan en çok
#: :data:`FRENCH_SLACK` daha uzak. (SCA ses sınıfları kabadır: 0,30'da
#: vatan~Bhoutan, sultan~question "yakın" çıkıyordu — ayar, 9e.)
FRENCH_NEAR = 0.15
FRENCH_SLACK = 0.10

#: Etiket adımında Fransızca kuralı (``data/cache/work/donor9e/PREREG.md``):
#: ``off``; ``f1`` — anlam havuzu dil başına ayrı sınır (``by_sense``
#: ``per_language``; paylaşılan 200'ü Arapça dolduruyordu); ``g1`` — paylaşılan
#: havuz + Fransızcaya ayrı 200'lük havuz; ``f2``/``g2`` — f1/g1 + "Fransızca
#: aracılı": D1 ateşlemediyse ve kazanan fa/hy/el adayı kendi dökümünde "from
#: French" ise ya da Fransızca havuzda yakın karşılığı varsa (bkz.
#: :data:`FRENCH_NEAR`) -> ``fr``, ``via`` = kazanan dil. Yalnız ETİKET; alıntı
#: gücü değişmez.
#:
#: ÖLÇÜLDÜ, ön kayıtlı (bf1eeea), KABUL: yeni Türkçe Wiktionary verici altını
#: (TDK+Nişanyan dışı, kör indeks), rapor bölümü bir kez (n=560)::
#:
#:              ayar n=290   rapor n=560   McNemar (yalnız aday / yalnız off)   Holm p
#:     off      0,466        0,463         —                                    —
#:     F2       0,528        0,550         68 / 19                              1e-7
#:     G2       0,548        0,548         57 /  9                              2e-9
#:
#: Korumalar (G2): Türkçe TDK+Nişanyan etiket train+dev 0,563 -> 0,628;
#: xturkic ayar verici tanıma 0,746 (aynı); Saha ``eval-donor`` 0,714 (aynı).
#: G2 önceden seçilmişti (Arapça havuzuna dokunmaz; F1'in dil başına havuzu
#: Türkçe train+dev'i 0,563 -> 0,529 düşürüyordu).
FRENCH_RULE = "g2"


def _french_via(comparison: str, lang: str, row: Any, distance: float,
                french: list[Any]) -> tuple[Any, float] | None:
    """Etiket Fransızcaya çevrilmeli mi? Evetse (temsilci satır, mesafe)."""
    if lang not in FRENCH_VIA_FROM:
        return None
    french = [r for r in french if r["comparison"]]

    def closest(members: list[Any]) -> tuple[Any, float]:
        best = min(members, key=lambda r: (label_distance(comparison, r["comparison"]), r["comparison"]))
        return best, label_distance(comparison, best["comparison"])

    near = [r for r in french if row["comparison"]
            and sca_distance(row["comparison"], r["comparison"]) <= FRENCH_NEAR
            and label_distance(comparison, r["comparison"]) <= distance + FRENCH_SLACK]
    if near:
        return closest(near)
    if (row["word"], row["gloss"] or "") in dump_loans(lang, "from french"):
        return closest(french) if french else (row, label_distance(comparison, row["comparison"]))
    return None


# --- Batı alıntısında Fransızca önceliği (9f) -----------------------------------

ITALIAN = "it"
#: Türkçe uluslararası sonek -> Fransızca karşılığı (karşılaştırma biçiminde).
#: Türkçe biçim bu sonekle bitiyor VE Fransızca havuzda karşılık sonekle biten
#: aday eşiğin (:data:`DONOR_DISTANCE_THRESHOLD`) altındaysa etiket ``fr``.
#: Gerekçe: Fransız imlası (``-tion``, ``-isme``, ``-ique``) Türkçe sesçil
#: yazıma İtalyancadan uzak düşer; ``organizasyon`` ~ it ``organizzazione``
#: 0,059, fr ``organisation`` 0,106. Soneki karşılıklı aramak şans benzerliğini
#: dışarıda tutar (yalnız biçim yakınlığı değil, biçimbilgisel koşutluk).
FRENCH_SUFFIXES = (
    ("syon", ("ion",)), ("zyon", ("ion",)),
    ("izm", ("isme",)), ("ist", ("iste",)),
    ("loji", ("logie",)), ("grafi", ("graphie",)), ("graf", ("graphe",)),
    ("metre", ("metre",)), ("metri", ("metrie",)),
    ("ör", ("eur",)), ("ik", ("ikue",)),
)
#: H2 "yakın beraberlik": İtalyanca kazandıysa ve en yakın Fransızca aday
#: sorguya en çok bu kadar daha uzaksa -> ``fr`` (Türkçedeki Batı alıntılarının
#: taban oranı: 9e ayar havuzunda Wiktionary fr 796 / it 89 -> %90 Fransızca).
FRENCH_TIE_EPSILON = 0.05

#: ``off`` | ``h1`` (sonek) | ``h2`` (yakın beraberlik) | ``h12`` (ikisi).
#: Ön kayıt ``data/cache/work/donor9f/PREREG.md`` (81fe3af). Yalnız ETİKET.
#:
#: ÖLÇÜLDÜ, ön kayıtlı, **H1 KABUL, H2 RED**. Yeni rapor: 9e'de kullanılmamış
#: (dil, etimon) gruplarından Türkçe Wiktionary fr 120 + it 120 (kör indeks),
#: bir kez; taban üretim (g2)::
#:
#:              9e ayar n=290   YENİ rapor n=240   McNemar (aday/off)   Holm p
#:     off      0,548           0,396              —                    —
#:     H1       0,583           0,438              10 / 0               0,004
#:     H2       0,579           0,400               7 / 6               1,0
#:     h12      0,600           0,433              15 / 6               (bilgi)
#:
#: H2 yeni veride İtalyancayı Fransızcaya çeviriyor (İtalyanca 21 -> 15/120):
#: taban oranı önceliği dengeli sınıflarda kazanç getirmedi. Korumalar (H1):
#: Türkçe TDK+Nişanyan etiket train+dev 0,628 -> 0,642; xturkic ayar verici
#: 0,746 ve Saha ``eval-donor`` 0,714 aynı; 9e raporu (görüldü, bilgi)
#: 0,548 -> 0,588. Kalan: yeni veride İtalyanca kaydı 21/120 (İtalyanca ->
#: Fransızca 52, -> Arapça 23, -> Yunanca 19).
WESTERN_RULE = "h1"


#: 9m R1 — H1'den çıkarılan Türkçe sonekler (Arapça/Farsça sözcüklerde de görülen
#: belirsiz sonlar). ``()`` = H1'in tüm sonekleri.
#: ⚠️ ÖLÇÜLDÜ (9m), RED: ``("ik",)`` (Vikisözlük'te -ik %45 Fransızca dışı) doğal oranlı rapor
#: 0,4648 -> 0,4614 (0/2); Türkçe train+dev 0,6416 -> 0,6348. H1 hiçbir altında Arapça bozmuyor.
WESTERN_SUFFIX_DROP: tuple[str, ...] = ()


def _french_suffix_pair(comparison: str) -> tuple[str, ...]:
    for turkish, french in FRENCH_SUFFIXES:
        if turkish in WESTERN_SUFFIX_DROP:
            continue
        if comparison.endswith(turkish) and len(comparison) > len(turkish) + 1:
            return french
    return ()


def _french_prior(comparison: str, lang: str, distance: float,
                  french: list[Any]) -> tuple[Any, float] | None:
    """Kazanan Fransızca değilken etiket Fransızcaya çevrilmeli mi (9f)?"""
    french = [r for r in french if r["comparison"]]
    if not french:
        return None

    def closest(members: list[Any]) -> tuple[Any, float]:
        best = min(members, key=lambda r: (label_distance(comparison, r["comparison"]), r["comparison"]))
        return best, label_distance(comparison, best["comparison"])

    if WESTERN_RULE in ("h1", "h12"):
        endings = _french_suffix_pair(comparison)
        if endings:
            paired = [r for r in french if r["comparison"].endswith(endings)]
            if paired:
                row, d = closest(paired)
                if d <= DONOR_DISTANCE_THRESHOLD:
                    return row, d
    if WESTERN_RULE in ("h2", "h12") and lang == ITALIAN:
        row, d = closest(french)
        if d <= distance + FRENCH_TIE_EPSILON:
            return row, d
    return None


# --- Dürüst etiket: şans düzeyindeki eşleşmede biçim gösterilmez (9n) -----------------------

#: 9n — şans düzeyindeki verici etiketinin gösterimi (ön kayıt ``data/cache/work/donor9n/PREREG.md``).
#: ``off``: etiket ve en yakın biçim her zaman gösterilir (4.3.1). Öbürlerinde "kesin" olmayan
#: (:func:`attribution_certain`) etikette verici BİÇİMİ gösterilmez ("yakın biçim bulunamadı") ve dil:
#: ``a1`` — yalnız aile düzeyinde (:data:`DONOR_FAMILIES`: "Arapça ya da Farsça", "Batı dili");
#: ``a2`` — doğal dağılım önseli + biçim ipuçları (:mod:`engine.nlp.donor_prior`), tek dil;
#: ``a3`` — a2'nin sonsalıyla seçilen AİLE. Yalnız Türkçe verici kümesinde (``TURKISH_DONORS``)
#: uygulanır (önsel Türkçenin); ``attribute_donor``un kendisi değişmez (Saha/xturkic aynı).
#:
#: ÖLÇÜLDÜ, ön kayıtlı (607da37), **A2 KABUL** (varsayılan). Yeni doğal oranlı TDK rapor altını
#: (n=600: ar 290 / fr 248 / fa 62; 9m dahil önceki altınların dışında), bir kez::
#:
#:            biçim kesinliği        doğal ağırlıklı dil doğruluğu   kapsama (biçimli etiket)
#:     off    0,451 (195/432)        0,458                           0,720
#:     a1     0,757 (171/226)        0,401 (18/87, p=5e-6) ✗         0,377
#:     a2     0,757 (171/226)        0,598 (94/10, p=5e-6)           0,377
#:     a3     0,757 (171/226)        0,467 (107/87, p=0,42)          0,377
#:
#: (i) Fisher: korunan 171/55 vs gizlenen 24/182, p=5e-44 (üçünde aynı kural). Belirsiz dilimde
#: (206 madde) a2 dili 171 doğru, en yakın biçimin dili 87. Gösterilen yanlış biçim 237 -> 55.
#: Korumalar: Türkçe train+dev doğal ağırlıklı 0,602 -> 0,726 (a1 0,552 ✗); Saha 0,7136 ve
#: xturkic 0,7459 aynı (Türkçe verici kümesi dışı). a3 de kabul koşullarını sağladı; kural
#: gereği rapor doğruluğu en yüksek olan (a2) seçildi.
DONOR_HONEST = "a2"

#: Kesin etiket: mesafe en çok bu, şans yüzdeliği 0 ve anlamda ortak içerik sözcüğü var ...
HONEST_MAX_DISTANCE = 0.25
#: ... ya da ünsüz iskeleti sorguyla aynı ve mesafe en çok bu (ortak içerik sözcüğüyle).
HONEST_SKELETON_MAX = 0.35
#: a2/a3'ün aday dilleri: biçim ipucu modelinde en az 100 eğitim sözcüğü olanlar
#: (İtalyanca/Yunanca/Ermenice Vikisözlük maddeleri önceki altınlarda tükendi).
HONEST_PRIOR_LANGS = ("ar", "fa", "fr")

#: Aile düzeyi etiketler (dil kodu -> aile üyeleri, gösterim).
DONOR_FAMILIES = {
    "ar": (("ar", "fa"), "Arapça ya da Farsça"), "fa": (("ar", "fa"), "Arapça ya da Farsça"),
    "fr": (("fr", "it"), "Batı dili (Fransızca ya da İtalyanca)"),
    "it": (("fr", "it"), "Batı dili (Fransızca ya da İtalyanca)"),
    "el": (("el", "hy"), "Yunanca ya da Ermenice"), "hy": (("el", "hy"), "Yunanca ya da Ermenice"),
}


def attribution_certain(attribution: DonorAttribution | None) -> bool:
    """Etiketin gösterdiği biçim şans düzeyinin üstünde mi? (9n tanısında seçilen kural)"""
    if attribution is None or attribution.sense_overlap < 1:
        return False
    if attribution.distance <= HONEST_MAX_DISTANCE and attribution.chance_percentile == 0:
        return True
    return attribution.skeleton_match and attribution.distance <= HONEST_SKELETON_MAX


@dataclass(frozen=True)
class HonestDonorLabel:
    """Kullanıcıya gösterilen verici etiketi (9n)."""

    #: Kesin etiket: dil + en yakın biçim gösterilir.
    certain: bool
    #: Tek dil kodu (kesin etikette ya da a2'de); aile etiketinde "".
    lang_code: str
    #: Aile etiketinde üyeler (a1/a3); öbür durumda ().
    family: tuple[str, ...] = ()
    #: ``yakınlık`` (en yakın biçim), ``önsel`` (a2), ``aile`` (a1), ``önsel-aile`` (a3).
    basis: str = "yakınlık"
    probability: float | None = None
    #: 9o: biçim-öncelikli aramanın bulduğu biçim (:func:`form_first_attribution`); gösterilen
    #: biçim ``attribute_donor``unki değil budur.
    form: DonorAttribution | None = None

    @property
    def code(self) -> str:
        """Ölçüm kodu: tek dil kodu ya da ``"ar|fa"``."""
        return self.lang_code or "|".join(self.family)

    @property
    def show_form(self) -> bool:
        return self.certain

    def text(self) -> str:
        from engine.nlp.borrowing_chain import language_name

        if self.family:
            return DONOR_FAMILIES[self.family[0]][1]
        return language_name(self.lang_code)

    def describe(self, attribution: DonorAttribution | None) -> str:
        if self.form is not None:
            return f"{self.form.describe()} (biçim-öncelikli arama: anlam havuzu dışından, dil önselle uyumlu)"
        if self.certain and attribution is not None:
            return attribution.describe()
        why = "verici belirsiz — şans düzeyinin üstünde yakın biçim bulunamadı"
        if self.basis == "önsel":
            p = f" {self.probability:.2f}".replace(".", ",") if self.probability is not None else ""
            return (f"muhtemelen {self.text()} (Türkçe alıntıların doğal dağılımı + biçim ipuçları,"
                    f" olasılık{p}); {why}")
        if self.basis == "önsel-aile":
            return f"{self.text()} (doğal dağılım + biçim ipuçları); {why}"
        return f"{self.text()}; {why}"


def honest_label(attribution: DonorAttribution | None, comparison: str, mode: str | None = None,
                 sense: str | None = None, form_first: str | None = None) -> HonestDonorLabel | None:
    """``attribute_donor`` çıktısının gösterilecek biçimi (bkz. :data:`DONOR_HONEST`).

    ``sense`` verilirse ve :data:`DONOR_FORM_FIRST` açıksa (9o) kesin olmayan etikette
    biçim-öncelikli arama denenir (a2'de); ``attribution`` ``None`` olabilir (anlam havuzu boş)
    ve o zaman yalnız biçim-öncelikli arama bir şey bulursa etiket döner, yoksa ``None``.
    """
    mode = DONOR_HONEST if mode is None else mode
    ff_mode = DONOR_FORM_FIRST if form_first is None else form_first
    if attribution is not None and (mode == "off" or attribution_certain(attribution)):
        return HonestDonorLabel(certain=True, lang_code=attribution.lang_code)
    if attribution is None and (mode != "a2" or ff_mode == "off" or sense is None):
        return None
    if mode == "a1":
        family = DONOR_FAMILIES.get(attribution.lang_code)
        if family is None:
            return HonestDonorLabel(False, attribution.lang_code, basis="yakınlık")
        return HonestDonorLabel(False, "", family[0], basis="aile")
    from engine.nlp import donor_prior

    post = donor_prior.posterior(comparison, list(HONEST_PRIOR_LANGS))
    if mode == "a2":
        lang = max(post, key=lambda k: (post[k], k))
        if ff_mode != "off" and sense is not None:
            found = form_first_attribution(comparison, sense, lang, ff_mode)
            if found is not None:
                return HonestDonorLabel(True, lang, basis="biçim-öncelikli", probability=post[lang], form=found)
        if attribution is None:
            return None
        return HonestDonorLabel(False, lang, basis="önsel", probability=post[lang])
    if mode == "a3":
        mass: dict[tuple[str, ...], float] = {}
        for lang, p in post.items():
            fam = DONOR_FAMILIES[lang][0]
            mass[fam] = mass.get(fam, 0.0) + p
        fam = max(mass, key=lambda k: (mass[k], k))
        return HonestDonorLabel(False, "", fam, basis="önsel-aile", probability=mass[fam])
    raise ValueError(f"bilinmeyen DONOR_HONEST: {mode}")


# --- 9o: biçim-öncelikli ikinci arama ----------------------------------------------------

#: Kesin olmayan (a2) etikette, anlam havuzundan BAĞIMSIZ ikinci arama. Aday: ünsüz iskeleti
#: sorgununkiyle aynı verici maddesi (:mod:`engine.db.donor_skeleton`), etiket mesafesi
#: :data:`FORM_FIRST_MAX` altında, anlamı DİLBİLGİSİ göndermesi değil ve sorgunun GENİŞ anlam
#: sözcükleriyle (:func:`form_first_tokens`) en az bir ortak içerik sözcüğü; dili a2'nin
#: önsel+ipucu dili (Farsça madde + önsel Arapça ise aynı Arap yazısı iskeletli Arapça madde,
#: D1 gibi). Bulunursa etiket KESİN sayılır ve bu biçim gösterilir; dil değişmez (a2'nin dili).
#: ``off`` · ``c1`` (geniş anlam: köprülü anlam + başlık köprüsü + Osmanlıca + özgün anlam,
#: d ≤ 0,15) · ``c2`` (c1 + Türkçe anlamın sözcük sözcük köprüsü, d ≤ 0,15) · ``c3`` (c2, d ≤ 0,20).
#: Bkz. ``data/cache/work/donor9o/PREREG.md``.
#:
#: ÖLÇÜLDÜ, ön kayıtlı (1b02f9d), **C2 KABUL**. Yeni doğal oranlı TDK rapor altını (n=600; ar 290,
#: fr 248, fa 62; önceki tüm altınların dışında, kör indeks, bir kez)::
#:
#:            kapsama   kazanç/kayıp (Holm p)   biçim kesinliği (gösterilen/yanlış)   acc_nat
#:     off    0,375     —                       0,693 (225/69)                         0,590
#:     c1     0,425     30/0 (2e-9)             0,718 (255/72)  ✗ < 0,72               0,590
#:     c2     0,458     50/0 (4e-15)            0,720 (275/77)                         0,608 (11/0)
#:     c3     0,477     61/0 (3e-18)            0,713 (286/82)  ✗ < 0,72               0,615 (15/0)
#:
#: Eklenen biçimlerin doğru etimon oranı c1 27/30, c2 42/50, c3 48/61 (hepsi tabanın üstünde; c1 ve c3
#: ön kayıttaki mutlak 0,72 eşiğini tabanın bu altında 0,693 olması yüzünden geçemedi). Kesin
#: etiketlerde gösterilen biçim hiç değişmedi; dil a2'ninki (acc_nat yalnız anlam havuzu boş
#: maddelerde artabilir). Türkçe train+dev (ayar): kapsama 0,580 -> 0,659, kesinlik 0,735 -> 0,736.
#: Korumalar: Saha ``eval-donor`` ve xturkic dört koşulda aynı (Türkçe verici kümesi dışı).
DONOR_FORM_FIRST = "c2"
FORM_FIRST_MAX = {"c1": 0.15, "c2": 0.15, "c3": 0.20}
#: Aday havuzunun (Arapça yazı eşi araması dahil) mesafe üst sınırı.
FORM_FIRST_POOL_MAX = 0.30


def _content_tokens(text: str) -> set[str]:
    from engine.db.donor_index import FUNCTION_WORDS, _sense_tokens

    return {t for t in _sense_tokens(text) if len(t) > 2 and t not in FUNCTION_WORDS}


@lru_cache(maxsize=20000)
def _word_bridge(token: str) -> frozenset[str]:
    from engine.db.sense_bridge import english_sense
    from engine.utils.orthography import to_comparison_form

    return frozenset(_content_tokens(english_sense(to_comparison_form(token))))


def form_first_tokens(comparison: str, sense: str, translate: bool) -> set[str]:
    """Geniş anlam sözcükleri: köprülü anlam ∪ başlığın İngilizce köprüsü ∪ Osmanlıca maddenin
    anlamı ∪ özgün anlam; ``translate`` ile Türkçe anlamın içerik sözcüklerinin tek tek köprüsü."""
    from engine.db.donor_index import _sense_tokens
    from engine.db.sense_bridge import english_sense, ottoman_sense

    out = (_content_tokens(bridged_sense(comparison, sense)) | _content_tokens(english_sense(comparison))
           | _content_tokens(ottoman_sense(comparison)) | _content_tokens(sense))
    if translate:
        for t in _sense_tokens(sense):
            if len(t) >= 3:
                out |= _word_bridge(t)
    return out


def form_first_attribution(comparison: str, sense: str, lang: str, mode: str | None = None) -> DonorAttribution | None:
    """Biçim-öncelikli arama (bkz. :data:`DONOR_FORM_FIRST`): ``lang`` dilinde biçimce çok yakın,
    anlamca örtüşen verici maddesi; yoksa ``None``."""
    from engine.db.donor_index import is_form_of
    from engine.db.donor_skeleton import by_skeleton

    mode = DONOR_FORM_FIRST if mode is None else mode
    if mode == "off" or not comparison or not sense or _pairwise() is None:
        return None
    index = _index()
    if not getattr(index, "exists", False):
        return None
    limit = FORM_FIRST_MAX[mode]
    tokens = form_first_tokens(comparison, sense, translate=mode in ("c2", "c3"))
    if not tokens:
        return None
    from engine.nlp.borrowing_detector import TURKISH_DONORS

    pool = []
    for row in by_skeleton(_query_skeletons(comparison), list(TURKISH_DONORS), donors=index.path):
        comp = row["comparison"] or ""
        if not comp or abs(len(comp) - len(comparison)) > max(3, len(comparison) // 2):
            continue
        d = label_distance(comparison, comp)
        if d <= FORM_FIRST_POOL_MAX:
            pool.append((d, row["id"], row))
    pool.sort(key=lambda x: (x[0], x[1]))
    near = [(d, row) for d, _, row in pool if d <= limit and not is_form_of(row["gloss"])]
    sensed = [(d, row) for d, row in near if tokens & _content_tokens(row["gloss"])]
    hit, via = None, ""
    for d, row in sensed:
        if row["lang_code"] == lang:
            hit = (d, row)
            break
    if hit is None and lang == ARABIC:
        for _, row in sensed:
            if row["lang_code"] != PERSIAN:
                continue
            sk = script_skeleton(row["word"])
            twin = [(d, r) for d, r in near if r["lang_code"] == ARABIC and sk and script_skeleton(r["word"]) == sk]
            if twin:
                hit, via = twin[0], PERSIAN
                break
    if hit is None:
        return None
    d, row = hit
    same = tuple(sorted({r["comparison"] for _, _, r in pool if r["lang_code"] == row["lang_code"]}))
    return DonorAttribution(
        lang_code=row["lang_code"], word=row["word"], comparison=row["comparison"], gloss=row["gloss"],
        distance=d, null_distance=_null_distance(len(comparison), same), source="kaikki",
        via=via, sense_overlap=len(tokens & _content_tokens(row["gloss"])), skeleton_match=True,
    )
