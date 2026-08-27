"""
Ata ses seçimi — karşılaştırmalı yöntemin karar katmanı.

Bir hizalama sütunundaki seslerden ata sesi seçer. Önceki uygulamanın üç
ölçülmüş kusuru vardı:

1. **Denklikler aşırı ateşleniyordu.** ``{d, y, z, t, r} -> *d̮`` kuralı,
   sütunda bu beş sesten *herhangi ikisi* bulunduğunda devreye giriyordu.
   ``t`` ile ``r`` sıradan kelimelerde sürekli yan yana gelir; sonuç
   ``*arka -> *arca``, ``*toprak -> *tobrau`` gibi bozulmalardı. Artık kural
   yalnız sütundaki **bütün sesleri açıklıyorsa** uygulanır.

2. **Tanıklar eşit ağırlıktaydı.** Karşılaştırmalı yöntemde arkaik tanık
   ağır basar: Çuvaşça ``-r`` görüyorsa ötekiler ne derse desin ata ses
   ``*ŕ``tir. Oysa çoğunluk oyu Çuvaşça'yı 30 dile karşı 1 sayıyordu.

3. **Ünlüler için hiç kural yoktu** — düz çoğunluk oyu ``*bil -> *bel``,
   ``*kül -> *kul`` gibi bozulmalar üretiyordu.

Karar sırası artık şudur::

    1. Oğur tanıklı tanısal denklik   (rotasizm / lambdaizm)  -- kesin
    2. Sütunu tam açıklayan denklik   (söz başı ötümlüleşme…)
    3. Arkaiklik ağırlıklı oy          (Halaçça/Çuvaşça/Eski Türkçe ağır)
    4. Düz çoğunluk                    (son çare)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from engine.logging_setup import get_logger
from engine.nlp.multi_alignment import AlignedColumn

logger = get_logger(__name__)

#: Oğur (Bulgar) kolu. Lir-Şaz ayrımının **tanısal** tanığıdır: rotasizm ve
#: lambdaizm ancak bu kol görüldüğünde türetilebilir. Yoksa ata biçim
#: Proto-Türkçe değil Ana Ortak Türkçe düzeyindedir.
OGHUR_CODES = frozenset({"cv", "wot"})

#: Oğur kolunun **yaşayan** (atteste) tanığı. ``wot`` bir rekonstrüksiyondur;
#: tanısal denkliği destekler ama tek başına ``*PT`` düğümünü taşımaz —
#: rekonstrüksiyondan rekonstrüksiyon türetmek zincirleme belirsizliktir.
#: Bkz. ``west_old_turkic.PT_REQUIRES_LIVE_OGHUR``.
LIVE_OGHUR_CODES = frozenset({"cv"})

#: Arkaiklik ağırlıkları — bir tanığın oyunun kaç sayılacağı.
#:
#: Gerekçe, her dil için ayrı bir Türkoloji yerleşiğidir:
#:
#: * ``cv``  Oğur kolu; *ŕ/*ĺ ayrımının tek tanığı
#: * ``klj`` Halaçça (Arguca); ünlü uzunluğunu ve söz başı *h-'yi korur (Doerfer)
#: * ``otk`` Eski Türkçe; 8.-11. yy'da TANIKLANMIŞ, rekonstrüksiyon değil
#: * ``sah``/``dlg`` Yakutça-Dolganca; ünlü uzunluğunu korur
#: * ``tk``  Türkmence; ünlü uzunluğunu korur
#: * ``tr``/``az`` Oğuz kolu; söz başı ötümlüleşme YENİLİĞİNİ yapmıştır,
#:   bu yüzden söz başı kararında en az güvenilir tanıktır
ARCHAISM_WEIGHTS: dict[str, float] = {
    "cv": 3.0,
    # Oğur kolunun ikinci tanığı, ama ATTESTE DEĞİL: Çuvaşçanın altında.
    "wot": 2.2,
    "klj": 2.5,
    "otk": 2.5,
    "sah": 2.0,
    "dlg": 1.8,
    "tk": 1.6,
    "kim": 1.5,
    "tyv": 1.4,
    "ybe": 1.4,
    "clw": 1.3,
    "qwm": 1.3,
    "chg": 1.3,
    "ota": 1.2,
    "khk": 1.1,
    "cjs": 1.1,
    "alt": 1.1,
    "atv": 1.0,
    "kdr": 1.0,
    "bay": 1.0,
    "slq": 1.0,
    "ky": 1.0,
    "kk": 1.0,
    "kaa": 1.0,
    "nog": 1.0,
    "kum": 1.0,
    "krc": 1.0,
    "crh": 1.0,
    "tt": 1.0,
    "ba": 1.0,
    "uz": 0.9,
    "ug": 0.9,
    "gag": 0.8,
    "az": 0.7,
    "tr": 0.7,
}

DEFAULT_WEIGHT = 1.0

#: Ünlüler için AYRI ağırlıklar.
#:
#: ⚠️ Çuvaşça ünsüzlerde en arkaik tanıktır ama **ünlülerde değildir**:
#: Oğur kolu kapsamlı bir ünlü kaymasından geçmiştir (*ö > u, *e > ĭ …).
#: Tek bir ağırlık tablosu kullanmak ``*köŕ`` yerine ``*kuŕ`` üretiyordu —
#: Çuvaşça'nın ``u``su, dört dilin ``ö``südüne karşı 3,0 ağırlıkla kazanıyordu.
#:
#: Ünlülerde ağır basan tanıklar uzunluğu ve nitelik ayrımını koruyanlardır:
#: Halaçça, Türkmence, Yakutça-Dolganca ve tanıklanmış Eski Türkçe.
VOWEL_ARCHAISM_WEIGHTS: dict[str, float] = {
    "cv": 0.5,  # ünsüzde 3,0 — ünlüde en güvenilmez tanık
    "klj": 3.0,
    "otk": 2.5,
    "tk": 2.2,
    "sah": 2.0,
    "dlg": 1.8,
    "kim": 1.5,
    "tyv": 1.4,
    "ybe": 1.2,
    "clw": 1.2,
    "qwm": 1.2,
    "chg": 1.2,
    "ota": 1.1,
    "uz": 0.8,  # Özbekçe ünlü sistemi yeniden düzenlenmiştir
}

#: Ünlü sesler — ağırlık tablosu seçimi buna göre yapılır.
VOWELS = frozenset("aeıioöuüâîûēīūōā")


def weight_for(lang: str, sound: str) -> float:
    """Bir tanığın belirli bir seste kaç oy sayılacağı.

    Arkaiklik sese göre değişir: Çuvaşça ünsüzde belirleyici, ünlüde
    güvenilmezdir. Tek tablo kullanmak ölçülen bir hata kaynağıydı.
    """
    if sound in VOWELS:
        return VOWEL_ARCHAISM_WEIGHTS.get(lang, ARCHAISM_WEIGHTS.get(lang, DEFAULT_WEIGHT))
    return ARCHAISM_WEIGHTS.get(lang, DEFAULT_WEIGHT)


#: Öğrenilmiş örüntü oyunun devreye girmesi için gereken güven.
#:
#: Ölçüldü (dev bölümü, sütun düzeyi, n=206)::
#:
#:     eşik   kural    öğrenilmiş   melez   öğrenilmiş oyun payı
#:     0,0    0,7621     0,7816    0,7961          %98
#:     0,5    0,7621     0,7816    0,8010          %85
#:     0,6    0,7621     0,7816    0,7864          %72
#:     0,7    0,7621     0,7816    0,7670          %60
#:
#: 0,5 en iyi melezi veriyor: öğrenilmiş oy çoğunlukta ama düşük güvenli
#: kararlar elle yazılmış kurala bırakılıyor.
LEARNED_MIN_CONFIDENCE = 0.5

_PATTERN_TABLE: object | None = None
_PATTERN_TABLE_LOADED = False


def _pattern_table() -> object | None:
    """Öğrenilmiş ata ses örüntü tablosu — yoksa ``None``.

    ⚠️ Tablo yoksa motor **tümüyle kural tabanlı** çalışır. Bu bir hata
    değildir ama ölçüm farkı yaratır; ``reset_pattern_cache`` ile
    sıfırlanabilir.
    """
    global _PATTERN_TABLE, _PATTERN_TABLE_LOADED
    if not _PATTERN_TABLE_LOADED:
        from engine.nlp.proto_patterns import load

        _PATTERN_TABLE = load()
        _PATTERN_TABLE_LOADED = True
        if _PATTERN_TABLE is None:
            logger.info("Öğrenilmiş örüntü tablosu yok; yalnız kural katmanı")
    return _PATTERN_TABLE


def reset_pattern_cache() -> None:
    global _PATTERN_TABLE, _PATTERN_TABLE_LOADED
    _PATTERN_TABLE = None
    _PATTERN_TABLE_LOADED = False


@dataclass(frozen=True)
class DiagnosticRule:
    """Oğur tanığına dayanan tanısal denklik.

    ``oghur_sounds`` Oğur kolunda, ``common_sounds`` Ortak Türkçe kolunda
    görülürse ata ses kesindir. Bu, Lir-Şaz ayrımının tanımıdır.
    """

    oghur_sounds: frozenset[str]
    common_sounds: frozenset[str]
    proto: str
    note: str
    #: Kuralın geçerli olduğu konumlar.
    #:
    #: ⚠️ Rotasizm ve lambdaizm **söz başında yoktur**. Çuvaşça ``ś-`` ~ Ortak
    #: Türkçe ``j-`` denkliği söz başında ``*j-`` demektir, ``*ŕ-`` değil.
    #: Konum kısıtı olmadan ``*jan`` yerine ``*ŕan``, ``*jaŋï`` yerine
    #: ``*ŕaŋı`` üretiliyordu — ölçümde 11 kelimede bu hata vardı.
    positions: frozenset[str] = frozenset({"medial", "final"})


#: Lir-Şaz tanısal denklikleri. Bunlar "kural" değil, **tanım**dır: Çuvaşça
#: ``-r`` ~ Ortak Türkçe ``-z`` denkliği Proto-Türkçe ``*-ŕ`` demektir.
DIAGNOSTIC_RULES: tuple[DiagnosticRule, ...] = (
    DiagnosticRule(
        frozenset({"r"}),
        frozenset({"z"}),
        "ŕ",
        "Lir-Şaz rotasizmi: Çuvaşça -r ~ Ortak Türkçe -z < Proto-Türkçe *-ŕ",
    ),
    DiagnosticRule(
        frozenset({"s", "ś"}),
        frozenset({"z"}),
        "ŕ",
        "Çuvaşça -ś (< -r) ~ Ortak Türkçe -z < Proto-Türkçe *-ŕ",
    ),
    DiagnosticRule(
        frozenset({"l"}),
        frozenset({"ş"}),
        "ĺ",
        "Lambdaizm: Çuvaşça -l ~ Ortak Türkçe -ş < Proto-Türkçe *-ĺ",
    ),
    DiagnosticRule(
        frozenset({"l"}),
        frozenset({"s", "ş"}),
        "ĺ",
        "Çuvaşça -l ~ Ortak Türkçe -s/-ş < Proto-Türkçe *-ĺ",
    ),
)


@dataclass(frozen=True)
class Correspondence:
    """Konuma bağlı ses denkliği.

    ``members`` sütundaki **bütün** sesleri kapsamalıdır; kısmi örtüşmede
    kural uygulanmaz. Bu kısıt olmadan ``{d,y,z,t,r}`` gibi geniş kümeler
    sıradan kelimeleri bozar.
    """

    members: frozenset[str]
    proto: str
    note: str
    position: str
    #: Kural yalnız Oğur (Çuvaşça) tanığı varken uygulanır mı?
    #:
    #: ⚠️ ``*ŕ`` ve ``*ĺ`` ancak Lir-Şaz ayrımı görülebiliyorsa türetilebilir.
    #: Oğur tanığı olmadan iddia edilebilecek en derin düğüm Ana Ortak
    #: Türkçe'dir ve o düğümde ``*ŕ``/``*r``/``*z`` ayrımı ZATEN BİRLEŞMİŞTİR.
    #: Onsuz ``*ŕ`` yazmak, veriden çıkmayan bir ayrımı iddia etmektir.
    requires_oghur: bool = False
    #: Kuralın ateşlenmesi için sütunda bulunması ZORUNLU seslerden en az biri.
    #: Boşsa kısıt yoktur.
    #:
    #: ⚠️ Bu alan olmadan geniş kümeler yanlış ateşleniyordu: ``*sub``ın söz
    #: başı sütunu ``{s, ş}`` iken ``{y,c,j,ç,ş,s} -> *j-`` kuralı devreye
    #: girip ``*jub`` üretiyordu. Oysa Çuvaşça ``ş-`` burada ``*s-``in düzenli
    #: refleksidir; ``*j-`` demek için sütunda gerçekten bir ``y/c/j`` olmalı.
    core: frozenset[str] = frozenset()


#: Yerleşik Proto-Türkçe denklikleri.
CORRESPONDENCES: tuple[Correspondence, ...] = (
    # --- Söz başı ---
    # Oğuz kolu söz başı ötümsüzleri ötümlüleştirdi; ata biçim ötümsüzdür.
    Correspondence(
        frozenset({"d", "t"}),
        "t",
        "Söz başı ötümlüleşme: Oğuz d- ~ diğer t- < Proto-Türkçe *t-",
        "initial",
    ),
    Correspondence(
        frozenset({"g", "k"}),
        "k",
        "Söz başı ötümlüleşme: Oğuz g- ~ diğer k- < Proto-Türkçe *k-",
        "initial",
    ),
    Correspondence(
        frozenset({"g", "k", "h"}),
        "k",
        "Söz başı k- ~ g- ~ h- < Proto-Türkçe *k-",
        "initial",
    ),
    Correspondence(
        frozenset({"y", "c", "j", "ç", "ş", "s"}),
        "j",
        "Söz başı akıcı: y- ~ c- ~ j- ~ ç- ~ ş- < Proto-Türkçe *j-",
        "initial",
        core=frozenset({"y", "c", "j"}),
    ),
    Correspondence(
        frozenset({"y", "c", "j"}),
        "j",
        "Söz başı akıcı: y- ~ c- ~ j- < Proto-Türkçe *j-",
        "initial",
    ),
    Correspondence(
        frozenset({"b", "m"}),
        "b",
        "Genizsilleşme: b- ~ m- < Proto-Türkçe *b-",
        "initial",
    ),
    Correspondence(
        frozenset({"b", "p"}),
        "b",
        "Söz başı b- ~ p- < Proto-Türkçe *b-",
        "initial",
    ),
    # --- Söz sonu / her yer ---
    Correspondence(
        frozenset({"z", "r"}),
        "ŕ",
        "Ortak Türkçe -z ~ -r < Proto-Türkçe *-ŕ",
        "final",
        requires_oghur=True,
    ),
    Correspondence(
        frozenset({"ş", "l"}),
        "ĺ",
        "Ortak Türkçe -ş ~ -l < Proto-Türkçe *-ĺ",
        "final",
        requires_oghur=True,
    ),
    Correspondence(
        frozenset({"n", "ŋ"}),
        "ŋ",
        "Genizsil denkliği: -n- ~ -ŋ- < Proto-Türkçe *-ŋ-",
        "any",
    ),
    Correspondence(
        frozenset({"b", "v", "w"}),
        "b",
        "Ünsüz yumuşaması: b ~ v ~ w < Proto-Türkçe *b",
        "any",
    ),
    Correspondence(
        frozenset({"g", "ğ"}),
        "g",
        "Ünlü arası yumuşama: g ~ ğ < Proto-Türkçe *g",
        "any",
    ),
    Correspondence(
        frozenset({"k", "g", "ğ"}),
        "g",
        "Ünlü arası ötümlüleşme: k ~ g ~ ğ < Proto-Türkçe *g",
        "medial",
    ),
    # ⚠️ ``*d̮`` denkliğinin TANISAL sesleri ``y`` ve ``z``dir. Bunlar sütunda
    # yoksa ortada yalnız ötümlülük değişimi vardır ve ata ses ``*t``tir.
    # ``core`` kısıtı olmadan ``{d, t}`` sütunu da ``*d`` veriyordu:
    # ``*jumurtka`` yerine ``*yumurdka`` üretiliyordu.
    Correspondence(
        frozenset({"d", "y", "z"}),
        "d",
        "Klasik *d̮ denkliği: d ~ y ~ z",
        "medial",
        core=frozenset({"y", "z"}),
    ),
    Correspondence(
        frozenset({"d", "y", "z", "t"}),
        "d",
        "Klasik *d̮ denkliği: d ~ y ~ z ~ t",
        "medial",
        core=frozenset({"y", "z"}),
    ),
    Correspondence(
        frozenset({"d", "y"}),
        "d",
        "*d̮ denkliği: d ~ y",
        "medial",
        core=frozenset({"y"}),
    ),
    # Söz içi ötümlülük değişimi: tanısal ses yoksa ata ses ÖTÜMSÜZDÜR.
    Correspondence(
        frozenset({"d", "t"}),
        "t",
        "Söz içi ötümlüleşme: -d- ~ -t- < Proto-Türkçe *-t-",
        "medial",
    ),
    Correspondence(
        frozenset({"b", "p"}),
        "p",
        "Söz içi ötümlüleşme: -b- ~ -p- < Proto-Türkçe *-p-",
        "medial",
    ),
)


@dataclass
class ColumnDecision:
    """Bir sütun için verilen karar ve gerekçesi."""

    sound: str
    note: str | None
    method: str
    agreement: float
    is_diagnostic: bool = False
    #: ``ses -> göreli puan`` — sütunun **öteki** adayları (Faz D5).
    #:
    #: ⚠️ Tek bir sesle dönmek ölçülmüş bir tavan koyuyordu: top-1 %28,4
    #: iken N-best oracle **%43,2** — yani doğru cevap adayların içinde ama
    #: sıralamanın tepesinde değil. Adaylar taşınmazsa yeniden sıralama
    #: yapılamaz.
    alternatives: tuple[tuple[str, float], ...] = ()


def _weighted_counts(column: AlignedColumn) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for lang, sound in column.present.items():
        counts[sound] += weight_for(lang, sound)
    return dict(counts)


def _merge_alternatives(
    winner: tuple[str, float], others: tuple[tuple[str, float], ...]
) -> tuple[tuple[str, float], ...]:
    """Kazananı başa alıp ötekileri arkasına dizer, tekrarları eler."""
    merged: dict[str, float] = {winner[0]: max(winner[1], 0.5)}
    for sound, score in others:
        if sound not in merged:
            merged[sound] = score * 0.5
    return tuple(sorted(merged.items(), key=lambda kv: (-kv[1], kv[0])))


def _agreement(column: AlignedColumn) -> float:
    """Sütunun ne kadar hemfikir olduğu — güven skorunun asıl sinyali."""
    present = column.present
    if not present:
        return 0.0
    counts = _weighted_counts(column)
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def pick_proto_sound(column: AlignedColumn, position: str) -> ColumnDecision:
    """Bir sütundan ata sesi seçer.

    :param position: ``"initial"`` | ``"medial"`` | ``"final"``
    """
    present = column.present
    agreement = _agreement(column)
    if not present:
        return ColumnDecision("", None, "bos", 0.0)

    distinct = column.distinct
    if len(distinct) == 1:
        sound = next(iter(distinct))
        return ColumnDecision(sound, None, "tek_ses", 1.0, alternatives=((sound, 1.0),))

    # 1 — Oğur tanıklı tanısal denklik. Kesindir, önceliklidir.
    #
    # ⚠️ Tanısal kural YALNIZ **atteste** Oğur tanığıyla ateşlenir. ``wot``
    # (Batı Eski Türkçe) Oğurdur ama Macarcadaki alıntılardan GERİ
    # KURULMUŞTUR; "kesin" damgasını taşıyacak gözlem değildir. Ölçüldü:
    # ``wot`` tanısal kurala sokulunca kısalmış bir türev biçim (``ïsï``
    # ~ ``*issig``) hizalamayı kaydırıp ``*iŕsi`` üretiyordu. ``wot`` yine de
    # aşağıdaki 2. ve 3. adımlarda ağırlıklı tanık olarak sayılır.
    oghur_sounds = {s for lang, s in present.items() if lang in LIVE_OGHUR_CODES}
    if oghur_sounds:
        common_sounds = {s for lang, s in present.items() if lang not in OGHUR_CODES}
        for rule in DIAGNOSTIC_RULES:
            if position not in rule.positions:
                continue
            if oghur_sounds & rule.oghur_sounds and common_sounds & rule.common_sounds:
                # ⚠️ Tanısal karar bir TANIMDIR; alternatifi yoktur.
                return ColumnDecision(
                    rule.proto,
                    rule.note,
                    "tanisal",
                    agreement,
                    True,
                    alternatives=((rule.proto, 1.0),),
                )

    # 2 — Sütundaki BÜTÜN sesleri açıklayan denklik.
    best: Correspondence | None = None
    for rule in CORRESPONDENCES:
        if rule.position not in ("any", position):
            continue
        if rule.requires_oghur and not oghur_sounds:
            continue
        if not distinct <= rule.members:
            continue  # kısmi örtüşme yeterli DEĞİL
        if rule.core and not (distinct & rule.core):
            continue  # kuralın çekirdek sesi sütunda yok
        # En dar kural tercih edilir: geniş kümeler yanlış ateşlenmeye yatkındır.
        if best is None or len(rule.members) < len(best.members):
            best = rule
    if best is not None:
        return ColumnDecision(
            best.proto,
            best.note,
            "denklik",
            agreement,
            alternatives=_merge_alternatives(
                (best.proto, 1.0), _ranked(_weighted_counts(column))
            ),
        )

    # 3 — ÖĞRENİLMİŞ örüntü oyu (Faz D2).
    #
    # ⚠️ Elle yazılmış denkliklerden SONRA gelir. İlk kurulumda önce
    # geliyordu ve dilbilimsel olarak yerleşik iki kararı BOZDU::
    #
    #     {tr: y, kk: z, otk: d}  ->  *j   (doğrusu *d̮)
    #     *teŋiŕ                  ->  *teniŕ  (ŋ sütunu kayboldu)
    #
    # Elle yazılmış denklikler dar ve küratörlüdür; 135 kümeden öğrenilmiş
    # bir sayım onları geçemez. Öğrenilmiş oyun asıl yeri **arkaiklik
    # ağırlıklı oyun önü**dür: o yol en çok kullanılan (426 sütun) ama en
    # zayıf (0,606) karar yoludur ve elle atanmış "arkaiklik" katsayılarına
    # dayanır. Öğrenilmiş tablo onu gerçek sayımla değiştirir.
    learned = _pattern_table()
    if learned is not None:
        sound, confidence, support = learned.vote(present)
        if sound and confidence >= LEARNED_MIN_CONFIDENCE and support >= 2:
            return ColumnDecision(
                sound,
                f"öğrenilmiş örüntü oyu (güven {confidence:.2f}, {support} tanık çifti)",
                "ogrenilmis_oruntu",
                agreement,
                alternatives=_merge_alternatives(
                    (sound, confidence), _ranked(_weighted_counts(column))
                ),
            )

    # 4 — Arkaiklik ağırlıklı oy.
    counts = _weighted_counts(column)
    winner = max(sorted(counts), key=lambda s: counts[s])
    method = "arkaik_agirlik" if len(set(counts.values())) > 1 else "cogunluk"
    return ColumnDecision(
        winner, None, method, agreement, alternatives=_ranked(counts)
    )


def _ranked(counts: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Sütun adaylarını normalize puanlarıyla sıralar (Faz D5)."""
    total = sum(counts.values())
    if not total:
        return ()
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple((sound, score / total) for sound, score in ordered)


# --- Ata biçmin kendi makullüğü -------------------------------------------


def proto_plausibility(proto_form: str) -> tuple[float, list[str]]:
    """Üretilen ata biçim Proto-Türkçe olarak makul mü? → [0, 1] ve gerekçeler.

    Sütun uyumu yalnız **tanıkların birbiriyle** ne kadar uyuştuğunu ölçer;
    ortaya çıkan biçmin Türkçe olup olmadığını ölçmez. Uydurma bir kelime
    (``zzzqx`` ~ ``zzzqy``) tanıkları arasında son derece uyumludur ve bu
    yüzden yüksek güven alıyordu.

    Denetlenen dört yerleşik kısıt:

    * **Ünlü uyumu** — Proto-Türkçe kökler art/ön uyumludur.
    * **Ünlü/ünsüz dengesi** — ünlüsüz veya neredeyse ünlüsüz kök yoktur.
    * **Söz başı kümesi** — Proto-Türkçe söz başında ünsüz kümesi yoktur.
    * **Yasak söz başı sesler** — ``*f-``, ``*v-``, ``*z-``, ``*ž-``, ``*h-``
      Proto-Türkçe'de bulunmaz (``*p-`` de tartışmalıdır).
    """
    import unicodedata

    from engine.utils.phonotactics import VOWELS, has_vowel_harmony

    # Uzunluk işareti sesbirim değil, nicelik işaretidir: ``ā`` bir ünlüdür.
    # Normalize edilmezse ``*kāpuk`` "ünlüsüz" ve "söz başı küme" sayılıyordu.
    decomposed = unicodedata.normalize("NFD", proto_form.lstrip("*").lower())
    form = unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ).replace("ː", "")
    if not form:
        return 0.0, ["boş biçim"]

    penalties: list[str] = []
    score = 1.0

    vowels = [ch for ch in form if ch in VOWELS]
    if not vowels:
        score -= 0.5
        penalties.append("hiç ünlü yok — Proto-Türkçe kök ünlüsüz olamaz")
    elif len(vowels) / len(form) < 0.25:
        score -= 0.2
        penalties.append("ünlü oranı çok düşük")

    if len(vowels) >= 2 and not has_vowel_harmony(form):
        score -= 0.2
        penalties.append("ünlü uyumu ihlali")

    if form[0] in PROHIBITED_INITIALS:
        score -= 0.3
        penalties.append(f"Proto-Türkçe'de söz başı *{form[0]}- bulunmaz")

    if len(form) >= 2 and form[0] not in VOWELS and form[1] not in VOWELS:
        score -= 0.25
        penalties.append("söz başı ünsüz kümesi — Proto-Türkçe'de bulunmaz")

    # Aynı ünsüzün üç kez üst üste gelmesi hiçbir doğal dilde olmaz.
    for i in range(len(form) - 2):
        if form[i] == form[i + 1] == form[i + 2]:
            score -= 0.4
            penalties.append(f"üç kez tekrarlanan ses: {form[i]!r}")
            break

    return max(0.0, round(score, 3)), penalties


#: Proto-Türkçe'de söz başında bulunmayan sesler (Clauson, Erdal).
PROHIBITED_INITIALS = frozenset("fvzžhğcñŋlr")
