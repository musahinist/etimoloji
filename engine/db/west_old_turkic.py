"""
Batı Eski Türkçe (Oğur/Bolgar) tanıkları — Oğur kolunun ikinci kaynağı.

Oğur kolunun **tek yaşayan tanığı Çuvaşçadır**. Ölçüldü: altın standardın
400 maddesinin yalnız **115'inde** (%28,7) Çuvaşça tanık var. Oğur tanığı
yoksa Lir-Şaz ayrımı görülemez ve iddia edilebilecek en derin düğüm
Proto-Türkçe değil **Ana Ortak Türkçe**dir (bkz.
``proto_phonology.DIAGNOSTIC_RULES``). Yani kapsam eksikliği doğrudan
``*PT`` iddiasının önünde duruyor.

Róna-Tas & Berta (2011) *West Old Turkic* Macarcadaki Türki alıntılardan
Oğur kolunu geri kurar: 430 kavram, 480 WOT biçimi, CLDF, CC-BY.

⚠️ **WOT biçimleri ATTESTE DEĞİLDİR.** Macarca alıntılardan geriye
kurulmuşlardır. Türkolojide standart bir kanıt türüdür (Çuvaşça ve Volga
Bulgarcası yazıtlarının yanında üçüncü sütun) ama rekonstrüksiyondan
rekonstrüksiyon türetmek zincirleme belirsizlik demektir. Bu yüzden:

* tanık kodu ``wot`` **ayrıdır**, ``cv`` ile karıştırılmaz;
* arkaiklik ağırlığı Çuvaşçadan (3,0) **düşüktür**;
* tek başına ``*PT`` iddiasını taşımaz — bkz. :data:`PT_REQUIRES_LIVE_OGHUR`.

## Bağlama sorunu

WOT ile ``savelyevturkic`` arasında ortak akrabalık kümesi kimliği YOKTUR.
Ortak olan tek anahtar Concepticon'dur. Ama kavram eşlemesi **tek başına
yetmez**: altın standartta bir kavramı birden çok küme paylaşır (ölçüldü —
GRASS 4, JUMP 4, WIND 3, BURN 3 küme). Kavrama göre bağlamak dört kümenin
dördüne de aynı WOT biçimini takardı; en çok üçü yanlış olurdu.

Bu yüzden biçim düzeyinde bir süzgeç var: WOT biçimi kümenin **Ortak Türkçe
tanıklarına** olan uzaklığıyla puanlanır (:func:`link_witness`).

⚠️ Puanlamada altın ``Root`` **kullanılmaz**. Kullanılsaydı tanık cevaba
benzediği için seçilirdi ve ölçüm sızıntılı olurdu.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from engine.config import CLDF_DIR
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

DATASET = "ronataswestoldturkic"

#: WOT'un CLDF dil kimliği ve Glottocode'u (bolg1249 = Bolgar).
WOT_LANGUAGE_ID = "WOT"

#: Motorun tanık kodu.
WOT_CODE = "wot"

#: ⚠️ ``*PT`` düğümü için YAŞAYAN Oğur tanığı (Çuvaşça) gerekir mi?
#:
#: ``True`` — WOT'un kendisi bir rekonstrüksiyon olduğu için tek başına
#: Lir-Şaz ayrımını "gözlemlemiş" saymayız. WOT tanısal denkliği
#: **destekler** ama Çuvaşça yoksa iddia ``*PCT`` düzeyinde kalır.
PT_REQUIRES_LIVE_OGHUR = True

#: ⚠️ **WOT tanığı varsayılan olarak girdiye EKLENMEZ.**
#:
#: Ölçüldü (altın standart, n=400, dürüst koşul):
#:
#:     WOT kapalı   tam 0,2350   NED 0,3809   BCFS 0,516   ED 1,933
#:     WOT açık     tam 0,2325   NED 0,3814   BCFS 0,516   ED 1,935
#:
#: Bağlanan 5 kümenin yalnız **biri** rekonstrüksiyonu değiştiriyor
#: (HOT: ``*issig`` -> ``*issi``) ve o değişiklik yanlış. Yani ölçülen net
#: etki tek bir maddedir — istatistiksel olarak gürültü, ama işareti
#: negatif ve **hiçbir kazanç ölçülemedi**.
#:
#: Kök neden: Róna-Tas & Berta külliyatı sözlük maddeleridir ve kümenin
#: sorduğu kökün türevini taşıyabilir; buna karşılık ``wot``un arkaiklik
#: ağırlığı yüksektir (2,2) ve eksik hecesi sütunu kazanır.
#:
#: Veri, künyesi ve bağlayıcı yerinde duruyor; bayrak açıldığında ölçüm
#: yeniden koşulabilir. Ölçülemeyen bir iyileştirmeyi sessizce açmak, bu
#: projede hiç açmamaktan kötü sayılır.
USE_WEST_OLD_TURKIC = False

#: Oğur ~ Ortak Türkçe denklikleri. Bir WOT biçimini Ortak Türkçe tanıklarla
#: karşılaştırmadan önce **beklenen Ortak Türkçe refleksine** çevirmek gerekir;
#: aksi hâlde düzenli denklik "fark" olarak sayılır ve doğru tanık elenir.
#:
#: ``śel`` ~ Türkçe ``yel``: ś- ~ j- söz başı denkliği.
#: ``tir`` ~ Türkçe ``diz``: -r ~ -z rotasizmi.
OGHUR_TO_COMMON: tuple[tuple[str, str, str], ...] = (
    ("s", "y", "initial"),   # Oğur ś- ~ Ortak Türkçe j-/y-
    ("ş", "y", "initial"),
    ("c", "y", "initial"),
    ("r", "z", "final"),     # rotasizm
    ("l", "ş", "final"),     # lambdaizm
    ("h", "k", "initial"),   # WOT χ- ~ Ortak Türkçe k-
)

#: Bir WOT biçiminin tanık sayılması için tanıklara olan azami normalize
#: düzenlenme uzaklığı.
#:
#: Önsel gerekçe: 0,35 kabaca "üç sesten biri farklı" demektir; bunun ötesi
#: aynı kavramın BAŞKA bir kökü olur.
#:
#: Ölçüldü (altın standart, n=400) — 0,50'de 8 bağ kuruluyor ve biri
#: yanlış: BREAST kümesine ``käbäl`` takılıyor (skor 0,400), oysa kümenin
#: gerçek Oğur tanığı Çuvaşça ``kəʷgəʷr``dir. 0,35'te o bağ düşüyor,
#: kalan 7'nin 7'si doğru. ⚠️ Bu bir eşik AYARIDIR ve n=8'de yapılmıştır;
#: bağımsız bir kümede doğrulanmadı.
LINK_MAX_DISTANCE = 0.35

#: Tanığın uzunluğu, kümenin tanıklarının **ortanca** uzunluğundan en çok
#: bu kadar ses ayrılabilir.
#:
#: Önsel gerekçe: WOT külliyatı sözlük maddeleridir ve kümenin sorduğu kökün
#: **türevini** taşıyabilir. ``ïsï`` ile ``*issig``, ``sek`` ile ``*sekir``
#: aynı köktendir ama aynı BİÇİM değildir. Bir tanık, taşımadığı bir ses
#: hakkında oy kullanamaz: eksik hece hizalamayı kaydırır.
#:
#: Ölçüldü: kısıt SAND (``χumakï`` ~ ``*kum``) ve JUMP (``sek`` ~ ``*sekir``)
#: bağlarını düşürüyor — ikisi de türev/taban uyuşmazlığı.
#:
#: ⚠️ Kısıt HOT'u **kurtarmıyor**: ``*issig`` kümesinde tanıkların ortancası
#: zaten ``issi`` (4 ses), yani Ortak Türkçenin çoğu ``-g``yi düşürmüştür;
#: ``ïsï`` (3 ses) ortancadan yalnız 1 sapıyor ve süzgeci geçiyor. Bu tek
#: bağ altın standartta ölçülen tek davranış değişikliğini yapıyor ve
#: YANLIŞ yöne götürüyor (bkz. :data:`USE_WEST_OLD_TURKIC`).
LINK_MAX_LENGTH_GAP = 1

#: Aynı kavramı paylaşan kümeler arasında seçim yapmak için gereken en az
#: puan farkı. Fark bundan küçükse **hiçbirine** bağlanmaz: yanlış kümeye
#: takılmış bir tanık, hiç tanık olmamasından kötüdür.
LINK_MIN_MARGIN = 0.10


@dataclass(frozen=True)
class WotForm:
    """Tek bir Batı Eski Türkçe biçimi."""

    form: str
    comparison: str
    concept: str
    concepticon_gloss: str
    parameter_id: str

    @property
    def is_reconstructed(self) -> bool:
        """Her zaman ``True`` — bütün WOT külliyatı geri kurulmuştur."""
        return True


def _normalised_distance(a: str, b: str) -> float:
    """Normalize Levenshtein (0 = aynı, 1 = tamamen farklı)."""
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def to_common_turkic_reflex(comparison: str) -> set[str]:
    """WOT biçmin Ortak Türkçede **beklenen** yazılışlarını üretir.

    Denklikler seçenekli olduğu için tek bir biçim değil bir **küme** döner;
    karşılaştırmada en yakını kullanılır. Kombinatorik patlamayı önlemek için
    her denklik ayrı ayrı uygulanır (çapraz çarpım alınmaz).
    """
    if not comparison:
        return set()
    out = {comparison}
    for oghur, common, position in OGHUR_TO_COMMON:
        if position == "initial" and comparison.startswith(oghur):
            out.add(common + comparison[1:])
        elif position == "final" and comparison.endswith(oghur):
            out.add(comparison[:-1] + common)
    return out


@lru_cache(maxsize=1)
def load_forms(directory: Path | None = None) -> dict[str, tuple[WotForm, ...]]:
    """Concepticon glossuna göre indekslenmiş WOT biçimleri.

    Veri kümesi indirilmemişse **boş sözlük** döner ve modül sessizce devre
    dışı kalır; eksik veri istisna değildir.
    """
    base = Path(directory) if directory else CLDF_DIR / DATASET
    forms_path, params_path = base / "forms.csv", base / "parameters.csv"
    if not forms_path.exists() or not params_path.exists():
        logger.info("WOT veri kümesi yok (%s) — Oğur ek tanığı devre dışı", base)
        return {}

    with params_path.open(encoding="utf-8", newline="") as handle:
        params = {
            row["ID"]: (row.get("Concepticon_Gloss") or "", row.get("Name") or "")
            for row in csv.DictReader(handle)
        }

    index: dict[str, list[WotForm]] = defaultdict(list)
    with forms_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Language_ID") != WOT_LANGUAGE_ID:
                continue
            gloss, name = params.get(row["Parameter_ID"], ("", ""))
            if not gloss:
                continue
            form = (row.get("Form") or "").strip()
            comparison = to_comparison_form(form)
            if not comparison:
                continue
            index[gloss].append(
                WotForm(form, comparison, name, gloss, row["Parameter_ID"])
            )
    logger.info("WOT: %d kavramda %d biçim", len(index), sum(map(len, index.values())))
    return {gloss: tuple(items) for gloss, items in index.items()}


def reset_cache() -> None:
    load_forms.cache_clear()


def candidates_for(concepticon_gloss: str) -> tuple[WotForm, ...]:
    return load_forms().get(concepticon_gloss or "", ())


def score_against(candidate: WotForm, witnesses: dict[str, str]) -> float:
    """WOT adayının Ortak Türkçe tanıklara **en yakın** uzaklığı.

    ⚠️ Yalnız tanıklar kullanılır; altın ``Root`` asla girmez.
    """
    reflexes = to_common_turkic_reflex(candidate.comparison)
    best = 1.0
    for form in witnesses.values():
        target = to_comparison_form(form)
        if not target:
            continue
        for reflex in reflexes:
            best = min(best, _normalised_distance(reflex, target))
    return best


def _median_witness_length(witnesses: dict[str, str]) -> int:
    lengths = sorted(len(to_comparison_form(f)) for f in witnesses.values() if f)
    if not lengths:
        return 0
    return lengths[len(lengths) // 2]


def link_witness(
    concepticon_gloss: str,
    witnesses: dict[str, str],
    *,
    max_distance: float = LINK_MAX_DISTANCE,
    min_margin: float = LINK_MIN_MARGIN,
    max_length_gap: int = LINK_MAX_LENGTH_GAP,
) -> WotForm | None:
    """Kümeye takılacak WOT tanığını seçer; emin değilse ``None`` döner.

    İki kısıt birden aranır:

    1. En iyi adayın uzaklığı ``max_distance``ın altında olmalı.
    2. İkinciyle arasındaki fark ``min_margin``dan büyük olmalı — aynı
       kavramda birden çok WOT biçimi varsa ve ikisi de benzer puandaysa
       hangisinin bu kümeye ait olduğu **veriden çıkmıyor** demektir.
    3. Uzunluğu tanıkların ortancasından ``max_length_gap``tan fazla
       ayrılmamalı — bkz. :data:`LINK_MAX_LENGTH_GAP`.

    ⚠️ Kavram düzeyinde eşleme tek başına yetmez: altın standartta GRASS'ı 4,
    JUMP'ı 4, WIND ve BURN'ü 3'er küme paylaşıyor. Süzgeçsiz bağlama o
    kümelerin hepsine aynı biçimi takardı.
    """
    pool = candidates_for(concepticon_gloss)
    if not pool or not witnesses:
        return None
    median = _median_witness_length(witnesses)
    usable = [
        c for c in pool if not median or abs(len(c.comparison) - median) <= max_length_gap
    ]
    if not usable:
        return None
    scored = sorted(((score_against(c, witnesses), c) for c in usable), key=lambda p: p[0])
    best_score, best = scored[0]
    if best_score > max_distance:
        return None
    if len(scored) > 1 and scored[1][0] - best_score < min_margin:
        return None
    return best


def coverage_report(items: list) -> dict[str, object]:
    """Altın standartta WOT'un Oğur kapsamına katkısı.

    ``items`` :class:`engine.evaluation.gold.GoldItem` listesidir; bağımlılık
    tersine dönmesin diye tip yazılmaz.
    """
    live = sum(1 for i in items if "Chuvash" in i.witnesses)
    linked = [i for i in items if link_witness(i.concepticon_gloss, i.witnesses)]
    added = [i for i in linked if "Chuvash" not in i.witnesses]
    total = len(items) or 1
    return {
        "n": len(items),
        "live_oghur": live,
        "live_oghur_rate": round(live / total, 4),
        "wot_linked": len(linked),
        "wot_only": len(added),
        "combined_rate": round((live + len(added)) / total, 4),
    }
