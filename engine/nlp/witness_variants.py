"""
Tanık yazım varyantları — çok biçimli madde ayrıştırma + sınırlı maliyetli bulanık eşleme.

TDK Tarama (13.-19. yy) ve Derleme (ağız) kayıtları sorgunun biçimini birebir
taşımaz: ``uçmak, (uçmağ, uçmah)``, ``gez (I), (kez)``, ``pıçak``,
``gaçırmak``, ``depik``. ``to_comparison_form`` statik ve global bir katlama
yapar; çok biçimli adımı TEK dizgiye yapıştırır (``uçmakuçmağuçmah``) ve ağız
alternasyonunu (söz sonu ``-ğ/-h/-k``, ötümlüleşme ``k-/g-``) tanımaz.

İki parça (bkz. ``data/cache/work/research/ZEMBEREK_ALGO.md`` §5, §2):

* ``split_forms(madde)`` — virgül/parantez/eğik çizgi ile yazılmış çok biçimli
  maddeyi ana biçim + varyantlara ayırır; eşsesli sıra no'ları (``(I)``,
  ``(II)``) atılır.
* ``variant_cost(a, b)`` — Zemberek ``CharacterGraphDecoder`` mantığı: ağırlıklı
  düzenleme uzaklığı, değiştirme maliyeti karakter ÇİFTİNE göre (ağız/Osmanlıca
  ikame sınıfları ucuz), toplam maliyet ``max_cost`` ile SINIRLI (sınır aşılınca
  erken çıkış). Hedef küme küçüktür (sorgu, gövdesi ve ``root_variants``
  yüzey/kök varyantları — birkaç biçim); trie gerekmez, çift başına DP yeter.
* ``link_witness(madde, sorgu)`` — maddenin biçimlerinden sorgunun hedef
  kümesine en ucuz bağı verir.

⚠️ Biçim bağı TEK BAŞINA tanıklık kanıtı değildir: ``kaz``/``gaz`` gibi
kısa biçimler ucuz ikameyle eşsesliye bağlanır. Bağ ancak anlam kontrolüyle
(Derleme ``meaning_check``) birlikte geçerlidir.

Ölçüm (2026-09-25; Türkçe altın TRAIN+DEV'den 90 kelime + uçmak/bıçak/
kaçırmak/tepik, motorun arama varyantlarıyla canlı Tarama/Derleme; 475 tekil
kayıt, 56 Tarama + 419 Derleme):

* Yaygınlık: Tarama kayıtlarının 21/56'sı çok biçimli adım. Derleme'de
  sorgunun kendisi (ya da ağız biçimi) olduğu ELLE doğrulanan ama anlam
  süzgecinden (MiniLM ≥ 0,50) geçemeyen 32 kayıt var — süzgeçten geçen 48
  kaydın yanında; rastgele 45 kayıtlık elle örneklemde doğru ağız biçimi olan
  11 kaydın yalnız 4'ü süzgeçten geçiyordu. Sebep biçim değil anlam: sözlük
  anlamı tek kelimelik standart biçimdir ("Bıçak", "Karın").
* Biçim bağı eşiği: süzgeçten geçen hiçbir Derleme kaydı biçimce bağsız
  değil (arama zaten varyantla yapılıyor) — biçim kapısı TEK başına bir şey
  değiştirmez. MiniLM + bulanık bağ ile ek kabul 64 kayıt / 57 doğru (0,89);
  yanlışlar ``is`` "İnsan" (es), ``tip`` "Pancar" (dip), ``gül`` (kül) —
  MiniLM'in zayıf eşleşmeleri. Bu yol BAĞLANMADI.
* Bağlanan yol (``dialect_names_query``): bulanık biçim bağı + sözlük anlamı
  sorgunun kendisi → 32 ek kayıt, elle 32/32 doğru (17'si ağız biçimi: pıçak,
  garın, gara, gabuk, gum, guru, ofak, suv, cıl, çine, yörü, bo…). Derleme
  tanığı olan kelime 27 → 47. Fiil sorgusunda anlam gövdeye İNMEZ:
  `yakınmak` ~ Derleme `yakın` "Yakın" eşseslisi bu yüzden elendi.
* audit2 alt örneklemi (40 kelime, canlı arama): virgül/parantezli tanık
  (G3) 9 → 1 kelime (kalan: Eski Yunanca verici biçimi); yeni bağ 2 kayıt
  (`gutlu` ~ kutlu, `zeynet` ~ ziynet), ikisi de doğru. `make eval-headline`
  ve `make eval-homonym` değişmedi (ikisi de yalnız yerel kaynakla koşar;
  Tarama/Derleme ağ kaynağıdır).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from engine.utils.orthography import to_comparison_form

#: Eşsesli sıra numarası: ``(I)``, ``(II)``, ``(IV)`` — biçim değil.
_SENSE_NO = re.compile(r"\(\s*[IVX]+\s*\)")
_SEPARATORS = re.compile(r"[,;/()]")


def split_forms(raw: str) -> list[str]:
    """Çok biçimli maddeyi ayrı biçimlere böler; ilk öğe ana biçimdir.

    ``uçmak, (uçmağ, uçmah)`` → ``[uçmak, uçmağ, uçmah]``;
    ``gez (I), (kez)`` → ``[gez, kez]``; ``dilmaç (dilmeç (I))`` →
    ``[dilmaç, dilmeç]``. Tek biçimli madde kendisini döndürür.
    """
    text = _SENSE_NO.sub(" ", raw or "")
    out: list[str] = []
    for part in _SEPARATORS.split(text):
        form = " ".join(part.split()).strip(" .-*")
        if form and to_comparison_form(form) and form not in out:
            out.append(form)
    return out


#: Ağız/Osmanlıca ikame sınıfları: aynı sınıftaki iki harf ucuz değişir.
_CHEAP_CLASSES: tuple[tuple[frozenset[str], float], ...] = (
    (frozenset("kgğh"), 0.3),   # ötümlüleşme / sızıcılaşma: kaçır- ~ gaçır-, uçmak ~ uçmah
    (frozenset("çc"), 0.3),
    (frozenset("pb"), 0.3),     # bıçak ~ pıçak
    (frozenset("td"), 0.3),     # tepik ~ depik
    (frozenset("ıiuü"), 0.4),   # dar ünlü kararsızlığı: aldırmak ~ aldurmak
    (frozenset("nŋ"), 0.3),     # Derleme genizsi ñ: çeñe ~ çene
    (frozenset("ei"), 0.5),     # ağızda e > i kapanması: irkek ~ erkek, çine ~ çene
)
#: Söz sonu ``-ğ/-h/-k`` (Tarama: uçmak ~ uçmağ ~ uçmah) daha da ucuz.
_FINAL_CLASS = frozenset("kğh")
_FINAL_COST = 0.2
_INDEL = 1.0
_SUBST = 1.0
#: Varsayılan sınır: en çok bir "tam" düzenleme ya da birkaç ucuz ikame.
DEFAULT_MAX_COST = 1.0


def _sub_cost(a: str, b: str, final: bool) -> float:
    if a == b:
        return 0.0
    if final and a in _FINAL_CLASS and b in _FINAL_CLASS:
        return _FINAL_COST
    for cls, cost in _CHEAP_CLASSES:
        if a in cls and b in cls:
            return cost
    return _SUBST


def variant_cost(a: str, b: str, max_cost: float = DEFAULT_MAX_COST) -> float:
    """İki biçim arasındaki ağırlıklı düzenleme maliyeti; sınır aşılırsa ``inf``.

    Biçimler önce ``to_comparison_form`` ile katlanır. Maliyet satır
    minimumu ``max_cost``u aşınca DP erken biter (sınırlı arama).
    """
    x, y = to_comparison_form(a), to_comparison_form(b)
    if not x or not y:
        return float("inf")
    if x == y:
        return 0.0
    if abs(len(x) - len(y)) * _INDEL > max_cost:
        return float("inf")
    prev = [j * _INDEL for j in range(len(y) + 1)]
    for i in range(1, len(x) + 1):
        cur = [i * _INDEL] + [0.0] * len(y)
        for j in range(1, len(y) + 1):
            final = i == len(x) and j == len(y)
            cur[j] = min(
                prev[j] + _INDEL,
                cur[j - 1] + _INDEL,
                prev[j - 1] + _sub_cost(x[i - 1], y[j - 1], final),
            )
        if min(cur) > max_cost:
            return float("inf")
        prev = cur
    return prev[-1] if prev[-1] <= max_cost else float("inf")


def _verb_stem(form: str) -> str:
    return form[:-3] if len(form) > 4 and form.endswith(("mak", "mek")) else form


def query_targets(word: str) -> list[str]:
    """Sorgunun bağlanabilir biçimleri: kendisi, mastarsız gövdesi ve
    ``root_variants`` yüzey/kök varyantları (``kitap → kitab``)."""
    from engine.nlp.root_variants import root_candidates, surface_variants

    own = to_comparison_form(word)
    if not own:
        return []
    out = [own]
    stem = _verb_stem(own)
    for form in (stem, *surface_variants(stem), *root_candidates(stem)):
        if form and form not in out:
            out.append(form)
    return out


@dataclass(frozen=True)
class WitnessLink:
    form: str        # maddenin bağlanan biçimi (ayrıştırılmış)
    target: str      # sorgunun hedef biçimi
    cost: float


def link_witness(raw: str, word: str, max_cost: float = DEFAULT_MAX_COST) -> WitnessLink | None:
    """Maddenin (çok biçimli olabilir) herhangi bir biçiminden sorguya en ucuz bağ.

    Fiil maddesinde ``-mak/-mek`` her iki tarafta da düşürülerek de denenir
    (``gaçırmak`` ~ ``kaçır``). Sınır içinde bağ yoksa ``None``.
    """
    targets = query_targets(word)
    best: WitnessLink | None = None
    for form in split_forms(raw):
        cf = to_comparison_form(form)
        for cand in dict.fromkeys((cf, _verb_stem(cf))):
            for target in targets:
                cost = variant_cost(cand, target, max_cost)
                if best is None or cost < best.cost:
                    best = WitnessLink(form, target, cost)
    if best is None or best.cost == float("inf"):
        return None
    return best


_SENSE_NUMBER = re.compile(r"^\s*\d+\s*\.\s*")
_GLOSS_SPLIT = re.compile(r"[,;:]")
#: Yalnız harflerden oluşan açıklama parçası: gönderme (``[-> dé (II)]``,
#: ``bk. gerçek``) ve niteleyicili açıklama (``O (Kuşu)`` — bir kuş adı) sayılmaz.
_PLAIN_GLOSS = re.compile(r"^[^\W\d_]+\.?$")


def gloss_names_query(meaning: str, word: str) -> bool:
    """Kaydın (ilk) anlamı sorgunun kendisi mi: Derleme ``pıçak`` "Bıçak.",
    ``gaçırmak`` "Kaçırmak, kaybetmek", ``tüş`` "Düş, rüya"."""
    first = _GLOSS_SPLIT.split(_SENSE_NUMBER.sub("", meaning or ""), 1)[0].strip()
    if not _PLAIN_GLOSS.match(first):
        return False
    gloss, own = to_comparison_form(first), to_comparison_form(word)
    # Fiil sorgusunda gövdeye inilmez: Derleme fiili mastarla verir
    # ("Kaçırmak"); `yakınmak` sorgusunda "Yakın" anlamlı `yakın` kaydı
    # eşseslidir (ölçüldü, audit2 alt örneklemi).
    return bool(gloss) and gloss == own


def dialect_names_query(form: str, meaning: str, word: str) -> WitnessLink | None:
    """Ağız kaydı sorgunun varyantı mı: biçim SINIRLI maliyetle bağlanır VE
    sözlük kaydın anlamını sorgunun kendisi olarak verir.

    Anlam kontrolünün sözlükçü eliyle yapılmış hâlidir: MiniLM ``Bıçak`` gibi
    tek kelimelik açıklamayı sorgunun TDK tanımına benzetemiyor (ölçüldü:
    ``pıçak`` "Bıçak" 0,39; ``garın`` "Karın" 0,15; ``gara`` "Kara" 0,11) ve
    doğru ağız biçimi düşüyordu. Biçim bağı şarttır: anlamı "Bıçak" olan her
    kayıt tanık olmaz, yalnız biçimi de bıçak'a ucuz ikameyle inenler.
    """
    if not gloss_names_query(meaning, word):
        return None
    return link_witness(form, word)
