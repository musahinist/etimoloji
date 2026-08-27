"""
Ünlü uzunluğu rekonstrüksiyonu — en zayıf halkanın verisi zaten elimizdeydi.

Proto-Türkçe'de ünlü uzunluğu ayırt edicidir (``*ōt`` "ateş" ≠ ``*ot`` "ot")
ama Ortak Türkçe kollarının çoğu onu kaybetmiştir. Uzunluğu koruyan diller
sayılıdır: **Halaçça, Türkmence, Yakutça, Dolganca, Tofaca, Tuvaca**.

⚠️ Ölçülmüş durum: ``savelyevturkic``in ``Segments`` sütununda 478 uzunluk
işaretli biçim var ama dağılım çarpık — Halaçça 73, Yakutça 67, ve
**Türkmence yalnız 2**. Oysa Türkmence klasik olarak birincil uzunluk
tanığıdır; o veri kümesinin çevriyazısı uzunluğu yazmıyor.

İndirilmiş kaikki dökümlerinde ise ``sounds[].ipa`` alanında **4.031** gerçek
uzun ünlü duruyordu ve hiç işlenmiyordu: Halaçça 426, Türkmence 261,
Tuvaca 174, Yakutça 168. Sıfır indirme maliyetiyle.

Ölçüldü: altın standarttaki 77 uzun ünlülü maddenin **%36'sında** tanıkların
en az biri sözlükte uzunluk gösteriyor (``*ōt`` ← Türkmence ``o:t``,
``*tīŕ`` ← Halaçça ``ti:iz``).

⚠️ Uzunluk **yalnız koruyan dillerden** okunur. Türkiye Türkçesinin bir
kelimesinde uzun ünlü görünmesi (ör. Arapça alıntılarda) ata biçim hakkında
hiçbir şey söylemez.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

#: Ünlü uzunluğunu koruyan diller ve tanıklık ağırlıkları.
#:
#: Halaçça (Arguca) en arkaik tanıktır: Doerfer'in gösterdiği gibi hem
#: uzunluğu hem söz başı ``*h-``yi korur. Türkmence Oğuz kolunda uzunluğu
#: koruyan tek dildir. Yakutça-Dolganca Sibirya kolunda korur.
LENGTH_PRESERVING: dict[str, float] = {
    "klj": 1.0,   # Halaçça
    "tk": 0.95,   # Türkmençe
    "sah": 0.9,   # Yakutça
    "dlg": 0.85,  # Dolganca
    "kim": 0.8,   # Tofaca
    "tyv": 0.75,  # Tuvaca
}

#: Uzunluğu KAYBETMİŞ diller. Bunlarda görünen uzun ünlü ata biçim hakkında
#: kanıt değildir (çoğu Arapça/Farsça alıntı kaynaklıdır).
LENGTH_LOSING = frozenset({"tr", "az", "gag", "kk", "ky", "tt", "ba", "uz", "ug", "cv"})

#: Ata biçimde uzunluk iddia etmek için gereken asgari ağırlıklı tanık gücü.
LENGTH_THRESHOLD = 0.9

#: ⚠️ **Uzunluk yalnız ANLAM KISITI varken uygulanır.**
#:
#: Ölçüldü — kısıtsız hâlde ZARAR veriyor, kısıtlı hâlde nötr:
#:
#:     kesinlik ölçümü (altın standart, n=400)
#:     anlam kısıtı KAPALI   TP=23 FP=53   kesinlik 0,303
#:     anlam kısıtı AÇIK     TP=14 FP=10   kesinlik **0,583**
#:
#:     uçtan uca (n=395)          tam        NED
#:     uzunluk uygulanmıyor      0,2329     0,3757
#:     anlam kısıtlı uzunluk     0,2354     0,3709
#:
#: NED farkı −0,0048, %95 GA [−0,0121, +0,0025] → **anlamlı DEĞİL**;
#: tam doğrulukta p=1,0. Yani kazanç gösterilemiyor, ama zarar da yok ve
#: uzunluk tanığını göstermek dilbilimsel olarak doğru.
#:
#: **Kök neden eşadlılık:** Türkmence ``ot`` hem "ateş" (uzun ünlülü
#: ``*ōt``) hem "ot/bitki" (kısa ``*ot``) demektir; yazılışa göre arama
#: ikisini ayıramaz. ``sense`` verilince sözlük maddesinin anlamı sorgunun
#: anlamıyla eşleştirilir ve kesinlik ikiye katlanır.
#:
#: Anlam verilmemişse uzunluk UYGULANMAZ — o durumda kesinlik 0,30'dur ve
#: ölçülen zararı geri getirir.
APPLY_LENGTH_TO_PROTO = True

#: Karşılaştırma biçimindeki ünlüler.
VOWELS = frozenset("aeıioöuü")


@dataclass
class LengthEvidence:
    """Bir kelime için ünlü uzunluğu kanıtı."""

    #: ``ünlü_sırası -> ağırlıklı tanık gücü``
    positions: dict[int, float] = field(default_factory=dict)
    witnesses: list[str] = field(default_factory=list)
    #: Kanıt **anlam kısıtıyla** mı toplandı? Kısıtsız kanıt ata biçme
    #: uygulanmaz: kesinliği 0,30'dur ve ölçülen zararı geri getirir.
    sense_constrained: bool = False

    def is_long(self, vowel_index: int) -> bool:
        return self.positions.get(vowel_index, 0.0) >= LENGTH_THRESHOLD

    @property
    def any_evidence(self) -> bool:
        return bool(self.positions)

    def describe(self) -> str:
        if not self.witnesses:
            return "uzunluk tanığı yok"
        return "uzunluk tanıkları: " + ", ".join(self.witnesses)


@lru_cache(maxsize=1)
def _index() -> Any:
    from engine.db.lexicon_index import LexiconIndex

    return LexiconIndex()


def reset_cache() -> None:
    _index.cache_clear()


def _long_vowel_indices(ipa_lengths: str) -> int:
    """Kaç uzun ünlü var? (``"aːiː"`` -> 2)"""
    return ipa_lengths.count("ː")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zçğıöşü]+", (text or "").lower()))


def gather_evidence(
    witnesses: dict[str, str],
    *,
    max_lookups: int = 8,
    sense: str = "",
) -> LengthEvidence:
    """Uzunluk koruyan tanıklardan ünlü uzunluğu kanıtı toplar.

    Her tanık için sözlükte IPA aranır; bulunan uzun ünlüler **ünlü sırasına**
    göre ata biçmin ünlülerine eşlenir. Bu kaba bir eşlemedir (hizalama
    yapılmaz) ama ünlü sayısı diller arasında büyük ölçüde korunur.

    :param sense: kelimenin anlamı (İngilizce gloss veya kavram adı).
        Verilirse **eşadlılık filtresi** devreye girer: bulunan sözlük
        maddesinin anlamı sorgunun anlamıyla örtüşmüyorsa kanıt sayılmaz.

        Ölçüldü — bu kısıt kesinliği neredeyse ikiye katlıyor::

            anlam kısıtı KAPALI   TP=23 FP=53   kesinlik 0,303
            anlam kısıtı AÇIK     TP=14 FP=10   kesinlik 0,583

        Sebep: Türkmence ``ot`` hem "ateş" (``*ōt``) hem "ot/bitki"
        (``*ot``) demektir ve yazılışa göre arama ikisini ayıramaz.
    """
    sense_tokens = _tokens(sense)
    evidence = LengthEvidence(sense_constrained=bool(sense_tokens))
    index = _index()
    if not getattr(index, "exists", False):
        return evidence

    looked_up = 0
    for lang, form in sorted(witnesses.items()):
        if lang not in LENGTH_PRESERVING or looked_up >= max_lookups:
            continue
        comparison = to_comparison_form(form)
        if not comparison:
            continue
        looked_up += 1
        rows = index.lookup(comparison, languages=[lang], limit=4)
        if not rows:
            rows = index.fuzzy_lookup(comparison, max_distance=1, languages=[lang])[:4]

        # Eşadlılık filtresi: anlam verilmişse örtüşmeyen madde elenir.
        if sense_tokens:
            rows = [r for r in rows if _tokens(r.get("gloss", "")) & sense_tokens]
        marked = next((r["long_vowels"] for r in rows if r.get("long_vowels")), "")
        if not marked:
            continue

        weight = LENGTH_PRESERVING[lang]
        # Uzun ünlüler tanıktaki ünlü SIRASINA göre kaydedilir.
        vowel_position = 0
        source = unicodedata.normalize("NFD", (rows[0].get("ipa") or ""))
        long_at: set[int] = set()
        for character_index, character in enumerate(source):
            from engine.db.lexicon_index import IPA_VOWELS

            if character.lower() not in IPA_VOWELS:
                continue
            following = source[character_index + 1 : character_index + 3]
            if "ː" in following or ":" in following:
                long_at.add(vowel_position)
            vowel_position += 1

        if not long_at and _long_vowel_indices(marked):
            long_at = {0}

        for position in long_at:
            evidence.positions[position] = evidence.positions.get(position, 0.0) + weight
        evidence.witnesses.append(f"{lang} {rows[0]['word']} [{marked}]")
    return evidence


def apply_length(
    proto_form: str, evidence: LengthEvidence, *, force: bool = False
) -> str:
    """Ata biçmin ünlülerine uzunluk işareti (makron) koyar.

    Türkolojik gelenekte uzunluk makronla yazılır: ``*ot`` -> ``*ōt``.

    ⚠️ Varsayılan olarak **hiçbir şey yapmaz** — bkz.
    :data:`APPLY_LENGTH_TO_PROTO`. Ölçüldü: uygulandığında doğruluğu
    düşürüyor çünkü eşadlılık yüzünden iddiaların üçte ikisi yanlış.
    ``force=True`` yalnız deney ve ölçüm içindir.
    """
    if not force:
        if not APPLY_LENGTH_TO_PROTO:
            return proto_form
        # Anlam kısıtı yoksa kesinlik 0,30'dur; uygulamak zarar verir.
        if not evidence.sense_constrained:
            return proto_form
    if not evidence.any_evidence:
        return proto_form

    prefix = "*" if proto_form.startswith("*") else ""
    body = proto_form.lstrip("*")
    out: list[str] = []
    vowel_position = 0
    for character in body:
        out.append(character)
        if character in VOWELS:
            if evidence.is_long(vowel_position):
                out.append("̄")  # birleşik makron
            vowel_position += 1
    return prefix + unicodedata.normalize("NFC", "".join(out))
