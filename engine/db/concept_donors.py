"""
Kavram hizalı Moğol ve Tunguz verici listeleri — YALNIZ verici ETİKETİ için.

Verici havuzları dengesiz: kaikki Rusçası ~441 bin madde, Moğolcası 6.480,
Evenkicesi 599. Verici etiketi (``donor_proximity.attribute_donor``) dil
başına en yakın maddeyi seçtiği için, doğru verici maddesi havuzda yoksa
etiket yanlış dile kayar: Moğolca alıntıların 45/166'sı Rusça, Tunguzca
alıntıların 1/14'ü doğru etiketleniyordu.

Bu modül iki yerel CLDF kaynağından kavram hizalı biçimleri okur:

* **NorthEuraLex** (Dellert ve ark. 2020): Halha, Buryatça, Kalmukça,
  Evenkice, Nanayca; dil başına ~1.000–1.300 kavram, IPA.
* **robbeetstriangulation** (Robbeets & Bouckaert, CC-BY-4.0): 253 kavram,
  Moğol ve Tunguz çeşitlerinin hepsi (Orta Moğolca dahil).

⚠️ Dil kodu çakışması: NorthEuraLex ``khk`` = Halha MOĞOLCASI; motorda
``khk`` = Hakasça. Burada hiçbir biçim kendi dil koduyla dönmez: her biçim
motorun verici HAVUZ koduna (``mn`` = Moğolca, ``evn`` = Tunguzca) bağlanır,
çeşidin adı ayrıca taşınır.

⚠️ **Alıntı GÜCÜNE girmez.** Starling Moğolcası güç havuzuna katıldığında
WOLD "alıntı mı?" F'si 0,615'ten 0,584'e düşmüştü: Türk-Moğol ortak
sözvarlığı miras kelimeleri de yakın gösterir. Bu listeler yalnız alıntı
olduğu zaten söylenmiş kelimenin vericisini seçerken kullanılır.

⚠️ robbeetstriangulation deposunun kuralı: Transavrasya verisi akrabalık
kanıtına katılmaz, yalnız temas analizinde kullanılır. Verici havuzu bu
kuralla uyumludur.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from engine.config import CLDF_DIR
from engine.nlp.donor_proximity import DONOR_DISTANCE_THRESHOLD
from engine.utils.orthography import to_comparison_form

#: NorthEuraLex dil kimliği -> motor verici havuzu.
NORTHEURALEX_POOLS = {
    "khk": "mn",  # Halha MOĞOLCASI (motorda khk = Hakasça)
    "bua": "mn",
    "xal": "mn",
    "evn": "evn",
    "gld": "evn",
}

#: Rusça alıntı süzgecinin SCA eşiği: üretimdeki verici yakınlığı eşiğinin
#: kendisi (aynı kavram — "SCA mesafesi bu kadar yakınsa alıntı").
RUSSIAN_LOAN_DISTANCE = DONOR_DISTANCE_THRESHOLD

#: robbeetstriangulation ``Family`` sütunu -> motor verici havuzu.
ROBBEETS_POOLS = {"Mongolic": "mn", "Tungusic": "evn"}

#: IPA bölütü -> karşılaştırma alfabesi, ``to_comparison_form``un IPA'yı
#: Türk yazımı sanıp yanlış okuduğu sesler için.
#:
#: ⚠️ ``to_comparison_form`` IPA ``j``yi ``j`` (= ʒ), IPA ``y``yi ``y``
#: (= j) bırakır; ``d͡ʒ``yi ``dj``, ``t͡ʃ``yi ``tş`` yapar ve ``ɮ ɬ ʊ ɐ ɒ``yi
#: siler. Karşılaştırma biçimi kaikki ve Starling ile aynı alfabede olmalı,
#: yoksa mesafe yazım farkını ölçer.
_IPA_SEGMENT = {
    "j": "y", "y": "ü", "ʏ": "ü", "ʉ": "ü", "ɥ": "y",
    "dʒ": "c", "d͡ʒ": "c", "ʤ": "c", "dʑ": "c", "d͡ʑ": "c", "ǯ": "c", "ǰ": "c",
    "tʃ": "ç", "t͡ʃ": "ç", "ʧ": "ç", "tɕ": "ç", "t͡ɕ": "ç", "č": "ç",
    "ʒ": "j", "ž": "j", "ʐ": "j",
    "ʂ": "ş", "š": "ş",
    "ɮ": "l", "ɬ": "l", "ʎ": "l", "ɫ": "l",
    "ʊ": "u", "ɐ": "a", "ɒ": "o", "ɤ": "ı", "ɘ": "e", "ɵ": "ö",
    "ɦ": "h", "ʔ": "", "ɲ": "n", "ń": "n",
}
_LENGTH_MARKS = "ːˑʰʲʷˠ̥̩̯̃̆̚"


def segment_comparison(segments: str) -> str:
    """CLDF ``Segments`` sütunu -> karşılaştırma biçimi.

    ``a/b`` gösteriminde sağ taraf (ses değeri) alınır; ``+`` ve ``_``
    biçimbirim/sözcük sınırıdır, atılır.
    """
    out: list[str] = []
    for token in (segments or "").split():
        if "/" in token:
            token = token.split("/")[-1]
        if token in {"+", "_", "#", "-"}:
            continue
        bare = "".join(ch for ch in token if ch not in _LENGTH_MARKS)
        mapped = _IPA_SEGMENT.get(bare)
        out.append(mapped if mapped is not None else to_comparison_form(bare))
    return "".join(out)


@dataclass(frozen=True)
class ConceptDonorForm:
    """Kavram hizalı bir verici biçimi."""

    #: Motorun verici havuz kodu (``mn`` / ``evn``).
    pool: str
    #: Çeşidin adı (``Buryat``, ``Nanai`` …).
    variety: str
    form: str
    comparison: str
    #: Kavram adı (Concepticon + kaynak adı), anlam eşleşmesi için.
    gloss: str
    source: str
    #: Concepticon kavramı (Rusça alıntı süzgeci için).
    concept: str = ""


def _gloss_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ]+", (text or "").lower()) if t)


def _parameters(path: Path) -> dict[str, tuple[str, str]]:
    """Kavram kimliği -> (anlam metni, Concepticon adı)."""
    out: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Name") or "").strip()
            gloss = (row.get("Concepticon_Gloss") or "").strip()
            out[row["ID"]] = (" ".join(dict.fromkeys(x for x in (gloss, name) if x)), gloss)
    return out


def _read(dataset: str, pools: dict[str, str], *, by_family: bool) -> list[ConceptDonorForm]:
    from engine.db.donor_index import MAX_LENGTH, MIN_LENGTH

    directory = CLDF_DIR / dataset
    forms_path = directory / "forms.csv"
    if not forms_path.exists():
        return []
    languages: dict[str, tuple[str, str]] = {}
    with (directory / "languages.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("Family") or "") if by_family else row["ID"]
            if key in pools:
                languages[row["ID"]] = (pools[key], row.get("Name") or row["ID"])
    glosses = _parameters(directory / "parameters.csv")
    out: list[ConceptDonorForm] = []
    with forms_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            language = languages.get(row.get("Language_ID") or "")
            if language is None:
                continue
            comparison = segment_comparison(row.get("Segments") or "")
            if not MIN_LENGTH <= len(comparison) <= MAX_LENGTH:
                continue
            gloss, concept = glosses.get(row.get("Parameter_ID") or "", ("", ""))
            out.append(ConceptDonorForm(
                pool=language[0], variety=language[1], form=row.get("Form") or row.get("Value") or "",
                comparison=comparison, gloss=gloss, source=dataset, concept=concept,
            ))
    return out


def _russian_forms() -> dict[str, list[str]]:
    """NorthEuraLex Rusçası: Concepticon kavramı -> karşılaştırma biçimleri."""
    out: dict[str, list[str]] = {}
    for form in _read("northeuralex", {"rus": "ru"}, by_family=False):
        if form.concept:
            out.setdefault(form.concept, []).append(form.comparison)
    return out


def _is_russian_loan(form: ConceptDonorForm, russian: dict[str, list[str]]) -> bool:
    """Aynı kavramın Rusça karşılığına alıntı eşiği kadar yakın mı?

    ⚠️ Moğol ve Tunguz listeleri Rusça alıntılarla doludur (Kalmukça
    ``ponetelnik`` "pazartesi"). Süzülmezse Saha'nın Rusça alıntısı Moğolca
    havuzda da yakın bir eş bulur ve etiket Rusçadan Moğolcaya kayar. Eşik
    alıntı gücünün SCA eşiğidir (0,35); ayrıca seçilmedi.
    """
    from engine.nlp.donor_proximity import sca_distance

    return any(
        sca_distance(form.comparison, candidate) <= RUSSIAN_LOAN_DISTANCE
        for candidate in russian.get(form.concept, ())
    )


@lru_cache(maxsize=1)
def load_concept_donors() -> tuple[tuple[ConceptDonorForm, frozenset[str]], ...]:
    """Bütün kavram hizalı verici biçimleri ve anlam sözcükleri.

    Dosyalar yoksa boş döner (``python scripts/download_cldf.py``).
    """
    forms = _read("northeuralex", NORTHEURALEX_POOLS, by_family=False)
    forms += _read("robbeetstriangulation", ROBBEETS_POOLS, by_family=True)
    russian = _russian_forms()
    return tuple(
        (form, _gloss_tokens(form.gloss)) for form in forms if not _is_russian_loan(form, russian)
    )


def pool_sizes() -> dict[str, int]:
    """Havuz başına madde sayısı (rapor için)."""
    sizes: dict[str, int] = {}
    for form, _ in load_concept_donors():
        sizes[form.pool] = sizes.get(form.pool, 0) + 1
    return sizes
