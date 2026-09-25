"""
Kök yüzey varyantları — Zemberek kök özniteliklerinden (yalnız veri).

Türkçe bir kök ek alınca yüzeyde değişebilir: ``kitap → kitab-ı`` (ünsüz
yumuşaması), ``hak → hakk-ı`` (ikizleşme), ``burun → burn-u`` (son ünlü
düşmesi), ``ayır- → ayr-ıl-`` (fiilde son ünlü düşmesi), ``ağla- → ağl-ıyor``
(şimdiki zamanda ünlü düşmesi). Hangi kökün hangi değişimi geçirdiği
sözlükseldir (``at → at-ı`` ama ``tat → tad-ı``); bu bilgi Zemberek-NLP'nin
kök sözlüğünden (Apache 2.0) alınır: ``scripts/download_zemberek_lexicon.py``
``data/zemberek/*.dict`` dosyalarını indirir. Zemberek koduna bağımlılık
yoktur; ``StemTransitionsBase.generateModifiedRootNodes`` ve
``TurkishDictionaryLoader.inferMorphemicAttributes`` mantığı burada
yeniden yazılmıştır.

İki yön:

* ``surface_variants(kök)`` — kökün kendisi + değişmiş biçimi
  (``kitap → {kitap, kitab}``, ``burun → {burun, burn}``).
* ``root_candidates(yüzey)`` — değişmiş biçimden sözlük köküne
  (``kitab → kitap``, ``burn → burun``, ``hakk → hak``, ``ağz → ağız``).

Kök sözlükte yoksa yalnız GÜVENLİ genel kural uygulanır: çok heceli ve
``p/ç/t/k`` ile biten kökte yumuşama (Zemberek'in de varsayılanı), tersine
``b/c/d/ğ`` ile biten yüzeyde sertleştirme. Ünlü düşmesi ve ikizleşme
sözlüksüz TAHMİN EDİLMEZ (``kavun → kavn`` yanlış olurdu).

``InverseHarmony`` (ters ünlü uyumu) yüzey kökü değiştirmez ve alıntı
sinyalidir; yüzey varyantlarında KULLANILMAZ, yalnız :func:`borrowing_marks`
ile alıntı dedektörüne verilir (``ters_uyum`` sinyali).

⚠️ Döngüsellik: Zemberek sözlüğü TDK tabanlıdır; burada yalnız ses
öznitelikleri kullanılır, köken bilgisi değil.

⚠️ OLUMSUZ SONUÇ — başlık köküne BAĞLANMADI (2026-09-24, `make eval-headline`
düzeni, 275 tekil kelime: 240 puanlı Starling + 32 savelyev dev; yalnız yerel kaynak):

* Başlık indeks yedeği (``search_engine._index_source_proto``): kök
  belirlenemeyen 83 kelimenin (Starling kapalı) HİÇBİRİNDE varyant sorgusu
  (``surface_variants`` + ``root_candidates``) tr/ota indeksinde Proto-Türkçe
  miras kaydı bulmadı. Sebep: ölçüm kelimeleri madde başı biçimindedir,
  indeks de madde başıyla tutulur; yumuşamış/düşmeli biçim madde başı
  olmaz. Bulunan dört satırın hepsi yanlış bağdı (`dala → dal`, `et → ed`).
* Tarihsel ek soyma (``historical_morphology.build_tree``: soyulan gövde
  tanıksızsa varyant köküne in): ölçüm kümesi + Starling TRK örneklemi (1065 kelime) içinde
  yalnız 3 soyma değişti (`tadım → tat` doğru; `edik → et`, `kabuk → kap`
  şüpheli); ölçüm kümesinde değişen iki kelimenin (edik, kabuk) başlığı
  Starling açık ve kapalı ayarda AYNI kaldı; başka hiçbir kelimenin başlığı
  değişmez (tam doğruluk farkı 0). Kabul ölçütü ("anlamlı artış") karşılanmadı.

Kullanım yeri kalıyorsa çekimli/ünlü düşmeli METİN biçimleridir (tanık
cümleleri, Derleme/Tarama ağız biçimleri) — orada ayrıca ölçülmeli.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

LEXICON_DIR = PROJECT_ROOT / "data" / "zemberek"
DICT_FILES = ("master-dictionary.dict", "non-tdk.dict")

#: Kullanılan ses öznitelikleri (diğerleri okunur ama yok sayılır).
PHONETIC_ATTRIBUTES = frozenset(
    {"Voicing", "NoVoicing", "Doubling", "LastVowelDrop", "ProgressiveVowelDrop"}
)

_VOWELS = frozenset("aeıioöuü")
_STOP = frozenset("çkpt")
#: Zemberek ``TurkishAlphabet.voice``: ç→c, g→ğ, k→ğ, p→b, t→d.
_VOICE = {"ç": "c", "g": "ğ", "k": "ğ", "p": "b", "t": "d"}
#: Genel ters kural: yumuşamış son ünsüzden sert olana.
_DEVOICE = {"b": "p", "c": "ç", "d": "t", "ğ": "k"}


@dataclass(frozen=True)
class RootItem:
    """Bir sözlük maddesi: kök (fiilde mastarsız), sözcük türü, öznitelikler."""

    lemma: str
    root: str
    pos: str
    attributes: frozenset[str]


def _vowel_count(word: str) -> int:
    return sum(ch in _VOWELS for ch in word)


def _parse_line(line: str) -> tuple[str, str, set[str]] | None:
    """``burun [P:Noun; A:LastVowelDrop]`` -> (``burun``, ``Noun``, {LastVowelDrop})."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    word, _, meta = line.partition("[")
    word = word.strip()
    pos, attributes = "", set()
    for part in meta.rstrip("]").split(";"):
        key, _, value = part.partition(":")
        key = key.strip()
        if key == "P":
            pos = value.split(",")[0].strip()
        elif key == "A":
            attributes |= {a.strip() for a in value.split(",") if a.strip()}
    return word, pos, attributes


def _item(word: str, pos: str, attributes: set[str]) -> RootItem | None:
    lemma = to_comparison_form(word)
    if not lemma or " " in word:
        return None
    if not pos:
        # Zemberek ``getPosData``: tür verilmemişse -mAk ile biten fiil, gerisi ad.
        pos = "Verb" if len(lemma) > 3 and lemma.endswith(("mak", "mek")) else "Noun"
    root = lemma[:-3] if pos == "Verb" and lemma.endswith(("mak", "mek")) else lemma
    if not root:
        return None
    # Ters uyumlu kök varsayılan yumuşamayı almaz (Zemberek); öznitelik
    # yalnız bu çıkarımda okunur, sonuçta tutulmaz.
    inverse_harmony = "InverseHarmony" in attributes
    attributes = set(attributes) & PHONETIC_ATTRIBUTES
    vowels = _vowel_count(root)
    last = root[-1]
    # ``TurkishDictionaryLoader.inferMorphemicAttributes`` (yalnız ses kısmı).
    if pos == "Verb":
        if last in _VOWELS:
            attributes.add("ProgressiveVowelDrop")
    elif pos in ("Noun", "Adj", "Dup"):
        if (vowels > 1 and last in _STOP and "NoVoicing" not in attributes
                and not inverse_harmony):
            attributes.add("Voicing")
        if root.endswith(("nk", "og")):
            if "NoVoicing" not in attributes:
                attributes.add("Voicing")
        elif vowels < 2 and "Voicing" not in attributes:
            attributes.add("NoVoicing")
    return RootItem(lemma, root, pos, frozenset(attributes))


def modified_root(item: RootItem, *, progressive: bool = True) -> str:
    """Zemberek ``generateModifiedRootNodes``: özniteliklerin uygulandığı biçim.

    Değişim yoksa kökün kendisi döner.
    """
    seq = item.root
    attrs = item.attributes
    if "Voicing" in attrs and "NoVoicing" not in attrs and seq[-1] in _VOICE:
        seq = seq[:-1] + ("g" if item.root.endswith("nk") else _VOICE[seq[-1]])
    if "Doubling" in attrs:
        seq += seq[-1]
    if "LastVowelDrop" in attrs and len(seq) > 2:
        seq = seq[:-1] if seq[-1] in _VOWELS else seq[:-2] + seq[-1]
    if progressive and "ProgressiveVowelDrop" in attrs and len(seq) > 1 and seq[-1] in _VOWELS:
        seq = seq[:-1]
    return seq


@lru_cache(maxsize=1)
def _lexicon() -> tuple[dict[str, tuple[RootItem, ...]], dict[str, tuple[str, ...]]]:
    """(kök -> maddeler, değişmiş biçim -> kökler). Sözlük yoksa boş."""
    by_root: dict[str, list[RootItem]] = {}
    by_modified: dict[str, set[str]] = {}
    for name in DICT_FILES:
        path = LEXICON_DIR / name
        if not path.is_file():
            logger.debug("Zemberek sözlüğü yok: %s (scripts/download_zemberek_lexicon.py)", path)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(line)
            item = _item(*parsed) if parsed else None
            if item is None:
                continue
            by_root.setdefault(item.root, []).append(item)
            # Şimdiki zaman düşmesi (`ağla → ağl-ıyor`) yalnız -Iyor önünde
            # olur; tersine kullanılınca kısa kökleri yanlış bağlıyordu
            # (`ad → ada-`, `al → ala-`). Ters haritaya girmez.
            changed = modified_root(item, progressive=False)
            if changed != item.root:
                by_modified.setdefault(changed, set()).add(item.root)
    return (
        {k: tuple(v) for k, v in by_root.items()},
        {k: tuple(sorted(v)) for k, v in by_modified.items()},
    )


def lexicon_available() -> bool:
    return bool(_lexicon()[0])


def lookup(root: str) -> tuple[RootItem, ...]:
    """Kökün (fiilde mastarsız gövde) sözlük maddeleri."""
    return _lexicon()[0].get(to_comparison_form(root), ())


def surface_variants(root: str) -> list[str]:
    """Kökün kendisi + ek alınca aldığı biçim(ler); ilk öğe her zaman kök.

    Sözlükte yoksa yalnız çok heceli ``p/ç/t/k`` sonlu kökte yumuşama.
    """
    stem = to_comparison_form(root)
    if not stem:
        return []
    out = [stem]
    items = lookup(stem)
    if items:
        for item in items:
            changed = modified_root(item)
            if changed not in out:
                out.append(changed)
    elif _vowel_count(stem) > 1 and stem[-1] in _VOICE and stem[-1] != "g":
        out.append(stem[:-1] + ("g" if stem.endswith("nk") else _VOICE[stem[-1]]))
    return out


def root_candidates(surface: str) -> list[str]:
    """Değişmiş yüzey gövdesinden aday sözlük kökleri (kendisi HARİÇ).

    Önce sözlük (``burn → burun``, ``hakk → hak``); sözlükte yoksa ve biçim
    kendisi de kök değilse ``b/c/d/ğ`` sonu sertleştirilir (``kitab →
    kitap``). Aday doğrulanmış değildir; çağıran kaydı arar.
    """
    form = to_comparison_form(surface)
    if not form:
        return []
    by_root, by_modified = _lexicon()
    found = [r for r in by_modified.get(form, ()) if r != form]
    if found or form in by_root:
        return found
    if _vowel_count(form) > 1 and form.endswith("ng"):
        return [form[:-1] + "k"]
    if _vowel_count(form) > 1 and form[-1] in _DEVOICE:
        return [form[:-1] + _DEVOICE[form[-1]]]
    return []


#: Ek alma davranışından okunan alıntı işaretleri (bkz. ``borrowing_marks``).
BORROWING_MARKS = frozenset({"InverseHarmony", "ImplicitPlural"})


@lru_cache(maxsize=1)
def _borrowing_marks() -> dict[str, frozenset[str]]:
    """Karşılaştırma biçimi (fiilde mastarsız gövde) -> işaretler."""
    out: dict[str, set[str]] = {}
    for name in DICT_FILES:
        path = LEXICON_DIR / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(line)
            if not parsed or " " in parsed[0]:
                continue
            word, pos, attributes = parsed
            marks = attributes & BORROWING_MARKS
            if not marks:
                continue
            key = to_comparison_form(word)
            if pos == "Verb" and key.endswith(("mak", "mek")):
                key = key[:-3]
            if key:
                out.setdefault(key, set()).update(marks)
    return {k: frozenset(v) for k, v in out.items()}


def borrowing_marks(word: str) -> frozenset[str]:
    """Kelimenin ``InverseHarmony`` / ``ImplicitPlural`` işaretleri (yoksa boş).

    ``InverseHarmony``: ek, kökün son ünlüsüne değil ince/kalın karşıtına
    uyar (``saat → saati``, ``kalp → kalbi``, ``alkol → alkolü``).
    ``ImplicitPlural``: madde zaten çoğuldur (Arapça kırık çoğul: ``ulema``,
    ``hayvanat``). İkisi de ek alma DAVRANIŞIDIR, köken beyanı değildir —
    ama sözlükçü onları alıntılarda gözlemiştir; döngüsellik ölçümü
    ``borrowing_detector._inverse_harmony_signal`` notunda.
    """
    return _borrowing_marks().get(to_comparison_form(word), frozenset())
