"""
CLDF Wordlist okuyucu — akraba kümeleri, hizalamalar ve ata biçimler.

``cldf_importer.py`` yalnızca **FormTable + BorrowingTable** okuyup donör tohum
verisi üretiyordu (WOLD içindi). Karşılaştırmalı yöntem için gereken asıl veri
başka yerde: **CognateTable**. Bu modül onu okur.

Bir Lexibank CLDF Wordlist'i dört tablodan oluşur::

    forms.csv       ID, Language_ID, Parameter_ID, Value, Form, Segments, ...
    cognates.csv    Form_ID, Cognateset_ID, Alignment, Root, Doubt, ...
    languages.csv   ID, Name, Glottocode, Family, ...
    parameters.csv  ID, Name, Concepticon_ID, Concepticon_Gloss

Uygulama açısından kritik üç ayrıntı (``savelyevturkic`` v2.1 üzerinde
ölçülerek doğrulandı):

1. **Ünlü uzunluğu yalnız ``Segments`` sütununda vardır.** ``Form`` sütununda
   478 uzun ünlülü biçimin yalnız 18'i görünür, ``Value`` ve ``Graphemes``
   sütunlarında hiç yoktur. Uzunluğa dayanan her iş ``Segments`` okumalıdır.
2. **``Root`` alanı ata biçimi taşır** ama hepsi rekonstrüksiyon değildir:
   843 benzersiz değerin yalnız **472'si** ``*`` ile başlar; geri kalanı
   Türkçe kavram etiketidir (``DIRNAK`` gibi). Altın standart yalnız
   ``*``lı olanlardan kurulur.
3. **Çuvaşça 905 kümenin yalnız 252'sinde vardır.** Oğur tanığı olmayan
   kümede rekonstrüksiyon Proto-Türkçe (*PT) değil, **Ana Ortak Türkçe**
   (*PCT) düzeyindedir. :attr:`CognateSet.proto_level` bunu etiketler.

Kullanım::

    wl = CldfWordlist.load("savelyevturkic")
    for cs in wl.cognate_sets(min_languages=15):
        print(cs.id, cs.root, cs.proto_level, len(cs.entries))
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from engine.config import CLDF_DIR
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: ``Value`` içindeki parantezli ve tırnaklı İngilizce açıklamalar.
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_QUOTED_GLOSS = re.compile(r"['\"\u2018\u2019\u201c\u201d][^'\"\u2018\u2019\u201c\u201d]*['\"\u2018\u2019\u201c\u201d]")

#: Oğur (Bulgar) kolu — bu tanık varsa rekonstrüksiyon Proto-Türkçe düzeyindedir.
#: Yoksa yalnızca Ana Ortak Türkçe (Proto-Common-Turkic) iddia edilebilir.
#:
#: Veri kümeleri aynı dili farklı kodlarla yazar: ``savelyevturkic`` "Chuvash",
#: ``hruschkaturkic`` "CHV", ``starostinaltaic`` "chuvash". Eşleme bu yüzden
#: normalize edilmiş biçim üzerinden yapılır.
OGHUR_LANGUAGES = frozenset({"chuvash", "chv", "chuv", "bulgar", "volgabulgarian", "bulgarian_volga", "chuvashanatri"})


def is_oghur_language(code: str) -> bool:
    """Dil kodu Oğur (Bulgar) koluna mı ait? Kod biçimi veri kümesine göre değişir."""
    normalized = "".join(ch for ch in code.lower() if ch.isalnum())
    return normalized in OGHUR_LANGUAGES


#: Uzunluk işaretleri. ``Segments`` sütununda ``ː`` veya birleşik makron olarak
#: görünür; ayrıca önceden bileşik uzun ünlüler kullanılabilir.
LENGTH_MARKS = frozenset("āēīōūǖȫǟ")
LENGTH_MODIFIER = "ː"
COMBINING_MACRON = "̄"


def has_length(text: str) -> bool:
    """Verilen dizgide ünlü uzunluğu işareti var mı?"""
    if not text:
        return False
    if LENGTH_MODIFIER in text or any(ch in LENGTH_MARKS for ch in text):
        return True
    return COMBINING_MACRON in unicodedata.normalize("NFD", text)


@dataclass(frozen=True)
class LanguageInfo:
    """``languages.csv`` satırı."""

    id: str
    name: str
    glottocode: str = ""
    family: str = ""
    subgroup: str = ""

    @property
    def is_oghur(self) -> bool:
        return is_oghur_language(self.id)


@dataclass(frozen=True)
class ConceptInfo:
    """``parameters.csv`` satırı — Concepticon bağlantısı buradan gelir."""

    id: str
    name: str
    concepticon_id: str = ""
    concepticon_gloss: str = ""


@dataclass(frozen=True)
class FormEntry:
    """Bir dildeki tek bir biçim, akraba kümesi bağlamıyla."""

    form_id: str
    language: str
    concept: str
    value: str
    form: str
    segments: tuple[str, ...]
    alignment: tuple[str, ...] = ()
    doubt: bool = False

    @property
    def has_length(self) -> bool:
        """Ünlü uzunluğu ``Segments``te taşınır — ``form`` değil, o okunur."""
        return has_length(" ".join(self.segments))

    @property
    def transliteration(self) -> str:
        """``Value`` sütunundan temizlenmiş Türkolojik çeviriyazı.

        ⚠️ Bu ayrım ölçüm için belirleyicidir. Aynı satırda üç gösterim var::

            Value    'tïrnaḳ'      Türkolojik çeviriyazı
            Form     'tɯrnaq'      IPA'ya yakın
            Segments 't ɯ r n a q' bölütlenmiş IPA

        Altın ``Root`` alanı **Value ile aynı geleneği** kullanır (``*tïrŋaḳ``,
        ``*Kāpuk``, ``*köŕ``). Motora ``Form`` verilirse ondan IPA'dan
        Türkolojik yazıya çeviri de beklenmiş olur; ölçülen şey
        rekonstrüksiyondan çeviriyazıya kayar.

        ``Value`` serbest metindir ve İNGİLİZCE açıklama taşıyabilir::

            "alïp + motion verb"
            "(køterip) alɯp bar (or other motion verbs)"

        Temizlenmezse rekonstrüksiyon ``*alıpmotionverb`` gibi çöp üretir.
        """
        text = _PARENTHETICAL.sub(" ", self.value)
        text = _QUOTED_GLOSS.sub(" ", text)
        text = text.split("+")[0].split(",")[0].split(";")[0]
        # ``?`` ve benzeri belirsizlik işaretleri sesbirim değildir; kaynak
        # "bu biçimden emin değilim" demek için koyar. Temizlenmezse çapa
        # kelimesi tek başına ``?`` olabiliyor ve rekonstrüksiyon hiç
        # başlamıyordu (ölçüldü: 24 madde bu yüzden cevapsız kalıyordu).
        text = text.replace("?", " ").replace("ˁ", "").replace("ˀ", "")
        text = text.strip().strip("-").strip()
        if " " in text:
            parts = [part for part in text.split() if part.strip("-")]
            text = parts[0] if parts else ""
        return text.strip().strip("-").strip()


@dataclass
class CognateSet:
    """Uzman kararıyla oluşturulmuş bir akraba kümesi."""

    id: str
    root: str = ""
    entries: list[FormEntry] = field(default_factory=list)
    concept: str = ""
    concepticon_gloss: str = ""

    @property
    def languages(self) -> set[str]:
        return {e.language for e in self.entries}

    @property
    def is_reconstruction(self) -> bool:
        """``Root`` bir rekonstrüksiyon mu, yoksa kavram etiketi mi?

        843 benzersiz ``Root`` değerinin yalnız 472'si ``*`` ile başlar.
        Kalanlar (``DIRNAK`` gibi) ata biçim değildir ve altın standarda
        girmemelidir.
        """
        return self.root.startswith("*")

    @property
    def has_oghur_witness(self) -> bool:
        return any(is_oghur_language(lang) for lang in self.languages)

    @property
    def proto_level(self) -> str:
        """``PT`` (Proto-Türkçe) veya ``PCT`` (Ana Ortak Türkçe).

        Çuvaşça/Oğur tanığı olmadan rotasizm ve lambdaizm türetilemez;
        bu durumda iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir.
        Bu ayrımı yapmayan çıktı Türkolog hakem önünde savunulamaz.
        """
        return "PT" if self.has_oghur_witness else "PCT"

    @property
    def has_length_witness(self) -> bool:
        """Kümede ünlü uzunluğunu koruyan en az bir tanık var mı?"""
        return any(e.has_length for e in self.entries)

    def forms_by_language(self, *, notation: str = "transliteration") -> dict[str, str]:
        """Dil -> biçim eşlemesi.

        :param notation: ``"transliteration"`` (``Value``, altın standartla
            aynı gelenek — **değerlendirmede bu kullanılır**), ``"form"``
            (IPA'ya yakın) veya ``"segments"`` (bölütlenmiş IPA).
        """
        if notation == "form":
            return {e.language: e.form for e in self.entries}
        if notation == "segments":
            return {e.language: "".join(e.segments) for e in self.entries}
        return {e.language: e.transliteration or e.form for e in self.entries}


class CldfWordlist:
    """Dört CLDF tablosunu birlikte okuyan Wordlist görünümü."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.languages: dict[str, LanguageInfo] = {}
        self.concepts: dict[str, ConceptInfo] = {}
        self.forms: dict[str, FormEntry] = {}
        self._sets: dict[str, CognateSet] = {}
        self.provenance: dict[str, object] = {}
        self._load()

    # -- yükleme ------------------------------------------------------------

    @classmethod
    def load(cls, name: str) -> CldfWordlist:
        """``data/cldf/<name>/`` altındaki veri kümesini açar."""
        directory = CLDF_DIR / name
        if not directory.exists():
            raise FileNotFoundError(f"{directory} yok. Önce indirin: python scripts/download_cldf.py {name}")
        return cls(directory)

    def _rows(self, filename: str) -> list[dict[str, str]]:
        path = self.dir / filename
        if not path.exists():
            logger.warning("CLDF tablosu yok: %s", path)
            return []
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def _load(self) -> None:
        prov = self.dir / "_provenance.json"
        if prov.exists():
            self.provenance = json.loads(prov.read_text(encoding="utf-8"))

        for row in self._rows("languages.csv"):
            self.languages[row["ID"]] = LanguageInfo(
                id=row["ID"],
                name=row.get("Name", row["ID"]),
                glottocode=row.get("Glottocode", ""),
                family=row.get("Family", ""),
                subgroup=row.get("SubGroup", "") or row.get("Subgroup", ""),
            )

        for row in self._rows("parameters.csv"):
            self.concepts[row["ID"]] = ConceptInfo(
                id=row["ID"],
                name=row.get("Name", row["ID"]),
                concepticon_id=row.get("Concepticon_ID", ""),
                concepticon_gloss=row.get("Concepticon_Gloss", ""),
            )

        raw_forms = {row["ID"]: row for row in self._rows("forms.csv")}

        cognate_rows = self._rows("cognates.csv")

        # ⚠️ ``Cognateset_ID`` tek başına yeterli anahtar DEĞİLDİR.
        # ``savelyevturkic`` v2.1'de 7 küme iki farklı ``Root`` taşıyor
        # (ör. 144 = "burn" hem ``*köń`` hem ``*jan``): veri kümesi orada
        # aslında iki ayrı etimolojiyi tek kimlik altında tutuyor. Yalnız
        # kimliğe göre gruplamak, ``*köń`` tanıklarını ``*jan`` altın biçmiyle
        # eşleştirir ve ölçümü sessizce bozar. Bu yüzden anahtar
        # ``(Cognateset_ID, Root)`` çiftidir.
        by_set: dict[tuple[str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
        for crow in cognate_rows:
            form_row = raw_forms.get(crow.get("Form_ID", ""))
            if form_row is None:
                continue
            key = (crow.get("Cognateset_ID", ""), (crow.get("Root") or "").strip())
            by_set[key].append((crow, form_row))

        # Aynı kimliğin birden çok kökü varsa kimliğe kök eki verilir, yoksa
        # kimlik olduğu gibi kalır (mevcut kayıtlarla uyum için).
        roots_per_id: dict[str, set[str]] = defaultdict(set)
        for set_id, root in by_set:
            roots_per_id[set_id].add(root)

        for (set_id, root), pairs in by_set.items():
            display_id = set_id if len(roots_per_id[set_id]) == 1 else f"{set_id}/{root}"
            entries: list[FormEntry] = []
            for crow, form_row in pairs:
                entry = FormEntry(
                    form_id=form_row["ID"],
                    language=form_row.get("Language_ID", ""),
                    concept=form_row.get("Parameter_ID", ""),
                    value=form_row.get("Value", ""),
                    form=form_row.get("Form", ""),
                    segments=tuple((form_row.get("Segments") or "").split()),
                    alignment=tuple((crow.get("Alignment") or "").split()),
                    doubt=(crow.get("Doubt", "").lower() == "true"),
                )
                entries.append(entry)
                self.forms[entry.form_id] = entry

            concept_id = entries[0].concept if entries else ""
            concept = self.concepts.get(concept_id)
            self._sets[display_id] = CognateSet(
                id=display_id,
                root=root,
                entries=entries,
                concept=concept_id,
                concepticon_gloss=concept.concepticon_gloss if concept else "",
            )

        logger.info(
            "CLDF yüklendi: %s — %d biçim, %d küme, %d dil, %d kavram",
            self.dir.name,
            len(self.forms),
            len(self._sets),
            len(self.languages),
            len(self.concepts),
        )

    # -- sorgular -----------------------------------------------------------

    def cognate_sets(
        self,
        *,
        min_languages: int = 1,
        reconstructed_only: bool = False,
        proto_level: str | None = None,
    ) -> list[CognateSet]:
        """Filtrelenmiş akraba kümeleri.

        :param min_languages: kümede en az kaç farklı dil bulunmalı
        :param reconstructed_only: yalnız ``*``lı ata biçmi olanlar
        :param proto_level: ``"PT"`` veya ``"PCT"`` ile sınırla
        """
        out = []
        for cs in self._sets.values():
            if len(cs.languages) < min_languages:
                continue
            if reconstructed_only and not cs.is_reconstruction:
                continue
            if proto_level and cs.proto_level != proto_level:
                continue
            out.append(cs)
        return sorted(out, key=lambda c: c.id)

    def summary(self) -> dict[str, object]:
        """Veri kümesinin ölçülmüş künyesi — rapor ve test için."""
        sets = list(self._sets.values())
        recon = [c for c in sets if c.is_reconstruction]
        return {
            "dataset": self.dir.name,
            "ref": self.provenance.get("ref", ""),
            "forms": len(self.forms),
            "cognate_sets": len(sets),
            "languages": len(self.languages),
            "concepts": len(self.concepts),
            "reconstructed_sets": len(recon),
            "unique_roots": len({c.root for c in recon}),
            "with_oghur_witness": sum(1 for c in sets if c.has_oghur_witness),
            "pt_ratio": round(sum(1 for c in sets if c.proto_level == "PT") / len(sets), 3) if sets else 0.0,
            "with_length_witness": sum(1 for c in sets if c.has_length_witness),
            "length_marked_forms": sum(1 for f in self.forms.values() if f.has_length),
            "aligned_forms": sum(1 for f in self.forms.values() if f.alignment),
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CLDF Wordlist künyesi")
    ap.add_argument("dataset", nargs="?", default="savelyevturkic")
    args = ap.parse_args()

    wl = CldfWordlist.load(args.dataset)
    for key, value in wl.summary().items():
        print(f"{key:24} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
