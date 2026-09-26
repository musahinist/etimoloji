"""
Yerel sözlük indeksi — akraba arama ve alıntı zinciri için.

``scripts/download_lexicons.py`` ile indirilen kaikki dökümlerini SQLite
FTS5 indeksine dönüştürür. İki soruya hızlı cevap verir:

1. **"Bu biçim Başkurtçada var mı?"** — ileri tahminle üretilen adayın
   gerçekten tanıklanıp tanıklanmadığı (Faz 5, öngörü testi).
2. **"Bu kelime hangi dilden, nasıl gelmiş?"** — kaikki'nin
   ``etymology_templates`` alanı verici dili ve özgün biçmi **yapılandırılmış**
   verir; serbest metin ayrıştırmaya gerek kalmaz::

       {"name": "bor", "args": {"1": "tr", "2": "ar", "3": "كتاب"}}

⚠️ Bu indeks **arama** içindir. Akrabalık kararı buradan gelmez: Wiktionary
türevi akraba kümeleri altın standart ağaçlarla tutarsız çıkıyor
(Häuser & Stamatakis 2025). Burada bulunan bir biçim "aday"dır; kararı
kümeleme ve rekonstrüksiyon katmanı verir.

Kullanım::

    python -m engine.db.lexicon_index --build
    python -m engine.db.lexicon_index --lookup köz
    python -m engine.db.lexicon_index --borrowings tr --limit 20
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import LEXICON_DIR, PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form
from engine.utils.transliteration import transliterate_to_latin

logger = get_logger(__name__)

#: ``ETY_LEXICON_INDEX`` başka bir indeks dosyası gösterir (ör. köken
#: etiketleri boşaltılmış kör indeks, ``scripts/build_blind_index.py``) —
#: alıntı ölçümünde döngüsellik denetimi (K2) için. Yoksa varsayılan.
INDEX_PATH = Path(
    os.environ.get("ETY_LEXICON_INDEX") or PROJECT_ROOT / "data" / "lexicons" / "index.db"
)

#: Wiktionary etimoloji şablonlarının anlamı.
#: ``args["1"]`` alan dil, ``args["2"]`` veren dil, ``args["3"]`` özgün biçim.
BORROWING_TEMPLATES: dict[str, str] = {
    "bor": "alıntı",
    "bor+": "alıntı",
    "ubor": "uyarlanmamış alıntı",
    "lbor": "öğrenilmiş alıntı",
    "slbor": "yarı öğrenilmiş alıntı",
    "obor": "orfografik alıntı",
    "calque": "öyküntü",
    "psm": "fono-semantik eşleme",
}

#: ``etymon`` / ``ety`` şablonları bambaşka bir yapı kullanır::
#:
#:     {"name": "etymon", "args": {"1": "tr", "2": ":inh", "3": "ota:كتاب"}}
#:     {"name": "ety",    "args": {"2": ":inh", "3": "ota:صلا\n<ety:der<ar:صَلاَة>>"}}
#:
#: İlişki ``args["2"]``te ``:`` önekiyle, dil ve biçim ``args["3"]``te
#: ``lang:form`` olarak gelir; daha derin halkalar ``<ety:REL<lang:form>>``
#: biçiminde İÇ İÇE gömülüdür.
#:
#: ⚠️ Bu biçim tanınmazsa ``kitap`` gibi temel alıntılar KAÇIRILIR: Türkçe
#: dökümünde `kitap`ın tek şablonu budur.
TREE_TEMPLATES = frozenset({"etymon", "ety"})
_TREE_NESTED = re.compile(r"<ety:(\w+)<([a-zA-Z-]+):([^<>]+)>>")
_TREE_HEAD = re.compile(r"^([a-zA-Z-]+):(.+)$")


def parse_tree_template(template: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``etymon``/``ety`` şablonundan ``(ilişki, dil, biçim)`` halkalarını çıkarır."""
    args = template.get("args", {}) or {}
    relation = str(args.get("2", "") or "").lstrip(":").strip().lower()
    raw = str(args.get("3", "") or "")
    if not raw:
        return []

    steps: list[tuple[str, str, str]] = []
    head = _TREE_HEAD.match(raw.split("\n")[0].split("<")[0].strip())
    if head:
        steps.append((relation or "inh", head.group(1), head.group(2).strip()))
    for nested_relation, lang, form in _TREE_NESTED.findall(raw):
        steps.append((nested_relation.lower(), lang, form.strip()))
    return steps


INHERITANCE_TEMPLATES: dict[str, str] = {
    "inh": "miras",
    "inh+": "miras",
    "der": "türev",
}

#: Etimoloji METNİNDE geçen verici dil adları -> kod.
#:
#: ⚠️ Yedek yoldur, birincil değil. Bazı maddelerde zincirin uzak halkaları
#: yalnız serbest metinde bulunur; ``kitap`` bunun tipik örneğidir::
#:
#:     Etymology tree
#:     Arabic كِتَاب (kitāb)bor.
#:     Ottoman Turkish كتاب
#:     Turkish kitap
#:
#: Şablon dizisi yalnız ``ota``ya kadar gider; Arapça halkası metindedir.
#: Bu yedek olmadan ``kitap`` "miras" sayılıyordu.
ETYMOLOGY_TEXT_DONORS: dict[str, str] = {
    "Arabic": "ar",
    "Persian": "fa",
    "Classical Persian": "fa-cls",
    "Middle Persian": "pal",
    "Ancient Greek": "grc",
    "Byzantine Greek": "gkm",
    "Greek": "el",
    "Latin": "la",
    "Italian": "it",
    "French": "fr",
    "English": "en",
    "German": "de",
    "Russian": "ru",
    "Armenian": "hy",
    "Georgian": "ka",
    "Hebrew": "he",
    "Aramaic": "arc",
    "Syriac": "syc",
    "Sanskrit": "sa",
    "Chinese": "zh",
    "Mongolian": "mn",
    "Sogdian": "sog",
    "Bulgarian": "bg",
    "Serbo-Croatian": "sh",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Spanish": "es",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Kurdish": "ku",
    "Adyghe": "ady",
}

#: Uzun adlar önce denenmeli: "Classical Persian" "Persian"dan önce.
_DONOR_NAMES_BY_LENGTH = sorted(ETYMOLOGY_TEXT_DONORS, key=len, reverse=True)


#: Köken zinciri burada biter; sonrası akraba/karşılaştırma listesidir.
_COGNATE_SECTION = re.compile(r"(?im)^\s*(?:cognates?|compare|see also|descendants|related terms)\b")
#: Zincir içindeki "Cognate with Kurdish gerek." / "Compare Persian …" /
#: "cf. …" kısımları — anahtar kelimeden CÜMLE SONUNA kadar.
#:
#: ⚠️ Cümlenin tamamı silinmez: "From Persian X, compare Y." cümlesinde
#: verici dil anahtar kelimeden ÖNCE durur; cümleyi baştan silen eski desen
#: vericiyi de götürüyordu. ``cf.`` noktayla bittiği için ardından ``\b``
#: gelemez; eski ``cf\.\b`` hiç eşleşmiyordu.
_COGNATE_SENTENCE = re.compile(r"(?i)(?:\bcognate\w*|\bcompare\b|\bcf\.)[^.]*\.?")

#: Wiktionary "Etymology tree" bloğu: başlık satırı + KÖKTEN BAŞLAYAN halka
#: satırları ("Aramaic קורבנא (qurbānā)bor."). Düzyazı ("Inherited from …")
#: ilk "from" içeren satırda başlar.
_TREE_HEADER = "Etymology tree"
_PROSE_LINE = re.compile(r"(?i)\bfrom\b")


def _split_tree_block(etymology_text: str) -> tuple[list[str], str]:
    """Metni ``(ağaç satırları, düzyazı)`` olarak ayırır."""
    lines = etymology_text.split("\n")
    if not lines or lines[0].strip() != _TREE_HEADER:
        return [], etymology_text
    index = 1
    while index < len(lines) and not _PROSE_LINE.search(lines[index]):
        index += 1
    return lines[1:index], "\n".join(lines[index:])


def _nearest_donor_name(text: str) -> tuple[str, int]:
    """Metinde EN ÖNCE geçen verici dil adı ve konumu; yoksa ``("", -1)``.

    Aynı konumda başlayan adlardan uzun olan kazanır ("Classical Persian").
    """
    best_name, best_index = "", -1
    for name in _DONOR_NAMES_BY_LENGTH:
        match = re.search(r"(?<![\w-])" + re.escape(name) + " ", text)
        if match is None:
            continue
        if best_index < 0 or match.start() < best_index:
            best_name, best_index = name, match.start()
    return best_name, best_index


def donor_from_text(etymology_text: str) -> tuple[str, str]:
    """Etimoloji metninden verici dili çıkarır. Bulamazsa ``("", "")``.

    ⚠️ Akraba listesi verici DEĞİLDİR. Metnin tamamı taranıyordu ve
    `gerek`in "Cognates … Northern Kurdish gerek" satırı kelimeyi Kürtçe
    alıntı yapıyordu. Ölçüldü: yalnız Türkçede 139 madde yanlışlıkla
    "alıntı" idi — `o` (Çince), `ben`, `bin`, `buz` (Moğolca), `don`.
    """
    if not etymology_text:
        return "", ""
    tree_lines, prose = _split_tree_block(etymology_text)
    prose = _COGNATE_SECTION.split(prose, maxsplit=1)[0]
    prose = _COGNATE_SENTENCE.sub(" ", prose)
    # Düzyazı zinciri Türkçeden geriye doğru okunur ("Inherited from Ottoman
    # Turkish …, borrowed from Arabic …, borrowed from Aramaic …"): EN ÖNCE
    # geçen verici, Türkçeye en yakın halka yani DOĞRUDAN vericidir. Adları
    # uzunluk sırasıyla denemek `kurban`ı Aramiceye bağlıyordu.
    name, index = _nearest_donor_name(prose)
    if name:
        rest = prose[index + len(name) + 1 :].strip()
        form = rest.split()[0] if rest else ""
        return ETYMOLOGY_TEXT_DONORS[name], form.strip("(),.")
    # Yalnız ağaç varsa: satırlar kökten başlar, doğrudan verici SONDAKİDİR.
    for line in reversed(tree_lines):
        name, index = _nearest_donor_name(line.strip() + " ")
        if name and index == 0:
            rest = line.strip()[len(name) + 1 :].strip()
            form = rest.split()[0] if rest else ""
            return ETYMOLOGY_TEXT_DONORS[name], form.strip("(),.")
    return "", ""


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id            INTEGER PRIMARY KEY,
    lang_code     TEXT NOT NULL,
    word          TEXT NOT NULL,
    comparison    TEXT NOT NULL,
    pos           TEXT,
    gloss         TEXT,
    ipa           TEXT,
    etymology     TEXT,
    long_vowels   TEXT,          -- IPA'dan çıkarılmış uzun ünlüler
    origin        TEXT,          -- 'alıntı' | 'miras' | 'diriltme' | NULL
    donor_lang    TEXT,
    donor_form    TEXT,
    formation     TEXT,          -- kendi dilindeki yapım: "biti- + -g"
    cognates      TEXT           -- JSON: sözlüğün andığı akrabalar (cog şablonu)
);
CREATE INDEX IF NOT EXISTS idx_comparison ON entries(comparison);
CREATE INDEX IF NOT EXISTS idx_lang ON entries(lang_code);
CREATE INDEX IF NOT EXISTS idx_origin ON entries(origin);
CREATE INDEX IF NOT EXISTS idx_length ON entries(long_vowels);
CREATE INDEX IF NOT EXISTS idx_donor ON entries(donor_lang);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    word, comparison, gloss, etymology,
    content='entries', content_rowid='id', tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS build_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass(frozen=True)
class LexiconEntry:
    """İndeksteki tek bir sözlük maddesi."""

    lang_code: str
    word: str
    comparison: str
    pos: str = ""
    gloss: str = ""
    ipa: str = ""
    etymology: str = ""
    #: IPA'dan çıkarılmış uzun ünlüler. Ünlü uzunluğu Proto-Türkçe
    #: rekonstrüksiyonunun en zayıf tarafıydı: ``savelyevturkic``te yalnız
    #: 478 uzunluk tanığı var ve Türkmence'de **yalnız 2**. Oysa indirilmiş
    #: kaikki dökümlerinde Halaçça 561, Türkmence 293 tanık duruyor.
    long_vowels: str = ""
    origin: str | None = None
    donor_lang: str = ""
    donor_form: str = ""
    formation: str = ""
    cognates: str = ""

    def as_row(self) -> tuple:
        return (
            self.lang_code,
            self.word,
            self.comparison,
            self.pos,
            self.gloss,
            self.ipa,
            self.etymology,
            self.long_vowels,
            self.origin,
            self.donor_lang,
            self.donor_form,
            self.formation,
            self.cognates,
        )


def _etymology_text(record: dict[str, Any]) -> str:
    """Etimoloji metni — iki şemayı da okur.

    ⚠️ İngilizce sürüm ``etymology_text`` (dizgi), Rusça sürüm
    ``etymology_texts`` (liste) kullanıyor. Yalnız ilkini okumak Rusça
    sürümün etimolojisini **sessizce** düşürürdü.
    """
    single = record.get("etymology_text")
    if single:
        return str(single)
    many = record.get("etymology_texts")
    if isinstance(many, list) and many:
        return " ".join(str(x) for x in many if x)
    return ""


#: Rusça Wiktionary'nin "etimoloji bilinmiyor" yer tutucusu ve şablon
#: artıkları. Dökümde 58 bin maddenin notu YALNIZ bundan ibaret
#: ("Происходит от ??" 32.798, "От ??" 16.507, "Из ??" 8.687) ve rapora
#: "Kaynak notu: Происходит от ??" diye basılıyordu.
_RU_PLACEHOLDER_NOTE = re.compile(r"^\s*(?:Происходит\s+)?(?:от|из)?\s*\?\?\s*\.?\s*$", re.IGNORECASE)
_RU_PLACEHOLDER_CLAUSE = re.compile(r",?\s*(?:далее\s+)?(?:от|из)\s+\?\?\s*\.?|\.?\?\?\.?", re.IGNORECASE)
_RU_BOILERPLATE = re.compile(
    r"Это болванка статьи[^\n]*|Это незаконченная статья[^\n]*|Статья нуждается в доработке\.?"
    r"|\(См\.\s*[Оо]бщепринятые правила\)\.?"
)
_LINE_END_PUNCT = (".", ":", ";", ",", "!", "?")


def _clean_note(text: str) -> str:
    """Sözlük notunu rapora basılabilir TEK satıra indirir.

    - "Etymology tree" bloğu atılır (aynı zincir düzyazıda tekrar ediyor,
      ağaç kökten başlayan ham satırlardır);
    - Rusça sürümün ``??`` yer tutucusu ve şablon cümleleri silinir;
    - satır sonları birleştirilir: noktalama ile biten satırdan sonra boşluk,
      liste satırlarında ("list of cognates / Azerbaijani körpü") " / ".

    ⚠️ Yalnız saklanan NOTU temizler; köken/verici çıkarımı ham metinden
    (``donor_from_text``) yapılır.
    """
    if not text:
        return ""
    tree_lines, prose = _split_tree_block(text)
    if tree_lines and not prose.strip():
        prose = "\n".join(tree_lines)
    prose = _RU_BOILERPLATE.sub(" ", prose)
    if _RU_PLACEHOLDER_NOTE.match(prose):
        return ""
    prose = _RU_PLACEHOLDER_CLAUSE.sub("", prose)
    out = ""
    for line in (x.strip() for x in prose.split("\n")):
        if not line:
            continue
        if out:
            # "; " DEĞİL: `;` ve `,` akrabalık beyanı desenlerinde cümle
            # sınırıdır; satır sonu eskiden sınır sayılmıyordu, sayılmaz.
            out += " " if out.endswith(_LINE_END_PUNCT) else " / "
        out += re.sub(r"\s+", " ", line)
    return out.strip()


def _first_gloss(record: dict[str, Any]) -> str:
    for sense in record.get("senses", []):
        glosses = sense.get("glosses") or []
        if glosses:
            return str(glosses[0])
    return ""


def _first_ipa(record: dict[str, Any]) -> str:
    for sound in record.get("sounds", []) or []:
        if sound.get("ipa"):
            return str(sound["ipa"])
    return ""


#: IPA ünlüleri. Uzunluk işareti ``ː`` bir ÜNSÜZDEN sonra gelirse ikizleşme
#: (gemination) demektir, ünlü uzunluğu değil — ``борщ [buɔɐ̯rɕː]`` bir
#: uzunluk tanığı DEĞİLDİR.
IPA_VOWELS = frozenset("aeiouɑɒæɐəɘɛɜɞɔɵøœyʉɨɯʌʊɪɤʏ")

#: Uzunluk işaretleri: modifier letter triangular colon ve düz iki nokta.
LENGTH_MARKS = ("ː", ":")

#: Ünlünün üstüne binen birleşik işaretler (ton, nazal, uzunluk işaretinden
#: önce gelebilir): ``aː`` ile ``ã ː`` arasında fark kalmasın diye atlanır.
_SKIPPABLE_COMBINING = frozenset(range(0x0300, 0x0370)) | frozenset(range(0x1AB0, 0x1B00))


def extract_long_vowels(ipa: str) -> str:
    """IPA gösteriminden **uzun ünlüleri** çıkarır.

    ⚠️ Yalnız ``ː`` aramak yetmez: o işaret ünsüzden sonra gelirse
    ikizleşmedir. Ölçüldü — indirilmiş dökümlerde ``ː`` taşıyan 5.506
    maddenin bir kısmı ``ɕː``, ``rː``, ``щ`` gibi ünsüz ikizleşmeleridir ve
    ünlü uzunluğu tanığı sayılamazlar.

    :returns: uzun ünlülerin sırayla birleştirilmiş hâli (``"aːiː"`` gibi),
        yoksa boş dizgi.
    """
    if not ipa:
        return ""
    text = unicodedata.normalize("NFD", ipa)
    found: list[str] = []
    for index, char in enumerate(text):
        if char not in LENGTH_MARKS:
            continue
        # İşaretten geriye doğru git, birleşik işaretleri atlayarak taban
        # sesi bul.
        cursor = index - 1
        while cursor >= 0 and ord(text[cursor]) in _SKIPPABLE_COMBINING:
            cursor -= 1
        if cursor >= 0 and text[cursor].lower() in IPA_VOWELS:
            found.append(text[cursor].lower() + "ː")
    return "".join(found)


#: Türki dil kodları. Zincirin bir halkası bu ailenin DIŞINA çıkıyorsa
#: kelime nihayetinde alıntıdır — ilk halka "miras" etiketli olsa bile.
#:
#: ⚠️ Şablonlardaki kodlar WIKTIONARY kodlarıdır: Hakasça ``kjh``, Salarca
#: ``slr``. ``khk`` Wiktionary'de Halha Moğolcasıdır, aileye girmez. Eskiden
#: ``khk``/``slq`` yazıyordu; Salarca içi türetme (`öxsirik` < Salarca
#: `öxsirğüsi`) "alıntı" çıkıyordu (ölçüldü: 2 kayıt).
TURKIC_FAMILY_CODES = frozenset(
    {
        "tr", "ota", "otk", "trk-pro", "trk-oat", "trk-ogz-pro", "trk-cmn-pro",
        "az", "tk", "gag", "kk", "kaa", "ky", "tt", "ba", "nog", "kum", "krc",
        "crh", "uz", "ug", "cv", "sah", "tyv", "alt", "kjh", "cjs", "slr", "slq",
        "chg", "klj", "dlg", "kim", "ybe", "clw", "atv", "bay", "qwm", "kdr",
    }
)


#: Türk dilinin TARİHÎ/ATA evreleri: bunlardan "alıntı" dil içi diriltmedir
#: (`betik` "learned borrowing from Old Turkic 𐰋𐰃𐱅𐰏", `bilge`, `tin`,
#: `başkan`), yabancı alıntı değildir. ``oui`` (Eski Uygurca), ``xqa``
#: (Karahanlıca) ve ``okm`` aile kodlarında yoktu; tek başına onlardan
#: "alıntı" aile dışına çıkış sayılıyordu. ``ota``/``chg`` BİLEREK yok:
#: Kırım Tatarcası ~ Osmanlıca (60 kayıt) gerçek Türk dilleri arası
#: alıntıdır; yalnız ÖĞRENİLMİŞ alıntı şablonuyla gelince diriltmedir
#: (`kamu` "learned borrowing from Ottoman Turkish قمو").
HISTORICAL_TURKIC_STAGES = frozenset(
    {"otk", "oui", "xqa", "okm", "trk-pro", "trk-cmn-pro", "trk-ogz-pro", "trk-oat", "trk-eog"}
)

#: Tarihî evreden ÖĞRENİLMEMİŞ (düz ``bor``) alıntı ancak alan dil o evrenin
#: soyundan geliyorsa diriltmedir; değilse Türk dilleri arası temas
#: alıntısıdır. Ölçüldü: Salarca (Oğuz) Eski Uygurcadan 8 kayıt (`eñgek`,
#: `yalañ-adaq`, `atıq` "Borrowed from Old Uyghur") diriltme çıkıyordu;
#: Salarca Eski Uygurcanın torunu değildir. Tabloda olmayan evre (ör. ``okm``)
#: yalnız öğrenilmiş şablonla diriltme olur. ``trk-pro``/``otk`` burada yok:
#: aile ortak atasıdır (Çuvaşça dahil sayılmaz, bkz. ``_COMMON_TURKIC_ANCESTORS``).
_OGHUZ_WEST = frozenset({"trk-oat", "ota", "tr", "az", "gag", "crh"})
HISTORICAL_STAGE_DESCENDANTS: dict[str, frozenset[str]] = {
    "oui": frozenset({"ug", "ybe"}),
    "xqa": frozenset({"chg", "uz", "ug"}),
    "trk-oat": _OGHUZ_WEST,
    "trk-eog": _OGHUZ_WEST | {"tk", "slq"},
    "trk-ogz-pro": _OGHUZ_WEST | {"trk-eog", "tk", "slq", "kdr"},
}

#: Bütün Türk dillerinin (``trk-cmn-pro``/``otk``: Çuvaşça hariç) atası.
_COMMON_TURKIC_ANCESTORS = frozenset({"trk-pro", "trk-cmn-pro", "otk"})


def _stage_is_ancestor(stage: str, lang_code: str) -> bool:
    """Tarihî evre ``stage`` alan dil ``lang_code``nin atası mı.

    Alan dil bilinmiyorsa (``""``) eski davranış korunur: ata sayılır.
    """
    if not lang_code or stage == "trk-pro":
        return True
    if stage in _COMMON_TURKIC_ANCESTORS:
        return lang_code != "cv"
    return lang_code in HISTORICAL_STAGE_DESCENDANTS.get(stage, frozenset())


#: Öğrenilmiş (bilinçli) alıntı şablonları.
LEARNED_BORROWING_TEMPLATES = frozenset({"lbor", "slbor"})

#: Türk dili İÇİ bilinçli diriltmenin köken etiketi: ne yabancı alıntı ne
#: kesintisiz miras. Alıntı sinyalleri bunu alıntı SAYMAZ
#: (bkz. `borrowing_detector._lexical_origin_rows`).
REVIVAL_ORIGIN = "diriltme"


def _is_revival(steps: list[tuple[str, str, str]], lang_code: str = "") -> bool:
    """Şablon zinciri Türk dilinin kendi tarihî evresinden bilinçli alıntı mı.

    Zincirin hiçbir halkası aile dışına çıkmaz ve her alıntı halkası ya alan
    dilin ATASI olan tarihî evreden (``_stage_is_ancestor``) ya da öğrenilmiş
    alıntı şablonuyla bir Türk dilinden gelir.
    """
    def turkic(code: str) -> bool:
        return code in TURKIC_FAMILY_CODES or code in HISTORICAL_TURKIC_STAGES or code.startswith("trk-")

    loans = [(name, donor) for name, donor, _ in steps if name in BORROWING_TEMPLATES]
    return bool(loans) and all(turkic(donor) for _, donor, _ in steps) and all(
        name in LEARNED_BORROWING_TEMPLATES
        or (donor in HISTORICAL_TURKIC_STAGES and _stage_is_ancestor(donor, lang_code))
        for name, donor in loans
    )


def _origin_from_templates(record: dict[str, Any], lang_code: str | None = None) -> tuple[str | None, str, str]:
    """``etymology_templates``ten köken, NİHAİ verici dil ve özgün biçmi çıkarır.

    ⚠️ **Zincirin tamamı taranır, ilk halkası değil.** Bu ayrım ölçüldü ve
    kritik çıktı: Türkçe ``sabun``un şablon dizisi şudur::

        ('inh', 'ota', 'صابون')      ilk halka MİRAS (Osmanlıcadan)
        ('der', 'ar',  'صَابُون')      ikinci halka Arapçaya çıkıyor

    Yalnız ilk halkaya bakan bir uygulama ``sabun``u **miras** sayar. Oysa
    kelime nihayetinde Arapçadan gelir; Osmanlıca yalnız aracıdır. Aynı
    hata ``kitap``, ``duvar``, ``çorap``, ``pencere`` ve ``çay``da da
    tekrarlanıyordu — negatif kontrol bataryasında alıntı tuzaklarının
    tamamı bu yüzden kaçırılıyordu.

    Ölçüt: zincirin herhangi bir halkası **alıntı şablonu** taşıyorsa ya da
    **Türki ailenin dışına** çıkıyorsa, kelime alıntıdır.
    """
    steps: list[tuple[str, str, str]] = []
    for template in record.get("etymology_templates", []) or []:
        name = str(template.get("name", "")).lower()
        if name in TREE_TEMPLATES:
            steps.extend(parse_tree_template(template))
            continue
        args = template.get("args", {}) or {}
        donor = str(args.get("2", "") or "").strip()
        form = str(args.get("3", "") or "").strip()
        if not donor:
            continue
        if name in BORROWING_TEMPLATES or name in INHERITANCE_TEMPLATES:
            steps.append((name, donor, form))

    text_donor, text_form = donor_from_text(_etymology_text(record))

    if not steps:
        if text_donor:
            return "alıntı", text_donor, text_form
        return None, "", ""

    explicit_borrowing = any(name in BORROWING_TEMPLATES for name, _, _ in steps)
    leaves_family = any(donor not in TURKIC_FAMILY_CODES for _, donor, _ in steps)

    # Nihai kaynak: zincirin en uzak ucundaki dil.
    final_lang, final_form = steps[-1][1], steps[-1][2]
    # Türk dili içi diriltme: verici alanları eskisi gibi (zincirin ucu)
    # kalır, yalnız köken sınıfı ayrılır. Eskiden `betik` "alıntı, verici
    # trk-pro" idi; soy koduyla süzen tüketiciler dışında her yerde yabancı
    # alıntı gibi görünüyordu. Metin aile dışı verici gösteriyorsa
    # (şablona yazılmamış uzak halka) diriltme değildir.
    recipient = str(lang_code if lang_code is not None else record.get("lang_code") or "")
    if _is_revival(steps, recipient) and not (
        text_donor and text_donor not in TURKIC_FAMILY_CODES and text_donor not in HISTORICAL_TURKIC_STAGES
    ):
        return REVIVAL_ORIGIN, final_lang, final_form
    if explicit_borrowing or leaves_family:
        # Aile dışına ilk çıkan halka gerçek vericidir.
        for _, donor, form in steps:
            if donor not in TURKIC_FAMILY_CODES:
                return "alıntı", donor, form
        return "alıntı", final_lang, final_form

    # Şablon zinciri aile içinde kalıyor ama metin aile dışı bir kaynak
    # gösteriyorsa, zincirin uzak halkası şablona yazılmamış demektir.
    #
    # ⚠️ Zincir aile ATA DİLİNDE (``trk-pro``) bitiyorsa uzak halka yazılmış
    # demektir: kelime kök düzeyine kadar mirastır ve metindeki yabancı dil
    # adı verici değil, akraba listesi / karşılaştırma / reddedilen
    # benzerliktir. Ölçüldü (tr dökümü): bu durumda metin vericisi alan 24
    # maddenin 24'ü yanlıştı — ``torun`` ("similarity to Old Armenian թոռն
    # is accidental" -> hy), ``koyun``/``eşek``/``kırağı`` (Moğolca akraba),
    # ``küçük``/``çene`` (Farsça karşılaştırma). Zincir ``ota``da bitenler
    # (161) değişmez: orada metin gerçekten şablonun devamıdır (``kitap``).
    reaches_proto = final_lang.endswith("-pro")
    if text_donor and text_donor not in TURKIC_FAMILY_CODES and not reaches_proto:
        return "alıntı", text_donor, text_form
    return "miras", final_lang, final_form


#: Kelimenin KENDİ dilinde nasıl yapıldığını anlatan şablonlar.
FORMATION_TEMPLATES = frozenset({
    "suf", "suffix", "af", "affix", "pre", "prefix", "com", "compound",
    "con", "confix", "inf", "infix", "blend", "univerbation",
})

#: Satır içi değiştirici: ``𐰋𐰃𐱅𐰃<ts:biti-><t:to write>``.
_INLINE_MODIFIER = re.compile(r"<(\w+):([^<>]*)>")


def _template_part(raw: str, args: dict[str, Any], index: int) -> str:
    """Yapım şablonunun bir parçasının OKUNABİLİR biçimi.

    Runik/Arap yazılı parça okunmaz; okunuşu satır içi (``<ts:…>``) ya da
    ``ts2``/``tr2`` argümanında durur.
    """
    modifiers = dict(_INLINE_MODIFIER.findall(raw))
    bare = _INLINE_MODIFIER.sub("", raw).strip()
    reading = (
        modifiers.get("ts") or modifiers.get("tr")
        or str(args.get(f"ts{index}") or args.get(f"tr{index}") or "")
    ).strip()
    return reading or bare


def _formation_from_templates(record: dict[str, Any], lang_code: str) -> str:
    """Kelimenin kendi dilindeki yapımı: ``{{suf|otk|biti-|-g}}`` -> ``biti- + -g``.

    ⚠️ Bu bilgi ``origin`` sütununu DEĞİŞTİRMEZ — ölçülerek karar verildi.
    "Yapım şablonu varsa alıntı değil türetmedir" kuralı denendi: tr'de
    192 kaydı alıntıdan çıkarıyordu ve rastgele 40 örneğin ancak yarısı
    gerçekten Türkçe içi yapımdı (*yayın*, *tören*, *merhametli*); öbür
    yarısı BÜTÜN OLARAK alınmış Arapça/Farsça yapılardı (*kırtasiye*,
    *hükümdar*, *velhasıl*), Wiktionary onlara da Türkçe `af` koyuyor.
    `origin` alıntı değerlendirmesinin altın etiketidir; yarı yanlış kural
    oraya giremez. Yapı yalnız bilgi olarak taşınır.

    Şablonun dil argümanı kelimenin kendi dili olmalı: ``tsunami``nin
    ``compound`` şablonu Japoncadaki yapıyı anlatır, Türkçedekini değil.
    """
    for template in record.get("etymology_templates", []) or []:
        if str(template.get("name", "")).lower() not in FORMATION_TEMPLATES:
            continue
        args = template.get("args", {}) or {}
        if str(args.get("1", "")) != lang_code:
            continue
        parts = []
        for i in range(2, 12):
            raw = str(args.get(str(i), "") or "").strip()
            if not raw:
                continue
            part = _template_part(raw, args, i - 1)
            if part:
                parts.append(part)
        if len(parts) >= 2:
            return " + ".join(parts)
    return ""


def _cognates_from_templates(record: dict[str, Any]) -> str:
    """Sözlüğün kendisinin andığı akrabalar (``cog`` şablonu), JSON dizisi."""
    out: list[dict[str, str]] = []
    for template in record.get("etymology_templates", []) or []:
        if str(template.get("name", "")).lower() not in ("cog", "cognate"):
            continue
        args = template.get("args", {}) or {}
        # Motor koduna çevrilir: `cog|kjh|…` Hakasça (`khk`), `cog|khk|…` Halha
        # Moğolcası (`mn`). Çevrilmeden 1.596 kaydın Hakasça/Salarca akrabası
        # arama motorunda atılıyordu.
        from engine.fetchers.base import lang_code_from_wiktionary

        lang = lang_code_from_wiktionary(str(args.get("1", "") or ""))
        raw = str(args.get("2", "") or "").strip()
        form = _INLINE_MODIFIER.sub("", raw).strip()
        reading = str(args.get("ts") or args.get("tr") or "").strip()
        gloss = str(args.get("t") or args.get("gloss") or args.get("4") or "").strip()
        if lang and (form or reading):
            out.append({"lang": lang, "form": form, "reading": reading, "gloss": gloss})
    return json.dumps(out, ensure_ascii=False) if out else ""


#: Bilimsel çeviriyazının seri işaretleri ve gırtlaksılları — arama anahtarı
#: olarak taşınmazlar.
_ROMANISATION_NOISE = str.maketrans("", "", "¹²³⁴ʾʿ")


def _best_romanisation(record: dict[str, Any]) -> str:
    """kaikki'nin kendi çevriyazısı; birden çoksa EN OKUNAKLI olanı.

    ⚠️ Bir kayıtta birden çok romanizasyon olabilir ve ilkini almak yanlış::

        𐰚𐰃𐰾𐰃  forms = ["k²is²i", "kişi"]   -> ilki alınırsa 'kisi', ş kaybolur
        𐱅𐰭𐰼𐰃  forms = ["t²ŋr²i", "Teŋri"]  -> ilki alınırsa 'tŋri'

    Üst simgeli rakamlar (¹²³⁴) Orhun yazısının ön/art ünsüz serisini
    işaretleyen BİLİMSEL çeviriyazı kuralıdır, okunabilir bir biçim değil.
    Bu yüzden üst simge taşımayan aday tercih edilir. Ölçüldü: otk'de
    370 kayıt çok romanizasyonlu; "en iyi" seçimi "ilk" seçimine göre
    kullanılabilir kayıt sayısını 302'den 312'ye çıkarıyor.
    """
    candidates: list[str] = []
    # `ts` (transcription) ünlüleri yazılmış okunuştur ve en güvenilir adaydır.
    # Ölçüldü: 𐰋𐰃𐱅𐰏 kaydında ne `romanization` etiketli biçim ne `tr` var,
    # yalnız ``ts: "bitig"``; bu alan okunmayınca anahtar runik çeviriyazıdan
    # ``bıtg`` kalıyor ve `bitig` araması 19 kaynakta da boş dönüyordu.
    # otk'de 470 kaydın 139'u `ts` taşıyor.
    #
    # `ts` bazen birden çok okunuş verir: "qaġan, xaġan", "bädiz, bediz",
    # "tögültün/ or /tügültün". Bölünmezse noktalama atılıp okunuşlar
    # BİRLEŞİYORDU (ölçüldü: `kaganhagan`, `bedizbediz`, `eşideşit`).
    # İlk okunuş alınır.
    for template in record.get("head_templates") or []:
        text = str((template.get("args") or {}).get("ts") or "").strip()
        first = next((p for p in re.split(r"\s*(?:,|/|\bor\b)\s*", text) if p.strip()), "")
        if first:
            candidates.append(first.strip())
    for form in record.get("forms") or []:
        if "romanization" in (form.get("tags") or []):
            text = (form.get("form") or "").strip()
            if text:
                candidates.append(text)
    for template in record.get("head_templates") or []:
        text = str((template.get("args") or {}).get("tr") or "").strip()
        if text:
            candidates.append(text)
    if not candidates:
        return ""
    clean = [c for c in candidates if not any(ch in c for ch in "¹²³⁴")]
    return (clean or candidates)[0]


def _romanised_comparison(record: dict[str, Any]) -> str:
    """Romanizasyondan türetilmiş karşılaştırma biçimi.

    ⚠️ KİRİL YAZILI KAYITLARDA KULLANILMAZ — ölçülerek karar verildi.
    Kısıtsız kural 24.280 kaydın karşılaştırma biçimini değiştiriyordu ve
    Kiril blokları İYİLEŞME DEĞİL, SÖZLEŞME DEĞİŞİKLİĞİ getiriyordu::

        ky  'çıçırkanak' -> 'cıcırkanak'   (ç -> c; Türkçe `ç` ile eşleşme bozulur)
        sah 'harıs'      -> 'karıs'
        ba  'yazmış'     -> 'yadmış'
        kk  'kivi'       -> 'kıyviy'

    Bizim Kiril tablomuz zaten Türkolojik karşılaştırma biçimi üretiyor;
    kaikki romanizasyonu ise dil-içi ya da İngilizce sözleşme izliyor.
    Abjad ve runik yazılarda ise durum tersi — oralarda BİZİM çeviriyazımız
    ünlü/ses kaybediyor (``эҥин`` -> ``ein`` gibi bir kayıp Kiril'de de
    görülüyor ama azınlıkta; Arap/Orhun yazısında kural).

    Kısıt sonrası etki (24.280 -> 12.855): ota 9029, ug 2278, klj 654,
    otk 345, chg 244 korunur; kk 6065->86, ba 2377->0, sah 1116->0,
    ky 593->16, alt 366->0, kum/nog/khk -> 0.
    """
    import re
    import unicodedata

    if re.search(r"[Ѐ-ӿ]", str(record.get("word") or "")):
        return ""

    raw = _best_romanisation(record)
    if not raw:
        return ""
    # Fiil kökleri sözlükte tireyle yazılır (``bil-``); tire arama anahtarı değil.
    text = unicodedata.normalize("NFC", raw.strip().strip("-").translate(_ROMANISATION_NOISE))
    return to_comparison_form(text)


#: Kiril imlasında ``ё``/``ю`` ünsüzden sonra (ve söz başında) ö/ü olan
#: diller: Karaçay-Balkarca кёз = köz, тюз = tüz, ёгюз = ögüz. Genel Kiril
#: tablosu bunları yo/yu okur (``kyoz``) ve hiçbir Türkçe biçimle eşleşmez.
#: Ünlüden ya da ь/ъ'dan sonra y+ünlüdür (аю = ayu). ⚠️ Kumukça da aynı
#: imlayı kullanır (гёз), ama indeksi yeniden kurmadan 3.330 kaydı
#: değiştirmemek için şimdilik yalnız Karaçay-Balkarca.
_FRONT_ROUNDED_YO_YU = frozenset({"krc"})
_AFTER_VOWEL = re.compile(r"(?<=[аеёиоуыэюяьъ])([ёю])")


def _front_rounded_cyrillic(word: str) -> str:
    """кёз -> кöз, ёгюз -> öгüз, аю -> аю (ünlüden sonra y+ünlü kalır)."""
    marked = _AFTER_VOWEL.sub(lambda m: {"ё": "\x00", "ю": "\x01"}[m.group(1)], word.lower())
    return marked.replace("ё", "ö").replace("ю", "ü").replace("\x00", "ё").replace("\x01", "ю")


def comparison_for(word: str, lang_code: str) -> str:
    """Kiril kaydın karşılaştırma biçimi, dile özgü ``ё``/``ю`` okumasıyla."""
    if lang_code in _FRONT_ROUNDED_YO_YU:
        word = _front_rounded_cyrillic(word)
    return to_comparison_form(word)


def iter_entries(path: Path, lang_code: str, *, skip_form_of: bool = False) -> Iterator[LexiconEntry]:
    """Bir kaikki JSONL dökümünü satır satır okur (bellekte tutmadan).

    :param skip_form_of: bütün anlamları çekim/biçim göndermesi olan maddeleri
        atla. Türkçe sürümde 325 bin Türkçe kaydın 114 bini (`boncuğu`,
        `boncuklar`) böyledir; indekse girerse arama gürültüsüdür.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            word = str(record.get("word", "")).strip()
            if not word:
                continue
            senses = record.get("senses") or []
            if skip_form_of and senses and all(
                s.get("form_of") or "form-of" in (s.get("tags") or []) for s in senses
            ):
                continue
            comparison = comparison_for(word, lang_code)
            if not comparison:
                # Arap veya Orhun yazısı: önce Latin'e çevir.
                # Ölçüldü: bu adım olmadan 4.215 Uygurca kaydın yalnız 114'ü
                # indekslenebiliyordu, Çağatayca'nın tamamı düşüyordu.
                comparison = to_comparison_form(transliterate_to_latin(word))

            # ⚠️ KENDİ ÇEVİRİYAZIMIZ KAYIPLI; kaynağınki otoriter.
            # Wiktionary editörlerinin yazdığı romanizasyon aynı kayıtta
            # duruyordu ve atılıyordu. Orhun yazısı ünlü niteliğini
            # kodlamadığı için bizim tablomuz `bıtı`, `tŋrı`, `kısı`
            # üretiyor; kaynak ise `biti`, `teŋri`, `kişi` diyor.
            #
            # Kural (ölçülerek sadeleşti): romanizasyon en az mevcut biçim
            # kadar uzunsa onu kullan, değilse mevcut kal. "Her zaman
            # romanizasyon" YANLIŞ olurdu — Çağataycada bazı okumalar
            # ünsüz iskeletidir (``ʾslʾm`` -> ``slm``, ``mn``) ve mevcut
            # Arap çeviriyazısından kötüdür.
            #
            # Ölçüm (>=3 harfli, yani aranabilir kayıt sayısı):
            #     otk   248 -> 312      chg   565 -> 575
            #     ota  9530 -> 9723
            # Nitelik düzeltmeleri: 'chad'->'cihad', 'amam'->'imam',
            # 'allh'->'allah', 'kgnlg'->'kağanlığ', 'myvh'->'meve'.
            romanised = _romanised_comparison(record)
            if len(romanised) >= len(comparison):
                comparison = romanised

            if not comparison:
                continue
            origin, donor_lang, donor_form = _origin_from_templates(record, lang_code)
            yield LexiconEntry(
                lang_code=lang_code,
                word=word,
                comparison=comparison,
                pos=str(record.get("pos", "")),
                gloss=_first_gloss(record),
                ipa=_first_ipa(record),
                etymology=_clean_note(_etymology_text(record)),
                long_vowels=extract_long_vowels(_first_ipa(record)),
                origin=origin,
                donor_lang=donor_lang,
                donor_form=donor_form,
                formation=_formation_from_templates(record, lang_code),
                cognates=_cognates_from_templates(record),
            )


def _within_one(a: str, b: str) -> int:
    """Levenshtein uzaklığı 0 ya da 1 ise onu, değilse 2 döndürür (O(n))."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 2
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        return 1 if a[i + 1:] == b[i + 1:] else 2  # tek değiştirme
    return 1 if a[i:] == b[i + 1:] else 2  # tek ekleme


class LexiconIndex:
    """FTS5 destekli yerel sözlük indeksi."""

    def __init__(self, path: Path | None = None):
        self.path = path or INDEX_PATH

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # -- kurulum ------------------------------------------------------------

    def build(
        self,
        *,
        sources: dict[str, Path] | None = None,
        batch: int = 5000,
        with_ru_edition: bool = True,
    ) -> dict[str, Any]:
        """İndeksi sıfırdan kurar.

        :param with_ru_edition: Rusça Wiktionary sürümü dökümleri de
            eklensin mi? ⚠️ O maddeler **yalnız tanık ve arama verisidir**;
            köken çıkarılamaz (şemada ``etymology_templates`` yok) ve
            anlamlar Rusçadır.
        """
        files = sources if sources is not None else discover_lexicons()
        ru_files = discover_ru_edition() if (with_ru_edition and sources is None) else {}
        # Türkçe sürüm Rusça sürümle aynı kurala tabidir (yalnız tanık/arama).
        edition_files = [(code, path, "ru") for code, path in sorted(ru_files.items())]
        if with_ru_edition and sources is None:
            edition_files += [
                (code, path, "tr") for code, path in sorted(discover_edition(TR_EDITION_SUBDIR).items())
            ]
        if not files:
            raise FileNotFoundError(
                f"{LEXICON_DIR} altında döküm yok. Önce indirin: "
                "python scripts/download_lexicons.py --all"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        counts: dict[str, int] = {}
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            for lang_code, source in sorted(files.items()):
                rows: list[tuple] = []
                total = 0
                for entry in iter_entries(source, lang_code):
                    rows.append(entry.as_row())
                    if len(rows) >= batch:
                        self._insert(connection, rows)
                        total += len(rows)
                        rows.clear()
                if rows:
                    self._insert(connection, rows)
                    total += len(rows)
                counts[lang_code] = total
                logger.info("indekslendi: %s -> %d kayıt", lang_code, total)

            # Rusça sürüm maddeleri İngilizce sürümün ÜSTÜNE eklenir.
            # ⚠️ Silme yok: aynı biçim iki sürümde de varsa ikisi de kalır.
            # İngilizce kayıt köken bilgisi taşır, Rusça kayıt taşımaz;
            # sorgular köken alanı dolu olanı zaten tercih eder.
            for lang_code, source, edition in edition_files:
                rows = []
                total = 0
                for entry in iter_entries(source, lang_code, skip_form_of=edition == "tr"):
                    rows.append(entry.as_row())
                    if len(rows) >= batch:
                        self._insert(connection, rows)
                        total += len(rows)
                        rows.clear()
                if rows:
                    self._insert(connection, rows)
                    total += len(rows)
                counts[lang_code] = counts.get(lang_code, 0) + total
                logger.info("indekslendi (%s sürümü): %s -> %d kayıt", edition, lang_code, total)

            connection.execute(
                "INSERT INTO entries_fts(entries_fts) VALUES('rebuild')"
            )
            info = {
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "languages": json.dumps(counts, ensure_ascii=False),
                "total_entries": str(sum(counts.values())),
                "note": (
                    "Arama indeksidir; akrabalık kararı buradan verilmez."
                ),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO build_info(key, value) VALUES (?, ?)",
                info.items(),
            )
        return {"languages": counts, "total": sum(counts.values())}

    def append(self, lang_code: str, source: Path, edition: str) -> int:
        """Kurulu indekse TEK bir sürüm dökümünü ekler (yeniden kurmadan).

        İndeksin kurulumu uzun ve indeks eval'lerin girdisidir; sonradan
        inen küçük bir döküm (ör. Rusça sürüm Karaçay-Balkarca, Türkçe
        sürüm Kumanca) için baştan kurmak gerekmez. ``build`` aynı dökümü
        ``discover_edition`` ile zaten bulur; buradaki ekleme onunla aynı
        satırları üretir. Aynı dosya (SHA-256) ikinci kez eklenmez.

        ⚠️ Veri açığı dökümleri (``gap/``) bununla eklenip ölçüldü ve
        BAĞLANMADI: yeni Karaçay-Balkarca/Kumanca tanıklarının kesinliği
        ~0,76 (eşsesli yazılış eşleşmesi). Ayrıntı ``GAP_LEXICONS``.

        :return: eklenen kayıt sayısı (zaten ekliyse 0).
        """
        import hashlib

        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        key = f"appended:{edition}:{lang_code}"
        with self.connect() as connection:
            done = connection.execute("SELECT value FROM build_info WHERE key = ?", (key,)).fetchone()
            if done and done[0] == sha:
                return 0
            if done:
                raise RuntimeError(f"{key} başka bir dökümle eklenmiş; indeksi --build ile yeniden kurun")
            first = connection.execute("SELECT COALESCE(MAX(id), 0) FROM entries").fetchone()[0]
            rows = [e.as_row() for e in iter_entries(source, lang_code, skip_form_of=edition == "tr")]
            self._insert(connection, rows)
            connection.execute(
                "INSERT INTO entries_fts(rowid, word, comparison, gloss, etymology) "
                "SELECT id, word, comparison, gloss, etymology FROM entries WHERE id > ?",
                (first,),
            )
            if lang_code in _FRONT_ROUNDED_YO_YU:
                self._refresh_comparisons(connection, lang_code, first)
            languages = json.loads(
                (connection.execute("SELECT value FROM build_info WHERE key = 'languages'").fetchone() or ["{}"])[0]
            )
            languages[lang_code] = languages.get(lang_code, 0) + len(rows)
            total = int(
                (connection.execute("SELECT value FROM build_info WHERE key = 'total_entries'").fetchone() or ["0"])[0]
            ) + len(rows)
            connection.executemany(
                "INSERT OR REPLACE INTO build_info(key, value) VALUES (?, ?)",
                [("languages", json.dumps(languages, ensure_ascii=False)),
                 ("total_entries", str(total)), (key, sha)],
            )
        return len(rows)

    @staticmethod
    def _refresh_comparisons(connection: sqlite3.Connection, lang_code: str, before_id: int) -> None:
        """Eklemeden ÖNCEKİ kayıtların karşılaştırma biçimini ``comparison_for``la
        günceller (kurulumda ``iter_entries`` bunu zaten yapar); FTS de."""
        rows = connection.execute(
            "SELECT id, word, comparison, gloss, etymology FROM entries "
            "WHERE lang_code = ? AND id <= ? AND (word LIKE '%ё%' OR word LIKE '%ю%' "
            "OR word LIKE '%Ё%' OR word LIKE '%Ю%')",
            (lang_code, before_id),
        ).fetchall()
        for row_id, word, old, gloss, etymology in rows:
            new = comparison_for(word, lang_code)
            if not new or new == old:
                continue
            connection.execute(
                "INSERT INTO entries_fts(entries_fts, rowid, word, comparison, gloss, etymology) "
                "VALUES('delete', ?, ?, ?, ?, ?)",
                (row_id, word, old, gloss, etymology),
            )
            connection.execute("UPDATE entries SET comparison = ? WHERE id = ?", (new, row_id))
            connection.execute(
                "INSERT INTO entries_fts(rowid, word, comparison, gloss, etymology) VALUES (?, ?, ?, ?, ?)",
                (row_id, word, new, gloss, etymology),
            )

    @staticmethod
    def _insert(connection: sqlite3.Connection, rows: list[tuple]) -> None:
        connection.executemany(
            "INSERT INTO entries(lang_code, word, comparison, pos, gloss, ipa, "
            "etymology, long_vowels, origin, donor_lang, donor_form, formation, cognates) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    # -- sorgular -----------------------------------------------------------

    def lookup(
        self, form: str, *, languages: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Karşılaştırma biçmiyle **tam** eşleşme arar."""
        comparison = to_comparison_form(form)
        if not comparison:
            return []
        query = "SELECT * FROM entries WHERE comparison = ?"
        params: list[Any] = [comparison]
        if languages:
            query += f" AND lang_code IN ({','.join('?' * len(languages))})"
            params.extend(languages)
        query += " LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def is_attested_stem(self, form: str, *, lang: str = "tr") -> bool:
        """Bu biçim gerçekten bir sözlükbirim mi?

        ⚠️ "İndekste var mı" diye sormak YETMEZ: kayıtların %69'u çekim
        satırıdır. ``barda`` indekste var ama tek anlamı
        ``locative singular of bar`` — yani kök değil, ``bar``ın bulunma hâli.

        Üç yollu kural (ölçüldü):
        1. Çekim olmayan bir anlamı varsa sözlükbirimdir.
        2. Fiil kökleri sözlükte MASTARLA durur (``taşı`` yok ama ``taşımak``
           var), o yüzden ``kök+mak/mek`` de kabul edilir.
        3. Hiçbiri yoksa sözlükbirim sayılmaz.

        ⚠️ ``lang`` varsayılanı ``tr``: Türkçe bir kelimeyi çözümlerken başka
        dildeki tesadüfi eşleşme tanık değildir. Ölçüldü — aşırı soymanın
        ürettiği sahte köklerin tamamı böyle eşleşiyordu: ``mene`` [az],
        ``avs`` [ota], ``köre`` [kdr], ``gara`` [tk].
        """
        from engine.utils.morphology import is_inflection_gloss

        rows = self.lookup(form, languages=[lang], limit=5) or []
        if any(not is_inflection_gloss(row.get("gloss") or "") for row in rows):
            return True
        return any(
            self.lookup(form + suffix, languages=[lang], limit=1)
            for suffix in ("mak", "mek")
        )

    def fuzzy_lookup(
        self, form: str, *, max_distance: int = 1, languages: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Yakın eşleşme arar — ileri tahmin bir harf yanılabilir.

        Ölçüldü: ileri tahminin **%47,6**'sı tam tutuyor ama **%75,5**'i bir
        harf içinde. Bulanık arama olmadan tahminlerin üçte biri boşa gider.
        """
        comparison = to_comparison_form(form)
        if not comparison:
            return []
        # Uzunluk penceresi indeksin taranmasını sınırlar.
        low, high = len(comparison) - max_distance, len(comparison) + max_distance
        query = "SELECT * FROM entries WHERE length(comparison) BETWEEN ? AND ?"
        params: list[Any] = [low, high]
        if languages:
            query += f" AND lang_code IN ({','.join('?' * len(languages))})"
            params.extend(languages)

        from engine.utils.edit_distance import edit_distance

        # ⚠️ Tam Levenshtein tablosu her aday için hesaplanıyordu: ölçüldü,
        # 30 kelimede 1.041.766 çağrı, sürenin %97'si (indeks 449 bin kayıt;
        # `make eval-borrowing` 16 -> 35 dk). Uzaklık ≤ 1 sorusu tek geçişte
        # cevaplanır; sonuç birebir aynıdır.
        within = _within_one if max_distance == 1 else None

        results: list[dict[str, Any]] = []
        with self.connect() as connection:
            for row in connection.execute(query, params):
                candidate = row["comparison"]
                if within is not None:
                    distance = within(comparison, candidate)
                else:
                    distance = edit_distance(comparison, candidate)
                if distance <= max_distance:
                    entry = dict(row)
                    entry["edit_distance"] = distance
                    results.append(entry)
        results.sort(key=lambda e: (e["edit_distance"], e["lang_code"], e["word"]))
        return results

    def search(self, text: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """FTS5 tam metin araması — anlam ve etimoloji metninde arar."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT e.* FROM entries_fts f JOIN entries e ON e.id = f.rowid "
                "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
                (text, limit),
            )
            return [dict(row) for row in rows]

    def borrowings(
        self, lang_code: str, *, donor: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Bir dilin alıntı kayıtları — verici dil ve özgün biçimle."""
        query = "SELECT * FROM entries WHERE lang_code = ? AND origin = 'alıntı'"
        params: list[Any] = [lang_code]
        if donor:
            query += " AND donor_lang = ?"
            params.append(donor)
        query += " ORDER BY word LIMIT ?"
        params.append(limit)
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def donor_counts(self, lang_code: str) -> list[tuple[str, int]]:
        """Verici dile göre alıntı sayısı — hangi dilden kaç kelime gelmiş."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT donor_lang, COUNT(*) AS n FROM entries "
                "WHERE lang_code = ? AND origin = 'alıntı' AND donor_lang != '' "
                "GROUP BY donor_lang ORDER BY n DESC",
                (lang_code,),
            )
            return [(row["donor_lang"], row["n"]) for row in rows]

    def stats(self) -> dict[str, Any]:
        if not self.exists:
            return {"exists": False}
        with self.connect() as connection:
            info = {
                row["key"]: row["value"] for row in connection.execute("SELECT * FROM build_info")
            }
            total = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            with_etymology = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE origin IS NOT NULL"
            ).fetchone()[0]
            borrowed = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE origin = 'alıntı'"
            ).fetchone()[0]
            with_length = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE long_vowels != ''"
            ).fetchone()[0]
        # ⚠️ ``**info`` ÖNCE gelmeli: ``build_info`` tablosunda da
        # ``total_entries`` anahtarı var ama değeri METİN. Sona konursa
        # hesaplanan tamsayıyı ezer ve ``stats()["total_entries"]`` bir
        # dizgi döner.
        return {
            **info,
            "exists": True,
            "path": str(self.path),
            "total_entries": total,
            "with_origin": with_etymology,
            "borrowed": borrowed,
            "with_long_vowels": with_length,
        }


#: Rusça Wiktionary sürümü dökümleri (Faz B3).
#:
#: ⚠️ Bu maddeler **yalnız tanık ve arama verisidir**. Şemada
#: ``etymology_templates`` yok, dolayısıyla köken (alıntı/miras) çıkarılamaz;
#: anlamlar Rusçadır, dolayısıyla anlam kısıtlı verici yakınlığı sinyali
#: bu maddelerde çalışmaz. İngilizce sürümdeki bir madde varsa **o
#: kazanır**: köken bilgisi taşıyan kayıt tercih edilir.
RU_EDITION_SUBDIR = "ru_edition"


#: Türkçe Wiktionary sürümü; Rusça sürümle AYNI kural: yalnız tanık ve arama.
TR_EDITION_SUBDIR = "tr_edition"

#: Veri açığı dökümleri (Karaçay-Balkarca ru, Kumanca tr): ``build`` BAKMAZ.
#: Ölçüldü, kesinlik eşiği tutmadı (bkz. ``scripts/download_lexicons.py``
#: ``GAP_LEXICONS``); yalnız ``--append gap <dil>`` ile elle eklenir.
GAP_SUBDIR = "gap"


def discover_ru_edition() -> dict[str, Path]:
    """Rusça sürüm dökümleri — yoksa boş sözlük."""
    return discover_edition(RU_EDITION_SUBDIR)


def discover_edition(subdir: str) -> dict[str, Path]:
    """İngilizce dışı bir Wiktionary sürümünün dökümleri — yoksa boş sözlük."""
    base = LEXICON_DIR / subdir
    if not base.exists():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(base.iterdir()):
        if path.name.endswith(".jsonl.gz"):
            found[path.name[: -len(".jsonl.gz")]] = path
        elif path.suffix == ".jsonl":
            found[path.stem] = path
    return found


def discover_lexicons() -> dict[str, Path]:
    """``data/lexicons/`` altındaki indirilmiş dökümleri bulur."""
    found: dict[str, Path] = {}
    if not LEXICON_DIR.exists():
        return found
    from engine.fetchers.base import TURKIC_LANGUAGES_MAP

    for path in sorted(LEXICON_DIR.glob("*.jsonl*")):
        code = path.name.split(".")[0]
        # ⚠️ Yalnız TANIK dilleri: `trk-pro` (Wiktionary Proto-Türkçe
        # sayfaları) rekonstrüksiyondur; indekse girerse arama onu bir dil
        # kaydı gibi tanık listesine taşır.
        if code not in TURKIC_LANGUAGES_MAP:
            continue
        found[code] = path
    return found


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Yerel sözlük indeksi")
    ap.add_argument("--build", action="store_true", help="indeksi kur")
    ap.add_argument("--stats", action="store_true", help="indeks künyesi")
    ap.add_argument(
        "--append",
        nargs=2,
        metavar=("SÜRÜM", "DİL"),
        help="kurulu indekse tek sürüm dökümü ekle, yeniden kurmadan (ör. --append ru krc)",
    )
    ap.add_argument("--lookup", help="tam biçim ara")
    ap.add_argument("--fuzzy", help="bir harf toleranslı ara")
    ap.add_argument("--borrowings", help="bir dilin alıntılarını listele (dil kodu)")
    ap.add_argument("--donors", help="bir dilin verici dillerini say (dil kodu)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    index = LexiconIndex()

    if args.build:
        result = index.build()
        print(f"indeks kuruldu: {result['total']:,} kayıt, {len(result['languages'])} dil")
        for code, count in sorted(result["languages"].items(), key=lambda kv: -kv[1]):
            print(f"  {code:5} {count:>7,}")
        return 0

    if not index.exists:
        print("İndeks yok. Önce: python -m engine.db.lexicon_index --build")
        return 1

    if args.append:
        edition, code = args.append
        subdir = {"ru": RU_EDITION_SUBDIR, "tr": TR_EDITION_SUBDIR, "gap": GAP_SUBDIR}[edition]
        source = discover_edition(subdir)[code]
        if edition == "gap":  # veri açığı dökümünün asıl sürümü künyesinde
            meta = json.loads((LEXICON_DIR / GAP_SUBDIR / f"{code}.provenance.json").read_text(encoding="utf-8"))
            edition = meta.get("gap_edition") or "ru"
        added = index.append(code, source, edition)
        print(f"{edition}:{code} -> {added:,} kayıt eklendi" if added else f"{edition}:{code} zaten ekli")
        return 0
    if args.stats:
        for key, value in index.stats().items():
            print(f"{key:16} {value}")
    if args.lookup:
        for row in index.lookup(args.lookup, limit=args.limit):
            print(f"  {row['lang_code']:5} {row['word']:20} {row['gloss'][:50]}")
    if args.fuzzy:
        for row in index.fuzzy_lookup(args.fuzzy)[: args.limit]:
            print(f"  d={row['edit_distance']} {row['lang_code']:5} {row['word']:20} {row['gloss'][:40]}")
    if args.borrowings:
        for row in index.borrowings(args.borrowings, limit=args.limit):
            print(f"  {row['word']:18} <- {row['donor_lang']:6} {row['donor_form'][:24]}")
    if args.donors:
        for donor, count in index.donor_counts(args.donors)[: args.limit]:
            print(f"  {donor:8} {count:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
