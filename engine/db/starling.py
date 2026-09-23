"""
Starling Türk etimoloji veritabanı (``turcet``) okuyucusu.

Dybo ve Starostin'in Altay paketindeki Türk etimolojisi (2005): 2.017 Proto-
Türkçe kök, her biri 32 dil/katman alanında tanıklarıyla (ATU Eski Türkçe,
KRH Karahanlıca, TRK Türkiye Türkçesi, CHV Çuvaşça, JAK Yakutça …) ve
kaynakça atıflarıyla (EDT = Clauson, ЭСТЯ = Sevortjan, VEWT = Räsänen).

Neden: motorun kendi rekonstrüksiyonu `boncuk` için *bonjuk kuruyordu;
Starling *bōnčok ve tanık olarak Orhun ve Eski Uygurca mončuq veriyor.

⚠️ Starling Altay hipotezi okulundandır (bkz. ``starostinaltaic``). Türkçe
iç veri ve Proto-Türkçe biçimler alanda kullanılır, ama Altay karşılaştırması
akrabalık kanıtı değildir; bu modül yalnız Türk tablosunu okur.

Biçim
-----
``turcet.dbf`` dBase III tablosudur; her metin alanı 6 baytlık bir
göstericidir (4 bayt konum + 2 bayt uzunluk) ve metin ``turcet.var``
içindedir. Karakter kodlaması Starling'e özgüdür: 0x80-0xAF ve 0xE0-0xEF
CP866 (Rusça), 0xB0-0xDF ve 0xF0-0xFF Starling fonetik işaretleri, bazı
baytlar kendinden önceki harfe binen birleşik işaretlerdir. Tablolar
``rhaver/Starling-cs`` (``StarlingDecoder.cs``) okuyucusundan alındı.
"""

from __future__ import annotations

import re
import struct
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from engine.config import PROJECT_ROOT

STARLING_DIR = PROJECT_ROOT / "data" / "starling"

#: Kendinden önceki harfe binen birleşik işaretler.
_COMBINING = {
    0x5E: "\u0302", 0x7E: "\u0303",
    0xB1: "\u0301", 0xB3: "\u0328", 0xBB: "\u0307", 0xBC: "\u0308", 0xBF: "\u032F",
    0xC4: "\u0304", 0xC8: "\u030A", 0xC9: "\u0325",
    0xDB: "\u0300", 0xDC: "\u0323", 0xDF: "\u0306",
    0xF6: "\u030C",
}

#: CP866 dışındaki Starling fonetik işaretleri (0xB0-0xDF, 0xF0-0xFF).
_SPECIAL = {
    0xB0: "ā", 0xB2: "ä", 0xB4: "ǟ", 0xB5: "c\u0323", 0xB6: "č", 0xB7: "č\u0323", 0xB8: "δ",
    0xB9: "ē", 0xBD: "ɛ", 0xBE: "ʡ",
    0xC0: "ç", 0xC1: "ɣ", 0xC2: "ʁ", 0xC3: "ħ", 0xC5: "ī", 0xC6: "ɨ", 0xC7: "ɨ\u0304",
    0xCA: "ḳ", 0xCB: "ʎ", 0xCC: "ƛ", 0xCD: "-", 0xCE: "ƛ\u0323", 0xCF: "ɫ",
    0xD0: "Ɫ", 0xD1: "ŋ", 0xD2: "ō", 0xD3: "ö", 0xD4: "ȫ", 0xD5: "ɔ", 0xD6: "ɔ\u0304",
    0xD7: "ṗ", 0xD8: "q\u0307", 0xD9: "ß", 0xDA: "~", 0xDD: "š", 0xDE: "ṭ",
    0xF0: "ϑ", 0xF1: "ū", 0xF2: "ü", 0xF3: "ǖ", 0xF4: "ə", 0xF5: "ə\u0304", 0xF7: "ʷ",
    0xF8: "ɦ", 0xF9: "χ", 0xFA: "ʒ", 0xFB: "ǯ", 0xFC: "ž", 0xFD: "ʔ", 0xFE: "ʕ", 0xFF: "ʌ",
}

_DOUBLE_BYTE_START = 0x01
_SPECIAL_NEXT = 0x1D
_LAYOUT = {0x09: "\t", 0x0A: "\n", 0x0D: "", 0x15: "\n\n"}

#: Biçimlendirme etiketleri: ``\Ibit-\i`` (italik) gibi.
_TAG_RE = re.compile(r"\\[A-Za-z]")


def decode(raw: bytes) -> str:
    """Starling baytlarını Unicode'a çevirir (NFC).

    Yunanca / Kilise Slavcası çift bayt dizileri Türk tablosunda yok
    denecek kadar azdır; bunlar ``?`` ile geçilir, uydurulmaz.
    """
    out: list[str] = []
    double = False
    i = 0
    while i < len(raw):
        b = raw[i]
        if b in _COMBINING:
            out.append(_COMBINING[b])
            i += 1
            continue
        if double:
            if b == _DOUBLE_BYTE_START:
                i += 1
            elif b <= 0x7F:
                double = False
            else:
                out.append("?")
                i += 2
            continue
        if b == _SPECIAL_NEXT:
            out.append("?")
            i += 2
            continue
        if b == _DOUBLE_BYTE_START:
            double = True
        elif b < 0x20:
            out.append(_LAYOUT.get(b, ""))
        elif b < 0x7F:
            out.append(chr(b))
        elif b == 0x7F:
            pass
        elif b <= 0xAF:
            out.append(chr(b + 0x390))
        elif b <= 0xDF or b >= 0xF0:
            out.append(_SPECIAL.get(b, "?"))
        else:
            out.append(chr(b + 0x360))
        i += 1
    return _TAG_RE.sub("", unicodedata.normalize("NFC", "".join(out))).strip()


#: Starling alan adı -> motorun dil kodu (yalnız motorun tanıdıkları).
FIELD_LANGUAGES = {
    "ATU": "otk", "TRK": "tr", "TAT": "tt", "CHG": "chg", "UZB": "uz", "UIG": "ug",
    "SJG": "ybe", "AZB": "az", "TRM": "tk", "HAK": "khk", "SHR": "cjs", "ALT": "alt",
    "KHAL": "klj", "CHV": "cv", "JAK": "sah", "DOLG": "dlg", "TUV": "tyv", "TOF": "kim",
    "KRG": "ky", "KAZ": "kk", "NOGX": "nog", "BAS": "ba", "BLKX": "krc", "GAGX": "gag",
    "KRMX": "crh", "KLPX": "kaa", "SAL": "slq", "QUM": "kum",
}

#: Tanık etiketlerinin yaklaşık tarihi (yalnız TARİHİ bilinen eserler).
#: Starling tanığın yanına kaynağını yazar: "mončuq (Orkh., OUygh.)".
SOURCE_DATES = {
    "Orkh.": 732,       # Orhun yazıtları (Köl Tigin 732, Bilge Kağan 735)
    "MK": 1072,         # Kâşgarlı Mahmud, Dîvânu Lugâti't-Türk
    "KB": 1069,         # Kutadgu Bilig
    "IM": 1245,         # İbn Mühennâ (yaklaşık)
    "AH": 1300,         # Atebetü'l-Hakayık (yaklaşık)
    "Sangl.": 1760,     # Sanglax
    "Abush.": 1500,     # Abuşka (16. yy)
}


@dataclass(frozen=True)
class StarlingEtymology:
    """Starling Türk tablosunda tek bir kök."""

    number: int
    proto: str
    meaning: str
    reflexes: dict[str, str] = field(default_factory=dict)
    reference: str = ""

    def earliest_dated_source(self) -> tuple[int, str] | None:
        """Eski Türkçe / Karahanlıca tanıklarındaki en eski tarihli eser."""
        text = " ".join(self.reflexes.get(k, "") for k in ("ATU", "KRH"))
        dated = [(year, tag) for tag, year in SOURCE_DATES.items() if tag in text]
        return min(dated) if dated else None


def _read_fields(dbf: bytes) -> list[tuple[str, str, int]]:
    fields, pos = [], 32
    while dbf[pos] != 0x0D:
        name = dbf[pos:pos + 11].split(b"\0")[0].decode("ascii")
        fields.append((name, chr(dbf[pos + 11]), dbf[pos + 16]))
        pos += 32
    return fields


def load_turcet(directory: Path = STARLING_DIR) -> list[StarlingEtymology]:
    """``turcet.dbf`` + ``turcet.var`` dosyalarını okur; yoksa boş liste."""
    dbf_path, var_path = directory / "turcet.dbf", directory / "turcet.var"
    if not (dbf_path.exists() and var_path.exists()):
        return []
    dbf, var = dbf_path.read_bytes(), var_path.read_bytes()
    count = struct.unpack("<I", dbf[4:8])[0]
    header, row = struct.unpack("<HH", dbf[8:12])
    fields = _read_fields(dbf)

    out: list[StarlingEtymology] = []
    for index in range(count):
        record = dbf[header + index * row: header + (index + 1) * row]
        if record[:1] == b"*":  # silinmiş kayıt
            continue
        values: dict[str, str] = {}
        offset = 1
        for name, kind, length in fields:
            raw = record[offset:offset + length]
            offset += length
            if kind == "N":
                values[name] = raw.decode("ascii").strip()
                continue
            start, size = struct.unpack("<IH", raw)
            values[name] = decode(var[start:start + size]) if size else ""
        if not values.get("PROTO"):
            continue
        out.append(StarlingEtymology(
            number=int(values.get("NUMBER") or 0),
            proto=values["PROTO"],
            meaning=values.get("MEANING", ""),
            reflexes={k: v for k, v in values.items() if k in FIELD_LANGUAGES or k == "KRH"},
            reference=values.get("REFERENCE", ""),
        ))
    return out


#: Starling transkripsiyonu -> Türkçe imla (yalnız TRK alanını eşlemek için).
_TO_TURKISH = str.maketrans({"š": "ş", "č": "ç", "ǯ": "c", "ɣ": "ğ", "ɨ": "ı", "ä": "e", "j": "y", "ŋ": "n"})


def _turkish_forms(field_text: str) -> set[str]:
    """TRK alanındaki biçimler: "kuš 1, dial. kuš-" -> {"kuş"}."""
    text = re.sub(r"'[^']*'", " ", field_text)       # tırnaklı Rusça anlamlar
    text = re.sub(r"\([^)]*\)", " ", text)            # (Osm.), (dial.)
    forms = set()
    for token in re.split(r"[,;/\s]+", text.translate(_TO_TURKISH).lower()):
        token = token.strip(".0123456789")
        # Fiil kökü tireyle yazılır ("uč-"); ad biçiminden ayrı anahtar olur.
        verb = token.endswith("-")
        token = token.strip("-")
        if token.isalpha() and len(token) >= 2:
            forms.add(f"{token}-" if verb else token)
    return forms


@lru_cache(maxsize=1)
def turkish_lookup() -> dict[str, tuple[StarlingEtymology, ...]]:
    """Türkiye Türkçesi biçimi -> o biçimi TRK alanında anan kökler."""
    table: dict[str, list[StarlingEtymology]] = {}
    for etym in load_turcet():
        for form in _turkish_forms(etym.reflexes.get("TRK", "")):
            table.setdefault(form, []).append(etym)
    return {k: tuple(v) for k, v in table.items()}


def lookup_turkish(word: str) -> tuple[StarlingEtymology, ...]:
    """Türkçe kelimenin Starling kökleri (``boncuk`` -> *bōnčok)."""
    key = (word or "").strip().lower()
    table = turkish_lookup()
    if key in table:
        return table[key]
    # Mastar: önce fiil kökü ("uçmak" -> "uç-"), yoksa aynı yazılışlı ad.
    stem = re.sub(r"m[ae]k$", "", key)
    if stem != key:
        return table.get(f"{stem}-", ()) or table.get(stem, ())
    return table.get(f"{key}-", ())
