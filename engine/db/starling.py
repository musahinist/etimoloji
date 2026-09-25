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
from engine.utils.attestation_dates import STARLING_SOURCE_DATES

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

#: Tanık etiketlerinin tarihi (yalnız TARİHİ bilinen eserler).
#: Starling tanığın yanına kaynağını yazar: "mončuq (Orkh., OUygh.)".
#: Yıllar ve gerekçeleri tek yerde: ``engine.utils.attestation_dates``.
SOURCE_DATES = STARLING_SOURCE_DATES


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


#: Anlam alanında fiil anlamı: "to fly", "1 to rise 2 jump up", "fly v.".
_VERBAL_MEANING = re.compile(r"(?:^|[\d;,]\s*)to\s|\bv\.")


def lookup_turkish(word: str) -> tuple[StarlingEtymology, ...]:
    """Türkçe kelimenin Starling kökleri (``boncuk`` -> *bōnčok).

    ``-mak/-mek`` ile biten kelimenin HEM ad HEM fiil okuması döner; hangisi
    kastedildiğini yazılış söyleyemez, anlam söyler (bkz.
    ``StarlingFetcher``): `kaymak` "krem" ve "kaymak (fiil)", `etmek`
    "yapmak" ve ağızdaki "ekmek" (*et-mek). Eskiden tam biçim tablodaysa
    yalnız o dönüyordu (`etmek` -> "bread"), değilse fiil kökü (`kaymak` ->
    *KAj- "to turn back").

    Kök fiil yazılmamışsa (TRK alanı `kalk` tiresiz) aynı yazılışlı ada
    bakılır, ama YALNIZ anlamı fiilse. Ölçüldü (indeksteki 81 Türkçe
    mastarın düştüğü bu yol): çoğu sahteydi — `kızmak` -> "girl",
    `bağırmak` -> "liver", `ırmak` -> *ɨr "song"; doğrular (`kalkmak` "to
    rise", `boğmak` "to strangle") fiil anlamlıdır.
    """
    key = (word or "").strip().lower()
    table = turkish_lookup()
    stem = re.sub(r"m[ae]k$", "", key)
    if stem == key:
        found = list(table.get(key, ())) or list(table.get(f"{key}-", ()))
    else:
        found = [*table.get(key, ()), *table.get(f"{stem}-", ())]
        if not found:
            found = [e for e in table.get(stem, ()) if _VERBAL_MEANING.search(e.meaning)]
    unique: dict[int, StarlingEtymology] = {}
    for etym in found:
        unique.setdefault(etym.number, etym)
    return tuple(unique.values())


# ---------------------------------------------------------------------------
# Starling tanıkları (YALNIZ GÖSTERİM)
# ---------------------------------------------------------------------------

#: Bir alan parçası: "köz 1", "kör- 2", "kösküt- 'to show'", "Guš (< Az.)".
_REFLEX_SPLIT = re.compile(r"[,;](?![^(]*\))(?![^']*'(?:[^']*'[^']*')*[^']*$)")


def _reflex_tokens(field_text: str) -> list[tuple[str, frozenset[str], str]]:
    """Alan -> [(biçim, anlam numaraları, tırnaklı anlam)].

    Alıntı işaretli parça (``(< Az.)``, ``< Pers.``) atılır: alıntı kök
    tanığı değildir.
    """
    out = []
    for part in _REFLEX_SPLIT.split(field_text or ""):
        part = part.strip()
        if not part or "<" in part:
            continue
        gloss = " ".join(re.findall(r"'([^']*)'", part))
        # Kelimeye bitişik ayraç isteğe bağlı sestir: qi(r)q -> qirq.
        part = re.sub(r"(?<=[^\s'(])\(([^)<]*)\)", r"\1", part)
        bare = re.sub(r"'[^']*'|\([^)]*\)", " ", part).split()
        if not bare:
            continue
        form = bare[0].strip(".,")
        if form in {"dial.", "dial", "?"} and len(bare) > 1:
            form = bare[1]
        nums = frozenset(t for t in bare[1:] if t.isdigit())
        if form and not form[0].isdigit():
            out.append((form, nums, gloss))
    return out


def reflex_witnesses(etym: StarlingEtymology, word: str) -> list[dict[str, str]]:
    """Seçilmiş Starling kökünün öbür dillerdeki biçimleri (dil kodu, biçim, anlam).

    Kökün numaralı anlamları varsa (*göŕ "1 eye 2 to see") sorgunun TRK
    alanındaki numarası seçilir; yalnız başka numaralı biçimi olan dil
    atlanır (`göz` için Dolganca kör- "2 to see" değil). Tırnaklı anlam
    taşıyan parça (türev ya da anlam kayması: kösküt- 'to show') atlanır.

    ⚠️ DÖNGÜSELLİK: bu biçimler başlık kökünü veren AYNI Starling
    kaydından gelir. Yayılım sayımına, A-HVP'ye, skora ve rekonstrüksiyon
    tanıklarına KATILMAZ; yalnız raporda "Starling tanığı" diye gösterilir.
    (Sütun modeli Starling alanlarıyla zaten ayrıca eğitilir.)
    """
    key = to_turkish_key(word)
    stem = re.sub(r"m[ae]k$", "", key)
    senses: set[str] = set()
    kinds: set[bool] = set()  # eşleşen TRK biçimi fiil mi (tireli)
    turkish = _reflex_tokens(etym.reflexes.get("TRK", ""))
    numbered = any(nums for _, nums, _ in turkish)
    for form, nums, gloss in turkish:
        verb = form.endswith("-")
        if to_turkish_key(form) != (stem if verb and stem != key else key):
            continue
        # Türkçe biçim kökün kendisi değil de yanına yazılmış bir türevse
        # (numarasız `tadɨm` "1 to taste" satırında) ya da anlamı kaymışsa
        # (`susak 'jar'`), öbür dillerin ilk biçimi o türevin akrabası olmaz.
        if gloss or (numbered and not nums):
            continue
        senses |= nums
        kinds.add(verb)
    if not kinds:
        return []
    out = []
    for field_name, code in FIELD_LANGUAGES.items():
        if code == "tr":
            continue
        for form, nums, gloss in _reflex_tokens(etym.reflexes.get(field_name, "")):
            # Başka numaralı anlam, ayrı anlamlı türev ('to show'), yeniden
            # kurulmuş (*) biçim ya da sorgunun türünden (ad/fiil) başka
            # biçim gösterilmez (`görmek` için Çağatayca göz değil).
            if (senses and nums and not nums & senses) or gloss or form.startswith("*"):
                continue
            if len(kinds) == 1 and form.endswith("-") not in kinds:
                continue
            out.append({"lang_code": code, "word": form, "gloss": gloss})
            break
    return out


def to_turkish_key(text: str) -> str:
    """Starling biçimi -> TRK eşlemesindeki anahtar (tire ve numara atılır)."""
    return (text or "").translate(_TO_TURKISH).lower().strip(".-0123456789 ")


# ---------------------------------------------------------------------------
# Moğol tablosu (``monget``) — YALNIZ verici dil ETİKETİ için
# ---------------------------------------------------------------------------

#: ``monget`` alanları: Yazı Moğolcası, Orta Moğolca, Halha, Buryatça, Kalmukça.
#: Öbür alanlar (Ordos, Dongxiang, Bao'an, Dagur, Moghol…) Saha'ya verici
#: olmadığından alınmaz.
MONGOLIC_FIELDS = ("WMO", "MMO", "HAL", "BUR", "KAL")


@dataclass(frozen=True)
class MongolicForm:
    """``monget`` tablosundan tek bir Moğolca biçim ve kökünün anlamı."""

    form: str
    meaning: str
    field: str
    proto: str


def _mongolic_forms(field_text: str) -> list[str]:
    """"dölü (L 272), döl (Khalkha)" -> ["dölü", "döl"].

    Kaynakça parantezleri, tırnaklı anlamlar ve köşeli notlar atılır.
    Starling ``j`` yazımı Saha karşılaştırmasındaki ``y`` ile hizalanır;
    ``ǯ`` ise ``ʤ`` olur (karşılaştırma biçiminde ikisi de ``c``).
    """
    text = re.sub(r"\([^)]*\)", "", field_text)
    text = re.sub(r"'[^']*'", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    out: list[str] = []
    for part in re.split(r"[,;~/]", text):
        token = part.strip().split(" ")[0] if part.strip() else ""
        if token:
            out.append(token.replace("j", "y").replace("ǯ", "ʤ"))
    return out


@lru_cache(maxsize=1)
def load_monget(directory: Path = STARLING_DIR) -> tuple[MongolicForm, ...]:
    """``monget.dbf`` + ``monget.var``: Moğolca biçimler ve İngilizce anlamları.

    ⚠️ **Yalnız verici dili etiketlemek içindir, alıntı gücüne girmez.**
    Ölçüldü (WOLD Saha, değerlendirme yarısı): bu biçimler verici yakınlığı
    havuzuna katılınca "alıntı mı?" F'si 0,615'ten 0,584'e düşüyor — Starling
    Moğol tablosu Türk-Moğol ortak sözvarlığıyla dolu ve miras Saha
    kelimeleri de ona yakın düşüyor. Ama alıntı olduğu zaten bilinen
    kelimenin vericisini seçerken kaikki Moğolcasından çok daha iyi:
    Saha Moğolca alıntıları Yazı Moğolcası biçiminden alınmıştır
    (``čakilɣan``), kaikki ise Halha Kirilini tutar (``цахилгаан``) ve
    6.480 maddesinin çoğu çekimli biçimdir.

    Dosyalar yoksa boş döner.
    """
    dbf_path, var_path = directory / "monget.dbf", directory / "monget.var"
    if not (dbf_path.exists() and var_path.exists()):
        return ()
    dbf, var = dbf_path.read_bytes(), var_path.read_bytes()
    count = struct.unpack("<I", dbf[4:8])[0]
    header, row = struct.unpack("<HH", dbf[8:12])
    fields = _read_fields(dbf)

    out: list[MongolicForm] = []
    for index in range(count):
        record = dbf[header + index * row: header + (index + 1) * row]
        if record[:1] == b"*":
            continue
        values: dict[str, str] = {}
        offset = 1
        for name, kind, length in fields:
            raw = record[offset:offset + length]
            offset += length
            if kind == "N":
                continue
            start, size = struct.unpack("<IH", raw)
            values[name] = decode(var[start:start + size]) if size else ""
        meaning = values.get("MEANING", "")
        if not meaning:
            continue
        for name in MONGOLIC_FIELDS:
            for form in _mongolic_forms(values.get(name, "")):
                out.append(MongolicForm(form, meaning, name, values.get("PROTO", "")))
    return tuple(out)
