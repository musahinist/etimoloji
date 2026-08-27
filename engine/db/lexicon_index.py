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

INDEX_PATH = PROJECT_ROOT / "data" / "lexicons" / "index.db"

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


def donor_from_text(etymology_text: str) -> tuple[str, str]:
    """Etimoloji metninden verici dili çıkarır. Bulamazsa ``("", "")``."""
    if not etymology_text:
        return "", ""
    for name in _DONOR_NAMES_BY_LENGTH:
        index = etymology_text.find(name + " ")
        if index < 0:
            continue
        rest = etymology_text[index + len(name) + 1 :].strip()
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
    origin        TEXT,          -- 'alıntı' | 'miras' | NULL
    donor_lang    TEXT,
    donor_form    TEXT
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
        )


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
TURKIC_FAMILY_CODES = frozenset(
    {
        "tr", "ota", "otk", "trk-pro", "trk-oat", "trk-ogz-pro", "trk-cmn-pro",
        "az", "tk", "gag", "kk", "kaa", "ky", "tt", "ba", "nog", "kum", "krc",
        "crh", "uz", "ug", "cv", "sah", "tyv", "alt", "khk", "cjs", "slq",
        "chg", "klj", "dlg", "kim", "ybe", "clw", "atv", "bay", "qwm", "kdr",
    }
)


def _origin_from_templates(record: dict[str, Any]) -> tuple[str | None, str, str]:
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

    text_donor, text_form = donor_from_text(str(record.get("etymology_text", "")))

    if not steps:
        if text_donor:
            return "alıntı", text_donor, text_form
        return None, "", ""

    explicit_borrowing = any(name in BORROWING_TEMPLATES for name, _, _ in steps)
    leaves_family = any(donor not in TURKIC_FAMILY_CODES for _, donor, _ in steps)

    # Nihai kaynak: zincirin en uzak ucundaki dil.
    final_lang, final_form = steps[-1][1], steps[-1][2]
    if explicit_borrowing or leaves_family:
        # Aile dışına ilk çıkan halka gerçek vericidir.
        for _, donor, form in steps:
            if donor not in TURKIC_FAMILY_CODES:
                return "alıntı", donor, form
        return "alıntı", final_lang, final_form

    # Şablon zinciri aile içinde kalıyor ama metin aile dışı bir kaynak
    # gösteriyorsa, zincirin uzak halkası şablona yazılmamış demektir.
    if text_donor and text_donor not in TURKIC_FAMILY_CODES:
        return "alıntı", text_donor, text_form
    return "miras", final_lang, final_form


def iter_entries(path: Path, lang_code: str) -> Iterator[LexiconEntry]:
    """Bir kaikki JSONL dökümünü satır satır okur (bellekte tutmadan)."""
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
            comparison = to_comparison_form(word)
            if not comparison:
                # Arap veya Orhun yazısı: önce Latin'e çevir.
                # Ölçüldü: bu adım olmadan 4.215 Uygurca kaydın yalnız 114'ü
                # indekslenebiliyordu, Çağatayca'nın tamamı düşüyordu.
                comparison = to_comparison_form(transliterate_to_latin(word))
            if not comparison:
                continue
            origin, donor_lang, donor_form = _origin_from_templates(record)
            yield LexiconEntry(
                lang_code=lang_code,
                word=word,
                comparison=comparison,
                pos=str(record.get("pos", "")),
                gloss=_first_gloss(record),
                ipa=_first_ipa(record),
                etymology=str(record.get("etymology_text", "")),
                long_vowels=extract_long_vowels(_first_ipa(record)),
                origin=origin,
                donor_lang=donor_lang,
                donor_form=donor_form,
            )


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

    def build(self, *, sources: dict[str, Path] | None = None, batch: int = 5000) -> dict[str, Any]:
        """İndeksi sıfırdan kurar."""
        files = sources if sources is not None else discover_lexicons()
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

    @staticmethod
    def _insert(connection: sqlite3.Connection, rows: list[tuple]) -> None:
        connection.executemany(
            "INSERT INTO entries(lang_code, word, comparison, pos, gloss, ipa, "
            "etymology, long_vowels, origin, donor_lang, donor_form) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

        from engine.evaluation.metrics import edit_distance

        results: list[dict[str, Any]] = []
        with self.connect() as connection:
            for row in connection.execute(query, params):
                distance = edit_distance(comparison, row["comparison"])
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


def discover_lexicons() -> dict[str, Path]:
    """``data/lexicons/`` altındaki indirilmiş dökümleri bulur."""
    found: dict[str, Path] = {}
    if not LEXICON_DIR.exists():
        return found
    for path in sorted(LEXICON_DIR.glob("*.jsonl*")):
        code = path.name.split(".")[0]
        found[code] = path
    return found


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Yerel sözlük indeksi")
    ap.add_argument("--build", action="store_true", help="indeksi kur")
    ap.add_argument("--stats", action="store_true", help="indeks künyesi")
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
