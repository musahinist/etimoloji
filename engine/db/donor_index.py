"""
Verici dil sözlüğü indeksi — alıntı tespitinin en güçlü ölçülmüş sinyali.

Miller & List (2023, EACL, ``sabor``) ölçtü: bir kelimenin **verici dil
sözlüğüne yakınlığı** tek başına F1 **0,806**, kesinlik **0,931** veriyor.
Bizim şu anki motorumuz WOLD/Sakha'da F 0,385 — yani "her şeye alıntı de"
diyen trivial sistemin (0,464) bile altında.

Sinyal verici sözlüğü olmadan hesaplanamaz; bu modül o sözlüğü kurar.

⚠️ **Verici indeksi Türki arama indeksinden AYRIDIR.** Ayrı dosya, ayrı
dizin (``data/lexicons/donors/``). Karışsalardı Rusça ``море`` Türki bir
akraba adayı olarak dönerdi ve akrabalık kararı sessizce bozulurdu.

## Hangi vericiler?

WOLD'da Sakha'nın 663 alıntısının kaynağı ölçüldü::

    Rusça 284 · Moğolca 253 · Evenkice 19 · Çince 4 · Arapça 3 · Farsça 3

Türkçe ölçütü için dağılım bambaşkadır (Arapça/Farsça/Fransızca ağırlıklı).

## ⚠️ Yön sorunu

Verici sözlüğü **Türkiden alınmış** kelimeleri de içerir ve bunlar sinyali
ters yönden tetikler. Ölçüldü::

    Türkçe göz   ~ Ermenice գյոզ (gyoz)  SCA 0,040   "From Ottoman Turkish"
    Türkçe demir ~ Farsça   تمر  (tamor) SCA 0,075   "Borrowed from Turkic"

Bu maddeler "alıntı kanıtı" değil, **tam tersinin kanıtıdır**. Süzgeç
verici maddesinin kendi etimoloji metnine bakar; Türki bir kaynağa işaret
ediyorsa madde ``from_turkic`` işaretlenir ve yakınlık kanıtından çıkarılır.

⚠️ Süzgeç eksiksiz değildir: Fransızca ``béluga`` Rusça ``белуга``dan gelir,
o da Türkiden — ama Fransızca madde bunu yazmaz, yalnız Rusçayı gösterir.
Zincirin ikinci halkasını görmüyoruz.

## Anlam kısıtı

Ham biçim benzerliği tek başına **şans benzerliğine** açıktır: yüz binlerce
maddelik bir sözlükte kısa bir CVC biçmine benzeyen bir şey her zaman
bulunur (bkz. ``evaluation.significance.chance_resemblance_test``). Bu yüzden
her madde **anlamıyla** birlikte saklanır ve sorgu anlam kısıtlı yapılabilir.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.config import LEXICON_DIR
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

DONOR_DIR = LEXICON_DIR / "donors"
DEFAULT_DB = DONOR_DIR / "donors.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS donor_entries (
    id          INTEGER PRIMARY KEY,
    lang_code   TEXT NOT NULL,
    word        TEXT NOT NULL,
    comparison  TEXT NOT NULL,
    length      INTEGER NOT NULL,
    gloss       TEXT,
    -- Madde TÜRKİDEN mi alınmış? Öyleyse yakınlık kanıtı DEĞİLDİR;
    -- tam tersinin kanıtıdır. Bkz. modül başlığındaki "Yön sorunu".
    from_turkic INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_donor_comparison ON donor_entries(comparison);
CREATE INDEX IF NOT EXISTS idx_donor_lang_len ON donor_entries(lang_code, length);
CREATE INDEX IF NOT EXISTS idx_donor_from_turkic ON donor_entries(from_turkic);

-- Anlam kısıtı için tam metin indeksi.
--
-- ⚠️ Kısıt olmadan sorgu 120.000-270.000 aday tarıyor (ölçüldü) ve şans
-- benzerliğine açık kalıyor. sabor'un yayınlanmış yöntemi zaten KAVRAM
-- kısıtlıdır; anlam indeksi hem yöntemi doğru uygular hem aramayı
-- binlerce kat küçültür.
CREATE VIRTUAL TABLE IF NOT EXISTS donor_gloss_fts USING fts5(
    gloss, content='donor_entries', content_rowid='id', tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS donor_build_info (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

#: Anlamsız/çok kısa biçimler indekse alınmaz: tek harflik bir madde her
#: sorguya yakın çıkar ve yalnız gürültü üretir.
MIN_LENGTH = 2

#: Çok uzun maddeler (deyim, çok kelimeli birim) alıntı adayı değildir.
MAX_LENGTH = 24


#: Verici maddesinin etimoloji metninde geçtiğinde "bu kelime Türkiden
#: alınmıştır" sayılan adlar. Küçük harfe indirilmiş metinde aranır.
#:
#: ⚠️ ``turkic`` tek başına yetmez: "Proto-Turkic" ve "Old Turkic" de
#: yakalanır ama "Turkish" ayrı yazılır. Liste `TURKIC_FAMILY_CODES`in
#: İngilizce adlarıdır.
TURKIC_SOURCE_MARKERS: tuple[str, ...] = (
    "turkic", "turkish", "ottoman turkish", "azerbaijani", "turkmen",
    "kazakh", "kyrgyz", "tatar", "bashkir", "uzbek", "uyghur", "uighur",
    "chuvash", "yakut", "sakha", "kumyk", "karachay", "nogai", "gagauz",
    "chagatai", "karakalpak", "crimean tatar", "karaim", "cuman", "kipchak",
)

#: Etimoloji metninde bu ifadeler varsa Türki ad **alıntı kaynağı değildir**,
#: yalnız karşılaştırma amaçlı anılmıştır.
TURKIC_MENTION_EXCUSES: tuple[str, ...] = ("compare ", "cf. ", "cognate with")


def is_from_turkic(etymology_text: str) -> bool:
    """Verici maddesi Türkiden mi alınmış?

    ⚠️ "Compare Turkish …" bir alıntı beyanı DEĞİLDİR; o cümle yalnız
    karşılaştırma yapar. Ayırmazsak Türki bir adı anan her madde elenir ve
    gerçek vericiler de kaybolur.
    """
    text = (etymology_text or "").lower()
    if not text:
        return False
    for marker in TURKIC_SOURCE_MARKERS:
        position = text.find(marker)
        while position != -1:
            prefix = text[max(0, position - 40) : position]
            if not any(excuse in prefix for excuse in TURKIC_MENTION_EXCUSES):
                return True
            position = text.find(marker, position + 1)
    return False


@dataclass(frozen=True)
class DonorEntry:
    lang_code: str
    word: str
    comparison: str
    gloss: str
    from_turkic: bool = False

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.lang_code,
            self.word,
            self.comparison,
            len(self.comparison),
            self.gloss,
            int(self.from_turkic),
        )


def _sense_tokens(sense: str) -> list[str]:
    """Kavram adını FTS'e verilebilir sözcüklere böler."""
    return [
        token
        for token in re.split(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ]+", (sense or "").lower())
        if token
    ]


# --- X4 temizliği (``ETY_DONOR_CLEAN``) ----------------------------------------
#
# Tanı (Türk dilleri arası altın, ayar bölümü; ``data/cache/work/xtr/PREREG_sense.md``):
# ``by_sense`` eşleşmesi işlev sözcükleriyle (the, for, make, one …) ve çekim
# açıklamalarıyla ("genitive plural of …") ateşleniyordu; 290/1.379 sorgu
# sırasız ``LIMIT 200``e dayanıyordu. Temizlik üç parçadır ve BİRLİKTE açılır:
#
# (a) eşleşme anahtarı yalnız İÇERİK sözcükleri (:data:`FUNCTION_WORDS` dışı);
#     sorgu yalnız işlev sözcüklerinden oluşuyorsa ("all", "much") eski
#     sözcükler kullanılır — kavramın kendisi işlev sözcüğüdür;
# (b) anlamı yalnız dilbilgisi açıklaması olan verici maddeleri (çekim/biçim
#     göndermesi: "genitive plural of X", "alternative form of X") havuzdan
#     çıkar; ":" sonrasında gerçek anlam taşıyanlar ("diminutive of нос: little
#     nose") kalır;
# (c) adaylar FTS ``bm25`` sırasıyla alınır (sırasız ilk 200 yerine en ilgili 200).

#: İngilizce işlev sözcükleri + tanıda eşleşmeyi taşıyan genel sözcükler.
FUNCTION_WORDS = frozenset("""
a about above after again against all also am an and another any are as at be because been before
being below between both but by can cause could did do does doing done down during each either else
especially etc even every few for from further get gets give given had has have having he her here
hers him his how however into its itself just kind less made make makes making many may might more
most much must not now off often once one ones only onto other others our out over own part per
perhaps person people rather same several she should since some someone something somebody somewhat
such than that the their them then there these they thing things this those though through thus too
towards under until upon usually use used very was way were what when where whether which while who
whom whose why will with within without would yet you your
""".split())

#: Dilbilgisi açıklaması sözcükleri: bir anlam parçası YALNIZ bunlardan +
#: "of <gönderme>" oluşuyorsa o parça anlam değildir.
_GRAMMAR_WORDS = frozenset("""
nominative genitive dative accusative instrumental prepositional locative vocative ablative partitive
plural singular dual inflection inflected form forms participle past present future perfective
imperfective imperative indicative subjunctive conditional infinitive gerund masculine feminine neuter
animate inanimate short long comparative superlative first second third person active passive
adverbial verbal noun adjective adverb agent alternative spelling obsolete archaic dated rare
misspelling abbreviation initialism acronym contraction clipping romanization transliteration
diminutive augmentative endearing pejorative nonstandard colloquial eye dialect variant of and or the
a an construct state definite indefinite possessive pronominal suffixed attributive predicative
dialectal female male equivalent
""".split())

_SEGMENT_SPLIT = re.compile(r"[;:]")
_REFERENCE = re.compile(r"\bof\b.*$")


def _grammar_words_only(segment: str) -> bool:
    head = _REFERENCE.sub("", segment.lower())
    words = [w for w in re.split(r"[^a-z-]+", head) if w and not w.endswith("-person")]
    return all(w in _GRAMMAR_WORDS for w in words)


def is_form_of(gloss: str) -> bool:
    """Verici anlamı YALNIZ dilbilgisi göndermesi mi? (X4 (b))

    En az bir parça "<dilbilgisi sözcükleri> of <gönderme>" biçiminde ve öbür
    bütün parçalar yalnız dilbilgisi sözcüklerinden ("nominative plural")
    oluşuyorsa evet.
    """
    segments = [s for s in _SEGMENT_SPLIT.split(gloss or "") if re.sub(r"[\W_]+", "", s)]
    if not segments:
        return False
    has_reference = any(re.search(r"\bof\b", s.lower()) and _grammar_words_only(s) for s in segments)
    return has_reference and all(_grammar_words_only(s) for s in segments)


def content_tokens(sense: str) -> list[str]:
    """Eşleşme anahtarı: 2 harften uzun İÇERİK sözcükleri (X4 (a))."""
    tokens = [t for t in _sense_tokens(sense) if len(t) > 2]
    content = [t for t in tokens if t not in FUNCTION_WORDS]
    return content or tokens


#: Temizliğin varsayılanı (bayrak verilmediğinde). Bkz. PREREG_x4.md.
CLEAN_DEFAULT = False

#: (c) sıralı sorguda süzgeç öncesi okunan en çok satır.
CLEAN_FETCH = 4000


def clean_enabled() -> bool:
    """``ETY_DONOR_CLEAN`` (1/0); verilmezse :data:`CLEAN_DEFAULT`."""
    import os

    value = os.environ.get("ETY_DONOR_CLEAN", "").strip().lower()
    if value in ("1", "on", "true", "yes"):
        return True
    if value in ("0", "off", "false", "no"):
        return False
    return CLEAN_DEFAULT


def sense_tokens_for_match(sense: str, clean: bool | None = None) -> list[str]:
    """``by_sense`` ve monget/kavram havuzlarının ortak eşleşme anahtarı."""
    if clean if clean is not None else clean_enabled():
        return content_tokens(sense)[:6]
    return [t for t in _sense_tokens(sense) if len(t) > 2][:6]


def _romanisation(record: dict[str, Any]) -> str:
    """kaikki'nin kendi çevriyazısı (``forms[].tags == ["romanization"]``).

    Wiktionary editörlerinin yazdığı çevriyazı, bizim tablomuzdan **daha
    doğrudur**; Arapça/Farsça/Yunanca/Ermenice için birincil kaynaktır.
    """
    for form in record.get("forms") or []:
        if "romanization" in (form.get("tags") or []):
            candidate = (form.get("form") or "").strip()
            if candidate:
                return candidate
    return ""


def comparison_for(record: dict[str, Any]) -> str:
    """Verici maddesinin karşılaştırma biçimi.

    ⚠️ ``to_comparison_form`` son elemede ``[^a-zçğıöşüŋŕĺ]`` dışını siler;
    Yunan ve Ermeni alfabeleri **tümden kaybolur** (``θάλασσα`` -> ``t``,
    ``գիրք`` -> ``""``). Ölçüldü: yalnız doğrudan çeviriyle Arapça 77.339
    maddenin **5'i**, Farsça ve Ermenice **sıfırı** indekse giriyordu.

    Bu yüzden doğrudan çeviri karakterlerin yarısından fazlasını kaybederse
    kaikki'nin kendi çevriyazısına düşülür.
    """
    word = (record.get("word") or "").strip()
    direct = to_comparison_form(word)
    letters = sum(1 for ch in word if ch.isalpha())
    if letters and len(direct) * 2 >= letters:
        return direct
    return to_comparison_form(_romanisation(record)) or direct


def _glosses(record: dict[str, Any]) -> str:
    out: list[str] = []
    for sense in record.get("senses") or []:
        for gloss in sense.get("glosses") or []:
            if gloss:
                out.append(str(gloss))
    return "; ".join(out[:3])


def iter_donor_entries(path: Path, lang_code: str) -> Iterator[DonorEntry]:
    """kaikki dökümünü akıtarak verici maddelerine çevirir.

    Bozuk satır **atlanır**; 900 MB'lik bir dökümde tek bozuk satır yüzünden
    indeks kurulmaması kabul edilemez.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            word = (record.get("word") or "").strip()
            if not word:
                continue
            comparison = comparison_for(record)
            if not MIN_LENGTH <= len(comparison) <= MAX_LENGTH:
                continue
            yield DonorEntry(
                lang_code,
                word,
                comparison,
                _glosses(record),
                is_from_turkic(record.get("etymology_text") or ""),
            )


def discover_donor_dumps(directory: Path | None = None) -> dict[str, Path]:
    base = Path(directory) if directory else DONOR_DIR
    if not base.exists():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(base.iterdir()):
        if path.name.endswith(".jsonl.gz"):
            found[path.name[: -len(".jsonl.gz")]] = path
        elif path.suffix == ".jsonl":
            found[path.stem] = path
    return found


class DonorIndex:
    """Verici dil sözlüklerinin aranabilir indeksi."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DEFAULT_DB

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def build(self, *, sources: dict[str, Path] | None = None, batch: int = 20000) -> dict[str, Any]:
        sources = sources if sources is not None else discover_donor_dumps()
        if not sources:
            raise FileNotFoundError(
                "verici dökümü yok: python scripts/download_lexicons.py --donors"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        counts: dict[str, int] = {}
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            for code, dump in sorted(sources.items()):
                rows: list[tuple[Any, ...]] = []
                total = 0
                for entry in iter_donor_entries(dump, code):
                    rows.append(entry.as_row())
                    if len(rows) >= batch:
                        connection.executemany(
                            "INSERT INTO donor_entries"
                            " (lang_code, word, comparison, length, gloss, from_turkic)"
                            " VALUES (?,?,?,?,?,?)",
                            rows,
                        )
                        total += len(rows)
                        rows.clear()
                if rows:
                    connection.executemany(
                        "INSERT INTO donor_entries"
                        " (lang_code, word, comparison, length, gloss, from_turkic)"
                        " VALUES (?,?,?,?,?,?)",
                        rows,
                    )
                    total += len(rows)
                counts[code] = total
                logger.info("verici %s: %d madde", code, total)
            connection.execute(
                "INSERT INTO donor_gloss_fts(rowid, gloss)"
                " SELECT id, gloss FROM donor_entries WHERE gloss IS NOT NULL AND gloss != ''"
            )
            connection.executemany(
                "INSERT OR REPLACE INTO donor_build_info (key, value) VALUES (?,?)",
                [("languages", json.dumps(counts, ensure_ascii=False))],
            )
        return {"languages": counts, "total": sum(counts.values())}

    def candidates(
        self,
        comparison: str,
        *,
        languages: list[str] | None = None,
        max_length_gap: int = 2,
        limit: int = 4000,
    ) -> list[sqlite3.Row]:
        """Uzunluğu yakın verici maddelerini döndürür.

        ⚠️ Uzunluk penceresi **kesinlik değil hız** içindir: ``n`` düzenleme
        uzaklığındaki bir biçmin uzunluk farkı da en çok ``n``dir, o yüzden
        pencere dışındakiler zaten elenirdi.
        """
        if not comparison or not self.exists:
            return []
        query = (
            "SELECT lang_code, word, comparison, gloss FROM donor_entries"
            " WHERE from_turkic = 0 AND length BETWEEN ? AND ?"
        )
        params: list[Any] = [len(comparison) - max_length_gap, len(comparison) + max_length_gap]
        if languages:
            query += f" AND lang_code IN ({','.join('?' * len(languages))})"
            params += languages
        query += " LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            return connection.execute(query, params).fetchall()

    def by_sense(
        self,
        sense: str,
        *,
        languages: list[str] | None = None,
        limit: int = 400,
        clean: bool | None = None,
    ) -> list[sqlite3.Row]:
        """Anlamı sorguyla örtüşen verici maddeleri.

        sabor'un (Miller & List 2023) yayınlanmış kurulumu **kavram
        kısıtlıdır**: aday yalnız aynı kavramın verici karşılığıdır. Kısıtsız
        arama şans benzerliğine açıktır — yüz binlerce maddelik bir sözlükte
        kısa bir biçme benzeyen bir şey her zaman bulunur.

        ⚠️ Kısıtın bedeli de ölçülmüştür: sabor'da kaçan alıntıların **%45'i**
        tam bu kısıttan kaynaklanıyor (verici maddenin anlamı sözlükte başka
        yazılmıştır). Bu yüzden :meth:`candidates` kısıtsız yol olarak durur.
        """
        clean = clean_enabled() if clean is None else clean
        tokens = sense_tokens_for_match(sense, clean)
        if not tokens or not self.exists:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        query = (
            "SELECT e.lang_code, e.word, e.comparison, e.gloss"
            " FROM donor_gloss_fts f JOIN donor_entries e ON e.id = f.rowid"
            " WHERE donor_gloss_fts MATCH ? AND e.from_turkic = 0"
        )
        params: list[Any] = [match]
        if languages:
            query += f" AND e.lang_code IN ({','.join('?' * len(languages))})"
            params += languages
        if clean:
            query += " ORDER BY f.rank"
        query += " LIMIT ?"
        params.append(CLEAN_FETCH if clean else limit)
        with self._connect() as connection:
            try:
                rows = connection.execute(query, params).fetchall()
                if clean:
                    rows = [r for r in rows if not is_form_of(r["gloss"] or "")][:limit]
                return rows
            except sqlite3.OperationalError:
                # FTS sorgu sözdizimi hatası (tırnaklı garip kavram adı):
                # sessizce boş dön, ölçüm çökmesin.
                logger.debug("verici anlam sorgusu ayrıştırılamadı: %r", sense)
                return []

    def stats(self) -> dict[str, Any]:
        if not self.exists:
            return {"exists": False, "path": str(self.path)}
        with self._connect() as connection:
            info = dict(
                connection.execute("SELECT key, value FROM donor_build_info").fetchall()
            )
            total = connection.execute("SELECT COUNT(*) FROM donor_entries").fetchone()[0]
            reverse = connection.execute(
                "SELECT COUNT(*) FROM donor_entries WHERE from_turkic = 1"
            ).fetchone()[0]
        return {
            **info,
            "exists": True,
            "total_entries": total,
            "from_turkic_excluded": reverse,
            "path": str(self.path),
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Verici dil sözlüğü indeksi")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    index = DonorIndex()
    if args.build:
        result = index.build()
        print(f"kuruldu: {result['total']:,} madde · {result['languages']}")
    print(index.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
