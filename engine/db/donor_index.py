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
        tokens = [t for t in _sense_tokens(sense) if len(t) > 2]
        if not tokens or not self.exists:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens[:6])
        query = (
            "SELECT e.lang_code, e.word, e.comparison, e.gloss"
            " FROM donor_gloss_fts f JOIN donor_entries e ON e.id = f.rowid"
            " WHERE donor_gloss_fts MATCH ? AND e.from_turkic = 0"
        )
        params: list[Any] = [match]
        if languages:
            query += f" AND e.lang_code IN ({','.join('?' * len(languages))})"
            params += languages
        query += " LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            try:
                return connection.execute(query, params).fetchall()
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
