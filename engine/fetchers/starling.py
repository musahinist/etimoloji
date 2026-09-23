import re
from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.utils.seed import load_seed_entries, seed_source_label

# Dahili Yıldız/Starling Proto-Türkçe Etimoloji Sözlük Dizini (Core Turkic Lexicon Index)
#: Tohum (seed) veri. Kod içinde değil, data/seed/lexicon/starling.json dosyasında tutulur.
#: ⚠️ Yalnız gerçek Starling veritabanı (`make starling`) indirilmemişse kullanılır.
SEED_PATH = "lexicon/starling.json"
STARLING_OFFLINE_LEXICON = load_seed_entries(SEED_PATH)

#: Kök alanındaki ilk yıldızlı biçim: "*göŕ ( = *gör-s) / *gör-" -> "*göŕ".
_FIRST_PROTO = re.compile(r"\*[^\s,/();]+")


class StarlingFetcher(BaseFetcher):
    def __init__(self, *, use_database: bool | None = None) -> None:
        """:param use_database: ``None`` = veritabanı indirilmişse onu kullan;
        ``False`` = tohum veri (testler makinedeki veriden bağımsız kalsın)."""
        from engine.db.starling import load_turcet

        #: Gerçek veritabanı varsa tohum veriye hiç bakılmaz.
        self.has_database = bool(load_turcet()) if use_database is None else use_database
        #: Tohum veri yerel ve elle yazılmıştır, canlı bir servis DEĞİLDİR.
        self.is_seed_source = not self.has_database

    @property
    def source_name(self) -> str:
        if self.has_database:
            return "Starling Türk Etimoloji Veritabanı (Dybo & Starostin 2005, yerel)"
        return seed_source_label("Starling Etymological Database", SEED_PATH)

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        if self.has_database:
            return self._fetch_database(word_clean)
        return self._fetch_seed(word_clean)

    def _fetch_database(self, word: str) -> dict[str, Any]:
        """Kök, anlam ve tarihli ilk tanıklama; TANIK ÜRETMEZ.

        Starling tanıkları kendi transkripsiyonundadır (`qɨrq`, `mončuq`);
        sözlük indeksindeki Kiril/Latin biçimlerle yan yana tanık sayılırsa
        aynı dil iki kez görünür. Tanık olarak kullanımı ayrıca ölçülmeli.
        """
        from engine.db.starling import lookup_turkish

        result = self.empty_result()
        matches = lookup_turkish(word)
        if not matches:
            return result
        etym = matches[0]
        proto = _FIRST_PROTO.search(etym.proto)
        result["root"]["proto_turkic"] = proto.group(0) if proto else etym.proto
        result["root"]["starling_proto"] = etym.proto
        result["root"]["meaning"] = etym.meaning
        old_turkic = etym.reflexes.get("ATU", "")
        result["root"]["reconstruction_notes"] = (
            f"Starling #{etym.number}: {etym.proto} “{etym.meaning}”"
            + (f"; Eski Türkçe: {old_turkic}" if old_turkic else "")
            + (f"; kaynakça: {etym.reference}" if etym.reference else "")
        )
        dated = etym.earliest_dated_source()
        if dated:
            year, tag = dated
            result["first_attestation"] = {
                "form": old_turkic.split("(")[0].strip() or etym.reflexes.get("KRH", ""),
                "meaning": etym.meaning,
                "source": f"{tag} (Starling #{etym.number})",
                "year": year,
            }
        return result

    def _fetch_seed(self, word_clean: str) -> dict[str, Any]:
        result = {
            "root": {
                "proto_turkic": "",
                "meaning": "",
                "reconstruction_notes": ""
            },
            "turkic_languages": []
        }

        if word_clean in STARLING_OFFLINE_LEXICON:
            entry = STARLING_OFFLINE_LEXICON[word_clean]
            result["root"]["proto_turkic"] = entry["proto_turkic"]
            result["root"]["meaning"] = entry["meaning"]
            result["root"]["reconstruction_notes"] = f"Starling / Tower of Babel Proto-Turkic reconstruction {entry['proto_turkic']}"

            for lang_code, cognate in entry["cognates"].items():
                if lang_code in TURKIC_LANGUAGES_MAP:
                    display_word = cognate["word"]
                    result["turkic_languages"].append({
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": cognate["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    })

        return result
