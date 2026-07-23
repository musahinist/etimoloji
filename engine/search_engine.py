import datetime
from typing import Dict, Any, List, Optional

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP
from engine.fetchers.academic_turkology import AcademicTurkologyFetcher
from engine.fetchers.tietze_altaica import TietzeAltaicaFetcher
from engine.fetchers.historical_modern import HistoricalModernLexiconFetcher
from engine.fetchers.etimoloji_turkce import EtimolojiTurkceFetcher
from engine.fetchers.wiktionary import WiktionaryFetcher
from engine.fetchers.starling import StarlingFetcher
from engine.fetchers.tdk_nisanyan import TdkFetcher, NisanyanFetcher
from engine.fetchers.tdk_historical import TdkTaramaFetcher, TdkDerlemeFetcher
from engine.fetchers.multilang_wiktionary import MultiLangWiktionaryFetcher
from engine.db.database import DatabaseManager

class SearchEngine:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.fetchers: List[BaseFetcher] = [
            AcademicTurkologyFetcher(),
            HistoricalModernLexiconFetcher(),
            TietzeAltaicaFetcher(),
            EtimolojiTurkceFetcher(),
            StarlingFetcher(),
            NisanyanFetcher(),
            TdkFetcher(),
            TdkTaramaFetcher(),
            TdkDerlemeFetcher(),
            WiktionaryFetcher(),
            MultiLangWiktionaryFetcher()
        ]

    def search(self, word: str, save_to_db: bool = True) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        if not word_clean:
            raise ValueError("Arama için geçerli bir kelime giriniz.")

        # 1. Önce veritabanında arat
        existing_finding = self.db.get_finding(word_clean)
        if existing_finding:
            existing_finding["from_cache"] = True
            return existing_finding

        # 2. Veri toplayıcıları çalıştır
        proto_root = ""
        root_meaning = ""
        reconstruction_notes = ""
        sources = []
        turkic_entries_map = {} # lang_code -> entry dict

        for fetcher in self.fetchers:
            try:
                res = fetcher.fetch(word_clean)
                root_info = res.get("root", {})
                if not proto_root and root_info.get("proto_turkic"):
                    proto_root = root_info.get("proto_turkic")
                if not root_meaning and root_info.get("meaning"):
                    root_meaning = root_info.get("meaning")
                if not reconstruction_notes and root_info.get("reconstruction_notes"):
                    reconstruction_notes = root_info.get("reconstruction_notes")

                for entry in res.get("turkic_languages", []):
                    code = entry.get("lang_code")
                    key = f"{code}_{entry.get('word')}"
                    if key not in turkic_entries_map:
                        turkic_entries_map[key] = entry
                    elif turkic_entries_map[key].get("meaning") in ["", f"Online {TURKIC_LANGUAGES_MAP.get(code, '')} Sözlük kaydı"]:
                        if entry.get("meaning") and not entry.get("meaning").startswith("Online"):
                            turkic_entries_map[key] = entry

                if res.get("turkic_languages") or root_info.get("proto_turkic"):
                    sources.append(fetcher.source_name)
            except Exception as e:
                pass

        # Türki diller listesini dili Türkçe isimlerine göre sıralayalım
        sorted_entries = sorted(
            list(turkic_entries_map.values()),
            key=lambda x: (0 if x["lang_code"] == "otk" else 1, x["lang_name"])
        )

        finding = {
            "query_word": word_clean,
            "root": {
                "proto_turkic": proto_root or f"*{word_clean}",
                "meaning": root_meaning or word_clean,
                "reconstruction_notes": reconstruction_notes or f"Proto-Turkic reconstruction for {word_clean}"
            },
            "turkic_languages": sorted_entries,
            "sources": sources or ["Local Turkic Etymology Engine"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "from_cache": False
        }

        # 3. Veritabanına kaydet
        if save_to_db and (sorted_entries or proto_root):
            self.db.save_finding(finding)

        return finding
