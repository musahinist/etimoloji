import re
import datetime
import concurrent.futures
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
from engine.utils.morphology import analyze_morphology
from engine.utils.transliteration import transliterate_to_latin
from engine.db.database import DatabaseManager

# İngilizce terimler için otomatik Türkçe semantik çeviri dizini
MEANING_TRANSLATIONS = {
    "beautiful": "güzel, alımlı, hoş",
    "water": "su, sıvı, akarsu",
    "water / liquid": "su, sıvı",
    "sea": "deniz, büyük göl",
    "sea / ocean": "deniz, okyanus",
    "eye": "göz, görme organı",
    "eye / sight": "göz, görüş, bakış",
    "hand": "el, tutma organı",
    "hand / forearm": "el, kol",
    "head": "baş, kafa, lider",
    "blood": "kan",
    "moon": "ay, uydu",
    "sun": "güneş, gün",
    "sky": "gök, gökyüzü",
    "god": "tanrı, ilah",
    "deity": "tanrı, ilahi varlık",
    "sign": "işaret, alamet",
    "mark": "damga, işaret",
    "alert": "uyanık, tetikte, keskin",
    "trigger": "tetik mekanizması",
    "realm": "ülke, il, memleket",
    "country": "ülke, memleket",
    "blessing": "kut, bereket, tanrı lütfu",
    "soul": "can, ruh, kut",
    "word": "kelime, söz"
}

def translate_meaning(meaning: str) -> str:
    m = meaning.strip()
    if not m or m.startswith("Online"):
        return m
    
    m_clean = re.sub(r'\{\{.*?\}\}', '', m).strip()
    m_clean = re.sub(r'\[\[(.*?)\]\]', r'\1', m_clean).strip()
    
    for eng, tr in MEANING_TRANSLATIONS.items():
        if eng.lower() in m_clean.lower():
            return tr
            
    return m_clean or meaning

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

    def _execute_fetcher(self, fetcher: BaseFetcher, word: str) -> tuple[BaseFetcher, Optional[Dict[str, Any]]]:
        try:
            res = fetcher.fetch(word)
            return fetcher, res
        except Exception:
            return fetcher, None

    def search(self, word: str, save_to_db: bool = True) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        if not word_clean:
            raise ValueError("Arama için geçerli bir kelime giriniz.")

        # Morfolojik kök ve ek analizi yap
        stem, suffixes = analyze_morphology(word_clean)

        # 1. Önce veritabanında arat
        existing_finding = self.db.get_finding(word_clean)
        if existing_finding:
            existing_finding["from_cache"] = True
            return existing_finding

        # 2. Tüm veri toplayıcıları PARALEL çalıştır (hem kelime hem de kökü için)
        proto_root = ""
        root_meaning = ""
        reconstruction_notes = ""
        sources = []
        turkic_entries_map = {} # lang_code -> entry dict

        target_words = [word_clean]
        if stem != word_clean:
            target_words.append(stem)

        for target in target_words:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.fetchers)) as executor:
                futures = [executor.submit(self._execute_fetcher, fetcher, target) for fetcher in self.fetchers]
                for future in concurrent.futures.as_completed(futures):
                    fetcher, res = future.result()
                    if not res:
                        continue

                    root_info = res.get("root", {})
                    if not proto_root and root_info.get("proto_turkic"):
                        proto_root = root_info.get("proto_turkic")
                    if not root_meaning and root_info.get("meaning"):
                        root_meaning = root_info.get("meaning")
                    if not reconstruction_notes and root_info.get("reconstruction_notes"):
                        reconstruction_notes = root_info.get("reconstruction_notes")

                    for entry in res.get("turkic_languages", []):
                        code = entry.get("lang_code")
                        entry_word = entry.get('word', '')
                        
                        # Latin Transkripsiyon Ekle (Kiril veya Arap Alfabesindeyse)
                        latin_trans = transliterate_to_latin(entry_word)
                        if latin_trans != entry_word and not "(" in entry_word:
                            entry["word"] = f"{entry_word} ({latin_trans})"

                        key = f"{code}_{entry.get('word')}"
                        
                        # Anlamı Türkçe semantik çeviriden geçir
                        raw_m = entry.get("meaning", "")
                        entry["meaning"] = translate_meaning(raw_m)

                        if key not in turkic_entries_map:
                            turkic_entries_map[key] = entry
                        elif turkic_entries_map[key].get("meaning") in ["", f"Online {TURKIC_LANGUAGES_MAP.get(code, '')} Sözlük kaydı"]:
                            if entry.get("meaning") and not entry.get("meaning").startswith("Online"):
                                turkic_entries_map[key] = entry

                    if res.get("turkic_languages") or root_info.get("proto_turkic"):
                        sources.append(fetcher.source_name)

        # 3. Kök Anlamını Çevir
        if root_meaning:
            root_meaning = translate_meaning(root_meaning)

        # Türki diller listesini dili Türkçe isimlerine göre sıralayalım
        sorted_entries = sorted(
            list(turkic_entries_map.values()),
            key=lambda x: (0 if x["lang_code"] == "otk" else 1, x["lang_name"])
        )

        morphology_info = f"Kök: {stem} + Ekler: {', '.join(suffixes)}" if suffixes else "Yalın Kök"

        finding = {
            "query_word": word_clean,
            "morphology": morphology_info,
            "root": {
                "proto_turkic": proto_root or f"*{stem}",
                "meaning": root_meaning or word_clean,
                "reconstruction_notes": f"[{morphology_info}] {reconstruction_notes or ('Proto-Turkic reconstruction for ' + word_clean)}"
            },
            "turkic_languages": sorted_entries,
            "sources": sorted(list(set(sources))),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "from_cache": False
        }

        # 4. Veritabanına kaydet
        if save_to_db and (sorted_entries or proto_root):
            self.db.save_finding(finding)

        return finding
