import json
import re
import urllib.parse
import urllib.request
from typing import Any

from engine import config
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.network import fetch as http_get
from engine.utils.seed import load_seed_entries, seed_source_label

logger = get_logger(__name__)

#: Tohum (seed) veri. Kod içinde değil, data/seed/lexicon/clauson_estja.json dosyasında tutulur.
SEED_PATH = "lexicon/clauson_estja.json"
ACADEMIC_TURKOLOGY_LEXICON = load_seed_entries(SEED_PATH)

#: Tohum dosyası okunuşu biçimin yanına yazıyor: "құт (qut)", "𐰸𐰆 (kut)".
_READING = re.compile(r"^(?P<form>[^()]+?)\s*\((?P<reading>[^()]+)\)\s*$")


def _split_reading(raw: str) -> tuple[str, str]:
    """``"құт (qut)"`` -> ``("құт", "qut")``.

    ⚠️ Parantezli okunuş biçimin parçası sayılıyordu: tanık adı "кут (kut)"
    basılıyor, karşılaştırma biçmi parantezle bozuluyordu (105 kelimelik
    denetim: `kutlu`da 5 tanık).
    """
    match = _READING.match(raw or "")
    if not match:
        return raw, ""
    return match.group("form").strip(), match.group("reading").strip()


class AcademicTurkologyFetcher(BaseFetcher):
    #: Bu kaynak yerel tohum veriden beslenir, canlı bir servis DEĞİLDİR.
    is_seed_source = True

    @property
    def source_name(self) -> str:
        return seed_source_label("Akademik Türkoloji Veri Bankası (Clauson EDPT & Sevortjan ЭСТЯ)", SEED_PATH)

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        if word_clean in ACADEMIC_TURKOLOGY_LEXICON:
            entry = ACADEMIC_TURKOLOGY_LEXICON[word_clean]
            result["root"]["proto_turkic"] = entry["proto_turkic"]
            result["root"]["meaning"] = entry["meaning"]
            notes = f"{entry.get('clauson_edpt', '')} | {entry.get('estja', '')}"
            result["root"]["reconstruction_notes"] = notes

            for lang_code, cognate in entry.get("cognates", {}).items():
                if lang_code in TURKIC_LANGUAGES_MAP:
                    display_word, reading = _split_reading(cognate["word"])
                    record = {
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": cognate["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    }
                    if reading:
                        record["latin_transliteration"] = reading
                    result["turkic_languages"].append(record)

        url = f"https://sozluk.gov.tr/terim?ara={urllib.parse.quote(word_clean)}"
        try:
            _body = http_get(url, timeout=config.HTTP_TIMEOUT_MEDIUM)
            # TDK bu ucu kaldırdı: HTTP 200 dönüyor ama gövde JSON değil, HTML
            # sayfası. `json.loads` her aramada JSONDecodeError yığın izi
            # basıyordu. Uç JSON dönmüyorsa kaynak sessizce atlanır; TDK geri
            # getirirse kod değişmeden yeniden çalışır.
            if _body is not None and _body.lstrip()[:1] in ("[", "{"):
                data = json.loads(_body)
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:2]:
                        soz = item.get("sozcuk", word_clean)
                        meaning = item.get("anlam", "")
                        sozluk_ad = item.get("sozluk_ad", "TDK Terim Sözlüğü")
                        if meaning:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"TDK Akademik Terim ({sozluk_ad})",
                                "word": soz,
                                "meaning": meaning,
                                "script": "Latin"
                            })
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name if hasattr(self, "source_name") else __name__, exc_info=True)
        return result
