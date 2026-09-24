"""
TDV İslâm Ansiklopedisi (İSAM) toplayıcısı.

İki yolu vardır:

* **Yerel tohum veri** (her zaman): ``data/seed/lexicon/isam.json`` içindeki
  elle derlenmiş birkaç madde. Ağ kullanmaz.
* **Canlı site** (yalnız ``ETY_LIVE_ISAM=1``): islamansiklopedisi.org.tr'de
  sorgu kelimesinin madde sayfası. Varsayılan KAPALI.

⚠️ Eskiden kaynak adı "yerel tohum veri" diyordu ama her aramada siteye 4
varyant için ayrı istek atıyordu. Ölçüldü (`kitap`, sıcak arama): 6,3 s'nin
6,1 s'si bu kaynaktı; eşleştirme de gürültülüydü (içinde "kök" ya da "Hun"
geçen herhangi bir paragraf "etimoloji" sayılıyordu). Artık canlı yol açıkça
istenmedikçe çalışmaz; açıksa yalnız madde başı sorgulanır, yanıt kalıcı
HTTP önbelleğine yazılır ve kaynak adı canlı olduğunu söyler.
"""
import re
import urllib.parse
from typing import Any

from engine import config
from engine.fetchers.base import BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.network import fetch as http_get
from engine.utils.seed import load_seed_entries, seed_source_label
from engine.utils.text import strip_html

logger = get_logger(__name__)

#: Tohum (seed) veri. Kod içinde değil, data/seed/lexicon/isam.json dosyasında tutulur.
SEED_PATH = "lexicon/isam.json"
ISAM_ENCYCLOPEDIA_INDEX = load_seed_entries(SEED_PATH)
BASE_NAME = "TDV İslam Ansiklopedisi (İSAM Tarih ve Etimoloji Külliyatı)"
LIVE_HOST = "islamansiklopedisi.org.tr"

#: Paragrafın etimoloji bilgisi taşıdığını gösteren işaretler. Tek başına
#: "kök" ya da "Hun" yetmez: bunlar sıradan tarih metinlerinde de geçer.
_ETYMOLOGY_MARKERS = ("etimoloji", "Eski Türkçe", "kökeni", "kelimesi", "sözlükte")


def _etymology_paragraph(html: str, word: str) -> str:
    """Sorgu kelimesini ADIYLA anan ve etimoloji işareti taşıyan ilk paragraf."""
    word_re = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    for p in re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)[:3]:
        clean = strip_html(p).strip()
        if len(clean) > 30 and word_re.search(clean) and any(m in clean for m in _ETYMOLOGY_MARKERS):
            return clean
    return ""


class IsamAnsiklopediFetcher(BaseFetcher):
    def __init__(self, *, live: bool | None = None) -> None:
        """:param live: ``None`` = ``config.LIVE_ISAM`` (``ETY_LIVE_ISAM``)."""
        self.live = config.LIVE_ISAM if live is None else live
        #: Canlı yol kapalıyken kaynak yalnız yerel tohum veridir.
        self.is_seed_source = not self.live
        #: Canlı yol açıksa yalnız madde başı sorgulanır (varyant başına istek yok).
        #: Kapalıyken tohum sözlüğüne varyantlarla bakmak bedava.
        self.exact_query_only = self.live

    @property
    def source_name(self) -> str:
        if self.live:
            return f"{BASE_NAME} [canlı: {LIVE_HOST} + yerel tohum veri]"
        return seed_source_label(BASE_NAME, SEED_PATH)

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = self.empty_result()

        # 1. Yerel tohum veri
        if word_clean in ISAM_ENCYCLOPEDIA_INDEX:
            text = ISAM_ENCYCLOPEDIA_INDEX[word_clean]
            result["root"]["reconstruction_notes"] = text
            result["turkic_languages"].append({
                "lang_code": "otk",
                "lang_name": "TDV İSAM Ansiklopedisi (M.Ö. Hun & Orhun Kayıtları)",
                "word": word_clean,
                "meaning": text,
                "script": "Latin",
                "origin": "seed",
            })

        if not self.live:
            return result

        # 2. Canlı İSAM madde sayfası (yalnız ETY_LIVE_ISAM=1)
        url = f"https://{LIVE_HOST}/{urllib.parse.quote(word_clean)}"
        try:
            html = http_get(url, timeout=config.HTTP_TIMEOUT_MEDIUM)
            clean = _etymology_paragraph(html, word_clean) if html else ""
            if clean:
                result["root"]["reconstruction_notes"] = f"TDV İSAM: {clean[:200]}..."
                result["turkic_languages"].append({
                    "lang_code": "otk",
                    "lang_name": "TDV İSAM Ansiklopedisi Metin Analizi",
                    "word": word_clean,
                    "meaning": clean[:150],
                    "script": "Latin",
                    "origin": "live",
                })
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
        return result
