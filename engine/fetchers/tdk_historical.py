import json
import re
import urllib.parse
import urllib.request
from typing import Any

from engine import config
from engine.fetchers.base import BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.network import fetch as http_get

logger = get_logger(__name__)

#: TDK'nın `sehir` alanı HTML taşır:
#: ``Çiftlik, Başmakçı *Dinar -<b>Afyon</b><br>*Eşme -<b>Uşak</b>``
#: Temizlenmezse etiketler kullanıcıya giden DİL ADINA sızıyor — ölçüldü:
#: `herkil` çıktısında ham ``<b>``/``<br>`` görünüyordu.
_HTML_TAG = re.compile(r"<[^>]+>")


def _clean_region(value: str) -> str:
    """Yöre alanındaki HTML etiketlerini ayıklar."""
    return " ".join(_HTML_TAG.sub(" ", value or "").split()).strip(" ,-")


class TdkTaramaFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK Tarama Sözlüğü (Tarihi Türkçe Metinler 13.-19. yy)"

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # ⚠️ Yeni arayüz (`sozluk.gov.tr`) JSON DEĞİL, SPA HTML kabuğu
        # döndürüyor; eski uç ayakta ve JSON veriyor. Ölçüldü.
        url = f"https://eski.sozluk.gov.tr/tarama?ara={urllib.parse.quote(word_clean)}"
        try:
            _body = http_get(url, timeout=config.HTTP_TIMEOUT_MEDIUM)
            # TDK bu ucu kaldırdı: HTTP 200 dönüyor ama gövde JSON değil, HTML
            # sayfası. `json.loads` her aramada JSONDecodeError yığın izi
            # basıyordu. Uç JSON dönmüyorsa kaynak sessizce atlanır; TDK geri
            # getirirse kod değişmeden yeniden çalışır.
            if _body is not None and _body.lstrip()[:1] in ("[", "{"):
                data = json.loads(_body)
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    tarama_list = item.get("tarama", [])
                    if tarama_list:
                        hist_word = tarama_list[0].get("kelime", word_clean)
                        meaning = tarama_list[0].get("anlam", "")
                        result["turkic_languages"].append({
                            "lang_code": "otk",
                            "lang_name": "Tarihi Türkçe / Osmanlıca (13.-19. yy)",
                            "word": hist_word,
                            "meaning": meaning,
                            "script": "Latin"
                        })
                        result["root"]["reconstruction_notes"] = f"TDK Tarama Sözlüğü (Tarihi Türkçe): {hist_word} - {meaning}"
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name if hasattr(self, "source_name") else __name__, exc_info=True)
        return result


class TdkDerlemeFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK Derleme Sözlüğü (Türk Ağızları ve Diyalektleri)"

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        # ⚠️ Yeni arayüz (`sozluk.gov.tr/derleme`) SPA HTML kabuğu döndürüyor;
        # eski uç JSON veriyor ve `scripts/harvest_derleme.py` zaten onu
        # kullanıyordu — canlı fetcher geride kalmıştı. Ölçüldü: `herkil`
        # eski uçta iki kayıt ("Erzak sandığı", "Erzak ambarı" — İstanbul,
        # Zonguldak, Kastamonu, Kocaeli, Bolu) verirken fetcher boş dönüyordu.
        url = f"https://eski.sozluk.gov.tr/derleme?ara={urllib.parse.quote(word_clean)}"
        try:
            _body = http_get(url, timeout=config.HTTP_TIMEOUT_MEDIUM)
            # TDK bu ucu kaldırdı: HTTP 200 dönüyor ama gövde JSON değil, HTML
            # sayfası. `json.loads` her aramada JSONDecodeError yığın izi
            # basıyordu. Uç JSON dönmüyorsa kaynak sessizce atlanır; TDK geri
            # getirirse kod değişmeden yeniden çalışır.
            if _body is not None and _body.lstrip()[:1] in ("[", "{"):
                data = json.loads(_body)
                if isinstance(data, list) and len(data) > 0:
                    for item in data[:3]:
                        m_word = item.get("madde", word_clean)
                        meaning = item.get("anlam", "")
                        city = _clean_region(item.get("sehir", ""))
                        if meaning:
                            result["turkic_languages"].append({
                                "lang_code": "tr",
                                "lang_name": f"Türk Ağızları ({city})" if city else "Türk Ağızları",
                                "word": m_word,
                                "meaning": meaning,
                                "script": "Latin",
                                # Ağız kaydı ölçünlü dil kaydıyla aynı `tr` kodunu
                                # taşıyor (Oğuz kolu sayımı için doğru); birleştirmede
                                # TDK sözlük maddesinin yerini kapmasın diye işaretli.
                                "dialect": True,
                            })
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name if hasattr(self, "source_name") else __name__, exc_info=True)
        return result
