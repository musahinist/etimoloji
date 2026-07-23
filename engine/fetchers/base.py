from abc import ABC, abstractmethod
from typing import Dict, Any, List

TURKIC_LANGUAGES_MAP = {
    "otk": "Eski Türkçe",
    "tr": "Türkiye Türkçesi",
    "az": "Azerbaycan Türkçesi",
    "kk": "Kazakça",
    "uz": "Özbekçe",
    "tk": "Türkmençe",
    "ky": "Kırgızca",
    "tt": "Tatarca",
    "ug": "Uygurca",
    "cv": "Çuvaşça",
    "sah": "Saha / Yakutça",
    "ba": "Başkurtça",
    "gag": "Gagavuzca",
    "krc": "Karaçay-Balkarca",
    "tyv": "Tuva Türkçesi",
    "alt": "Altay Türkçesi",
    "khk": "Hakasça",
    "chg": "Çağatayca"
}

class BaseFetcher(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Fetcher kaynağının adı (örn. Wiktionary, Starling)."""
        pass

    @abstractmethod
    def fetch(self, word: str) -> Dict[str, Any]:
        """
        Verilen kelime için Türki dillerdeki karşılıkları ve anlamları toplar.
        Dönüş formatı:
        {
            "root": {"proto_turkic": str, "meaning": str, "reconstruction_notes": str},
            "turkic_languages": [
                {"lang_code": str, "lang_name": str, "word": str, "meaning": str, "script": str}
            ]
        }
        """
        pass
