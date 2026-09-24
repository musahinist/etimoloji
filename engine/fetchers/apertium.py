"""
Apertium iki dilli sözlüklerinden Türk dili tanıkları (yerel, ``make apertium``).

Türkçe sorgunun sözlükteki karşılıkları tanık adayıdır. ⚠️ Karşılık ÇEVİRİDİR,
akrabalık değil: Kırgızca 'pencere' karşılığı *терезе*dir. Aday bu yüzden
sorguyla ya da o dil için ses denklikleriyle tahmin edilen biçimle yazılışça
benzer olmalıdır (``NorthEuraLexFetcher`` ile aynı ölçüt, eşik 0,50).

Fiiller Apertium'da kök olarak durur (`uç`, ``vblex``): `uçmak` sorgusu
yalnız FİİL kayıtlarında `uç`a bakar; `kırkmak` -> ad `kırk` 'kırk (sayı)'
hatası böylece oluşmaz.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from engine.config import PROJECT_ROOT
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher, detect_script
from engine.fetchers.northeuralex import _predicted_forms, _similarity
from engine.nlp.cognate_clustering import COGNATE_THRESHOLD
from engine.utils.orthography import to_comparison_form

DATA_DIR = PROJECT_ROOT / "data" / "apertium"

_ENTRY = re.compile(r"<e\b[^>]*>\s*<p>\s*<l>(.*?)</l>\s*<r>(.*?)</r>\s*</p>\s*</e>", re.S)
_TAG = re.compile(r'<s n="([^"]+)"\s*/>')
_VERB_TAGS = ("vblex", "vbmod", "v", "tv", "iv")

#: Dile özgü sıkı eşik. Çuvaşça sözlük tüm söz varlığını kapsar ve akraba
#: oranı temel listeden çok düşüktür; kısa kelimede tahmin edilen biçimle 0,50
#: kolay tutuyordu. Elle bakıldı (~40 aday): 0,50'de 9 Çuvaşça tanığın 6'sı
#: sahte (kazma ~ кайла, gevşek ~ кӑпка, yankı ~ ян); 0,67'de sahteler büyük
#: ölçüde gider, köpük ~ кӑпӑк (0,6) ve gece ~ каҫ (0,5) gibi doğrular da
#: kaybolur. ⚠️ Küçük örneklemle seçildi.
LANGUAGE_THRESHOLDS = {"cv": 0.67}


def _side(raw: str) -> tuple[str, frozenset[str]]:
    tags = frozenset(_TAG.findall(raw))
    text = re.sub(r"<b\s*/>", " ", raw)
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text, tags


@lru_cache(maxsize=1)
def _table() -> dict[tuple[str, bool], list[tuple[str, str]]]:
    """(Türkçe sözcükbirim, fiil mi) -> [(dil, biçim)]."""
    provenance = DATA_DIR / "_provenance.json"
    if not provenance.exists():
        return {}
    table: dict[tuple[str, bool], list[tuple[str, str]]] = {}
    for pair, meta in json.loads(provenance.read_text(encoding="utf-8")).get("files", {}).items():
        path = DATA_DIR / f"{pair}.dix"
        if not path.exists():
            continue
        left, right = meta["left"], meta["right"]
        other = right if left == "tr" else left
        for match in _ENTRY.finditer(path.read_text(encoding="utf-8")):
            (l_text, l_tags), (r_text, r_tags) = _side(match.group(1)), _side(match.group(2))
            tr_text, tr_tags, o_text = (l_text, l_tags, r_text) if left == "tr" else (r_text, r_tags, l_text)
            if not tr_text or not o_text or " " in o_text:
                continue
            is_verb = bool(tr_tags & set(_VERB_TAGS))
            table.setdefault((tr_text.lower(), is_verb), []).append((other, o_text))
    return table


class ApertiumFetcher(BaseFetcher):
    #: Yerel veri.
    is_seed_source = True
    #: Kök varyantı almaz (fiil kökü kendi içinde, türüne bakarak çıkarılır).
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "Apertium iki dilli sözlükleri (yerel)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        query = (word or "").strip().lower()
        table = _table()
        if not query or not table:
            return result
        stem = re.sub(r"m[ae]k$", "", query)
        candidates = table.get((stem, True), []) if stem != query else table.get((query, False), [])
        own = to_comparison_form(query if stem == query else stem)
        predicted = _predicted_forms(query)
        seen: set[tuple[str, str]] = set()
        for code, form in candidates:
            # Büyük harfli karşılık özel addır (Deniz, Yol, Qor).
            if code not in TURKIC_LANGUAGES_MAP or (code, form) in seen or form[:1].isupper():
                continue
            comparison = to_comparison_form(form)
            expected = predicted.get(code, "")
            expected_stem = re.sub(r"(m[ae]k|m[ae]q|moq|ma[ʁғ]|mäk)$", "", expected) if stem != query else expected
            score = max(_similarity(own, comparison),
                        _similarity(expected_stem, comparison) if expected_stem else 0.0)
            if not comparison or score < LANGUAGE_THRESHOLDS.get(code, COGNATE_THRESHOLD):
                continue  # çeviri karşılığı, akraba değil
            seen.add((code, form))
            entry = self.make_entry(code, form, f"Türkçe: {stem if stem != query else query}",
                                    script=detect_script(form))
            entry["comparison"] = comparison
            result["turkic_languages"].append(entry)
        return result
