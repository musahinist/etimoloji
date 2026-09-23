"""
NorthEuraLex (Dellert ve ark. 2020) — kavram hizalı çağdaş Türk dili tanıkları.

Sorgu kelimesi NorthEuraLex'in Türkçe biçimlerinden biriyse, AYNI kavramın
öbür Türk dillerindeki biçimleri döner. Çuvaşçayı ve Sahacayı Swadesh
listesinin ötesine (~1.000 kavram) taşır.

⚠️ Aynı kavram akraba demek DEĞİLDİR: 'pencere' kavramının Kazakça biçimi
*терезе*dir ve *pencere* ile akraba değildir. Bu yüzden aday, sorguya
yazılışça da benzemelidir: ``cognate_clustering``'in ölçülmüş eşiği
(normalize düzenleme benzerliği ≥ 0,50) kullanılır.

⚠️ Dil kodu çakışması: NorthEuraLex ``khk`` = Halha MOĞOLCASI; motorda
``khk`` = Hakasça. Eşleme açıkça yazılır, Moğol dilleri tanık yapılmaz.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from typing import Any

from engine.config import CLDF_DIR
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher, detect_script
from engine.nlp.cognate_clustering import COGNATE_THRESHOLD
from engine.nlp.donor_lexicon import levenshtein
from engine.utils.orthography import to_comparison_form

DATA_DIR = CLDF_DIR / "northeuralex"

#: NorthEuraLex dil kimliği -> motor dil kodu (yalnız Türk dilleri).
LANGUAGES = {
    "tur": "tr", "azj": "az", "uzn": "uz", "kaz": "kk",
    "bak": "ba", "tat": "tt", "sah": "sah", "chv": "cv",
}


def _similarity(a: str, b: str) -> float:
    longest = max(len(a), len(b)) or 1
    return 1 - levenshtein(a, b) / longest


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, set[str]], dict[str, list[tuple[str, str]]], dict[str, str]]:
    """(Türkçe biçim -> kavramlar, kavram -> [(dil, biçim)], kavram -> anlam)."""
    forms_path = DATA_DIR / "forms.csv"
    if not forms_path.exists():
        return {}, {}, {}
    glosses: dict[str, str] = {}
    params = DATA_DIR / "parameters.csv"
    if params.exists():
        with params.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                glosses[row["ID"]] = (row.get("Concepticon_Gloss") or row.get("Name") or "").strip()

    turkish: dict[str, set[str]] = {}
    by_concept: dict[str, list[tuple[str, str]]] = {}
    with forms_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            code = LANGUAGES.get(row.get("Language_ID") or "")
            # `Value` imladır (Kazakça көз, Türkçe göz); `Form` IPA'dır (ɟøz) ve
            # Türkçe sorguyla eşleşmez.
            form = (row.get("Value") or row.get("Form") or "").strip()
            if not code or not form:
                continue
            concept = row.get("Parameter_ID") or ""
            by_concept.setdefault(concept, []).append((code, form))
            if code == "tr":
                turkish.setdefault(form.lower(), set()).add(concept)
    return turkish, by_concept, glosses


class NorthEuraLexFetcher(BaseFetcher):
    #: Yerel veri, canlı servis değil.
    is_seed_source = True
    #: Arama motoru bu kaynağa KÖK varyantlarını göndermez: `yankı` -> `yan`
    #: "side" kavramının Çuvaşça аяк'ı tanık oluyordu (ölçüldü, 50 kelime).
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "NorthEuraLex (kavram hizalı, yerel CLDF)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        query = (word or "").strip().lower()
        turkish, by_concept, glosses = _load()
        # ⚠️ Mastar eki soyulmaz: `kırkmak` -> `kırk` "kırk (sayı)" oluyordu.
        concepts = turkish.get(query)
        if not concepts:
            return result
        own = to_comparison_form(query)
        seen: set[tuple[str, str]] = set()
        for concept in sorted(concepts):
            for code, form in by_concept.get(concept, []):
                if code == "tr" or code not in TURKIC_LANGUAGES_MAP or (code, form) in seen or " " in form or "_" in form:
                    continue
                comparison = to_comparison_form(form)
                if not comparison or _similarity(own, comparison) < COGNATE_THRESHOLD:
                    continue  # aynı kavram, başka kök
                seen.add((code, form))
                entry = self.make_entry(code, form, glosses.get(concept, ""), script=detect_script(form))
                entry["comparison"] = comparison
                result["turkic_languages"].append(entry)
        return result
