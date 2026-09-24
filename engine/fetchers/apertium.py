"""
Apertium iki dilli sözlüklerinden Türk dili tanıkları (yerel, ``make apertium``).

Türkçe sorgunun sözlükteki karşılıkları tanık adayıdır. ⚠️ Karşılık ÇEVİRİDİR,
akrabalık değil: Kırgızca 'pencere' karşılığı *терезе*dir. Aday bu yüzden
sorguyla ya da o dil için ses denklikleriyle tahmin edilen biçimle yazılışça
benzer olmalıdır (``NorthEuraLexFetcher`` ile aynı ölçüt, eşik 0,50).

Fiiller Apertium'da kök olarak durur (`uç`, ``vblex``): `uçmak` sorgusu
FİİL kayıtlarında `uç`a bakar; `kırkmak` -> ad `kırk` 'kırk (sayı)'
hatası böylece oluşmaz. -mAk'lı sorgunun ad okuması (`parmak`) da ayrıca
aranır.

⚠️ Karşılık alıntı kelimede paralel alıntı olabilir (Tatarca бинт ~ bant);
bu fetcher kelimenin alıntı olup olmadığını bilmez, ayrımı motor yapmalı.
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
#: Yeni tahminciyle yeniden ölçüldü (altın kümeden 400 kelime, seed 33; 47
#: Çuvaşça aday elle): 0,50'de kesinlik 30/47, 0,60'ta 20/22, 2/3'te 19/20.
#: Tam 2/3 benzerlik (3 harfte 1 fark: gün ~ кун; saymak ~ сӑв şüpheli —
#: сӑв- "sağmak" olabilir) 0,67'nin altında kalıyordu; bu banttaki 35
#: adayın (NorthEuraLex'in 33'ü dahil) sahtesi görülmedi. Eski sahteler (kazma, gevşek, yankı, dadanmak) ≤ 0,60 — dışarıda.
LANGUAGE_THRESHOLDS = {"cv": 0.66}

#: Tahmin edilen mastar eki (Apertium'da fiil kökü eksiz durur). Çuvaşça
#: mastar -ма/-ме'dir; ortak ek listesi onu soymuyordu (dadanmak -> tudenme).
_INFINITIVE_ANY = r"(m[ae]k|m[ae]q|moq|ma[ʁғ]|mäk)$"
_INFINITIVE = {"cv": r"m[aeă]$"}

#: Fiil adayında yazılış benzerliği + tahmin benzerliği toplamı için alt
#: sınır. Çuvaşça fiil kökleri kısadır (CVC) ve tek ölçüt kolay tutar.
#: Elle bakıldı (Apertium'daki 2.282 Çuvaşça fiil kökünün tamamı, 127 aday
#: ≥ 0,66): yalnız en büyük benzerlik ile kesinlik 0,71 (sahte: bezmek ~ пар,
#: çizmek ~ ҫыр, taşımak ~ тулт, denemek ~ сӑна), -ма soyulmadan da 0,80.
#: Toplam ≥ 1,2 -> 0,90 (52 aday), ≥ 1,3 -> 0,94 (36 aday, sahte: basmak ~
#: бас, yaşlanmak ~ ҫуллан). Sorun kısa kök değil tek ölçüt: uzun türevlerde
#: tahmin -lAn/-lA ekleri yüzünden tutuyor (kirlenmek ~ варлан). Bedeli
#: gerçek kısa köklerin (yazmak ~ ҫыр, durmak ~ тӑр) düşmesi.
VERB_EVIDENCE_SUM = {"cv": 1.3}


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
    #: Yerel, indirilmiş veri (tohum değil).
    is_local = True
    #: Kök varyantı almaz (fiil kökü kendi içinde, türüne bakarak çıkarılır).
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "Apertium iki dilli sözlükleri (yerel)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        query = (word or "").strip().lower()
        try:
            table = _table()
        except Exception:
            # Sözleşme: fetch() istisna atmaz (bozuk künye ya da .dix dosyası).
            from engine.logging_setup import get_logger

            get_logger(__name__).warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
            return result
        if not query or not table:
            return result
        predicted = _predicted_forms(query)
        # -mAk ile biten sorgu hem fiil (`kaymak` 'kaymak' -> `kay`) hem ad
        # (`kaymak` 'süt kaymağı', `parmak`, `ırmak`) olabilir: iki yorum da
        # aranır, her aday KENDİ lemmasıyla benzerlik denetiminden geçer.
        # Eskiden her -mAk'lı sorgu fiil sayılıyordu; `parmak` hiç tanık almıyordu.
        readings: list[tuple[str, bool]] = [(query, False)]
        stem = re.sub(r"m[ae]k$", "", query)
        if stem != query:
            readings.insert(0, (stem, True))
        seen: set[tuple[str, str]] = set()
        for lemma, is_verb in readings:
            own = to_comparison_form(lemma)
            for code, form in table.get((lemma, is_verb), []):
                # Büyük harfli karşılık özel addır (Deniz, Yol, Qor).
                if code not in TURKIC_LANGUAGES_MAP or (code, form) in seen or form[:1].isupper():
                    continue
                comparison = to_comparison_form(form)
                expected = predicted.get(code, "")
                if is_verb:
                    expected = re.sub(_INFINITIVE.get(code, _INFINITIVE_ANY), "", expected)
                raw = _similarity(own, comparison)
                guided = _similarity(expected, comparison) if expected else 0.0
                if not comparison or max(raw, guided) < LANGUAGE_THRESHOLDS.get(code, COGNATE_THRESHOLD):
                    continue  # çeviri karşılığı, akraba değil
                if is_verb and raw + guided < VERB_EVIDENCE_SUM.get(code, 0.0):
                    continue  # tek ölçüte dayanan fiil eşleşmesi
                seen.add((code, form))
                # ⚠️ Anlam alanı BOŞ: "Türkçe: master" başlık anlamına ve anlam
                # süzgecine giriyordu (`master` sorgusunda başlık "Türkçe:
                # master" oldu). Çevirinin Türkçe lemması ayrı alanda durur.
                entry = self.make_entry(code, form, "", script=detect_script(form))
                entry["comparison"] = comparison
                entry["translation_of"] = lemma
                result["turkic_languages"].append(entry)
        return result
