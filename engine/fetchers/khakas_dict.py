"""
Hakasça sözlük tanıkları (yerel, ``make khakas``).

Kaynak: Hakasça–Rusça sözlük (~22 bin madde) ve Hakasça açıklamalı sözlük
(~12,5 bin madde), ikisi de Hugging Face'te CC-BY-4.0 (``adeshkin``). Motorun
Hakasça kodu ``khk``tır (veri kümesi ISO ``kjh`` kullanır).

Türkçe sorgu için aday: karşılaştırma biçimi sorguya YA DA ses
denklikleriyle tahmin edilen Hakasça biçime (``_predicted_forms``) yazılışça
≥0,50 benzeyen madde başları. ⚠️ Yazılış benzerliği akrabalık değildir
(kısa kelimede çok eşsesli tutar); her tanık Rusça karşılığıyla
``meaning_check`` taşır ve arama motoru onu sorgunun anlamıyla doğrular
(``LOCAL_WITNESS_FLOOR`` / ``LOCAL_WITNESS_MARGIN``). Rusça alıntı olarak
etiketlenen maddeler (``etym == 'rus'``) ve özel adlar tanık olamaz.

⚠️ PORTFÖYDE DEĞİL — ölçüldü (Türkçe altın küme, 2026-09-24):
* 120 kelime (tohum 33), motorun süzgeci (anlam ≥0,50, en iyiden 0,35
  içinde): 3.826 aday, 851 "doğrulanmış" tanık, 79 kelime tanık kazanıyor;
  ama 506'sı ALINTI etiketli kelimelerde (badire ~ падырбах 'грубый',
  âdet ~ иде 'хорошенько'). Rusça anlamla İngilizce/Türkçe sorgu anlamı
  arasındaki çok dilli MiniLM benzerliği ilgisiz çiftlerde de 0,5-0,7
  veriyor; 0,50 alt sınırı Rusça anlamda süzgeç işlevi görmüyor.
* Sıkılaştırma, tüm küme (699 kelime): yazılış ≥0,67 ve anlam ≥0,80 ->
  47 tanık, elle bakıldı 19'u doğru (%40; elek ~ илік 'косуля', kısa ~
  хыра 'мелкий' sahte). Doğru tanıkların çoğu (кӱн, сас, пычах, харлығ,
  паба) Wiktionary'den zaten sözlük indeksinde; YENİ doğru tanık kazanan
  kelime 4/279 miras (dondurmak, gülüş, tadım, unutulmak).
Sonuç: ikinci kez eklemeye değmiyor; sözlük, anlamı Rusçadan güvenle
eşleyen bir yol (ör. Rusça–Türkçe karşılık tablosu) bulunana dek yalnız
yerel veri olarak durur.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from engine.config import PROJECT_ROOT
from engine.fetchers.base import BaseFetcher, detect_script
from engine.fetchers.northeuralex import _predicted_forms, _similarity
from engine.utils.orthography import to_comparison_form

DATA_DIR = PROJECT_ROOT / "data" / "khakas"
LANG = "khk"
SIMILARITY_THRESHOLD = 0.50
#: Aynı sorgu için en çok bu kadar aday (en benzerler); anlam süzgeci pahalı.
MAX_CANDIDATES = 40

_VERB_PARTS = ("VERBUM",)
#: Hakasça fiil madde başı mastardır: кил-ерге, пар-арға, істе-ирге (> істирге).
#: Karşılaştırma kök üzerinden yapılır (ünlüyle biten kökte son ünlü düşer;
#: yaklaşık kalır).
_KHAKAS_LETTERS = re.compile(r"[іӧӱғңҷӌҶ]", re.IGNORECASE)
_INFINITIVE = re.compile(r"[аеиоуыіэӧӱ]?р(?:ға|ге|ха|ке)$")
#: Rusça karşılıktaki numara, kısaltma ve açıklama kırpıntıları.
_NOISE = re.compile(r"^\s*(?:\d+\)\s*)?(?:(?:[а-яё]{1,6}\.\s*)+)?")


def _clean_gloss(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    first = re.split(r";|\s//\s|\d+\)", _NOISE.sub("", text, count=1), maxsplit=1)[0]
    return first.strip(" .,:;-")


def _explanatory_entry(row: dict[str, Any]) -> tuple[str, str, bool] | None:
    """Açıklamalı sözlük maddesi -> (madde başı, Rusça karşılık, fiil mi).

    Biçim: ``**БАШ** [\\[kök-\\]] *tür.* Hakasça açıklama. -- Rusça çeviri. *Örnek.*``
    Fiilin türü ``иділ.``, madde başı mastar, kökü köşeli ayraçta verilir.
    """
    head = re.sub(r"\\?\[.*?\\?\]|\s+[IІVХ]+$", "", str(row.get("headword_fix") or "")).strip()
    field = str(row.get("field_fix") or "")
    if not head or " -- " not in field:
        return None
    is_verb = "иділ" in field.split(" -- ", 1)[0][:200]
    tail = field.split(" -- ", 1)[1]
    tail = re.sub(r"^(?:\s*\*[^*]{1,20}\*\s*)+", "", tail)  # *ист.* gibi kısaltmalar
    gloss = _clean_gloss(re.split(r"\s\*|\.\s|\.$", tail, maxsplit=1)[0])
    return head.lower(), gloss[:1].lower() + gloss[1:], is_verb


def _usable(word: str, gloss: str) -> bool:
    """Tanık olabilir mi: anlam Rusça olmalı (Hakasça harf yok), özel ad ya
    da Rusça alıntı olmamalı (karşılık madde başıyla aynı: лента ~ лента)."""
    return bool(
        word and gloss and " " not in word
        and not _KHAKAS_LETTERS.search(gloss)
        and not gloss[:1].isupper()
        and gloss.lower() != word.lower()
    )


def _read_jsonl(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def _index() -> dict[int, list[tuple[str, str, str, bool]]]:
    """Uzunluk -> [(karşılaştırma biçimi, madde başı, Rusça karşılık, fiil mi)]."""
    entries: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in _read_jsonl("khakas_russian.jsonl"):
        word = str(row.get("word") or "").strip().rstrip("-")
        if str(row.get("etym") or "").strip() == "rus" or re.search(
            r"\bимя\b|фамилия", str(row.get("semgloss") or "")
        ):
            continue
        gloss = _clean_gloss(row.get("semgloss") or "") or _clean_gloss(row.get("rest") or "")
        if _usable(word, gloss):  # büyük harfli madde başı: özel ad
            entries.setdefault((word.lower(), gloss), (gloss, str(row.get("part") or "") in _VERB_PARTS))
    known = {w for w, _ in entries}
    for row in _read_jsonl("khakas_explanatory.jsonl"):
        parsed = _explanatory_entry(row)
        if parsed and parsed[0] not in known and _usable(parsed[0], parsed[1]):
            entries.setdefault(parsed[:2], parsed[1:])
    index: dict[int, list[tuple[str, str, str, bool]]] = defaultdict(list)
    for (word, _), (gloss, is_verb) in entries.items():
        comparison = to_comparison_form(_INFINITIVE.sub("", word) if is_verb else word)
        if comparison:
            index[len(comparison)].append((comparison, word, gloss, is_verb))
    return dict(index)


def _near(form: str, index: dict[int, list[tuple[str, str, str, bool]]]):
    """``form``a ≥ eşik benzeyen kayıtlar; uzunluk kovası aramayı daraltır."""
    n = len(form)
    # 1 - d/max(n,m) ≥ t  =>  |n-m| ≤ (1-t)·max(n,m)
    low, high = int(n * SIMILARITY_THRESHOLD), int(n / SIMILARITY_THRESHOLD) + 1
    for length in range(max(1, low), high + 1):
        for record in index.get(length, ()):
            score = _similarity(form, record[0])
            if score >= SIMILARITY_THRESHOLD:
                yield score, record


class KhakasDictFetcher(BaseFetcher):
    #: İndirilmiş yerel veri; elle yazılmış tohum değil.
    is_seed_source = False
    is_local = True
    #: Kök varyantı almaz; fiil kökünü kendisi çıkarır.
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return "Hakasça–Rusça ve açıklamalı sözlük (yerel, CC-BY-4.0)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        query = (word or "").strip().lower()
        index = _index() if query else {}
        if not index:
            return result
        stem = re.sub(r"m[ae]k$", "", query)
        is_verb = stem != query and len(stem) >= 2
        own = to_comparison_form(stem if is_verb else query)
        # Fiilde kök tahmin edilir (`gel` -> `kil`); mastar ekinin tahmini
        # (`gelmek` -> `kilmih`) Hakasça mastarla (-ерге) örtüşmez.
        predicted = _predicted_forms(stem if is_verb else query).get(LANG, "")
        # (madde, anlam) -> [sorguya benzerlik, tahmine benzerlik]
        best: dict[tuple[str, str], list[float]] = {}
        for slot, form in ((0, own), (1, predicted)):
            if not form:
                continue
            for score, (_, headword, gloss, verb) in _near(form, index):
                if verb != is_verb:  # `taş` ~ тастирға 'bırakmak' değil
                    continue
                scores = best.setdefault((headword, gloss), [0.0, 0.0])
                scores[slot] = max(scores[slot], score)
        ranked = sorted(best.items(), key=lambda kv: (-max(kv[1]), kv[0]))[:MAX_CANDIDATES]
        for (headword, gloss), (own_score, predicted_score) in ranked:
            entry = self.make_entry(LANG, headword, gloss, script=detect_script(headword))
            entry["comparison"] = to_comparison_form(headword)
            entry["form_similarity"] = round(max(own_score, predicted_score), 3)
            entry["via_prediction"] = predicted_score >= SIMILARITY_THRESHOLD
            entry["meaning_check"] = True
            result["turkic_languages"].append(entry)
        return result
