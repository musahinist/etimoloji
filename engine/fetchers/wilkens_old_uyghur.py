"""
Eski Uygurca tanıklar — Wilkens 2021, *Handwörterbuch des Altuigurischen*
(yerel, ``make wilkens``; CC BY-SA 4.0).

Sözlükte ~19 bin gerçek madde var (hatalı okuma ve göndermeler hariç);
yerel indeksin Eski Uygurca katmanı (kaikki) yalnız 446 madde. Her maddenin
Almanca VE Türkçe anlamı var (``Auge || göz``).

Eşleşme (yalnız bu ikisi birlikte tutarsa tanık):

1. **Anlam** — maddenin Türkçe anlamlarından biri sorgunun KENDİSİ (fiilde
   mastarı: ``bil`` ~ "bilmek"). Sözcük eşitliğidir, model yok. Anlam
   modeli (``paraphrase-multilingual-MiniLM-L12-v2``) çok dillidir ve
   Almancayı da okur, ama Hakasça denemesinde (``khakas_dict``) yabancı dilde
   anlamla ilgisiz çiftlere 0,5-0,7 verdi; burada Türkçe anlam hazır olduğu
   için gerek yok.
2. **Biçim** — madde başı sorguya ya da sorgunun öğrenilmiş ses
   denklikleriyle tahmin edilen Eski Türkçe biçimine yazılışça
   ``FORM_THRESHOLD`` kadar benzer. Anlam tek başına çeviri eşdeğeri bulur
   (``kapı`` ~ *ešik* "kapı"), akraba değil.

Özel adlar, yer adları, hatalı okumalar (``†``) ve göndermeler (``→``)
tanık olmaz. Köken zinciri (``< TochB ajite < Skt. ajita``) tanığa
``etymology``/``donor_chain`` olarak taşınır.

Tarih: sözlük tanık yeri VERMEZ (Vorbemerkungen s. II), yani maddeye özgü
yıl yoktur. Tanık NOKTA YIL DEĞİL, dönem aralığıdır ("Eski Uygurca dönemi
(9.–14. yy) içinde tanıklı; kesin yer yok"); kronolojide yalnız üst sınır
(en geç 1350) olarak kullanılır ve nokta tarih (Starling, Orhun, DLT) varsa
her zaman o kazanır (``HistoricalAttestationVerifier``). Tanık, Eski Uygurca döneminin (``PERIOD``) ETİKETİYLE tarihlenir
ve yıl motorun var olan yüzyıl ayrıştırıcısıyla
(``ChronologicalTimeLock.parse_year_or_century``) bu etiketten çıkar: "9.-14. yy"
-> 1350, dönemin SONU. Dönemin başı (850) seçilseydi her Moğolca (temas
~1200) ve Arapça/Farsça (~900) alıntı A-HVP zaman kilidinde "ANAKRONİZM"
alırdı; son sınır ise yalnız "en geç 14. yy'da tanıklı" der, bu doğrudur.

Ölçüldü (2026-09-24, ad243fd, aynı ağaçta Wilkens'siz/Wilkens'li):
* ``make eval-chronology`` Starling'siz kapsam 0,060 -> 0,455 (79 kelime
  Wilkens'ten); "yüzyıl içi" 0 -> 0. Başvuru Orhun (732) / MK (1072) / KB
  (1069) etiketleri; dönem düzeyinde TEK bir yıl ikisine birden 100 yıldan
  yakın olamaz (850 seçilse de 0 kalırdı). Yani kapsam artar, yıl isabeti
  artmaz: bu kaynak "Eski Uygurcada tanıklı" der, hangi yüzyılda demez.
  Starling'li: kapsam 0,995 -> 1,000, yüzyıl içi 0,955 -> 0,950 (eklenen tek
  kelime 1350 ile).
* ``make eval-headline``: değişmedi (Starling'siz tam 0,354 -> 0,358).
* Tanık kesinliği, elle (kronoloji + Türkçe altın küme eşleşmeleri, eşik
  ayarından bağımsız yeni 50 örnek): 43/50 = 0,86. Hatalar türetme farkı
  (*ädärtä-* ~ eyerlemek, *bakıt-* ~ baktırmak, *tägšil-* ~ değiş), eşses
  (*yara-* "yaramak" ~ yara) ve bir sahte biçim (*kadgu* ~ tutku).
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from engine.fetchers.base import BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

LANG = "oui"
#: Eski Uygurca yazılı dönemi (Uygur Kağanlığı sonrası Turfan/Dunhuang
#: metinleri). Yıl bundan ``parse_year_or_century`` ile çıkar.
PERIOD = "9.-14. yy"
#: Dönemin yıl aralığı: 9. yüzyıl başı - üst sınır (``attestation_year``).
PERIOD_START = 800
SOURCE_LABEL = "Wilkens 2021, Handwörterbuch des Altuigurischen"
#: Madde başı ile sorgu (ya da tahmini Eski Türkçe biçim) arasındaki en düşük
#: yazılış benzerliği (1 - Levenshtein / uzunluk).
FORM_THRESHOLD = 0.6
#: Bir sorgu için en çok tanık.
MAX_WITNESSES = 3



def gloss_items(gloss: str) -> list[str]:
    """Türkçe anlamın virgül/noktalı virgülle ayrılmış öğeleri, parantez
    içi (``(Skt. …)``, ``(ler)``) atılmış hâliyle.

    Eşleşme SÖZCÜK değil ÖĞE üzerinden: "göz" sözcüğü *kök* maddesinin
    "göz bebeklerinin perdelenmesi" anlamında da geçer, öğe olarak geçmez.
    """
    text = _tr_lower(gloss)
    previous = None
    while previous != text:  # iç içe parantez
        previous, text = text, re.sub(r"\([^()]*\)", " ", text)
    items = []
    for item in re.split(r"[,;]", text):
        if "…" in item:
            # "göz …" Almanca "Augen-" karşılığıdır: türemiş sıfat (*közlüg*),
            # sorgunun kendisi değil.
            continue
        item = re.sub(r"[?!.‘’'\"]", " ", item)
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            items.append(item)
    return items



def _tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _headword_forms(record: dict[str, Any]) -> list[str]:
    """Madde başı ve yazılış varyantlarının karşılaştırma biçimleri.

    ``ad(ı)nagu`` -> ``adınagu`` ve ``adnagu``; köşeli ayraç (tamamlanmış
    harf) açılır. Fiil tiresi ``to_comparison_form``da düşer.
    """
    out: list[str] = []
    for raw in [record.get("headword") or "", *(record.get("variants") or [])]:
        raw = raw.replace("[", "").replace("]", "")
        for form in (re.sub(r"[()]", "", raw), re.sub(r"\([^)]*\)", "", raw)):
            comparison = to_comparison_form(form)
            if comparison and comparison not in out:
                out.append(comparison)
    return out


def _usable(record: dict[str, Any]) -> bool:
    return bool(
        record.get("tr")
        and not record.get("error")
        and not record.get("see")
        and not record.get("proper_name")
        and " " not in str(record.get("headword") or "").strip()
    )


@lru_cache(maxsize=1)
def _index() -> dict[str, list[dict[str, Any]]]:
    """Türkçe anlam sözcüğü -> o sözcüğü anlamında taşıyan maddeler."""
    from engine.db.wilkens import load_records

    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records():
        if not _usable(record):
            continue
        record["_forms"] = _headword_forms(record)
        for item in {i for gloss in record["tr"] for i in gloss_items(gloss)}:
            index[item].append(record)
    return dict(index)


#: Kaba ses sınıfları: Eski Uygurca ~ Türkiye Türkçesi düzenli denklikleri
#: (köz ~ göz, tüz ~ düz, ačıg ~ acı, taŋ ~ tan, äv ~ ev) fark sayılmasın.
#: Ünlüler yalnız ön/art uyumunu değil yuvarlaklık ve yüksekliği tutar
#: (ı=i, u=ü, o=ö, a≠e: bičäk ~ bıçak farkı küçük kalır).
_COARSE = str.maketrans({
    "g": "K", "k": "K", "ğ": "K", "q": "K", "d": "T", "t": "T", "b": "P", "p": "P",
    "c": "C", "ç": "C", "č": "C", "j": "C", "ş": "S", "š": "S", "ŋ": "N", "n": "N", "w": "V", "v": "V",
    "ä": "E", "e": "E", "ı": "I", "i": "I", "u": "U", "ü": "U", "o": "O", "ö": "O", "a": "A",
})


def coarse(form: str) -> str:
    """Karşılaştırma biçimini kaba ses sınıflarına indirger; ünlüden sonraki
    son ``-g`` düşer (Eski Uygurca *ačıg*, *sarıg*, *kapıg* ~ acı, sarı, kapı)."""
    form = re.sub(r"(?<=[aeıioöuüä])[gğ]$", "", form)
    return form.translate(_COARSE)


def _similarity(a: str, b: str) -> float:
    from engine.fetchers.northeuralex import _similarity as similarity

    return similarity(coarse(a), coarse(b))


def _predicted_old_turkic(stem: str) -> str:
    try:
        from engine.fetchers.northeuralex import _predicted_forms

        return _predicted_forms(stem).get("otk", "")
    except Exception:  # tahminci yoksa yalnız sorgu biçimi kullanılır
        return ""


@lru_cache(maxsize=1)
def attestation_year() -> int | None:
    """Eski Uygurca dönem yılı: ortak kaynak→yıl haritasında (``attestation_dates``)
    varsa oradan; yoksa ``PERIOD`` etiketinden motorun yüzyıl ayrıştırıcısıyla
    (ikisi de 1350)."""
    try:
        from engine.utils.attestation_dates import work

        return work("oui").year
    except (ImportError, KeyError):
        pass
    from engine.nlp.hypothesis_validation_protocol import ChronologicalTimeLock

    return ChronologicalTimeLock().parse_year_or_century(PERIOD)


def _etymology(record: dict[str, Any]) -> str:
    chain = record.get("donor_chain") or []
    return " ".join(
        f"{'<' if step.get('direct') else '<<'} {step.get('lang')} {step.get('form') or ''}".strip()
        for step in chain
    )


class WilkensOldUyghurFetcher(BaseFetcher):
    """Wilkens sözlüğünden sorguyla aynı Türkçe anlamlı, biçimce yakın madde."""

    is_seed_source = False
    is_local = True
    #: Kendi fiil kökünü ve tahmini biçimini üretir; motorun ses varyantları
    #: (`göz` -> `gör`) "görmek" anlamlı *kör-* maddesini `göz`e bağlardı.
    exact_query_only = True

    @property
    def source_name(self) -> str:
        return f"{SOURCE_LABEL} (yerel, Eski Uygurca)"

    def matches(self, word: str) -> list[tuple[float, dict[str, Any]]]:
        query = _tr_lower((word or "").strip())
        if not query or " " in query:
            return []
        index = _index()
        if not index:
            return []
        stem = re.sub(r"m[ae]k$", "", query)
        is_verb = stem != query and len(stem) >= 2
        keys = [query]
        own = to_comparison_form(stem)
        predicted = _predicted_old_turkic(stem)
        found: dict[int, tuple[float, dict[str, Any]]] = {}
        # Çıplak sorgu önce ad olarak aranır; fiil maddesi (`taş` ~ *taš-*
        # "taşmak") yalnız ad eşleşmesi yoksa: `bil` ~ *bil-* "bilmek".
        if not is_verb:
            keys = [query, f"{query}mek", f"{query}mak"]
        for key in keys:
            if found and key != query and not is_verb:
                break
            for record in index.get(key, ()):
                verb_entry = str(record.get("headword") or "").endswith("-")
                # "bilmek" anlamı fiil maddesine, "göz" anlamı ad maddesine
                if verb_entry != key.endswith(("mek", "mak")):
                    continue
                score = max(
                    (_similarity(target, form) for target in (own, predicted) if target
                     for form in record["_forms"]),
                    default=0.0,
                )
                if score >= FORM_THRESHOLD:
                    previous = found.get(id(record))
                    if previous is None or previous[0] < score:
                        found[id(record)] = (score, record)
        ranked = sorted(found.values(), key=lambda item: (-item[0], item[1].get("page", 0)))
        # Yalnız en iyi biçim benzerliğindekiler: `göl` ~ *köl* (1,0) varken
        # *kölmän* "küçük göl" (0,5) türevdir; `kara` ~ *kurgak* "kara (toprak)"
        # eşsestir.
        return [item for item in ranked if item[0] >= ranked[0][0] - 1e-9][:MAX_WITNESSES] if ranked else []

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        try:
            matched = self.matches(word)
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
            return result
        if not matched:
            return result
        year = attestation_year()
        for score, record in matched:
            meaning = "; ".join(record["tr"][:3])
            entry = self.make_entry(LANG, record["headword"], meaning)
            entry["comparison"] = record["_forms"][0]
            entry["form_similarity"] = round(score, 3)
            entry["meaning_de"] = "; ".join(record.get("de", [])[:3])
            entry["source_page"] = record.get("page")
            # Nokta yıl değil: doğrulayıcı eser adından ayrıca yıl çıkarmasın.
            entry["attestation_precision"] = "period"
            etymology = _etymology(record)
            if etymology:
                entry["etymology"] = f"Wilkens: {etymology}"
                entry["donor_chain"] = record["donor_chain"]
            result["turkic_languages"].append(entry)
        best = matched[0][1]
        result["root"]["reconstruction_notes"] = "Eski Uygurca tanık (Wilkens 2021): " + ", ".join(
            f"{r['headword']} “{r['tr'][0]}” (s. {r.get('page')})" for _, r in matched
        )
        if year is not None:
            result["first_attestation"] = {
                "form": best["headword"],
                "meaning": best["tr"][0],
                "source": f"Eski Uygurca ({PERIOD}), {SOURCE_LABEL} s. {best.get('page')}",
                # Yıl yalnız ÜST SINIR; doğrulayıcı nokta tarihi (Starling, Orhun,
                # DLT) her zaman öne alır, rapor aralığı gösterir.
                "year": year,
                "precision": "period",
                "range": [PERIOD_START, year],
                "label": (f"Eski Uygurca dönemi ({PERIOD.replace('-', '–')}) içinde tanıklı; "
                          f"kesin yer yok ({SOURCE_LABEL}, s. {best.get('page')})"),
            }
        return result
