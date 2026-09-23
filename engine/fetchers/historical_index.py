"""
Tarihî Katman Fetcher'ı — yerel sözlük indeksinden Eski/Osmanlı/Çağatay tanığı

Neden gerekli
-------------
Sözlük indeksi (``data/lexicons/index.db``, 150.458 kayıt) **kullanıcının
gördüğü arama çıktısına hiç düşmüyordu**: ``engine/search_engine.py`` içinde
``LexiconIndex`` geçmiyor. İndeksi yalnız alıntı tespiti
(``nlp/borrowing_detector.py``), ünlü uzunluğu (``nlp/vowel_length.py``) ve
değerlendirme hattı kullanıyor. Arama çıktısına giren tek yol fetcher'dır.

Ölçüldü: indekste ``𐰚𐰇𐰕 köz "eye"`` ve ``𐰚𐰇𐰼 kör "to see"`` runik tanıkları
duruyor ama ``search göz`` çıktısında görünmüyorlardı; oradaki Eski Türkçe
kayıtlar tohum dosyalarından ve Nişanyan'dan geliyordu.

Bu fetcher indeksin **tarihî katmanını** (Eski Türkçe, Osmanlı Türkçesi,
Çağatayca) aramaya taşır. Motor her fetcher'ı ses varyantlarıyla ayrı ayrı
çağırdığı için ``göz`` sorgusu ``köz`` varyantı üzerinden runik kaydı bulur;
``lookup()`` karşılaştırma biçmiyle eşleşir.

⚠️ Bu kaynak ata biçim (``proto_turkic``) ÖNERMEZ. Tanık sunar; ata biçim
kararı karşılaştırmalı yöntemin işidir. Aksi hâlde köken damgası kaydı
"tanıklı" diye işaretler ve türetilmiş bir kökü tanık gibi gösterirdi.
"""
from __future__ import annotations

import json
from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.variant_expander import generate_dynamic_phonetic_variants

logger = get_logger(__name__)

#: Yerel taramada denenecek en çok ses varyantı.
#:
#: ⚠️ Motor varyant listesini ``config.MAX_VARIANTS`` (=4) ile kırpar, çünkü
#: her varyant BÜTÜN fetcher'lara ayrı bir AĞ İSTEĞİ demektir. Ölçüldü: ``göz``
#: 22 varyant üretiyor ama yalnız ilk dördü kullanılıyor
#: (``göz, gör, gös, göŕ``) — Oğuz ~ Kıpçak ``g- ~ k-`` denkliğinin karşılığı
#: olan ``köz`` kesiliyor ve indeksteki runik tanık (``𐰚𐰇𐰕``) hiç bulunamıyordu.
#: Bu kaynak YEREL bir SQLite sorgusudur; ağ maliyeti yoktur, dolayısıyla o
#: kırpma burada geçerli değildir.
MAX_LOCAL_VARIANTS = 24

#: İndeksteki tarihî katmanlar. Çağdaş diller BURAYA GİRMEZ: onlar için zaten
#: canlı Wiktionary fetcher'ları var ve indeks Wiktionary türevi olduğu için
#: aynı kanıt iki kez sayılırdı.
HISTORICAL_LANGUAGES: tuple[str, ...] = ("otk", "ota", "chg", "oui", "trk-oat")

#: Aynı sözlükbirimin ikinci kaydı: runik maddenin Latin okunuşu. Tanık olarak
#: saymak aynı kanıtı iki kez saymaktır.
_SKIPPED_POS = frozenset({"romanization"})

#: Tek bir dil için en çok kaç tanık alınsın.
MAX_PER_LANGUAGE = 3


class HistoricalIndexFetcher(BaseFetcher):
    """Yerel sözlük indeksinin tarihî katmanını tanık olarak sunar."""

    #: Canlı bir servis değil, yerel veri.
    is_seed_source = True
    #: Sorgulanan diller (alt sınıf değiştirir).
    languages: tuple[str, ...] = HISTORICAL_LANGUAGES

    @property
    def source_name(self) -> str:
        return "Tarihî Katman (yerel sözlük indeksi: Eski Türkçe, Osmanlıca, Çağatayca)"

    def fetch(self, word: str) -> dict[str, Any]:
        result = self.empty_result()
        word_clean = (word or "").strip().lower()
        if not word_clean:
            return result

        try:
            from engine.db.lexicon_index import LexiconIndex

            index = LexiconIndex()
            if not index.exists:
                logger.debug("Sözlük indeksi yok; tarihî katman atlandı.")
                return result

            candidates = list(
                dict.fromkeys(
                    [word_clean, *generate_dynamic_phonetic_variants(word_clean)]
                )
            )[:MAX_LOCAL_VARIANTS]

            rows: list[dict[str, Any]] = []
            for candidate in candidates:
                rows.extend(
                    index.lookup(
                        candidate,
                        languages=list(self.languages),
                        limit=MAX_PER_LANGUAGE * len(self.languages),
                    )
                )

            seen: dict[str, int] = {}
            seen_forms: set[tuple[str, str]] = set()
            attestations: list[str] = []
            for row in rows:
                lang_code = str(row.get("lang_code") or "")
                if str(row.get("pos") or "") in _SKIPPED_POS:
                    continue
                if seen.get(lang_code, 0) >= MAX_PER_LANGUAGE:
                    continue
                surface = str(row.get("word") or "").strip()
                if not surface:
                    continue
                # Aynı biçim birden çok varyanttan gelebilir (ör. `köz` hem
                # kendi hem `küz` taramasında) — tanık iki kez sayılmasın.
                if (lang_code, surface) in seen_forms:
                    continue
                seen_forms.add((lang_code, surface))
                seen[lang_code] = seen.get(lang_code, 0) + 1
                entry = self.make_entry(lang_code, surface, str(row.get("gloss") or ""))
                # Runik/Arap yazılı biçimin indeksteki Latin anahtarı (𐰋𐰃𐱅𐰏 ->
                # `bitig`); motor kaydın sorguyu adlandırıp adlandırmadığını
                # buna bakarak anlar.
                if row.get("comparison"):
                    entry["comparison"] = str(row["comparison"])
                # Sözlüğün KENDİ etimoloji notu. İndekste duruyordu ama kayda
                # taşınmıyordu; `bitig` için "biti- + -g, Orta Çince 筆 (pit)"
                # bilgisi rapora hiç ulaşmıyordu.
                if row.get("etymology"):
                    entry["etymology"] = str(row["etymology"])
                # Sözlüğün verdiği yapı ("biti- + -g") ve andığı akrabalar.
                # Bilgi olarak taşınır; tanık sayımına girmez.
                if row.get("formation"):
                    entry["formation"] = str(row["formation"])
                # Sözlük kaydının köken sınıfı ve verici (varsa). `origin`
                # alanı fetcher sözleşmesinde seed/live anlamında kullanıldığı
                # için ayrı adla taşınır.
                if row.get("origin"):
                    entry["lexicon_origin"] = str(row["origin"])
                    entry["donor_lang"] = str(row.get("donor_lang") or "")
                    entry["donor_form"] = str(row.get("donor_form") or "")
                if row.get("cognates"):
                    try:
                        entry["source_cognates"] = json.loads(row["cognates"])
                    except ValueError:
                        pass
                result["turkic_languages"].append(entry)
                attestations.append(f"{lang_code}: {surface}")

            if attestations:
                result["root"]["reconstruction_notes"] = (
                    "Tarihî tanık (yerel indeks): " + ", ".join(attestations)
                )
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)

        return result


class ModernIndexFetcher(HistoricalIndexFetcher):
    """Çağdaş Türk dillerinin yerel sözlük kayıtları (İngilizce, Rusça ve Türkçe
    Wiktionary dökümleri), canlı 14 Wiktionary sorgusunun yerel karşılığı.

    ⚠️ Yazılış eşleşmesi akrabalık değildir: başka dilde aynı yazılan kelime
    sahte akraba olabilir. Bu yüzden her kayıt ``meaning_check`` ile işaretlenir
    ve arama motoru onu sorgunun anlamıyla karşılaştırır (eşsesli süzgeci);
    anlamı olmayan kayıt doğrulanamaz ve elenir.
    """

    #: Kök varyantı almaz (bkz. `NorthEuraLexFetcher`); kendi ses
    #: varyantlarını (`göz` -> `köz`) üst sınıf zaten üretir.
    exact_query_only = True

    languages = tuple(
        code for code in TURKIC_LANGUAGES_MAP
        if code not in HISTORICAL_LANGUAGES and code not in ("tr", "wot")
    )

    @property
    def source_name(self) -> str:
        return "Çağdaş Türk Dilleri (yerel sözlük indeksi: İngilizce/Rusça/Türkçe Wiktionary dökümleri)"

    def fetch(self, word: str) -> dict[str, Any]:
        import re

        result = super().fetch(word)
        result["root"] = self.empty_result()["root"]  # tarihî not bu kaynağa ait değil
        kept = []
        for entry in result["turkic_languages"]:
            gloss = str(entry.get("meaning") or "")
            # Özel ad (Kazakça `Дәнеш` "a male given name" `deniz`e 0,42 ile
            # geçiyordu) ve yönlendirme ("Arabic spelling of …") tanık değildir.
            if str(entry.get("word") or "")[:1].isupper() or re.search(
                r"given name|surname|\b(?:form|spelling) of\b", gloss, re.IGNORECASE
            ):
                continue
            entry["meaning_check"] = True
            kept.append(entry)
        result["turkic_languages"] = kept
        return result
