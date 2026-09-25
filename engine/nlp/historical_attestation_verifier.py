"""
Tarihsel İlk Tanıklama Doğrulayıcı (Historical Attestation Verifier)

Bir kelimenin **gerçek** ilk yazılı tanıklamasını, veri katmanından gelen
kayıtlardan çıkarır.

Yeniden yazılma gerekçesi
-------------------------
Önceki uygulama, eşleşme bulamadığında şu cümleyi UYDURUYORDU::

    "'<kelime>' için ilk tanıklama 13.-19. yüzyıl Osmanlı/Çağatay metinleri
     veya Cumhuriyet dönemi özleştirme kayıtlarındadır."

Bu metin daha sonra A-HVP kronolojik zaman kilidine girdi olarak veriliyor,
oradaki yıl ayrıştırıcısı "19. yüzyıl" ifadesini yakalayıp 1850 yılını
üretiyordu. Yani **tamamen uydurulmuş bir tarih bilimsel skora dönüşüyordu**.

Artık kanıt yoksa ``verified: False`` ve ``record: None`` döner; A-HVP bu
durumda kronoloji aşamasının ağırlığını toplam skordan düşer.
"""
from __future__ import annotations

import re
from typing import Any

from engine.logging_setup import get_logger
from engine.utils.attestation_dates import POINT_WORKS, canonical_year, is_old_turkic_runic, work

logger = get_logger(__name__)

#: Bilinen tarihî kaynaklar ve yılları. Bunlar kelime değil KAYNAK
#: bilgisidir. Yıllar ve gerekçeleri tek yerde:
#: ``engine.utils.attestation_dates`` (eskiden burada DLT 1074 / Orhun 735 /
#: Atebet 1150 yazıyordu, Starling'de 1072 / 732 / 1300).
DATED_SOURCES: list[tuple[re.Pattern[str], int, str]] = [
    (w.pattern, w.year, f"{w.year} {w.label}") for w in POINT_WORKS
]

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9]|[6-9][0-9]{2})\b")


_VOWELS = str.maketrans("", "", "aeıioöuüâîûïäéë")
_VOICING = str.maketrans({"g": "k", "d": "t", "ğ": "k", "ġ": "k", "ɣ": "k", "q": "k"})


def _skeleton(text: str) -> str:
    """Ünsüz iskeleti (ötümlülük birleşik): runik yazı ünlüleri çoğu yerde
    yazmaz, Oğuz g-/d- Eski Türkçe k-/t- karşılığıdır (güz ~ küz)."""
    return (text or "").lower().strip("-* ").translate(_VOWELS).translate(_VOICING)


def _names_query(entry: dict[str, Any], word: str) -> bool:
    """Runik madde sorgunun kendisi mi (ünsüz iskeleti örtüşüyor mu)?

    Yalnız ses varyantıyla bulunan indeks kaydı (``comparison`` taşır)
    sınanır; Proto-Türkçe kökün torun listesindeki runik biçim köke bağlı
    geldiği için ve Latin okunuşu olmadığı için olduğu gibi kabul edilir."""
    comparison = str(entry.get("comparison") or "")
    return not comparison or _skeleton(comparison) == _skeleton(word)


class HistoricalAttestationVerifier:
    def verify_attestation(
        self,
        word: str,
        live_entries: list[dict[str, Any]] | None = None,
        fetcher_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Kelimenin bilinen en erken yazılı tanıklamasını bulur.

        :param live_entries: Fetcher'lardan gelen dil kayıtları.
        :param fetcher_results: Ham fetcher çıktıları; ``first_attestation``
            alanı taşıyanlar (ör. EtimolojiTürkçe) doğrudan kullanılır.
        :returns: ``verified`` alanı, bu kaydın A-HVP'de kanıt sayılıp
            sayılamayacağını belirtir. Kanıt yoksa tarih UYDURULMAZ.
        """
        w = (word or "").strip().lower()
        candidates: list[tuple[int, str, str]] = []
        #: Dönem düzeyindeki tanıklar (Wilkens: "9.-14. yy"): nokta yıl DEĞİL.
        periods: list[dict[str, Any]] = []
        #: Runik (atıfsız) Eski Türkçe madde: dönem tanığı, ama yalnız serbest
        #: metin eşleşmesinden gelen (``corpus``) nokta yılları eler. Kaynağın
        #: kelime düzeyinde verdiği yıl (Starling, EtimolojiTürkçe) kazanır:
        #: runik eşleşme ses varyantıyla bulunur, eşsesli olabilir.
        runic_periods: list[dict[str, Any]] = []

        # 1. Fetcher'ın doğrudan sağladığı tarihli tanıklama (en güvenilir)
        for res in fetcher_results or []:
            att = (res or {}).get("first_attestation")
            if att and att.get("year"):
                if att.get("precision") == "period":
                    periods.append(att)
                    continue
                # Eser tanınıyorsa kaynağın yazdığı değil haritanın yılı
                # (EtimolojiTürkçe DLT'ye 1070 diyor, Starling MK 1072).
                year = canonical_year(str(att.get("source") or ""), int(att["year"]))
                candidates.append((year, att.get("source", ""), "fetcher"))

        # 2. Dil kayıtlarının kaynak/ad alanlarında geçen bilinen tarihî eserler
        for entry in live_entries or []:
            if entry.get("attestation_precision") == "period":
                continue  # fetcher'ın dönem tanığı; eser adı ayrıca yıl vermesin
            # ``attestation_ref``: sözlüğün kendi tanık atfı (Vikisözlük
            # otk: "8th century CE, Kültegin Inscription, S5").
            haystack = " ".join(
                str(entry.get(k, "")) for k in ("lang_name", "meaning", "source", "word", "attestation_ref")
            )
            dated = False
            runic_form = is_old_turkic_runic(str(entry.get("word") or ""))
            if runic_form and not _names_query(entry, w):
                # Runik madde ses varyantıyla bulunur (`iz` -> 𐰃𐰾 iş, `ye` ->
                # 𐰲𐰀 çe): ünsüz iskeleti sorguyla örtüşmüyorsa başka kelimedir,
                # ne atfı ne dönemi bu kelimeye yıl verir.
                continue
            for pattern, year, label in DATED_SOURCES:
                if pattern.search(haystack):
                    candidates.append((year, label, "corpus"))
                    dated = True
            if not dated and is_old_turkic_runic(str(entry.get("word") or "")):
                # Runik madde, atfında tarihli yazıt yok: dönem tanığı (8.-10.
                # yy, üst sınır 1000). Eskiden tarihsizdi; Starling kapalıyken
                # Orhun'da tanıklı kelime Kumanca tanıktan 1303 alıyordu.
                runic = work("otk_runic")
                runic_periods.append({
                    "year": runic.year,
                    "precision": "period",
                    "range": list(runic.range or (None, runic.year)),
                    "source": f"{runic.label}: {entry.get('word')}",
                    "label": (f"Eski Türkçe runik yazıt dönemi (8.–10. yy) içinde tanıklı; "
                              f"tarihli yazıt atfı yok ({entry.get('word')})"),
                })

        runic_bound = min((int(p["year"]) for p in runic_periods), default=None)
        if runic_bound is not None:
            candidates = [c for c in candidates if c[2] != "corpus" or c[0] <= runic_bound]
        bound = min((int(p["year"]) for p in periods), default=None)
        periods = periods + runic_periods
        if bound is not None or runic_bound is not None:
            # Dönem tanığı yalnız ÜST SINIRDIR ("en geç 1350"). Nokta tarih
            # (Starling, Orhun, DLT) her zaman kazanır; yalnız sınırdan geç
            # olan nokta tarih ilk tanıklık olamaz (kelime zaten tanıklı).
            # Eskiden Wilkens 1350'yi nokta yıl olarak veriyordu: Orhun'da
            # (732) tanıklı kelime "ilk tanıklık 1350" görünüyordu.
            candidates = [c for c in candidates if bound is None or c[0] <= bound]
            if not candidates:
                best = min(periods, key=lambda p: int(p["year"]))
                return {
                    "word": w,
                    "verified": True,
                    "first_attestation_record": best.get("label") or best.get("source", ""),
                    "first_attestation_year": int(best["year"]),
                    "first_attestation_precision": "period",
                    "first_attestation_range": list(best.get("range") or [None, int(best["year"])]),
                    "evidence_origin": "fetcher",
                    "candidate_count": len(periods),
                }

        if not candidates:
            logger.debug("'%s' için tarihli tanıklama bulunamadı", w)
            return {
                "word": w,
                "verified": False,
                "first_attestation_record": None,
                "first_attestation_year": None,
                "reason": "Veri katmanında tarihli bir yazılı tanıklama bulunamadı.",
            }

        year, source, origin = min(candidates, key=lambda c: c[0])
        return {
            "word": w,
            "verified": True,
            "first_attestation_record": source,
            "first_attestation_year": year,
            "evidence_origin": origin,
            "first_attestation_precision": "point",
            "candidate_count": len(candidates),
        }

    @staticmethod
    def parse_year(text: str) -> int | None:
        """Serbest metinden yıl çıkarır (sayfa/cilt numaralarını dışlar)."""
        if not text:
            return None
        # "s. 456", "sayfa 130", "cilt 2", "p. 88", "nr. 12" bağlamlarını maskele
        masked = re.sub(r"\b(?:s|sf|sayfa|p|pp|cilt|c|nr|no|vol|II|III|IV)\.?\s*\d+", " ", text, flags=re.I)
        m = _YEAR_RE.search(masked)
        return int(m.group(1)) if m else None
