"""
Negatif kontrol bataryası — motor "hayır" demeyi biliyor mu?

Tek bir uydurma kelime (``zzzqx``) yetmez: o kadar bariz bir örneği herhangi
bir sistem eler. Gerçek sınav, **elenmesi zor** olanlardır.

Beş batarya:

``fonotaktik_gecerli_sahte``
    Türkçenin ses dizimine **uyan** ama var olmayan kelimeler (``kalgır``,
    ``sötüm``). Motor bunlara ata biçim uydurmamalı; uyduruyorsa güveni
    düşük ve rozeti ⚪ olmalı.
``bariz_sahte``
    Fonotaktiği ihlal eden uydurmalar (``zzzqx``, ``ftrxq``). Taban çizgi.
``sahte_akraba``
    Benzeyen ama akraba OLMAYAN çiftler (Türkçe ``ay`` ~ İngilizce ``eye``).
    Rastlantısal benzerliğin klasik tuzağı.
``alinti_tuzagi``
    Alıntı olduğu **kesin** ama miras gibi görünen kelimeler (``kitap``,
    ``duvar``, ``çorap``). Motor bunları miras sayarsa alıntı katmanı işe
    yaramıyor demektir.
``eşadlı``
    Aynı yazılışta hem miras hem alıntı kelime bulunanlar (``çay`` = hem
    "tea" Farsça alıntı, hem "brook" Proto-Türkçe miras). Doğru cevap
    "belirsiz"dir; kesin karar vermek burada **hata**dır.

Ölçülen büyüklük **yanlış-pozitif oranı**: motorun "bu rekonstrükte edilebilir"
dediği ve bunu güçlü bir rozetle yaptığı kontrol maddelerinin oranı. Ana
sonucun yanında raporlanır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ControlItem:
    """Tek bir negatif kontrol maddesi."""

    query: str
    witnesses: list[tuple[str, str]]
    battery: str
    reason: str


#: Fonotaktik olarak GEÇERLİ ama var olmayan kökler. Türkçenin ünlü uyumuna,
#: hece yapısına ve söz başı kısıtlarına uyarlar — yani "Türkçe gibi"dirler.
PHONOTACTICALLY_VALID: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=witnesses,
        battery="fonotaktik_gecerli_sahte",
        reason="Türkçenin ses dizimine uyuyor ama böyle bir kök yok",
    )
    for query, witnesses in [
        ("kalgır", [("kk", "kalgır"), ("tt", "qalğır"), ("ky", "kalgır")]),
        ("sötüm", [("kk", "sötüm"), ("tt", "sötem"), ("ba", "hötöm")]),
        ("tirbek", [("kk", "tirbek"), ("ky", "tirbek"), ("uz", "tirbak")]),
        ("yomgan", [("kk", "jomgan"), ("tt", "yomğan"), ("ky", "comgon")]),
        ("bürtel", [("kk", "bürtel"), ("tt", "börtel"), ("ba", "bürtäl")]),
        ("kañtar", [("kk", "kañtar"), ("ky", "kaŋtar"), ("tyv", "kaŋdar")]),
        ("üsper", [("kk", "üsper"), ("tt", "üsper"), ("uz", "uspar")]),
        ("dolgaş", [("az", "dolgaş"), ("tk", "dolgaş"), ("kk", "dolgas")]),
    ]
)

#: Fonotaktiği açıkça ihlal edenler — taban çizgi, elenmeleri kolay olmalı.
OBVIOUSLY_FAKE: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=witnesses,
        battery="bariz_sahte",
        reason="Türkçenin ses dizimini ihlal ediyor",
    )
    for query, witnesses in [
        ("zzzqx", [("kk", "zzzqy"), ("tt", "zzzqz")]),
        ("ftrxq", [("kk", "ftrxp"), ("tt", "ftrxm")]),
        ("vlkrn", [("kk", "vlkrm"), ("ky", "vlkrl")]),
        ("psxth", [("kk", "psxtl"), ("uz", "psxtn")]),
    ]
)

#: Benzeyen ama akraba OLMAYAN çiftler. Rastlantısal benzerliğin klasik
#: tuzağı: ses benzerliği tek başına akrabalık kanıtı değildir.
FALSE_FRIENDS: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=witnesses,
        battery="sahte_akraba",
        reason=reason,
    )
    for query, witnesses, reason in [
        ("ay", [("en", "eye"), ("de", "auge")], "Türkçe 'ay' ~ İng. 'eye': salt rastlantı"),
        ("bad", [("en", "bad"), ("fa", "bad")], "Farsça 'bād' rüzgâr ~ İng. 'bad': ilgisiz"),
        ("kol", [("en", "call"), ("de", "kohl")], "salt ses benzerliği"),
        ("gel", [("de", "gel"), ("en", "gel")], "salt ses benzerliği"),
    ]
)

#: Alıntı olduğu KESİN ama miras gibi görünen kelimeler. Motor bunlara ata
#: biçim türetip yüksek güven verirse alıntı katmanı işe yaramıyor demektir.
LOANWORD_TRAPS: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=witnesses,
        battery="alinti_tuzagi",
        reason=reason,
    )
    for query, witnesses, reason in [
        ("kitap", [("kk", "kitap"), ("tt", "kitap"), ("uz", "kitob"), ("ky", "kitep")],
         "Arapça kitāb — bütün Türki dillerde var ama MİRAS DEĞİL"),
        ("duvar", [("az", "divar"), ("tk", "diwar"), ("uz", "devor")],
         "Farsça dīwār"),
        ("çorap", [("az", "corab"), ("kk", "şorap"), ("tt", "çorap")],
         "Farsça/Arapça ǧurāb"),
        ("pencere", [("az", "pəncərə"), ("tk", "penjire")],
         "Farsça panǧara"),
        ("sabun", [("kk", "sabın"), ("tt", "sabın"), ("uz", "sovun")],
         "Arapça ṣābūn (nihayetinde Latince)"),

    ]
)

#: **Gerçek eşadlılar.** Aynı yazılışta hem miras hem alıntı bir kelime
#: bulunur. Doğru cevap "alıntı" da "miras" da değildir — **belirsiz**tir.
#:
#: ⚠️ Bu batarya, sistemi "kararlı görünsün" diye zorlamamak için ayrıdır:
#: eşadlı bir kelimeye kesin karar vermek, karar vermemekten kötüdür.
#: Ölçüldü — Türkçe `çay` sözlükte iki ayrı maddedir: "tea" (Farsça alıntı)
#: ve "brook, small river" (Proto-Türkçe miras). Motorun "belirsiz" demesi
#: DOĞRU davranıştır; ilk kontrol listesi bunu yanlışlıkla hata sayıyordu.
HOMONYM_CASES: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=witnesses,
        battery="eşadlı",
        reason=reason,
    )
    for query, witnesses, reason in [
        ("çay", [("kk", "şay"), ("tt", "çäy"), ("uz", "choy"), ("ky", "çay")],
         "'tea' Farsça alıntı, 'brook' Proto-Türkçe miras — aynı yazılış"),
        ("yaş", [("kk", "jas"), ("tt", "yäş"), ("ky", "jaş")],
         "'age' ve 'wet/tear' ayrı köklerdir"),
        ("kat", [("kk", "kat"), ("tt", "qat"), ("ky", "kat")],
         "'layer' ve 'hard' ayrı köklerdir"),
    ]
)

ALL_BATTERIES: dict[str, tuple[ControlItem, ...]] = {
    "fonotaktik_gecerli_sahte": PHONOTACTICALLY_VALID,
    "bariz_sahte": OBVIOUSLY_FAKE,
    "sahte_akraba": FALSE_FRIENDS,
    "alinti_tuzagi": LOANWORD_TRAPS,
    "eşadlı": HOMONYM_CASES,
}


@dataclass
class BatteryResult:
    """Bir bataryanın sonucu."""

    battery: str
    n: int = 0
    reconstructed: int = 0
    strong_badge: int = 0
    fallback: int = 0
    unattested: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def false_positive_rate(self) -> float:
        """Motorun KARŞILAŞTIRMALI YÖNTEMLE kök türettiği maddelerin oranı.

        ⚠️ `anchor_fallback` bu orana GİRMEZ. O yol, motorun "karşılaştırmalı
        yöntem uygulanamadı, aşağıdaki biçim sorgu kelimesinin kendisidir"
        dediği etiketli geri-dönüştür: `evidence_available=False`,
        `confidence=0.0`, rozet ⚪. Bunu yanlış pozitif saymak yanlış şeyi
        ölçer — motor zaten yapamadığını söylüyor.

        Ölçüldü: `sahte_akraba` bataryasının 4/4'ü bu yoldan geçiyordu
        (tanıkları en/de/fa, yani Türki tanık sayısı 0) ve oran %100
        görünüyordu. Yedekler gizlenmiyor, `fallback` alanında ayrıca
        raporlanıyor.
        """
        return self.reconstructed / self.n if self.n else 0.0

    @property
    def fallback_rate(self) -> float:
        """Etiketli geri-dönüşe düşen maddelerin oranı (kök İDDİA EDİLMEZ)."""
        return self.fallback / self.n if self.n else 0.0

    @property
    def unattested_rate(self) -> float:
        """TANIKSIZ rekonstrüksiyon oranı — bataryalar arası asıl ayırt edici.

        Bir kök, tanıkların yalnız birbiriyle uyumundan doğmuşsa ve o
        tanıkların HİÇBİRİ sözlük indeksinde yoksa, ortada tanıklı bir
        dayanak yoktur. Ölçüldü::

            fonotaktik_gecerli_sahte  8 rekons, 7'si TANIKSIZ  <- gerçek kusur
            eşadlı                    3 rekons, 0'ı tanıksız   (çay 4, yaş 4,
                                                                kat 1 tanıklı)
            alinti_tuzagi             0 rekons
            bariz_sahte               0 rekons

        Bu, `yanlış-poz` oranının ayıramadığı şeyi ayırır: `eşadlı` ve
        `fonotaktik_gecerli_sahte` ikisi de 1.000 yanlış-poz verir, ama
        biri tanıklı gerçek kelimelerdir, öbürü uydurma köklerdir.

        ⚠️ `verdict` alanı bu iş için DENENDİ VE ÇÜRÜTÜLDÜ: `kalgır` ve
        `sötüm` de tıpkı `çay`/`kat` gibi "belirsiz" dönüyor.
        ⚠️ Dürüst kayıt: `kañtar` 1 tanıkla bu ölçütten kaçıyor (7/8).
        """
        return self.unattested / self.n if self.n else 0.0

    @property
    def strong_claim_rate(self) -> float:
        """Üstüne bir de GÜÇLÜ/ORTA rozet verdiklerinin oranı — asıl tehlike."""
        return self.strong_badge / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "battery": self.battery,
            "n": self.n,
            "reconstructed": self.reconstructed,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "strong_claim_rate": round(self.strong_claim_rate, 4),
            "fallback": self.fallback,
            "fallback_rate": round(self.fallback_rate, 4),
            "unattested": self.unattested,
            "unattested_rate": round(self.unattested_rate, 4),
        }


def run_battery(reconstructor, items: tuple[ControlItem, ...], name: str) -> BatteryResult:
    """Tek bir bataryayı koşar."""
    result = BatteryResult(battery=name)
    for item in items:
        entries = [{"lang_code": code, "word": form} for code, form in item.witnesses]
        try:
            output = reconstructor(item.query, entries)
        except Exception:
            logger.warning("Negatif kontrol çöktü: %s", item.query, exc_info=True)
            continue
        result.n += 1
        # ⚠️ `anchor_fallback` KÖK İDDİASI DEĞİLDİR: motor "karşılaştırmalı
        # yöntem uygulanamadı, bu biçim sorgu kelimesinin kendisidir" diyor
        # (`evidence_available=False`, `confidence=0.0`, rozet ⚪).
        # Yanlış pozitif sayılırsa yanlış şey ölçülür. Gizlenmiyor: ayrı
        # `fallback` sayacında raporlanıyor.
        method = str(output.get("method") or "comparative")
        is_fallback = method == "anchor_fallback"
        reconstructed = bool(output.get("is_reconstructible")) and not is_fallback
        badge = str(output.get("confidence_badge", ""))
        strong = reconstructed and ("GÜÇLÜ" in badge or "ORTA" in badge)
        # `attested_witness_count is None` = ÖLÇÜLEMEDİ (yedek/alıntı yolu),
        # 0 = ölçüldü ve hiçbir tanık sözlükte yok. İkisi karıştırılmamalı.
        attested = output.get("attested_witness_count")
        result.reconstructed += reconstructed
        result.fallback += is_fallback
        result.unattested += reconstructed and attested == 0
        result.strong_badge += strong
        result.details.append(
            {
                "query": item.query,
                "reconstructed": reconstructed,
                "method": method,
                "evidence_available": bool(output.get("evidence_available")),
                "root": output.get("reconstructed_root") or output.get("withheld_reconstruction", ""),
                "badge": badge,
                "calibrated": output.get("calibrated_confidence"),
                "reason": item.reason,
            }
        )
    return result


def main() -> int:
    import argparse

    from engine.evaluation.harness import comparative_reconstructor
    from engine.evaluation.report import EVAL_DIR

    ap = argparse.ArgumentParser(description="Negatif kontrol bataryası")
    ap.add_argument("--verbose", action="store_true", help="madde madde göster")
    args = ap.parse_args()

    reconstructor = comparative_reconstructor()
    results = [
        run_battery(reconstructor, items, name) for name, items in ALL_BATTERIES.items()
    ]

    # `yedek` sütunu ŞART: `anchor_fallback` yanlış pozitif sayılmıyor,
    # ama sayılmıyor diye görünmez de olmamalı. Motorun "yapamadım, işte
    # sorgu biçimi" dediği maddeler burada açıkça durur.
    print(
        f"\n{'batarya':30} {'n':>4} {'rekonstrükte':>13} {'yanlış-poz':>11} "
        f"{'güçlü iddia':>12} {'yedek':>7} {'tanıksız':>9}"
    )
    print("-" * 92)
    for result in results:
        print(
            f"{result.battery:30} {result.n:>4} {result.reconstructed:>13} "
            f"{result.false_positive_rate:>11.3f} {result.strong_claim_rate:>12.3f} "
            f"{result.fallback:>7} {result.unattested:>9}"
        )

    if args.verbose:
        for result in results:
            print(f"\n--- {result.battery}")
            for detail in result.details:
                if detail["reconstructed"]:
                    mark = "!!"
                elif detail.get("method") == "anchor_fallback":
                    mark = "~~"  # kök iddia edilmedi, etiketli geri-dönüş
                else:
                    mark = "ok"
                print(
                    f"  {mark} {detail['query']:10} {str(detail['root']):14} "
                    f"{detail['badge']:20} {detail['calibrated']}"
                )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "negative_controls.json"
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-negative-controls/v1",
                "note": (
                    "Yanlış-pozitif oranı ANA SONUCUN YANINDA raporlanır. "
                    "Tek bir uydurma kelime yetmez; asıl sınav fonotaktik olarak "
                    "geçerli sahte kökler ve alıntı tuzaklarıdır."
                ),
                "batteries": [r.as_dict() for r in results],
                "details": {r.battery: r.details for r in results},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
