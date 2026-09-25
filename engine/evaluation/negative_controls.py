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
    Benzeyen ama akraba OLMAYAN çiftler (Farsça ``bād`` ~ İngilizce ``bad``).
    Rastlantısal benzerliğin klasik tuzağı.
``alinti_tuzagi``
    Alıntı olduğu **kesin** ama miras gibi görünen kelimeler (``saat``,
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
from pathlib import Path
from typing import Any

from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

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

#: Sözlük biçimi (mastar/şimdiki zaman) ekleri; tanıklık denetimiyle AYNI
#: liste. Üretilen sahte kök bu eklerle de indekste bulunmamalıdır.
def _citation_suffixes() -> tuple[str, ...]:
    from engine.nlp.comparative_reconstruction import CITATION_SUFFIXES

    return CITATION_SUFFIXES


#: Üretilmiş sahte köklerin tohumu. DEĞİŞTİRİLİRSE liste yeniden üretilmeli.
GENERATED_FAKE_SEED = 20260924

#: Üretilecek sahte kök sayısı.
GENERATED_FAKE_COUNT = 50


def _sample_markov(counts: dict[str, dict[str, int]], rng: Any, order: int = 3) -> str:
    """Miras modelinin ham n-gram sayımlarından tek kelime örnekler.

    Yumuşatma KULLANILMAZ: yumuşatılmış kütle her karakteri mümkün kılar ve
    Türkçeye benzemeyen diziler üretir. Yalnız eğitimde görülmüş geçişler.
    """
    from engine.nlp.phonotactic_lm import BOUNDARY

    context = BOUNDARY * (order - 1)
    out = ""
    for _ in range(12):
        bucket = counts.get(context) or {}
        if not bucket:
            break
        chars = sorted(bucket)
        character = rng.choices(chars, weights=[bucket[c] for c in chars])[0]
        if character == BOUNDARY:
            break
        out += character
        context = (context + character)[-(order - 1):]
    return out


#: Söz sonunda Türkçede görülen ünsüz kümeleri; diğerleri (``-lp``,
#: ``-şk``) üretilmiş kelimeyi yabancı gösterir.
_FINAL_CLUSTERS = frozenset({"rt", "lt", "nt", "rk", "lk", "nk", "rç", "nç", "st", "rs", "ls", "rp", "ft", "şt"})


def _bad_final_cluster(word: str) -> bool:
    from engine.utils.phonotactics import VOWELS

    tail = word[-2:]
    return len(tail) == 2 and not (set(tail) & VOWELS) and tail not in _FINAL_CLUSTERS


def fake_witnesses(root: str) -> list[tuple[str, str]]:
    """Sahte köke düzenli ses denklikleriyle üç "akraba" biçim türetir.

    Denklikler bilinçli olarak GERÇEKÇİDİR (Kıpçak ``y- > j-``, ``ş > s``,
    ``ç > ş``; Tatar/Başkurt ünlü kayması): tanıklar birbiriyle kusursuz
    uyumlu olmalı ki sınav tanık UYUMUNU değil tanık GERÇEKLİĞİNİ ölçsün.
    """
    kk = root.replace("ş", "s").replace("ç", "ş")
    if kk.startswith("y"):
        kk = "j" + kk[1:]
    ky = "j" + root[1:] if root.startswith("y") else root
    tt = root.translate(str.maketrans({"o": "u", "ö": "ü", "e": "ä"}))
    return [("kk", kk), ("ky", ky), ("tt", tt)]


def generate_phonotactic_fakes(
    n: int = GENERATED_FAKE_COUNT,
    *,
    seed: int = GENERATED_FAKE_SEED,
    model_file: Path | None = None,
) -> list[str]:
    """Eğitilmiş Türkçe fonotaktik modelden (miras yarısı) sahte kökler üretir.

    Süzgeç: 4–7 harf, en az iki ünlü, büyük ünlü uyumu, kesin söz başı
    ihlali yok, ve **kök de türetilen tanıkları da** sözlük indeksinde
    (mastar ekli biçimleri dahil) YOK. İndeks yoksa üretim reddedilir:
    "indekste yok" doğrulanamayan bir sahte kök sahte olduğu bilinmeyen köktür.
    """
    import random

    from engine.db.lexicon_index import LexiconIndex
    from engine.nlp.phonotactic_lm import MarkovModel, model_path
    from engine.utils.phonotactics import (
        VOWELS,
        has_vowel_harmony,
        initial_consonant_violation,
    )

    source = model_file or model_path("tr")
    data = json.loads(source.read_text(encoding="utf-8"))
    model = MarkovModel.from_dict(data["inherited"])
    index = LexiconIndex()
    if not index.exists:
        raise RuntimeError("sözlük indeksi yok; sahte kökün yokluğu doğrulanamaz")
    suffixes = _citation_suffixes()
    rng = random.Random(seed)
    seen: set[str] = set()
    fakes: list[str] = []
    with index.connect() as connection:

        def attested(form: str) -> bool:
            candidates = [form, *(form + suffix for suffix in suffixes)]
            query = f"SELECT 1 FROM entries WHERE comparison IN ({','.join('?' * len(candidates))}) LIMIT 1"
            return connection.execute(query, candidates).fetchone() is not None

        for _ in range(200_000):
            if len(fakes) >= n:
                break
            word = _sample_markov(model.counts, rng, model.order)
            if word in seen or not 4 <= len(word) <= 7:
                continue
            seen.add(word)
            if sum(ch in VOWELS for ch in word) < 2 or not has_vowel_harmony(word):
                continue
            if initial_consonant_violation(word)[0]:
                continue
            if _bad_final_cluster(word):
                continue
            forms = {word, *(to_comparison_form(w) for _, w in fake_witnesses(word))}
            if any(attested(f) for f in forms):
                continue
            # Çekimli gerçek kelimeyi (``bayırı`` = ``bayır`` + iyelik) sahte
            # kök saymamak için: 1–3 harf kısaltılmış biçim de (``ğ > k`` ile)
            # indekste olmamalı.
            stems = {
                stem[:-1] + "k" if stem.endswith("ğ") else stem
                for cut in (1, 2, 3)
                if len(stem := word[:-cut]) >= 3
            }
            if any(attested(stem) for stem in stems):
                continue
            fakes.append(word)
    return fakes


#: :func:`generate_phonotactic_fakes` çıktısı (tohum
#: :data:`GENERATED_FAKE_SEED`, ``data/models/phonotactic_tr.json`` commit
#: ``b0e42d4`` sürümü). Modül yüklenirken indekse DOKUNULMAZ: liste burada
#: sabittir, yeniden üretmek için
#: ``python -m engine.evaluation.negative_controls --regenerate-fakes
#: --model <o sürümün dosyası>``.
GENERATED_FAKES: tuple[str, ...] = (
    "afka", "akçık", "kıca", "iğitli", "açkut", "afkay",
    "yençeği", "apulua", "balçava", "apul", "apulga", "oklık",
    "arnıra", "arutçuk", "korbağı", "sirciği", "ayaymaç", "açkur",
    "balmaçı", "oğuk", "apaçkut", "iğit", "acaskın", "ayışlık",
    "açkabaş", "iğer", "aşyayıl", "afkar", "başyara", "afga",
    "arnırık", "alçu", "başyana", "yençiçe", "osmaçık", "amyokya",
    "yıtmak", "apulan", "amyokla", "alçuk", "asunacı", "apuluğu",
    "otaşçık", "afgara", "oklım", "acaarna", "bençile", "bençeği",
    "amsuna", "tupyırt",
)

GENERATED_PHONOTACTIC_FAKES: tuple[ControlItem, ...] = tuple(
    ControlItem(
        query=query,
        witnesses=fake_witnesses(query),
        battery="fonotaktik_gecerli_sahte",
        reason="fonotaktik modelden üretildi; kök ve tanıkları sözlük indeksinde yok",
    )
    for query in GENERATED_FAKES
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
    # "ay", "kol", "gel" burada YOKTUR: bunlar gerçek miras Türkçe kelimelerdir
    # (Proto-Türkçe köklere dayanırlar), sahte akraba örneği değildirler. Motor
    # bunlara "YETERSİZ KANIT" derse bu bir KAÇIRMADIR (yanlış negatif), ama
    # `sahte_akraba` bataryası bunu başarı sayardı — ölçüt tersine dönerdi.
    for query, witnesses, reason in [
        ("bad", [("en", "bad"), ("fa", "bad")], "Farsça 'bād' rüzgâr ~ İng. 'bad': ilgisiz"),
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
        # "kitap" burada YOKTUR: data/seed/donor/donor_etymology.json TOHUM
        # dosyasında `kitap` doğrudan Arapça alıntı olarak kayıtlı — motor
        # gerçek karşılaştırmalı akıl yürütmeden, tohum sızıntısıyla doğru
        # cevabı verebilir. "saat" aynı yayılım desenini taşır (kk/tt/uz/ky
        # tümünde ayrı ayrı ödünçlenmiş Arapça sā'a, indekste origin='alıntı'
        # donor_lang='ar' olarak kayıtlı) ama tohum dosyasında YOKTUR.
        ("saat", [("kk", "sagat"), ("tt", "segat"), ("uz", "soat"), ("ky", "saat")],
         "Arapça sā'a — bütün Türki dillerde ayrı ayrı ödünçlenmiş, MİRAS DEĞİL"),
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
    "fonotaktik_gecerli_sahte": PHONOTACTICALLY_VALID + GENERATED_PHONOTACTIC_FAKES,
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

        **Tanıksız kök yasağından sonra** (``UNATTESTED_BAN``) bu sayaç
        sıfıra iner: tanıksız maddeler artık `anchor_fallback`e düşer ve
        `yedek` sütununda görünür. Ölçüldü (58 sahte, 50'si fonotaktik
        modelden üretilmiş)::

            önce   58/58 rekons, 57 tanıksız   yanlış-poz 1.000
            sonra   1/58 rekons (kañtar), 57 yedek  yanlış-poz 0.017
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


def borrowing_detector_report() -> int:
    """Alıntı tespitçisini her bataryada koşar, madde madde yazdırır.

    Önceden ``borrowing_detector --controls`` idi; üretim modülü ölçüm
    bataryasını içe aktarmasın diye buraya taşındı.
    """
    from engine.nlp.borrowing_detector import BorrowingDetector

    detector = BorrowingDetector()
    for name, items in ALL_BATTERIES.items():
        print(f"\n--- {name}")
        for item in items:
            entries = [{"lang_code": c, "word": w} for c, w in item.witnesses]
            verdict = detector.detect(item.query, entries)
            if name == "alinti_tuzagi":
                expected = verdict.is_borrowed
            elif name == "eşadlı":
                # Eşadlıda doğru cevap KESİN KARAR DEĞİL, belirsizliktir.
                expected = not verdict.blocks_inherited_reconstruction
            else:
                expected = not verdict.is_borrowed
            mark = "OK " if expected else "!! "
            print(f"  {mark}{verdict.word:10} {verdict.verdict:12} {verdict.score:.2f}")
    return 0


def main() -> int:
    import argparse

    from engine.evaluation.harness import comparative_reconstructor
    from engine.evaluation.report import EVAL_DIR

    ap = argparse.ArgumentParser(description="Negatif kontrol bataryası")
    ap.add_argument("--verbose", action="store_true", help="madde madde göster")
    ap.add_argument("--model", type=Path, help="--regenerate-fakes için model dosyası")
    ap.add_argument(
        "--regenerate-fakes",
        action="store_true",
        help="GENERATED_FAKES listesini fonotaktik modelden yeniden üret ve yazdır",
    )
    ap.add_argument(
        "--borrowing-detector",
        action="store_true",
        help="alıntı tespitçisini bataryalarda koş (eski borrowing_detector --controls)",
    )
    args = ap.parse_args()

    if args.borrowing_detector:
        return borrowing_detector_report()

    if args.regenerate_fakes:
        print(generate_phonotactic_fakes(model_file=args.model))
        return 0

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
