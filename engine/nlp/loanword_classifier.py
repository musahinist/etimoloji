"""
Alıntı Kelime Sınıflandırıcısı — Katman 1 + 2 (Loanword Classifier)

Alıntı keşif hattının **Katman 1** (fonotaktik ihlal analizi) ve
**Katman 2** (çapraz lehçe yayılımı) uygulamasıdır.

Düzeltilen kritik hata
----------------------
Önceki sürümde ünlü uyumsuzluğu gibi ihlaller ``score_native``'i düşürüyor ama
**hiçbir alıntı kovasına puan eklemiyordu**. Olasılık dağılımı toplam skora
bölünerek normalize edildiği için pay ve payda birlikte küçülüyor,
``p_native`` yeniden **1.0**'a çıkıyordu. Sonuç::

    kitap      -> "Asli Öz Türkçe"  p_native = 1.0   (1 ihlal tespit edilmiş!)
    televizyon -> "Asli Öz Türkçe"  p_native = 1.0
    müdür      -> "Asli Öz Türkçe"  p_native = 1.0
    kalem      -> "Asli Öz Türkçe"  p_native = 1.0

Artık her ihlal, hangi kaynak dil ailesini işaret ediyorsa O KOVAYA puan
ekler; öz Türkçe kovası yalnızca ihlal yokluğundan beslenir.

Ayrıca ``score_greek_latin`` eskiden yalnızca ``score_arabic_persian`` ile
BİRLİKTE artıyordu; bu yüzden ``p_greek_latin > p_arabic_persian`` dalı hiçbir
zaman doğru olamıyor (ölü kod), ``==`` dalı ise her Arapça alıntıyı
"Akdeniz/Ermenice/Grekçe" diye sınıflandırıyordu. Grekçe/Latince artık
bağımsız sinyallerden beslenir.
"""
from __future__ import annotations

from typing import Any

from engine.logging_setup import get_logger
from engine.utils.phonotactics import (
    PERSIAN_SUFFIXES,
    STRICT_NON_TURKIC_INITIALS,
    WEAK_NON_TURKIC_INITIALS,
    WESTERN_SUFFIXES,
    has_vowel_harmony,
    initial_cluster_violation,
    match_arabic_pattern,
    match_greek_latin_pattern,
    match_suffix,
)

logger = get_logger(__name__)

#: Söz başı ünsüzlerinin işaret ettiği kaynak dil aileleri.
INITIAL_HINTS: dict[str, tuple[str, ...]] = {
    "f": ("arabic_persian", "western"),
    "h": ("arabic_persian", "greek_latin"),
    "p": ("arabic_persian", "greek_latin", "western"),
    "v": ("arabic_persian", "western"),
    "j": ("western",),
    "z": ("arabic_persian", "greek_latin"),
    "c": ("arabic_persian",),
    "ğ": ("arabic_persian",),
    "r": ("arabic_persian", "western"),
    "l": ("arabic_persian", "western"),
    "m": ("arabic_persian",),
    "n": ("arabic_persian",),
}

CLASSIFICATION_LABELS = {
    "native": "Asli Öz Türkçe (Native Turkic)",
    # Aile puanları berabereyse hangi ailenin işaret edildiği BİLİNMİYOR
    # demektir; rastgele bir aile adı vermek uydurma kesinliktir.
    "loan_undetermined": "Alıntı — verici dil ailesi belirlenemedi",
    "arabic_persian": "Arapça / Farsça Alıntısı (Doğu Alıntısı)",
    "greek_latin": "Grekçe / Bizans / Latince / Ermenice Alıntısı",
    "western": "Batı Dilleri Alıntısı (Fransızca / İngilizce / İtalyanca)",
}

#: p_native bu değerin üzerindeyse "asli Öz Türkçe" sayılır (ikili karar).
NATIVE_THRESHOLD = 0.55

#: ``LoanwordDetector`` aynı p_native üzerinde ÜÇLÜ karar verir: bu bandın
#: üstü "native", altı "loanword", arası "uncertain". Ikili eşik
#: (``NATIVE_THRESHOLD``) bandın İÇİNDEdir; bu yüzden iki modül hiçbir
#: kelimede zıt hüküm vermez (sınıflayıcı "asli" derken dedektör en kötü
#: "belirsiz" der). Dedektörün güveni her durumda max(p, 1-p) olduğundan
#: bant sınırları rozet girdisini (``detect_conf``) DEĞİŞTİRMEZ; yalnız
#: ``verdict`` etiketini belirler (bkz. docs/THRESHOLDS.md §2).
DETECTOR_NATIVE_FLOOR = 0.70
DETECTOR_LOAN_CEILING = 0.35

#: Katman 2 — çapraz Türki yayılım oranı (``TURKIC_LANGUAGE_COUNT``'a
#: bölünmüş). ≥ ``SPREAD_NATIVE_EVIDENCE`` asli kanıtı (p_native'e artı),
#: ≤ ``SPREAD_LOAN_EVIDENCE`` alıntı kanıtı. A-HVP 4. aşaması yayılım
#: puanını aynı ``SPREAD_NATIVE_EVIDENCE``'ta doyurur.
#: ``cognate_alignment``'ın 0,70 / 0,20'si yalnız açıklama METNİdir (hiçbir
#: karara girmez). Bu iki sabiti 0,70 / 0,20'ye çekmek ölçüldü (eval-badge,
#: ön-kayıt data/cache/work/thresholds/PREREG.md): train'de rozet AUC 0,8871 →
#: 0,8816, hiçbir ölçüt anlamlı artmadı → kabul edilmedi (THRESHOLDS §2).
SPREAD_NATIVE_EVIDENCE = 0.40
SPREAD_LOAN_EVIDENCE = 0.12

#: Sözlükteki `donor_lang` kodu -> kaynak dil ailesi kovası.
DONOR_CODE_FAMILY: dict[str, str] = {
    # Doğu
    "ar": "arabic_persian", "arz": "arabic_persian", "apc": "arabic_persian",
    "fa": "arabic_persian", "fa-cls": "arabic_persian", "fa-ira": "arabic_persian",
    "pal": "arabic_persian", "peo": "arabic_persian", "ku": "arabic_persian",
    "kmr": "arabic_persian", "he": "arabic_persian", "arc": "arabic_persian",
    "akk": "arabic_persian", "sem-pro": "arabic_persian",
    # Akdeniz (Grek / Latin / Ermeni)
    "el": "greek_latin", "grc": "greek_latin", "gkm": "greek_latin",
    "pnt": "greek_latin", "la": "greek_latin", "la-cla": "greek_latin",
    "la-vul": "greek_latin", "la-med": "greek_latin", "la-new": "greek_latin",
    "hy": "greek_latin",
    # Batı
    "fr": "western", "fro": "western", "frm": "western", "it": "western",
    "vec": "western", "scn": "western", "lij": "western", "en": "western",
    "enm": "western", "ang": "western", "de": "western", "goh": "western",
    "gmh": "western", "nl": "western", "dum": "western", "es": "western",
    "pt": "western", "ca": "western", "ro": "western", "ru": "western",
    "orv": "western", "uk": "western", "pl": "western", "cs": "western",
    "sk": "western", "bg": "western", "sr": "western", "sh": "western",
    "cu": "western", "sq": "western", "hu": "western", "sv": "western",
    "da": "western", "no": "western", "non": "western", "is": "western",
    "fi": "western",
}

#: **Sözlükte YÖNÜ TERS kaydedilmiş alıntılar.**
#:
#: Wiktionary bu maddeleri `origin='alıntı'` diye işaretliyor ama alıntı
#: yönü terstir: Macarca ve Sırp-Hırvatça bu sözcükleri TÜRKÇEDEN almıştır
#: (Macarcadaki Eski Türkçe katmanı iyi bilinir). Tanıklık kapısı bunları
#: süzmezse çekirdek Türkçe sözvarlığı "Batı Dilleri Alıntısı" damgalanır.
#:
#: ⚠️ Bu liste bir KESTİRME DEĞİL, son çaredir. Önce iki ilkeli ayırt edici
#: denendi ve ÖLÇÜLEREK ÇÜRÜTÜLDÜ:
#:   1. Türki diller arası yayılım — ayırmıyor: `kitap` 5 Türki dile yayılmış
#:      (pan-İslamik Arapça alıntı), `okul` yalnız 2. Aralıklar iç içe.
#:   2. Verici dilin Balkan/Macar olması — ayırmıyor: `hu`+`sh` etiketli 42
#:      satırın yarısı GERÇEK alıntıdır (`haydut` hajdúk, `soba` szoba,
#:      `tabur` tábor, `varoş` város, `çete` četa, `voyvoda`). Toptan elemek
#:      bunları kırardı.
#: Geriye indirgenemez veri hatası kalıyor; kapsamı dar ve adı adına yazılı.
KNOWN_REVERSED_LOAN_DIRECTION = frozenset({
    # Ortak Türkçe çekirdek sözvarlığı — Macarca bunları Türkçeden aldı
    "yaz", "öküz", "diz", "gece", "buzağı", "kazan", "saz", "sekmek",
    "ermek", "düş", "yapağı", "dazlak", "kendir",
    # Sırp-Hırvatça Türkçeden aldı
    "yastık", "balta", "pınar",
    # `okul` 1930'lar Türkçe türetmesidir (oku- + -l); Fransızca `école`
    # benzerliği tartışmalı bir etkidir, alıntı değildir.
    "okul",
})

#: Tanıklık süreç ömrü boyunca önbelleklenir (bkz. historical_morphology'deki
#: aynı desen); aynı kelime birçok kez sınıflandırılabiliyor.
_ATTEST_CACHE: dict[str, tuple[str | None, str]] = {}


def _attested_loan_family(word: str) -> tuple[str | None, str]:
    """Sözlük bu kelimeyi ALINTI olarak tanıklıyor mu, hangi aileden?

    ⚠️ Bu sınıf tasarımı gereği FONOTAKTİKTİR ve tamamen Türkçeleşmiş
    alıntıları göremez. Ölçüldü (387 gerçek alıntı, özel adlar elendi,
    indeksin kendisi `origin='alıntı'` + `donor_lang` diyor):

        çekimser (aile belirlenemedi)  158  %40,8   [meşru]
        doğru aile                      96  %24,8
        yanlış aile                      9  %2,3
        "ASLİ ÖZ TÜRKÇE" dedi          124  %32,0   <- ciddi hata

    `elektrik`, `makas`, `bot`, `ahır`, `maruz`, `tasa` gibi kelimeler
    fonotaktik olarak kusursuz Türkçedir; ihlal aramak onları bulamaz.
    Ama sözlük bunların alıntı olduğunu ZATEN SÖYLÜYOR. Tanıklık,
    üretilmiş tahmini ezmelidir — bu, `borrowing_detector`'daki soy
    süzgeci ve `cognates`'teki uydurma eleme ile aynı ilkedir.

    Korumalar (hepsi ölçülmüş sorunlardan geliyor):
      * Özel ad elemesi — küçük harfli cins ad sorgusu için `Aya`, `İhsan`
        gibi kayıtlar kanıt değildir (bkz. borrowing_detector'daki not).
      * Türki ata kodları — `trk-eog`, `trk-pro`, `otk`... verici dil
        değildir; süzülmezse `bardak` yine alıntı olur.
      * Eşadlılık oranı — `tin` (Türkçe "ruh" / Arapça تين "incir") gibi
        çiftlerde miras kayıt varsa ve alıntı payı %60'ın altındaysa
        tanıklık belirsizdir, hüküm fonotaktiğe bırakılır.
    """
    key = (word or "").strip().lower()
    if key in _ATTEST_CACHE:
        return _ATTEST_CACHE[key]
    if key in KNOWN_REVERSED_LOAN_DIRECTION:
        return (None, "")

    result: tuple[str | None, str] = (None, "")
    try:
        from engine.db.lexicon_index import LexiconIndex
        from engine.nlp.borrowing_chain import TURKIC_LINEAGE_CODES, language_name

        index = LexiconIndex()
        if index.exists:
            rows = index.lookup(key, languages=["tr"], limit=10) or []
            rows = [
                r
                for r in rows
                if r.get("pos") != "name"
                and not str(r.get("word") or "")[:1].isupper()
            ]
            borrowed = [
                r
                for r in rows
                if r.get("origin") == "alıntı"
                and str(r.get("donor_lang") or "") not in TURKIC_LINEAGE_CODES
            ]
            # Soy kodlu satır (otk, trk-pro, trk-eog...) VERİCİ değildir ama
            # MİRAS KANITIDIR. Yalnız süzüp yok saymak, eşadlılarda tek
            # yabancı satırın hükmü ele geçirmesine yol açıyordu.
            # Ölçüldü: `tin` -> [otk 𐱅𐰃𐰤 "soul", ar تين "fig"]. Soy satırı
            # miras sayılınca 1/(1+1)=0.5 < 0.6 ve koruma devreye giriyor,
            # hüküm fonotaktiğe kalıyor (doğru sonuç: öz Türkçe).
            inherited = [
                r
                for r in rows
                if r.get("origin") == "miras"
                or str(r.get("donor_lang") or "") in TURKIC_LINEAGE_CODES
            ]
            if borrowed and not (
                inherited and len(borrowed) / (len(borrowed) + len(inherited)) < 0.6
            ):
                donor = str(borrowed[0].get("donor_lang") or "")
                family = DONOR_CODE_FAMILY.get(donor)
                result = (
                    family or "loan_undetermined",
                    f"sözlükte alıntı olarak tanıklanmış: verici dil {language_name(donor)}",
                )
    except Exception:
        logger.debug("Alıntı tanıklığı okunamadı: %s", key, exc_info=True)

    _ATTEST_CACHE[key] = result
    return result


class LoanwordClassifier:
    @staticmethod
    def _is_native_compound(word: str) -> bool:
        """Kelime Türkçe bir bileşik mi? (ünlü uyumu istisnası için)"""
        from engine.nlp.neologism_detector import NeologismDetector

        res = NeologismDetector().detect(word)
        return bool(res and res.get("components"))

    def classify(
        self, word: str, spreading_ratio: float | None = None
    ) -> dict[str, Any]:
        """
        Kelimenin kaynak dil ailesi olasılık dağılımını hesaplar.

        :param spreading_ratio: Katman 2'den gelen çapraz Türki lehçe yayılım
            oranı (0.0–1.0). Verilirse öz Türkçe kanıtını güçlendirir/zayıflatır.
            Bu, plan dokümanının Katman 1 + Katman 2 birleşimidir.
        """
        w = (word or "").strip().lower()
        if not w:
            return {
                "word": "",
                "classification": None,
                "evidence_available": False,
                "probabilities": {},
                "phonotactic_violations": [],
                "reason": "Boş sorgu.",
            }

        # İki AYRI hesap yapılır:
        #   1) nativeness  : kelime Öz Türkçe fonotaktiğine ne kadar uyuyor?
        #   2) donor_scores: uymuyorsa hangi kaynak dil ailesini işaret ediyor?
        # Eskiden ikisi tek havuzda toplanıyordu; ihlal puanı aileler arasında
        # bölününce seyreliyor ve p_native yeniden 1.0'a dönüyordu.
        nativeness = 1.0
        donor_scores = {"arabic_persian": 0.0, "greek_latin": 0.0, "western": 0.0}
        violations: list[str] = []

        def add(families: tuple[str, ...] | str, points: float) -> None:
            fams = (families,) if isinstance(families, str) else families
            share = points / len(fams)
            for fam in fams:
                donor_scores[fam] += share

        # --- Katman 1a: söz başı ünsüz kısıtı ---
        first = w[0]
        if first in STRICT_NON_TURKIC_INITIALS:
            nativeness -= 0.55
            add(INITIAL_HINTS.get(first, ("arabic_persian",)), 5.0)
            violations.append(f"Söz başı '{first}-' Öz Türkçede bulunmaz")
        elif first in WEAK_NON_TURKIC_INITIALS:
            nativeness -= 0.25
            add(INITIAL_HINTS.get(first, ("arabic_persian",)), 2.5)
            violations.append(f"Söz başı '{first}-' Öz Türkçede alışılmadıktır")

        # --- Katman 1b: söz başı ünsüz kümesi ---
        has_cluster, cluster_reason = initial_cluster_violation(w)
        if has_cluster:
            nativeness -= 0.60
            add("western", 6.0)
            violations.append(cluster_reason)

        # --- Katman 1c: büyük ünlü uyumu ---
        # Öz Türkçe kelimeler ünlü uyumuna uyar; ihlal GÜÇLÜ alıntı kanıtıdır.
        # İSTİSNA: bileşik kelimeler (bilgi+sayar, baş+öğretmen) uyumu öğe
        # sınırında doğal olarak bozar; bu bir alıntı göstergesi değildir.
        is_compound = self._is_native_compound(w)
        if is_compound:
            violations.append("Bileşik yapı: ünlü uyumu öğe sınırında bozulur (alıntı göstergesi değil)")
        if not is_compound and not has_vowel_harmony(w):
            nativeness -= 0.50
            add(("arabic_persian", "western", "greek_latin"), 3.0)
            violations.append("Büyük ünlü uyumu ihlali")

        # --- Katman 1d: Arapça vezin ---
        arabic = match_arabic_pattern(w)
        if arabic:
            nativeness -= 0.35
            add("arabic_persian", 6.0)
            violations.append(f"Arapça {arabic}")

        # --- Katman 1e: Farsça yapım eki ---
        persian = match_suffix(w, PERSIAN_SUFFIXES)
        if persian:
            nativeness -= 0.35
            add("arabic_persian", 5.0)
            violations.append(f"Farsça {persian}")

        # --- Katman 1f: Batı dili eki ---
        western = match_suffix(w, WESTERN_SUFFIXES)
        if western:
            nativeness -= 0.50
            add("western", 6.5)
            violations.append(f"Batı dili {western}")

        # --- Katman 1g: Grek/Latin kalıbı ---
        greek = match_greek_latin_pattern(w)
        if greek:
            nativeness -= 0.25
            add("greek_latin", 4.0)
            violations.append(f"Grekçe/Latince {greek}")

        # --- Katman 2: çapraz Türki lehçe yayılımı ---
        spread_note = None
        if spreading_ratio is not None:
            if spreading_ratio >= SPREAD_NATIVE_EVIDENCE:
                nativeness += 0.35 * spreading_ratio
                spread_note = f"Geniş çapraz-lehçe yayılımı (%{spreading_ratio * 100:.0f}) öz Türkçe kanıtı"
            elif spreading_ratio <= SPREAD_LOAN_EVIDENCE:
                nativeness -= 0.15
                add(("arabic_persian", "western", "greek_latin"), 1.5)
                spread_note = f"Dar yayılım (%{spreading_ratio * 100:.0f}) alıntı göstergesi"

        p_native = round(min(1.0, max(0.02, nativeness)), 3)
        loan_mass = round(1.0 - p_native, 3)

        donor_total = sum(donor_scores.values())
        if donor_total > 0:
            probabilities = {
                k: round(loan_mass * v / donor_total, 3) for k, v in donor_scores.items()
            }
        else:
            probabilities = dict.fromkeys(donor_scores, round(loan_mass / 3, 3))
        probabilities["native"] = p_native

        # Verici aile seçimi BERABERE kalabilir: "ünlü uyumu ihlali" gibi bir
        # ihlal hangi aileyi işaret ettiğini söylemez, puanı üçe eşit dağıtır
        # (`kitap`, `kalem`: 1.5 / 1.5 / 1.5). `max()` bu durumda sözlükteki
        # İLK anahtarı -- arabic_persian -- döndürüyordu; rastgele bir seçim
        # "Arapça / Farsça Alıntısı" diye kesin bir hüküm gibi sunuluyordu.
        # --- Tanıklık, fonotaktik tahmini ezer --------------------------------
        # Sözlük bu kelimeyi alıntı olarak tanıklıyorsa "Asli Öz Türkçe"
        # hükmü VERİLEMEZ; fonotaktik ihlal bulamamış olması kelimenin
        # Türkçeleşmiş olduğunu gösterir, öz Türkçe olduğunu değil.
        attested_family, attest_note = _attested_loan_family(w)

        if attested_family is not None:
            best = attested_family
            # Olasılık dağılımı hükümle ÇELİŞMEMELİ: aksi hâlde kullanıcı
            # "Alıntı" hükmünün yanında "Öz Türkçe: %100" görür. (Aynı sınıf
            # çelişki d408f06'da başlık/karar arasında düzeltilmişti.)
            # ⚠️ Burada önce SERT KELEPÇE vardı (`min(p_native, 0.15)`) ve
            # sıralamayı TERSİNE ÇEVİRİYORDU: geniş lehçe yayılımı p_native'i
            # eşiğin üstüne çıkarıp kelepçeye sokuyor (0.15), dar yayılım ise
            # eşiğin altında kalıp dokunulmadan geçiyordu (0.50). Sonuç:
            # "geniş yayılım öz Türkçe kanıtıdır" varsayımı bozuluyordu.
            # Oranlı düşürme sıralamayı korur ve hükümle de çelişmez.
            p_native = round(p_native * 0.3, 3)
            loan_mass = round(1.0 - p_native, 3)
            probabilities["native"] = p_native
            for fam_key in donor_scores:
                probabilities[fam_key] = (
                    round(loan_mass if fam_key == best else 0.0, 3)
                    if best in donor_scores
                    else round(loan_mass / 3, 3)
                )
        elif p_native >= NATIVE_THRESHOLD or donor_total <= 0:
            best = "native"
        else:
            ranked_families = sorted(donor_scores.items(), key=lambda kv: kv[1], reverse=True)
            best = ranked_families[0][0]
            if len(ranked_families) > 1 and abs(ranked_families[0][1] - ranked_families[1][1]) < 1e-9:
                best = "loan_undetermined"

        return {
            "word": w,
            "classification": CLASSIFICATION_LABELS[best],
            "classification_key": best,
            "evidence_available": True,
            "probabilities": {
                "p_native_turkic": probabilities["native"],
                "p_arabic_persian": probabilities["arabic_persian"],
                "p_greek_latin": probabilities["greek_latin"],
                "p_western": probabilities["western"],
            },
            "phonotactic_violations": violations,
            "cross_dialect_note": spread_note,
            "attestation_note": attest_note,
        }
