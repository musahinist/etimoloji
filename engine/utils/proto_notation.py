"""Ata biçim gösterimini karşılaştırılabilir hâle getirme.

``evaluation.metrics`` bunu ölçüm için, ``column_model`` üretimde
kullanır; üretim kodunun ``evaluation``'ı içe aktarmaması için burada durur.
``evaluation.metrics`` aynı adları buradan yeniden dışa verir.
"""

from __future__ import annotations

import unicodedata

#: Rekonstrüksiyon karşılaştırmasında yok sayılan işaretler. Ata biçim
#: gösteriminde ``*`` yalnızca "bu bir rekonstrüksiyondur" demektir, sesin
#: parçası değildir; parantez ise belirsizlik/isteğe bağlılık işaretidir.
#: ⚠️ Tire de atılır. Kaynaklar biçimbirim sınırını tireyle işaretler
#: (``*ḳalï-``, ``*āt-la-``), motor ise hiçbir zaman tire üretmez; tire
#: fonolojik içerik değil **yazım kuralıdır** ve karşılaştırmada kalırsa
#: doğru bir rekonstrüksiyonu yapısı gereği tutturulamaz kılar.
#:
#: Etkisi ölçüldü (savelyevturkic, dev, n=83): NED 0,306 -> 0,304,
#: ED 1,4819 -> 1,4699, FER 0,2627 -> 0,2606, tam 0,3855 -> 0,3976.
#: Hiçbir ölçüt kötüleşmiyor. Küçük ama GERÇEK bir taban değişikliğidir:
#: savelyev'de tire içeren altın madde yalnız 1/400 olduğu için etki
#: tek maddeliktir.
STRIP_CHARS = "*()[]{}?-"

#: **Salt yazım geleneği farkları.** Aynı sesi farklı yazan okulların
#: uzlaştırılması. Bunlar EXACT ölçümde de eşitlenir: ``*yol`` ile ``*jol``
#: aynı rekonstrüksiyondur, farklı yazımdır. Bunları hata saymak dilbilimi
#: değil, çeviriyazı geleneğini ölçmek olurdu.
#:
#: Ölçüldü: "söz başı yanlış" sayılan 93 hatanın 31'i yalnız buydu
#: (``y``/``j`` 18, ``c``/``č`` 5, ``ı``/``ï`` 3, ``k``/``ḳ`` 2 …).
TRANSCRIPTION_VARIANTS: dict[str, str] = {
    "ï": "ı", "ɨ": "ı", "ɯ": "ı",
    "y": "j", "ǰ": "j", "ɟ": "j",
    "č": "ç", "c": "ç", "ʧ": "ç",
    "š": "ş", "ʃ": "ş",
    "ñ": "ŋ", "ń": "ŋ", "ṅ": "ŋ",
    "ḳ": "k", "q": "k", "ḵ": "k",
    "ġ": "g", "ǧ": "ğ", "ɣ": "ğ",
    "ẹ": "e", "ė": "e", "ạ": "a", "ǝ": "e", "ə": "e",
    "ẓ": "z", "ṣ": "s", "ṭ": "t", "ḏ": "d",
    "ʼ": "", "ʔ": "", "ʲ": "", "ˊ": "", "'": "", "ʻ": "",
}


def fold_transcription(text: str) -> str:
    return "".join(TRANSCRIPTION_VARIANTS.get(ch, ch) for ch in text)


def normalize_proto(form: str, *, strip_length: bool = False) -> str:
    """Ata biçmi karşılaştırılabilir hâle getirir.

    ``*Kāpuk`` -> ``kāpuk``. ``strip_length=True`` ise uzunluk da atılır
    (``kapuk``) — "uzunluk dışında doğru mu?" sorusunu ayrıca ölçmek için.

    Yazım geleneği farkları (:data:`TRANSCRIPTION_VARIANTS`) burada eşitlenir.
    ⚠️ ``ŕ`` ve ``ĺ`` **eşitlenmez**: Türkolojide bunlar ``r``/``l``den ayrı
    sesbirimlerdir; birleştirmek ``*ar`` ile ``*aŕ``ı aynı sayardı.
    """
    text = unicodedata.normalize("NFC", form.strip())
    for ch in STRIP_CHARS:
        text = text.replace(ch, "")
    text = text.split(",")[0].split("/")[0].strip()  # "*Kūrɨk,gak" -> "*Kūrɨk"
    text = fold_transcription(text.casefold())
    if strip_length:
        decomposed = unicodedata.normalize("NFD", text)
        text = unicodedata.normalize("NFC", "".join(c for c in decomposed if c != "̄")).replace("ː", "")
    return text


# --- Gelenekten bağımsız eşdeğerlik (Wiktionary ↔ Starling/EDAL) ----------------
#
# Aynı Proto-Türkçe kök iki yayın geleneğinde farklı yazılır. Tablo VERİYE
# BAKILMADAN, yalnız iki geleneğin kendi gösterim tanımlarından yazıldı:
#
# * Wiktionary, "Proto-Turkic entry guidelines" (WT:ATRK; eski adı
#   "About Proto-Turkic"), Transcription bölümü, 2026-09 sürümü:
#   trk-pro ünsüzleri *nʸ, *lᶴ, *rᶻ, *y; trk-cmn-pro'da aynı sesler *š, *z;
#   *d (~ *ð); söz başı *d-/*g- "normalde olmaz", yalnız Oğuz'a özgü sözde
#   (Oğuz söz başı ötümlüleşmesi) yazılır; "*ạ kullanılmaz"; ünlü uzunluğu
#   "sorunlu", farklı uzunluklu biçimler "alternatif rekonstrüksiyon" sayılır;
#   örnek denklik: *bẹńi (EDAL) = *benʸi (Wiktionary).
# * Starostin, Dybo & Mudrak 2003, *EDAL*, Proto-Türkçe gösterimi (Starling
#   turcet aynı gösterimi kullanır): *ŕ, *ĺ, *ń, *j (= y), *ǯ, *ẹ (kapalı e),
#   *ä, *ɨ, uzun ünlü makronla, kısa/yarı uzun ünlü breve ile (*ă);
#   söz başı *d- Oğuz d-'sinden rekonstrükte edilir (*dāt-, Wiktionary *tāt-).
# * Söz başı ötümlü/ötümsüz patlayıcı tartışması: Doerfer (1971, "Khalaj
#   Materials"; 1975) Oğuz d-/g-'yi ata ses sayar, Tekin (1979, "Once more on
#   Proto-Turkic initial voiced stops") ve Clauson (1972) *t-/*k- yazar.
#   *b-/*p- iki gelenekte de *b- olduğu için tabloda YOKTUR.
# * *ŕ/*z ve *ĺ/*š: Tekin 1969 ("Zetacism and sigmatism in Proto-Turkic"),
#   Johanson 1998 ("The history of Turkic"): ŕ/ĺ ata sesi Ortak Türkçede
#   z/š olur; Wiktionary trk-cmn-pro başlıkları *z/*š yazar.
#
# ⚠️ Bu anahtar ölçümde "aynı kök, farklı gösterim" sorusunu cevaplar; ``exact``
# ve ``acceptable``in YERİNE geçmez. *ŕ'yi *z'ye katladığı için *tuz ~ *tūŕ'u
# eşit sayar — bu, iki geleneğin aynı kökü yazma biçimidir (Wiktionary
# trk-cmn-pro *tuz = EDAL *tūŕ); rotasizmi kaçırma hatası ölçülmek isteniyorsa
# ``acceptable`` kullanılır.

#: Çok karakterli Wiktionary gösterimlerinin EDAL karşılığı (önce uygulanır).
TRADITION_DIGRAPHS: dict[str, str] = {
    "nʸ": "ń", "lᶴ": "ĺ", "rᶻ": "ŕ",   # Wiktionary trk-pro ↔ EDAL
    "r₂": "ŕ", "l₂": "ĺ",               # eski Türkoloji r²/l² gösterimi
}

#: Karakter denklikleri (``normalize_proto`` katlamasından SONRA uygulanır;
#: ï/ı, y/j, č/ç, š/ş, ẹ/e, ñ/ń/ŋ, ḳ/k orada zaten eşitlenmiştir).
TRADITION_EQUIVALENTS: dict[str, str] = {
    "ŕ": "z",   # EDAL *ŕ = Wiktionary *rᶻ / trk-cmn-pro *z
    "ĺ": "ş",   # EDAL *ĺ = Wiktionary *lᶴ / trk-cmn-pro *š
    "ä": "e",   # EDAL/Räsänen *ä ~ Wiktionary *e (bkz. *bänʸi ~ *benʸi)
    "δ": "d", "ð": "d",   # Wiktionary *d (~ *ð)
    "ǯ": "j", "ʒ": "j",   # EDAL *ǯ ~ Wiktionary (*ǰ) / *y
}

#: Söz başında gelenek farkı olan patlayıcılar (yalnız ilk ses).
TRADITION_INITIAL: dict[str, str] = {"d": "t", "g": "k"}

_VOWELS = set("aeıioöuüäẹɨïāēīōūǟȫǖ")


def tradition_key(form: str) -> str:
    """İki geleneğin aynı kökü aynı anahtara düşsün diye katlanmış biçim.

    ``*tāt-`` / ``*dāt-`` -> ``tat``; ``*tāĺ`` / ``*tāš`` -> ``taş``;
    ``*ăl`` / ``*āl`` -> ``al``; ``*benʸi`` / ``*bẹńi`` -> ``beŋi``.
    """
    text = unicodedata.normalize("NFC", form.strip())
    for src, dst in TRADITION_DIGRAPHS.items():
        text = text.replace(src, dst)
    text = normalize_proto(text, strip_length=True)
    # Breve (kısa/yarı uzun ünlü işareti) yalnız ÜNLÜDEN atılır: ğ'yi koru.
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", text):
        if ch == "̆" and out and out[-1] in _VOWELS:
            continue
        out.append(ch)
    text = unicodedata.normalize("NFC", "".join(out)).replace(":", "")
    text = "".join(TRADITION_EQUIVALENTS.get(ch, ch) for ch in text)
    if text:
        text = TRADITION_INITIAL.get(text[0], text[0]) + text[1:]
    return text


def same_root_across_traditions(a: str, b: str) -> bool:
    """``a`` ve ``b`` gösterim geleneği dışında aynı kök mü?"""
    ka, kb = tradition_key(a), tradition_key(b)
    return bool(ka) and ka == kb
