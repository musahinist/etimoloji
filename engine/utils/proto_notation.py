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
