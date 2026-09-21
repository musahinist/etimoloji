"""
Türki Diller Otomatik Alfabe ve Transkripsiyon Çevirici (Transliteration Engine)
Kiril, Arap ve Orhun Göktürk Alfabesini Latin Fonetik Okunuşuna Çevirir.
"""
import re

CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q", "л": "l", "м": "m",
    "н": "n", "ң": "ñ", "о": "o", "ө": "ö", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ү": "ü", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ç", "ш": "ş",
    "щ": "şç", "ъ": "", "ы": "ı", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ә": "ə", "і": "i", "ҫ": "ś", "ӳ": "ü", "ӑ": "ă", "ӗ": "ĕ"
}

ARABIC_TO_LATIN = {
    "ا": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "c", "چ": "ç", "ح": "h",
    "خ": "h", "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "j", "س": "s", "ش": "ş",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "ğ", "ف": "f", "ق": "q",
    "ک": "k", "گ": "g", "ڭ": "ñ", "ل": "l", "م": "m", "ن": "n", "و": "v", "ه": "h",
    "ی": "y", "ي": "y", "ئ": "e", "ە": "e", "ۇ": "u", "ۈ": "ü", "ۆ": "ö", "ى": "ı",
    # --- Eksik harfler (Faz 4'te eklendi) ---------------------------------
    # ⚠️ Tabloda yalnız Farsça kef (ک U+06A9) vardı; ARAP kefi (ك U+0643) yoktu.
    # Uygurca ve Çağatayca dökümlerinin neredeyse tamamı bu harfle yazılır ve
    # çeviriyazıdan sonra söz başı ünsüz kayboluyordu: 'كۆز' -> 'كöz' -> 'öz'.
    # Ölçüldü: 4.215 Uygurca kaydın yalnız 114'ü indekslenebiliyordu.
    "ك": "k",   # ARAP kefi — en kritik eksik
    "ۋ": "w",   # Uygurca we
    "ھ": "h",   # heh doachashmee
    "ې": "e",   # Uygurca e
    "ۅ": "ö", "ۉ": "ü", "ۊ": "u",
    "آ": "a", "أ": "a", "إ": "i", "ؤ": "ü", "ء": "",
    "ة": "e", "ﻻ": "la", "ﷲ": "allah",
    "ـ": "",    # tatwil (uzatma çizgisi) — harf değil
    # Harekeler: sesbirim değil, okuma yardımıdır.
    "\u064b": "", "\u064c": "", "\u064d": "", "\u064e": "a", "\u064f": "u",
    "\u0650": "i", "\u0651": "", "\u0652": "", "\u0653": "", "\u0670": "a",
}

#: Orhun–Yenisey (Eski Türk) yazısı — Unicode bloğu U+10C00–U+10C48.
#:
#: ⚠️ Bu tablo YOKTU. Modül başlığı "Orhun Göktürk Alfabesini çevirir" diyor,
#: README "Kiril/Arap/Runik" diyor, `lexicon_index.py:393` yorumu "Arap veya
#: Orhun yazısı" diyordu — ama runik metin aşağıdaki "zaten Latin mi" testini
#: geçemediği için dokunulmadan geri dönüyordu. Sonuç: `to_comparison_form`
#: boş dönüyor ve kayıt indekse HİÇ girmiyordu. Ölçüldü: Eski Türkçe
#: dökümünün 470 kaydından yalnız 1'i (Latin harfli bir "romanization"
#: satırı) indekslenebilmişti.
#:
#: Harf adları sesi kodlar: "ORKHON AB" ile "ORKHON AEB" aynı /b/ sesinin
#: art/ön ünlü değişkeleridir, ikisi de `b`'ye çöker; Yenisey değişkeleri de
#: Orhun karşılıklarıyla birleşir.
#: ⚠️ Çıktı `to_comparison_form`'un hedef alfabesinde kalmalıdır
#: (`a-zçğıöşüŋŕĺ`): geniz n'si için `ñ` DEĞİL `ŋ` kullanılır — `ñ` o
#: süzgeçte silinir ve ses tamamen kaybolurdu.
#: ⚠️ BU TABLO SÖZLÜKSEL ARAMA İÇİN YETERLİ DEĞİLDİR — ölçüldü, kurtarılamaz.
#:
#: Çeviriyazı runik kayıtları indekslenebilir yapar (1/470 -> 470/470) ama
#: üretilen biçimler Latin köklerle EŞLEŞMEZ::
#:
#:     𐰋𐰃𐱅𐰃 -> 'bıtı'   (biti- "yazmak")     𐰋𐰃𐰼 -> 'bır'  (bir)
#:     𐱅𐰭𐰼𐰃 -> 'tŋrı'   (teŋri)              𐰚𐰃𐰾𐰃 -> 'kısı' (kişi)
#:
#: Sebep tabloda değil KAYNAK YAZIDA:
#:   * `U+10C03 ORKHON I` tek işaret olarak hem `i` hem `ı` içindir; tabloda
#:     `i` ve `u`/`ü` hiç yoktur. Ölçüldü: otk'nin 470 kaydında `ı` geçen 92,
#:     `i` geçen 1 — bir dil için imkânsız bir oran.
#:   * Ünsüz işaretleri `AEB`, `AET`, `AER`, `AEY`, `AEK`, `AES` diye
#:     adlandırılmıştır: "AE" hem A hem E bağlamı demektir, yani ön/art
#:     uyumunu TAŞIMAZLAR. Ünsüz niteliği de kaybolur (`kısı` <- `kişi`,
#:     AES işareti `ş` sesini de karşılıyor).
#:   * Orhun yazısı ünlülerin çoğunu hiç yazmaz: alfabe dışı 397 otk
#:     kaydının 150'si (%38) iki harf veya daha kısa — `tg` (tağ "dağ"),
#:     `lg` (elig "el"), `sç` (saç), `dg` (adıg "ayı"), `bş` (beş).
#:
#: **Uyum-duyarlı çeviriyazı DENENDİ ve ÖLÇÜLEREK ÇÜRÜTÜLDÜ**: ünsüz
#: serisinden ünlü niteliği türetilebilseydi iş görürdü, ama yukarıdaki
#: Unicode adları o bilginin kod noktalarında bulunmadığını gösteriyor.
#: Tek gerçek çözüm bilimsel bir edisyondan KÜRATÖRLÜ LATİN OKUMA eklemek,
#: yani yeni veri. Bu yüzden Eski Türkçe fiil kökleri (`biti-` gibi) Latin
#: kökle aranamaz; bkz. `nlp/historical_morphology.py` fiil kapısı notu.
OLD_TURKIC_TO_LATIN = {
    # --- Ünlüler ---
    "\U00010C00": "a", "\U00010C01": "a", "\U00010C02": "e",
    "\U00010C03": "ı", "\U00010C04": "ı", "\U00010C05": "e",
    "\U00010C06": "o", "\U00010C07": "ö", "\U00010C08": "ö",
    # --- Ünsüzler: art/ön değişkeler aynı sese çöker ---
    "\U00010C09": "b", "\U00010C0A": "b", "\U00010C0B": "b", "\U00010C0C": "b",
    "\U00010C0D": "g", "\U00010C0E": "g", "\U00010C0F": "g", "\U00010C10": "g",
    "\U00010C11": "d", "\U00010C12": "d", "\U00010C13": "d",
    "\U00010C14": "z", "\U00010C15": "z",
    "\U00010C16": "y", "\U00010C17": "y", "\U00010C18": "y", "\U00010C19": "y",
    "\U00010C1A": "k", "\U00010C1B": "k", "\U00010C1C": "k", "\U00010C1D": "k",
    "\U00010C1E": "l", "\U00010C1F": "l", "\U00010C20": "l",
    "\U00010C22": "m",
    "\U00010C23": "n", "\U00010C24": "n", "\U00010C25": "n",
    "\U00010C2C": "ŋ", "\U00010C2D": "ŋ", "\U00010C2E": "ŋ",
    "\U00010C2F": "p", "\U00010C30": "p",
    "\U00010C31": "ç", "\U00010C32": "ç", "\U00010C33": "ç",
    "\U00010C34": "q", "\U00010C35": "q", "\U00010C36": "q",
    "\U00010C37": "q", "\U00010C38": "q", "\U00010C39": "q",
    "\U00010C3A": "r", "\U00010C3B": "r", "\U00010C3C": "r",
    "\U00010C3D": "s", "\U00010C3E": "s",
    "\U00010C3F": "ş", "\U00010C40": "ş", "\U00010C41": "ş", "\U00010C42": "ş",
    "\U00010C43": "t", "\U00010C44": "t", "\U00010C45": "t",
    "\U00010C46": "t", "\U00010C47": "t",
    # --- Küme harfleri: tek işaret, birden çok ses ---
    "\U00010C21": "lt",                        # ORKHON ELT
    "\U00010C26": "nt", "\U00010C27": "nt",    # ENT
    "\U00010C28": "nç", "\U00010C29": "nç",    # ENC
    "\U00010C2A": "ny", "\U00010C2B": "ny",    # ENY
    "\U00010C48": "baş",                       # ORKHON BASH (hece işareti)
}


def transliterate_to_latin(text: str) -> str:
    if not text:
        return text

    # Runik metin aşağıdaki Kiril/Arap testinden geçemez ve "zaten Latin"
    # sayılıp değişmeden dönerdi; bu yüzden ÖNCE burada ele alınır.
    if any(ch in OLD_TURKIC_TO_LATIN for ch in text):
        return "".join(OLD_TURKIC_TO_LATIN.get(ch, ch) for ch in text)

    # Eğer zaten Latin harfleri ağırlıklıysa dokunma
    if not re.search(r'[\u0400-\u04FF\u0600-\u06FF]', text):
        return text

    res = []
    for ch in text:
        ch_lower = ch.lower()
        if ch_lower in CYRILLIC_TO_LATIN:
            trans = CYRILLIC_TO_LATIN[ch_lower]
            res.append(trans.upper() if ch.isupper() else trans)
        elif ch_lower in ARABIC_TO_LATIN:
            res.append(ARABIC_TO_LATIN[ch_lower])
        elif ch in OLD_TURKIC_TO_LATIN:
            # Karışık yazılı metinler için (ör. runik + Kiril açıklama).
            res.append(OLD_TURKIC_TO_LATIN[ch])
        else:
            res.append(ch)

    return "".join(res)
