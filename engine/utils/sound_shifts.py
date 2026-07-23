"""
Türki Diller Fonetik ve Ses Denkliği Matrisi (Turkic Sound Shift & Cognate Generator)
Tüm Türki dil kolları (Oğuz, Kıpçak, Karluk, Sibirya, Oğur/Çuvaş) için otomatik türev ve ses kayması alternatifi üretir.
"""
import re
from typing import List, Set

# Türki diller arası genel fonetik denklik haritası
SOUND_SHIFT_RULES = [
    # 1. Baş ünsüz değişimleri (d~t, b~m~p~v, g~k)
    (r'^d', 't'), (r'^t', 'd'),
    (r'^b', 'm'), (r'^b', 'p'), (r'^b', 'v'),
    (r'^g', 'k'), (r'^k', 'g'),
    (r'^y', 'c'), (r'^y', 'j'), (r'^y', 'zh'),

    # 2. Orta/Son ünsüz ve ünlü denkliği (z~s~ş~r, g~ğ~w~v, e~i~ä~ə)
    (r'z$', 's'), (r'z$', 'ş'), (r'z$', 'r'), (r'z$', 'z'),
    (r'e', 'i'), (r'i', 'e'), (r'e', 'ə'), (r'e', 'ä'),
    (r'o', 'u'), (r'u', 'o'), (r'ö', 'ü'), (r'ü', 'ö'),
    (r'g', 'ğ'), (r'g', 'w'), (r'g', 'v')
]

CYRILLIC_TRANSCRIPTIONS = {
    "a": "а", "b": "б", "c": "дж", "ç": "ч", "d": "д", "e": "е", "f": "ф", "g": "г", "ğ": "ғ",
    "h": "х", "ı": "ы", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "ñ": "ң",
    "o": "о", "ö": "ө", "p": "п", "r": "р", "s": "с", "ş": "ш", "t": "т", "u": "у", "ü": "ү",
    "v": "в", "y": "й", "z": "з", "ə": "ә", "ä": "ә"
}

def generate_turkic_cognate_candidates(word: str) -> List[str]:
    w = word.strip().lower()
    candidates: Set[str] = {w}

    # 1. Ek ve takı varyasyonları (ge/gi/gü/gə/g/k/ğ/q/lık/lik/lik/luk)
    if w.endswith("gi") or w.endswith("ge"):
        stem = w[:-2]
        candidates.update([stem + "g", stem + "ğ", stem + "gі", stem + "gə", stem + "gi", stem + "ge", stem + "gü", stem + "im", stem + "ik", stem + "ig", stem + "iğ"])
    elif w.endswith("lik") or w.endswith("lık"):
        stem = w[:-3]
        candidates.update([stem + "lıq", stem + "lik", stem + "ліқ", stem + "лык", stem + "лық"])

    # 2. Fonetik kural denklikleri
    for pat, rep in SOUND_SHIFT_RULES:
        if re.search(pat, w):
            mod = re.sub(pat, rep, w)
            candidates.add(mod)

    # 3. Kiril ve Arap alfabesi transkripsiyonu
    base_set = list(candidates)
    for item in base_set:
        cyr = "".join([CYRILLIC_TRANSCRIPTIONS.get(ch, ch) for ch in item])
        candidates.add(cyr)

    # 4. Yaygın kök matrisi
    if "belg" in w or "bəlg" in w or "bilg" in w:
        candidates.update(["белгі", "белги", "билге", "билдә", "bəlgə", "belgi", "паллӑ", "бэлиэ", "білім", "билим", "bilim", "bilik", "biliğ", "билик", "belgü", "biliğ"])
    elif "deniz" in w or "dəniz" in w:
        candidates.update(["dəniz", "теңіз", "деңиз", "тиңез", "тиңис", "deňiz", "тинӗс", "teŋiz"])
    elif "su" in w:
        candidates.update(["suw", "су", "суу", "һыу", "шыв", "уу", "суг", "sub", "suv"])
    elif "göz" in w or "köz" in w:
        candidates.update(["köz", "kөз", "көз", "күз", "куҫ", "көс", "göŕ"])
    elif "tetik" in w:
        candidates.update(["тетік", "тетик", "تتیك", "tətik", "tät-", "tetiq"])

    return list(candidates)
