"""
Türki Diller Morfolojik Analizör ve Kök Ayrıştırıcı (Turkic Stemmer & Morphological Analyzer)
Karmaşık türemiş ve çekim ekli kelimeleri köklerine ayrıştırır.
"""
import re
from typing import List, Tuple

# Türkçe ve Türki diller yapım ve çekim ekleri
TURKIC_SUFFIXES = [
    # İsimden İsim ve Fiilden İsim Yapım Ekleri
    "cılık", "cilik", "çılık", "çilik", "culuk", "cülük",
    "daş", "deş", "taş", "teş",
    "lik", "lık", "luk", "lük", "ci", "cı", "cu", "cü", "çi", "çı",
    "li", "lı", "lu", "lü",
    "siz", "sız", "suz", "süz",
    "gi", "gı", "gu", "gü", "ki", "kı", "ku", "kü",
    "sel", "sal", "gil",
    "ler", "lar", "dan", "den", "tan", "ten",
    "daki", "deki", "taki", "teki",
    "mak", "mek", "ma", "me", "ış", "iş", "uş", "üş"
]

def analyze_morphology(word: str) -> Tuple[str, List[str]]:
    w = word.strip().lower()
    detected_suffixes = []
    stem = w

    for suf in TURKIC_SUFFIXES:
        if stem.endswith(suf) and len(stem) - len(suf) >= 2:
            detected_suffixes.append("-" + suf)
            stem = stem[:-len(suf)]

    return stem, detected_suffixes
