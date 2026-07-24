"""
Türki Diller Derin Akraba Kelime ve Türev Ağı (Cognate Network Builder)
Araması yapılan kelimelerin aynı kökten türeyen akraba sözcük kümesini
sıfır sabit kelime haritası ile morfolojik türetim kalıpları ve fonetik denklik matrisi üzerinden dinamik oluşturur.
"""
from typing import List
from engine.utils.sound_shifts import generate_turkic_cognate_candidates
from engine.utils.morphology import analyze_morphology, TURKIC_SUFFIXES

def get_related_cognates(word: str) -> List[str]:
    """Herhangi bir kelime için morfolojik kök türetme ekleri ve fonetik denklik varyantlarını %100 jenerik hesaplar."""
    w = (word or "").strip().lower()
    if not w:
        return []

    stem, _ = analyze_morphology(w)
    base_stem = stem or w

    cognate_set = []

    # 1. Kök üzerinden jenerik Türki diller fonetik denkliği üretimi
    turkic_phonetic_cognates = generate_turkic_cognate_candidates(base_stem)
    for c in turkic_phonetic_cognates:
        if c != w and c not in cognate_set:
            cognate_set.append(c)

    # 2. Jenerik morfolojik ek türetimi (Kök + Türk dili yapım ekleri)
    for suf in ["-lik", "-li", "-siz", "-ci", "-daş", "-sal", "-gi", "-mak"]:
        clean_suf = suf.lstrip("-")
        derived = f"{base_stem}{clean_suf}"
        if derived != w and derived not in cognate_set:
            cognate_set.append(derived)

    return cognate_set[:14]
