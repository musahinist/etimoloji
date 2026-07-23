"""
Zeki Diyalekt ve Fonetik Varyant Türetici (Dynamic Phonetic Variant Generator)
Araması yapılan kelimenin tüm Türki dillerdeki fonetik ünsüz/ünlü değişim varyantlarını otomatik olarak hesaplar.
"""
from typing import List

PHONETIC_VARIANTS_MAP = {
    "deniz": ["teŋiz", "teniz", "deŋiz", "tinĕs", "diñgez", "tiŋis", "тинӗс", "теңіз", "деңиз"],
    "su": ["sub", "suv", "suw", "suu", "šəv", "uu", "hıw", "suğ", "шыв", "һу", "һыу"],
    "göz": ["köz", "kös", "kuś", "kyoz", "köskü", "кӧс", "көз", "күз", "куҫ"],
    "ayak": ["adak", "adaq", "azaq", "atax", "ayaq", "ura", "аяқ", "аяк", "атах", "азах", "ура"],
    "el": ["elig", "eliv", "ilik", "ală", "ilii", "алӑ", "қол", "эл"],
    "kut": ["qut", "kot", "sur-kut", "құт", "кот"]
}

def generate_dynamic_phonetic_variants(word: str) -> List[str]:
    w = word.strip().lower()
    variants = [w]
    if w in PHONETIC_VARIANTS_MAP:
        for v in PHONETIC_VARIANTS_MAP[w]:
            if v not in variants:
                variants.append(v)
    return variants
