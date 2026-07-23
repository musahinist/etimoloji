"""
Fonetik Ses Kayması ve Dilbilimsel Dönüşüm Analizcisi (Phonetic Shift Detector)
Türkiye Türkçesindeki kelime ile Türki dillerdeki akraba kelimeler arasındaki ses dönüşüm kurallarını tespit eder.
"""
from typing import List, Dict

def analyze_phonetic_shifts(source_word: str, target_word: str, lang_name: str) -> str:
    s = source_word.strip().lower()
    t = target_word.strip().lower()

    explanations = []

    # 1. Baş ünsüz ötümlüleşme/ötümsüzleşme (g->k, d->t, b->p)
    if s.startswith("g") and (t.startswith("k") or t.startswith("к")):
        explanations.append("Baştaki ötümlü 'g-' konsonantının ötümsüz 'k-' sesine dönüşmesi (Oğuz -> Kıpçak/Sibirya ses kuralı)")
    elif s.startswith("d") and (t.startswith("t") or t.startswith("т")):
        explanations.append("Baştaki 'd-' sesinin 't-' biçimine sertleşmesi")
    elif s.startswith("b") and (t.startswith("m") or t.startswith("м")):
        explanations.append("Baştaki 'b-' ünsüzünün genizsilleşerek 'm-' sesine dönüşmesi (ben -> men)")

    # 2. Sızıcılaşma (z -> s / ś)
    if s.endswith("z") and (t.endswith("s") or t.endswith("с") or t.endswith("ҫ") or t.endswith("ś")):
        explanations.append("Sondaki '-z' ünsüzünün sızıcı '-s / -ś' sesine dönüşmesi (Proto-Turkic r-z denkliği)")

    # 3. Çuvaşça / Oğur s- -> ş- kayması
    if "Çuvaş" in lang_name and s.startswith("s") and (t.startswith("ш") or t.startswith("š")):
        explanations.append("Oğur/Çuvaş koluna özgü baştaki 's-' ünsüzünün 'ş-' (š-) sesine kayması")

    if explanations:
        return "; ".join(explanations)
    return "Standart Lehçe Ses Uyumu"
