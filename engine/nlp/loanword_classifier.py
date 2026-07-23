"""
NLP Alıntı Kelime Sınıflandırıcısı ve Olasılık Hesaplayıcısı (NLP Loanword Classifier)
Kelimenin Öz Türkçe mi yoksa Alıntı (Loanword) mı olduğunu ve hangi dil ailesinden geçtiğini
fonotaktik ihlaller, hece yapıları ve vezin kalıpları üzerinden matematiksel olarak hesaplar.
"""
import re
from typing import Dict, Any, List

NON_TURKIC_INITIALS = {'f', 'h', 'p', 'v', 'j', 'z', 'r', 'l', 'm', 'n'}
INITIAL_CONSONANT_CLUSTERS = ['tr', 'sp', 'st', 'kr', 'pr', 'dr', 'gr', 'sk', 'str', 'fl', 'fr', 'pl', 'kl']

ARABIC_PATTERNS = [
    (r'^mu[a-z]a[a-z]{2,}$', "Mufa'al / Mufa'il Vezni"),
    (r'^ta[a-z]ī[a-z]$', "Taf'īl Vezni"),
    (r'^[a-z]i[a-z]ā[a-z]$', "Fi'āl Vezni"),
    (r'^in[a-z]i[a-z]ā[a-z]$', "Infi'āl Vezni"),
    (r'^ist[a-z]i[a-z]ā[a-z]$', "Istif'āl Vezni")
]

PERSIAN_SUFFIXES = ["gār", "stān", "khāneh", "dān", "bān", "pərvar", "šənās"]
WESTERN_SUFFIXES = ["tion", "sion", "izm", "izim", "ist", "loji", "grafi", "tik", "sek"]

class LoanwordClassifier:
    def classify(self, word: str) -> Dict[str, Any]:
        w = word.strip().lower()
        if not w:
            return {"word": "", "p_native": 1.0, "p_loan": 0.0, "classification": "Öz Türkçe"}

        score_native = 10.0
        score_arabic_persian = 0.0
        score_greek_latin = 0.0
        score_western = 0.0
        violations = []

        # 1. Söz Başı İhlali Kontrolü (Söz başı f, h, p, v, j, z, r, l ise Öz Türkçe olasılığı düşer)
        if w[0] in NON_TURKIC_INITIALS:
            score_native -= 8.5
            violations.append(f"Söz başı '{w[0]}' harfi Öz Türkçede bulunmaz")
            if w[0] in ['f', 'h', 'p', 'v']:
                score_arabic_persian += 4.0
                score_greek_latin += 4.0
            elif w[0] in ['j', 'z', 'r', 'l']:
                score_western += 5.0

        # 2. Çift Ünsüz Başlangıcı (Consonant Clusters)
        for cluster in INITIAL_CONSONANT_CLUSTERS:
            if w.startswith(cluster):
                score_native -= 9.0
                score_western += 6.0
                violations.append(f"Söz başı çift ünsüz '{cluster}' (Batı Alıntısı)")
                break

        # 3. Ünlü Uyumu Kontrolü
        vowels = [c for c in w if c in 'aeıioöuü']
        back = [c for c in vowels if c in 'aıou']
        front = [c for c in vowels if c in 'eiöü']
        if back and front:
            score_native -= 4.5
            violations.append("Büyük ünlü uyumsuzluğu")

        # 4. Arapça Vezin Kontrolü
        for pat, desc in ARABIC_PATTERNS:
            if re.search(pat, w):
                score_arabic_persian += 6.0
                violations.append(f"Arapça {desc} tespit edildi")

        # 5. Farsça Sonek Kontrolü
        for suf in PERSIAN_SUFFIXES:
            if w.endswith(suf):
                score_arabic_persian += 5.0
                violations.append(f"Farsça -{suf} eki tespit edildi")

        # 6. Batı Sonek Kontrolü
        for suf in WESTERN_SUFFIXES:
            if w.endswith(suf):
                score_western += 6.0
                violations.append(f"Batı dili -{suf} soneki tespit edildi")

        # Olasılık Dağılımını Normalize Et
        native_valid_score = max(score_native, 0.5)
        total_score = native_valid_score + score_arabic_persian + score_greek_latin + score_western

        p_native = round(native_valid_score / total_score, 3)
        p_arabic_persian = round(score_arabic_persian / total_score, 3)
        p_greek_latin = round(score_greek_latin / total_score, 3)
        p_western = round(score_western / total_score, 3)

        if p_native >= 0.65:
            classification = "Asli Öz Türkçe (Native Turkic)"
        elif p_arabic_persian >= max(p_greek_latin, p_western):
            classification = "Arapça / Farsça Alıntısı (Doğu Alıntısı)"
        elif p_greek_latin >= p_western:
            classification = "Grekçe / Bizans / Latince / Ermenice Alıntısı"
        else:
            classification = "Batı Dilleri Alıntısı (Fransızca / İngilizce)"

        return {
            "word": w,
            "classification": classification,
            "probabilities": {
                "p_native_turkic": p_native,
                "p_arabic_persian": p_arabic_persian,
                "p_greek_latin": p_greek_latin,
                "p_western": p_western
            },
            "phonotactic_violations": violations
        }
