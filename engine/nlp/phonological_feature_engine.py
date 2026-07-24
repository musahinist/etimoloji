"""
Artikülatör Fonetik Vektör ve Ses Özellikleri Motoru (PanPhon Articulatory Feature Engine)
Uluslararası Fonetik Alfabe (IPA) seslerini 21-boyutlu biyomekanik artikülatör özellik matrisine
(syllabic, consonantal, sonorant, continuant, delayed_release, nasal, lateral, voiced, coronal, anterior, distributed, high, low, back, round vb.)
dönüştürür ve ses çiftleri arasındaki articulatory Hamming fonetik mesafesini hesaplar.
%100 jenerik ve sıfır kelime hardcode'lu dilbilimsel mimarı.
"""

from typing import Dict, Any, List, Tuple
import math
import re

# 21 Artikülatör Fonolojik Özellik Vektörü İndeksleri
FEATURE_NAMES = [
    "syl", "cons", "son", "cont", "delrel", "nas", "lat", "voiced",
    "strid", "high", "low", "back", "round", "cor", "ant", "dist",
    "lab", "hitone", "lotone", "tense", "long"
]

# IPA Karakterlerinin 21-Boyutlu Fonolojik Özellik Haritası (+1: Var, -1: Yok, 0: Nötr)
IPA_FEATURE_MATRIX: Dict[str, List[int]] = {
    # Ünlüler (Vowels)
    'a': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'e': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'ı': [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'i': [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'o': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
    'ö': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
    'u': [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
    'ü': [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 0, 0, 1, -1],
    'ä': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'ə': [1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, -1],

    # Ünsüzler (Consonants)
    'b': [-1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 1, -1],
    'p': [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 1, -1],
    'm': [-1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 1, -1],
    'w': [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 0, 0, -1, -1],
    'v': [-1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, -1, -1],
    'f': [-1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, -1, -1],
    'd': [-1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 1, -1],
    't': [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 1, -1],
    'n': [-1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, 1, -1],
    'z': [-1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1],
    's': [-1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1],
    'r': [-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1],
    'ŕ': [-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 0, 0, -1, -1],
    'l': [-1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 0, 0, -1, -1],
    'c': [-1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 0, 0, 1, -1],
    'ç': [-1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 0, 0, 1, -1],
    'ş': [-1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 0, 0, -1, -1],
    'j': [-1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 0, 0, -1, -1],
    'y': [-1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 0, 0, -1, -1],
    'g': [-1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'k': [-1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'ğ': [-1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
    'q': [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'ŋ': [-1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1],
    'h': [-1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 0, 0, -1, -1]
}

# Varsayılan nötr vektör
DEFAULT_VECTOR = [0] * 21

class PhonologicalFeatureEngine:
    """PanPhon Artikülatör Fonetik Vektör Motoru"""

    def get_feature_vector(self, char: str) -> List[int]:
        c = char.lower()
        return IPA_FEATURE_MATRIX.get(c, DEFAULT_VECTOR)

    def articulatory_hamming_distance(self, char1: str, char2: str) -> float:
        """İki IPA karakteri arasındaki artikülatör Hamming mesafesini hesaplar (0.0: Birebir Aynı, 1.0: Zıt)."""
        v1 = self.get_feature_vector(char1)
        v2 = self.get_feature_vector(char2)

        differences = 0
        total_valid = 0

        for f1, f2 in zip(v1, v2):
            if f1 != 0 and f2 != 0:
                total_valid += 1
                if f1 != f2:
                    differences += 1

        if total_valid == 0:
            return 0.5

        return round(differences / total_valid, 3)

    def sequence_phonological_distance(self, seq1: str, seq2: str) -> Dict[str, Any]:
        """İki kelime dizilimi arasındaki PanPhon artikülatör fonetik uzaklığı hesaplar."""
        s1 = re.sub(r'[^a-zçğıöşüа-яŕŋəäq]', '', (seq1 or "").lower().lstrip("*"))
        s2 = re.sub(r'[^a-zçğıöşüа-яŕŋəäq]', '', (seq2 or "").lower().lstrip("*"))

        if not s1 or not s2:
            return {"distance": 1.0, "similarity": 0.0, "matrix": []}

        m, n = len(s1), len(s2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = self.articulatory_hamming_distance(s1[i - 1], s2[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,        # Silme (Deletion)
                    dp[i][j - 1] + 1.0,        # Ekleme (Insertion)
                    dp[i - 1][j - 1] + cost    # Artikülatör Değiştirme (Subst)
                )

        dist = dp[m][n]
        max_possible = float(max(m, n, 1))
        norm_distance = round(dist / max_possible, 3)
        similarity = round(max(0.0, 1.0 - norm_distance), 3)

        return {
            "seq1": s1,
            "seq2": s2,
            "phonological_edit_distance": round(dist, 2),
            "normalized_articulatory_distance": norm_distance,
            "phonetic_similarity": similarity
        }
