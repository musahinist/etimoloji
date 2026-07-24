"""
CLDF & LingPy Standartlarında Hesaplamalı Fonetik Dizilim Hizalayıcısı (Phonetic Sequence Aligner)
LingPy SCA (Sound-Class-Based Alignment - Dolgopolsky Ses Sınıfları) ve Needleman-Wunsch
algoritmalarıyla ses dizilimlerini hizalar ve fonetik denklik matrisi üretir.
%100 jenerik ve sıfır kelime hardcode'lu dilbilimsel mimari.
"""

from typing import Dict, Any, List, Tuple
import re
from engine.nlp.phonological_feature_engine import PhonologicalFeatureEngine

# LingPy / Dolgopolsky Ses Sınıfları (Sound Classes - SCA Model)
DOLGOPOLSKY_SOUND_CLASSES: Dict[str, str] = {
    'a': 'V', 'e': 'V', 'ı': 'V', 'i': 'V', 'o': 'V', 'ö': 'V', 'u': 'V', 'ü': 'V', 'ä': 'V', 'ə': 'V',
    'p': 'P', 'b': 'P', 'f': 'P', 'v': 'P', 'm': 'M', 'w': 'W',
    't': 'T', 'd': 'T', 's': 'S', 'z': 'S', 'c': 'S', 'ç': 'S', 'ş': 'S', 'j': 'S', 'r': 'R', 'ŕ': 'R', 'l': 'L', 'n': 'N',
    'k': 'K', 'g': 'K', 'ğ': 'K', 'q': 'K', 'ŋ': 'N', 'h': 'H', 'y': 'Y'
}

class CldfLingPyAligner:
    """CLDF / LingPy uyumlu SCA (Sound-Class-Based Alignment) ve PanPhon fonetik dizi hizalama motoru"""

    def __init__(self):
        self.panphon_engine = PhonologicalFeatureEngine()

    def to_sound_classes(self, seq: str) -> str:
        """Kelimeyi Dolgopolsky SCA ses sınıfları dizilimine dönüştürür (örn: 'suv' -> 'VPW')."""
        clean = re.sub(r'[^a-zçğıöşüа-яŕŋəäq]', '', (seq or "").lower().lstrip("*"))
        return "".join([DOLGOPOLSKY_SOUND_CLASSES.get(ch, 'X') for ch in clean])

    def align_sequences(self, seq1: str, seq2: str) -> Dict[str, Any]:
        """Needleman-Wunsch + SCA + PanPhon artikülatör fonetik dizi hizalamasını hesaplar."""
        s1 = re.sub(r'[^a-zçğıöşüа-яŕŋəäq]', '', (seq1 or "").lower().lstrip("*"))
        s2 = re.sub(r'[^a-zçğıöşüа-яŕŋəäq]', '', (seq2 or "").lower().lstrip("*"))

        if not s1 or not s2:
            return {
                "seq1": s1,
                "seq2": s2,
                "aligned_seq1": s1,
                "aligned_seq2": s2,
                "alignment_score": 0.0,
                "phonetic_similarity": 0.0,
                "sound_class_seq1": "",
                "sound_class_seq2": "",
                "aligned_pairs": []
            }

        sc1 = self.to_sound_classes(s1)
        sc2 = self.to_sound_classes(s2)

        # PanPhon artikülatör fonetik uzaklık hesabı
        panphon_res = self.panphon_engine.sequence_phonological_distance(s1, s2)

        # LingPy SCA Needleman-Wunsch Hizalaması
        m, n = len(s1), len(s2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        match_score = 3
        class_match_score = 2
        mismatch_penalty = -1
        gap_penalty = -2

        for i in range(m + 1):
            score_matrix[i][0] = i * gap_penalty
        for j in range(n + 1):
            score_matrix[0][j] = j * gap_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char1, char2 = s1[i - 1], s2[j - 1]
                sc_char1, sc_char2 = sc1[i - 1], sc2[j - 1]

                if char1 == char2:
                    score = match_score
                elif sc_char1 == sc_char2:
                    score = class_match_score
                else:
                    # PanPhon artikülatör yakınlık skoru
                    hamm = self.panphon_engine.articulatory_hamming_distance(char1, char2)
                    score = 1 if hamm <= 0.35 else mismatch_penalty

                score_matrix[i][j] = max(
                    score_matrix[i - 1][j - 1] + score,
                    score_matrix[i - 1][j] + gap_penalty,
                    score_matrix[i][j - 1] + gap_penalty
                )

        # Traceback (Hizalama Oluşturma)
        i, j = m, n
        align1, align2 = [], []
        aligned_pairs = []

        while i > 0 and j > 0:
            char1, char2 = s1[i - 1], s2[j - 1]
            sc_char1, sc_char2 = sc1[i - 1], sc2[j - 1]

            if char1 == char2:
                exp = match_score
            elif sc_char1 == sc_char2:
                exp = class_match_score
            else:
                hamm = self.panphon_engine.articulatory_hamming_distance(char1, char2)
                exp = 1 if hamm <= 0.35 else mismatch_penalty

            if score_matrix[i][j] == score_matrix[i - 1][j - 1] + exp:
                align1.append(char1)
                align2.append(char2)
                aligned_pairs.append((char1, char2))
                i -= 1
                j -= 1
            elif score_matrix[i][j] == score_matrix[i - 1][j] + gap_penalty:
                align1.append(char1)
                align2.append("-")
                aligned_pairs.append((char1, "-"))
                i -= 1
            else:
                align1.append("-")
                align2.append(char2)
                aligned_pairs.append(("-", char2))
                j -= 1

        while i > 0:
            align1.append(s1[i - 1])
            align2.append("-")
            aligned_pairs.append((s1[i - 1], "-"))
            i -= 1
        while j > 0:
            align1.append("-")
            align2.append(s2[j - 1])
            aligned_pairs.append(("-", s2[j - 1]))
            j -= 1

        aligned_seq1 = "".join(reversed(align1))
        aligned_seq2 = "".join(reversed(align2))
        aligned_pairs.reverse()

        max_possible = max(len(s1), len(s2)) * match_score
        raw_score = score_matrix[m][n]
        normalized_sim = max(0.0, min(1.0, raw_score / max(max_possible, 1)))

        # PanPhon benzerliği ile LingPy SCA benzerliğinin ağırlıklı ortalaması
        combined_similarity = round((normalized_sim * 0.5) + (panphon_engine_sim := panphon_res.get("phonetic_similarity", 0.0)) * 0.5, 3)

        return {
            "seq1": s1,
            "seq2": s2,
            "sound_class_seq1": sc1,
            "sound_class_seq2": sc2,
            "aligned_seq1": aligned_seq1,
            "aligned_seq2": aligned_seq2,
            "alignment_score": raw_score,
            "phonetic_similarity": combined_similarity,
            "panphon_articulatory_distance": panphon_res.get("normalized_articulatory_distance", 1.0),
            "aligned_pairs": aligned_pairs
        }
