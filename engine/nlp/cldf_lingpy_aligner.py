"""
CLDF & LingPy Standartlarında Hesaplamalı Fonetik Dizilim Hizalayıcısı (Phonetic Sequence Aligner)
Ses dizilimlerini (IPA/harf) Needleman-Wunsch / Smith-Waterman algoritmalarıyla hizalar ve fonetik denklik matrisi üretir.
"""

from typing import Dict, Any, List, Tuple
import re

class CldfLingPyAligner:
    """CLDF / LingPy uyumlu fonetik dizi hizalama ve karakter denklik motoru"""

    def align_sequences(self, seq1: str, seq2: str) -> Dict[str, Any]:
        s1 = re.sub(r'[^a-zçğıöşüа-я]', '', (seq1 or "").lower().lstrip("*"))
        s2 = re.sub(r'[^a-zçğıöşüа-я]', '', (seq2 or "").lower())

        if not s1 or not s2:
            return {
                "aligned_seq1": s1,
                "aligned_seq2": s2,
                "alignment_score": 0.0,
                "phonetic_similarity": 0.0,
                "aligned_pairs": []
            }

        # Needleman-Wunsch Fonetik Hizalama Matrisi
        m, n = len(s1), len(s2)
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]

        match_score = 2
        mismatch_penalty = -1
        gap_penalty = -2

        for i in range(m + 1):
            score_matrix[i][0] = i * gap_penalty
        for j in range(n + 1):
            score_matrix[0][j] = j * gap_penalty

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char1, char2 = s1[i - 1], s2[j - 1]
                if char1 == char2:
                    score = match_score
                elif (char1 in 'aeıioöuü' and char2 in 'aeıioöuü') or (char1 in 'kptgdb' and char2 in 'kptgdb'):
                    score = 1 # Ses sınıfı yakınlığı
                else:
                    score = mismatch_penalty

                score_matrix[i][j] = max(
                    score_matrix[i - 1][j - 1] + score,
                    score_matrix[i - 1][j] + gap_penalty,
                    score_matrix[i][j - 1] + gap_penalty
                )

        # Traceback (Hizalamayı Geriye Doğru Oluşturma)
        i, j = m, n
        align1, align2 = [], []
        aligned_pairs = []

        while i > 0 and j > 0:
            score_current = score_matrix[i][j]
            score_diag = score_matrix[i - 1][j - 1]
            score_up = score_matrix[i - 1][j]
            score_left = score_matrix[i][j - 1]

            char1, char2 = s1[i - 1], s2[j - 1]
            expected_score = 2 if char1 == char2 else (1 if (char1 in 'aeıioöuü' and char2 in 'aeıioöuü') else -1)

            if score_current == score_diag + expected_score:
                align1.append(char1)
                align2.append(char2)
                aligned_pairs.append((char1, char2))
                i -= 1
                j -= 1
            elif score_current == score_up + gap_penalty:
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

        return {
            "seq1": s1,
            "seq2": s2,
            "aligned_seq1": aligned_seq1,
            "aligned_seq2": aligned_seq2,
            "alignment_score": raw_score,
            "phonetic_similarity": round(normalized_sim, 3),
            "aligned_pairs": aligned_pairs
        }
