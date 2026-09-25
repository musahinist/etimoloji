"""Düz Levenshtein uzaklığı — projedeki tek kaynak.

Önceden aynı algoritmanın birebir kopyaları ``evaluation.metrics``,
``donor_lexicon``, ``donor_proximity``, ``nbest_reranking`` ve
``west_old_turkic`` içinde ayrı ayrı yazılıydı. Üretim kodu bunun için
``evaluation`` paketini içe aktarıyordu (katman ihlali).

⚠️ Yalnız **birim maliyetli** uzaklık buradadır. Ağırlıklı ya da özellik
tabanlı uzaklıklar (``phonological_feature_engine``, SCA,
``witness_variants``) anlamca farklıdır; buraya taşınmaz.
"""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein uzaklığı. Sürüm bağımlılığı olmasın diye elde yazılmıştır."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # silme
                    current[j - 1] + 1,  # ekleme
                    previous[j - 1] + (ca != cb),  # değiştirme
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """ED'yi daha uzun dizginin uzunluğuna böler → [0, 1].

    Normalizasyon olmadan uzun kelimeler kısa kelimelerden "daha kötü"
    görünür ve veri kümeleri arası karşılaştırma anlamsızlaşır (List 2019).
    İki boş dizgi için 0, yalnız biri boşsa 1 döner.
    """
    longest = max(len(a), len(b))
    return edit_distance(a, b) / longest if longest else 0.0
