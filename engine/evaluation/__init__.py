"""
Değerlendirme koşum takımı — yayın şartı.

Bu paket olmadan hiçbir doğruluk iddiası savunulamaz. Literatürün zorunlu
kıldığı çıtayı kurar:

* :mod:`metrics` — ED + NED + B-Cubed F + accuracy dörtlüsü. **Salt edit
  distance kabul edilmiyor** (List 2019, *Computational Linguistics*).
* :mod:`significance` — bootstrap güven aralığı, permütasyon testi, FDR.
  Aralıksız tek sayı raporlanmaz.
* :mod:`calibration` — ECE, Brier, güvenilirlik diyagramı, risk-coverage.
* :mod:`gold` — altın standardı **kavram bazlı** train/dev/test'e ayırır ve
  test bölümünü dondurur (sızıntı önleme).
* :mod:`harness` — motoru altın standarda karşı koşar; altın alanlar motordan
  gizlenir.

Tasarım ilkesi: **aynı kaynak hem sinyal hem sınav olamaz.** Bu paketin varlık
sebebi budur.
"""

from engine.evaluation.metrics import (
    ReconstructionScore,
    bcubed_fscore,
    edit_distance,
    feature_error_rate,
    normalized_edit_distance,
    score_reconstructions,
)

__all__ = [
    "ReconstructionScore",
    "bcubed_fscore",
    "edit_distance",
    "feature_error_rate",
    "normalized_edit_distance",
    "score_reconstructions",
]
