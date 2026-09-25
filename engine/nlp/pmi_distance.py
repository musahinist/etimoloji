"""
ASJP-PMI mesafesi — SCA'nın yanında ikinci ses benzerliği ölçüsü.

Jäger (2013, 2018) ASJP ses sınıfları arasındaki noktasal karşılıklı bilgi
(PMI) puanlarını dünya dillerindeki olası akraba çiftlerinden öğrendi;
Wientzek (2025, arXiv 2512.01713) aynı matrisle alıntı tespitinde SCA'yı
geçti (Miller & List verisi 0,826 vs 0,781; WOLD İng->Jap/Tay 0,773 vs 0,644),
ama Türki/İrani verisinde fark yok (0,819 vs 0,821).

Matris ``data/asjp/pmi_scores.csv`` ve boşluk cezaları
``data/asjp/gap_penalties.csv``: Jäger 2018, *Scientific Data* 5:180189
(https://doi.org/10.1038/sdata.2018.189, OSF rb3n4 / 9bvfe), CC-BY-4.0;
dosyalar github.com/TGH-2020/CWE_BorDetect (e3f9c39) kopyasından alındı.
Burada **hiçbir değer öğrenilmedi ya da ayarlanmadı**.

Hizalama: afin boşluklu Needleman-Wunsch (Gotoh); ilk boşluk ``gp1``, devamı
``gp2`` (Biopython ``globalds`` ile aynı). Mesafe::

    d(a, b) = max(0, 1 − s(a, b) / ((s(a, a) + s(b, b)) / 2))

⚠️ Girdi motorun karşılaştırma alfabesidir (``to_comparison_form``), IPA
değil. ASJP'ye dönüşüm :data:`COMPARISON_TO_ASJP` tablosuyla yapılır; ASJP
ön-yuvarlak ünlü ayırmaz (ö -> o, ü -> u) ve ı/ɨ'yı merkezî ünlü ``3`` sayar.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from engine.config import PROJECT_ROOT

ASJP_DIR = PROJECT_ROOT / "data" / "asjp"

#: Karşılaştırma alfabesi -> ASJP sembolü.
COMPARISON_TO_ASJP = {
    "a": "a", "b": "b", "c": "j", "ç": "C", "d": "d", "e": "e", "f": "f",
    "g": "g", "ğ": "x", "h": "h", "ı": "3", "i": "i", "j": "Z", "k": "k",
    "l": "l", "ĺ": "l", "m": "m", "n": "n", "ŋ": "N", "o": "o", "ö": "o",
    "p": "p", "q": "q", "r": "r", "ŕ": "r", "s": "s", "ş": "S", "t": "t",
    "u": "u", "ü": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z",
}


def to_asjp(comparison: str) -> str:
    """Karşılaştırma biçimi -> ASJP dizgisi (bilinmeyen harf atılır)."""
    return "".join(COMPARISON_TO_ASJP.get(ch, "") for ch in comparison)


@lru_cache(maxsize=1)
def _matrix(directory: Path = ASJP_DIR) -> tuple[dict[tuple[str, str], float], float, float] | None:
    scores_path, gaps_path = directory / "pmi_scores.csv", directory / "gap_penalties.csv"
    if not (scores_path.exists() and gaps_path.exists()):
        return None
    with scores_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0][1:]
    table = {(row[0], col): float(value) for row in rows[1:] for col, value in zip(header, row[1:], strict=True)}
    gaps: dict[str, float] = {}
    with gaps_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 2 and row[0] in {"gp1", "gp2"}:
                gaps[row[0]] = float(row[1])
    return table, gaps["gp1"], gaps["gp2"]


def available() -> bool:
    return _matrix() is not None


def pmi_score(a: str, b: str) -> float:
    """İki ASJP dizgisinin en iyi global hizalama PMI puanı (Gotoh)."""
    matrix = _matrix()
    if matrix is None or not a or not b:
        return 0.0
    table, gap_open, gap_extend = matrix
    neg = float("-inf")
    n, m = len(a), len(b)
    # M: eşleşme ile biten, X: a'da boşluk dışı (b'ye boşluk), Y: tersi.
    prev_m = [neg] * (m + 1)
    prev_x = [neg] * (m + 1)
    prev_y = [neg] * (m + 1)
    prev_m[0] = 0.0
    for j in range(1, m + 1):
        prev_y[j] = gap_open + (j - 1) * gap_extend
    for i in range(1, n + 1):
        cur_m = [neg] * (m + 1)
        cur_x = [neg] * (m + 1)
        cur_y = [neg] * (m + 1)
        cur_x[0] = gap_open + (i - 1) * gap_extend
        ai = a[i - 1]
        for j in range(1, m + 1):
            best_prev = max(prev_m[j - 1], prev_x[j - 1], prev_y[j - 1])
            cur_m[j] = best_prev + table.get((ai, b[j - 1]), -5.0)
            cur_x[j] = max(prev_m[j] + gap_open, prev_x[j] + gap_extend, prev_y[j] + gap_open)
            cur_y[j] = max(cur_m[j - 1] + gap_open, cur_y[j - 1] + gap_extend, cur_x[j - 1] + gap_open)
        prev_m, prev_x, prev_y = cur_m, cur_x, cur_y
    return max(prev_m[m], prev_x[m], prev_y[m])


@lru_cache(maxsize=200000)
def pmi_distance(a: str, b: str) -> float:
    """İki karşılaştırma biçimi arasındaki normalize PMI mesafesi.

    0 = özdeş; 1 = hizalama puanı sıfır; negatif PMI'da 1'in üstüne çıkar
    (ilgisiz çiftler arasında sıralama korunsun diye kırpılmaz). Matris
    yoksa ``1.0`` ("kanıt yok").
    """
    x, y = to_asjp(a), to_asjp(b)
    if not available() or not x or not y:
        return 1.0
    self_score = (pmi_score(x, x) + pmi_score(y, y)) / 2
    if self_score <= 0:
        return 1.0
    return max(0.0, 1.0 - pmi_score(x, y) / self_score)
