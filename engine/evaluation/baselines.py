"""
Trivial taban çizgileri — "%X aldık" cümlesini anlamlı kılan şey.

Cui ve ark. (2024) rastgele taban çizginin **%3,68**, modelin **%54** aldığını
gösteriyor. Bu farkı göstermeden bildirilen bir doğruluk sayısı yorumlanamaz:
kolay bir veri kümesinde aptal bir sistem de yüksek puan alabilir.

Buradaki taban çizgileri kasten aptaldır. Motorun bunları **belirgin biçimde**
geçmesi gerekir; geçemiyorsa karşılaştırmalı yöntem hiçbir şey katmıyor demektir.

============================  ==============================================
``copy_random_daughter``      rastgele bir kız dilin biçmini ata biçim say
``copy_longest``              en uzun tanığı ata biçim say
``copy_anchor``               sorgu kelimesinin kendisini ata biçim say
``majority_character``        her konumda en sık geçen sesi seç (hizalamasız)
============================  ==============================================

Rastgelelik **tohumlanır**: aynı veri, aynı sonuç. Yeniden üretilemeyen taban
çizgisi taban çizgisi değildir.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

#: Tüm rastgele taban çizgileri bu tohumu kullanır.
BASELINE_SEED = 20260827


def _forms(entries: list[dict[str, str]]) -> list[str]:
    return [e["word"] for e in entries if e.get("word")]


def copy_anchor(word: str, entries: list[dict[str, str]]) -> dict[str, Any]:
    """Sorgu kelimesini olduğu gibi ata biçim ilan eder.

    En dürüst "hiçbir şey yapmama" taban çizgisi: modern biçimin ata biçme
    eşit olduğunu varsayar. Motorun bunu geçememesi, ses kanunlarının hiç
    uygulanmadığı anlamına gelir.
    """
    return {
        "reconstructed_root": f"*{word}",
        "is_reconstructible": bool(word),
        "confidence": 0.0,
        "witness_count": len(entries),
    }


def copy_longest(word: str, entries: list[dict[str, str]]) -> dict[str, Any]:
    """En uzun tanığı ata biçim sayar — "ata biçim en dolu olandır" sezgisi."""
    forms = _forms(entries) or [word]
    longest = max(forms, key=len)
    return {
        "reconstructed_root": f"*{longest}",
        "is_reconstructible": bool(longest),
        "confidence": 0.0,
        "witness_count": len(entries),
    }


def copy_random_daughter(word: str, entries: list[dict[str, str]]) -> dict[str, Any]:
    """Rastgele bir kız dilin biçmini ata biçim sayar (Cui ve ark. 2024 taban çizgisi)."""
    forms = _forms(entries) or [word]
    rng = random.Random(f"{BASELINE_SEED}:{word}")
    chosen = rng.choice(sorted(forms))
    return {
        "reconstructed_root": f"*{chosen}",
        "is_reconstructible": bool(chosen),
        "confidence": 0.0,
        "witness_count": len(entries),
    }


def majority_character(word: str, entries: list[dict[str, str]]) -> dict[str, Any]:
    """Her konumda en sık geçen karakteri seçer — **hizalama yapmadan**.

    Karşılaştırmalı yöntemin hizalama katmanının ne kattığını ölçer: motor
    bunu geçemiyorsa hizalama bir işe yaramıyor demektir.
    """
    forms = _forms(entries) + [word]
    if not forms:
        return {"reconstructed_root": "", "is_reconstructible": False, "confidence": 0.0}
    length = Counter(len(f) for f in forms).most_common(1)[0][0]
    chars: list[str] = []
    for i in range(length):
        column = Counter(f[i] for f in forms if i < len(f))
        if column:
            chars.append(column.most_common(1)[0][0])
    return {
        "reconstructed_root": "*" + "".join(chars),
        "is_reconstructible": bool(chars),
        "confidence": 0.0,
        "witness_count": len(entries),
    }


#: Rapor sırası — kolaydan zora.
BASELINES: dict[str, Any] = {
    "copy_anchor": copy_anchor,
    "copy_random_daughter": copy_random_daughter,
    "copy_longest": copy_longest,
    "majority_character": majority_character,
}
