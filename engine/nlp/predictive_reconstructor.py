"""
Tanıklanmamış Kelimeler İçin Rekonstrüksiyon Cephesi

Sözlüklerde etimolojik çözümü bulunmayan kelimeler için ata biçim türetir.
Hesaplama :mod:`engine.nlp.comparative_reconstruction` motorunda yapılır.

Not: Bu modül eskiden akraba listesini yalnızca ``len()`` almak için kullanıp
içeriğini atıyor, güven skorunu ise akraba SAYISINA göre sabit ``0.88`` /
``0.75`` olarak veriyordu. Artık ata biçim gerçekten akraba biçimlerden
türetilir ve güven skoru kanıttan hesaplanır.
"""
from __future__ import annotations

from typing import Any

from engine.nlp.comparative_reconstruction import ComparativeReconstructor


class PredictiveReconstructor:
    def __init__(self) -> None:
        self._engine = ComparativeReconstructor()

    def reconstruct_unattested_proto_form(
        self, word: str, cognate_entries: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        res = self._engine.reconstruct(word, cognate_entries)

        # ⚠️ Bu cephe TANIKLANMAMIŞ kelimeler içindir. Motor, ölçümde
        # cevapsızlığın mümkün olan en kötü değeri alması yüzünden tanık
        # yokken sorgu biçmini "aday" olarak döndürüyor (`anchor_fallback`).
        # Ama BURADA onu ata biçim diye sunmak, tam olarak bu testin
        # koruduğu uydurmadır: akraba tanığı olmadan `*korak` üretmek.
        # Aday, `withheld_candidate` alanında saklanır.
        is_fallback = res.get("method") == "anchor_fallback"
        proto_form = "" if is_fallback else res.get("reconstructed_root", "")

        # Geriye dönük uyumlu anahtar adları
        return {
            "target_word": res.get("word", word),
            "reconstructed_proto_form": proto_form,
            "withheld_candidate": res.get("reconstructed_root", "") if is_fallback else "",
            "method": res.get("method", ""),
            "reconstruction_confidence": None if is_fallback else res.get("confidence"),
            "evidence_available": res.get("evidence_available", False),
            "witness_count": res.get("witness_count", 0),
            "applied_historical_rules": res.get("applied_correspondences", []),
            "reconstruction_notes": res.get("reconstruction_notes", ""),
        }
