"""
Veri ve isteğe bağlı kütüphane bekçileri (test atlama koşulları).

CI'daki ``test`` işi yalnız ``.[dev]`` kurar ve git'e alınmayan verileri
(sözlük indeksi, verici dökümleri, CLDF, Starling) indirmez. Bunlara
dayanan testler veri yokken sessizce YANLIŞ sonuç üretip düşüyordu
(örn. `kitap` indeks olmadan alıntı sayılmıyor). Bu testler burada
tanımlı bayraklarla AÇIKÇA atlanır; veriyle koşan ``data-checks`` işinde
ve yerelde (veri varken) eskisi gibi koşar.

Kullanım::

    from engine.tests.data_guards import HAS_INDEX, needs_index

    @needs_index
    class TestX(unittest.TestCase): ...
"""
from __future__ import annotations

import importlib.util
import unittest

from engine.config import CLDF_DIR, LEXICON_DIR, PROJECT_ROOT


def _importable(name: str) -> bool:
    """Modülü İÇE AKTARMADAN kurulu olup olmadığını söyler (torch yavaştır)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --- Veri --------------------------------------------------------------------
HAS_INDEX = (LEXICON_DIR / "index.db").exists()
HAS_DONORS = (LEXICON_DIR / "donors" / "donors.db").exists()
HAS_CLDF = (CLDF_DIR / "savelyevturkic" / "forms.csv").exists()
HAS_STARLING = (PROJECT_ROOT / "data" / "starling" / "turcet.dbf").exists()

# --- İsteğe bağlı kütüphaneler ----------------------------------------------
HAS_PANPHON = _importable("panphon")
HAS_PDFMINER = _importable("pdfminer")
HAS_TORCH = _importable("torch")
HAS_SEMANTIC = HAS_TORCH and _importable("sentence_transformers")

needs_index = unittest.skipUnless(
    HAS_INDEX, "sözlük indeksi kurulmamış (python -m engine.db.lexicon_index --build)"
)
needs_donors = unittest.skipUnless(HAS_DONORS, "verici dil indeksi kurulmamış")
needs_cldf = unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş (make data)")
needs_starling = unittest.skipUnless(HAS_STARLING, "Starling veritabanı yok")
needs_panphon = unittest.skipUnless(HAS_PANPHON, "panphon kurulu değil (.[phon])")
needs_pdfminer = unittest.skipUnless(HAS_PDFMINER, "pdfminer kurulu değil (.[pdf])")
needs_torch = unittest.skipUnless(HAS_TORCH, "torch kurulu değil")
needs_semantic = unittest.skipUnless(HAS_SEMANTIC, "sentence-transformers kurulu değil")
