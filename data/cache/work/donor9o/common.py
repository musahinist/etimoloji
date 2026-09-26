"""9o ortak ortam: 9n `diag` yardımcıları + KENDİ indeks kopyalarımız (paralel ajan indeksi yeniden kurabilir).

Kör indeks = data/cache/work/donor9o/index_blind.db (xtr/index_blind.db'nin kopyası, sha256 9566…4839),
tam indeks (yalnız PUANLAMA referansı) = index_full.db (data/lexicons/index.db kopyası, f956…a85).
"""
import os, sys
from pathlib import Path

OUT = Path(__file__).parent
W = OUT.parent
sys.path.insert(0, str(W / "donor9n"))
sys.path.insert(0, str(W / "donor9m"))
import diag  # noqa: E402  (ağ kapalı, env kurar)

if not diag._SAHA:
    os.environ["ETY_LEXICON_INDEX"] = str(OUT / "index_blind.db")
    diag.BLIND = OUT / "index_blind.db"
diag.FULL = OUT / "index_full.db"

CLASS_CODES = {"Arapça": ("ar",), "Farsça": ("fa",), "Fransızca": ("fr",), "İtalyanca": ("it",),
               "Yunanca": ("el",), "Ermenice": ("hy",)}
