"""9o: `tr_donor_eval` koşusu KÖR İNDEKS KOPYASIYLA (donor9o/index_blind.db = 9o öncesi xtr/index_blind.db,
sha256 9566…4839); paralel iş xtr/index_blind.db'yi yeniden kurdu. Kullanım: trdonor.py capture|report ..."""
import sys
from pathlib import Path
from engine.evaluation import tr_donor_eval as t
t.BLIND_INDEX = Path(__file__).parent / "index_blind.db"
raise SystemExit(t.main())
