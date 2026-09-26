"""9o koruma: eval-borrowing F'leri DONOR_FORM_FIRST off / c2 ile (yalnız etiket gösterimi değişir).

Çıktı JSON'u data/eval/ yerine donor9o/evalb/<mod>/ altına yazılır (paralel işin data/eval dosyalarını ezmemek için).
"""
import sys
from pathlib import Path
mode = sys.argv.pop(1)
from engine.evaluation import report  # noqa: E402
report.EVAL_DIR = Path(__file__).parent / "evalb" / mode
from engine.nlp import donor_proximity as dp  # noqa: E402
dp.DONOR_FORM_FIRST = mode
from engine.evaluation import borrowing_eval  # noqa: E402
raise SystemExit(borrowing_eval.main())
