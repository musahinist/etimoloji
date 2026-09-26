"""9n koruma: eval-borrowing F'leri DONOR_HONEST off / a2 ile (yalnız etiket gösterimi değişir)."""
import sys
mode = sys.argv.pop(1)
from engine.nlp import donor_proximity as dp  # noqa: E402
dp.DONOR_HONEST = mode
from engine.evaluation import borrowing_eval  # noqa: E402
raise SystemExit(borrowing_eval.main())
