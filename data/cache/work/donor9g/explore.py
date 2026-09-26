"""9g keşif: tr/ota/az/crh dökümlerinde ilk verici şablonu el/grc/gkm/hy/xcl olan maddeler."""
import gzip, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "data/cache/work/donor9e"))
import build_gold as B
FAM = {"el": "el", "grc": "el", "gkm": "el", "hy": "hy", "xcl": "hy", "axm": "hy", "el-kal": "el", "el-pap": "el", "pnt": "el", "cpg": "el"}
B.FAMILY = FAM
B.CLASS = {"el": "Yunanca", "hy": "Ermenice"}
out = Counter()
dial = Counter()
for lang in ("tr", "ota", "az", "crh"):
    with gzip.open(ROOT / f"data/lexicons/{lang}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            ety = r.get("etymology_templates") or []
            d = B.donor_of(ety) if ety else None
            tags = set(r.get("tags") or []) | {t for s in r.get("senses") or [] for t in s.get("tags") or []}
            isd = "dialectal" in tags or bool(r.get("categories") and any("dialect" in c.lower() for c in r.get("categories") if isinstance(c, str)))
            if isd:
                dial[(lang, "dialectal-tüm")] += 1
            if d:
                raw = next((t["args"].get("2") for t in ety if t.get("name") in B.DONOR_T and t["args"].get("2") in FAM), "?")
                out[(lang, d[0], raw)] += 1
                if isd:
                    dial[(lang, d[0])] += 1
for k, v in sorted(out.items()): print(k, v)
print(dict(dial))
