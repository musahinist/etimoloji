"""TETTL ayrıştırma denetimi: cilt başına 30 rastgele (hash sırası) kayıt + paragraf başı (yerel ekrana; dosyaya yazılmaz)."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import parse_tettl as P

vol = int(sys.argv[1])
rows = [r for r in json.loads((P.RAW / "tettl_parsed.json").read_text()) if r["vol"] == vol]
paras = dict(P.paragraphs((P.RAW / f"t{vol:02d}.txt").read_text(encoding="utf-8")))
rows.sort(key=lambda r: hashlib.sha256(f"9j-audit:{vol}:{r['para']}".encode()).hexdigest())
for i, r in enumerate(rows[:30]):
    print(f"{i+1:2d} {'/'.join(r['head'])} => {r['donor']}:{r['etymon']}{' V' if r['venetian'] else ''} | {paras[r['para']][:170]}")
