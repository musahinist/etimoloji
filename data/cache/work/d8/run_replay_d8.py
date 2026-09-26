"""Arama yolu koruması (D8): ranker/replay.py 'new dev', aday açıkken; birleştirici
run_beval_d8.py'nin o aday için yazdığı modeldir. Çıktı d8/replay_w/.
Kullanım: run_replay_d8.py <off|pred|hn:...>"""
import importlib.util, json, sys
from pathlib import Path
spec = sys.argv[1]
ROOT = Path("/Users/mshn/Documents/etimoloji")
import engine.nlp.borrowing_combiner as bc, engine.nlp.phonotactic_lm as pl, engine.nlp.donor_proximity as dp
beval = ROOT / "data/cache/work/d8/beval" / spec.replace(":", "_")
bc.MODEL_PATH = beval / "borrowing_combiner.json"
pl.MODEL_DIR = beval
if spec == "pred":
    dp.STRENGTH_DISTANCE = "pred"
dp.reset_cache()
m = importlib.util.spec_from_file_location("replay", ROOT / "data/cache/work/ranker/replay.py")
replay = importlib.util.module_from_spec(m)
m.loader.exec_module(replay)
replay.W = ROOT / "data/cache/work/d8/replay_w"
replay.W.mkdir(parents=True, exist_ok=True)
import shutil; shutil.copy(ROOT / "data/cache/work/ranker/sample.json", replay.W / "sample.json")  # cap/ -> ranker/cap bağlantısı
print("D8 replay", spec, bc.MODEL_PATH, dp.STRENGTH_DISTANCE, flush=True)
out = replay.run(["new"], "dev")
(replay.W / f"replay_dev_new_{spec.replace(':', '_')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
