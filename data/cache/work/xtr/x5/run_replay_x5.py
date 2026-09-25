"""Arama yolu koruması (X5): ranker/replay.py 'new dev' — bayrak kapalı (off) ve
ETY_SEARCH_DONOR_PROXIMITY açık (on) — birleşik aday ya da taban ayarla.
Çıktı x5/replay_w/ altına (ranker/ dosyalarına yazmaz). Kullanım: run_replay_x5.py <off|x5c>"""
import importlib.util, os, sys
from pathlib import Path
spec = sys.argv[1]
ROOT = Path("/Users/mshn/Documents/etimoloji")
env = {"ETY_DONOR_CLEAN": "0", "ETY_DONOR_RAMP_CHANCE": "0", "ETY_DONOR_SENSE_FILTER": "off"}
if spec == "x5c":
    env = {"ETY_DONOR_CLEAN": "1", "ETY_DONOR_RAMP_CHANCE": "1", "ETY_DONOR_SENSE_FILTER": "s3:fallback"}
os.environ.update(env)
import engine.nlp.donor_proximity as dp
if spec == "x5c":
    dp.STRENGTH_DISTANCE, dp.DONOR_DISTANCE_THRESHOLD, dp.DONOR_DISTANCE_CEILING = "mean", 0.60, 0.85
else:
    dp.STRENGTH_DISTANCE, dp.DONOR_DISTANCE_THRESHOLD, dp.DONOR_DISTANCE_CEILING = "sca", 0.35, 0.60
dp.reset_cache()
m = importlib.util.spec_from_file_location("replay", ROOT / "data/cache/work/ranker/replay.py")
replay = importlib.util.module_from_spec(m)
m.loader.exec_module(replay)
replay.W = ROOT / "data/cache/work/xtr/x5/replay_w"
print("X5 replay", spec, env, dp.STRENGTH_DISTANCE, flush=True)
out = replay.run(["new"], "dev")
(replay.W / f"replay_dev_new_{spec}.json").write_text(__import__("json").dumps(out, ensure_ascii=False, indent=1))
