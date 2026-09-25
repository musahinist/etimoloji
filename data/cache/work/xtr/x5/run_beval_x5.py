"""Koruma ölçümü (X5): make eval-borrowing'i (WOLD + Türkçe) birleşik aday ya da
bir bileşen açıkken, izlenen dosyalara yazmadan koşar.
Kullanım: run_beval_x5.py <off|x5c|x4a2|mean|s3fallback>"""
import os, shutil, sys
from pathlib import Path
spec = sys.argv.pop(1)
out = Path("data/cache/work/xtr/x5/beval") / spec
out.mkdir(parents=True, exist_ok=True)
for f in ("borrowing_combiner.json", "phonotactic_sah.json", "phonotactic_tr.json"):
    src = Path("data/models") / f
    if src.exists():
        shutil.copy(src, out / f)
import engine.nlp.phonotactic_lm as pl, engine.nlp.borrowing_combiner as bc, engine.evaluation.report as rp
import engine.nlp.donor_proximity as dp
pl.MODEL_DIR = out
bc.MODEL_PATH = out / "borrowing_combiner.json"
rp.EVAL_DIR = out
env = {"ETY_DONOR_CLEAN": "0", "ETY_DONOR_RAMP_CHANCE": "0", "ETY_DONOR_SENSE_FILTER": "off"}
if spec in ("x5c", "x4a2"):
    env.update(ETY_DONOR_CLEAN="1", ETY_DONOR_RAMP_CHANCE="1")
if spec in ("x5c", "s3fallback"):
    env["ETY_DONOR_SENSE_FILTER"] = "s3:fallback"
os.environ.update(env)
if spec in ("x5c", "mean"):
    dp.STRENGTH_DISTANCE = "mean"
    dp.DONOR_DISTANCE_THRESHOLD = 0.60
    dp.DONOR_DISTANCE_CEILING = 0.85
else:
    dp.STRENGTH_DISTANCE = "sca"
    dp.DONOR_DISTANCE_THRESHOLD = 0.35
    dp.DONOR_DISTANCE_CEILING = 0.60
dp.reset_cache()
print("X5", spec, env, dp.STRENGTH_DISTANCE, dp.DONOR_DISTANCE_THRESHOLD, dp.DONOR_DISTANCE_CEILING, flush=True)
from engine.evaluation.borrowing_eval import main
raise SystemExit(main())
