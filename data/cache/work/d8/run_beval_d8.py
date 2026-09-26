"""Koruma (D8): make eval-borrowing (WOLD + Türkçe) aday açıkken, izlenen dosyalara
yazmadan. Kullanım: run_beval_d8.py <off|pred|hn:<ağırlık>:<kural>[:thr]>"""
import os, shutil, sys
from pathlib import Path
spec = sys.argv.pop(1)
out = Path("data/cache/work/d8/beval") / spec.replace(":", "_")
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
os.environ["ETY_COMBINER_HARD_NEG"] = spec[3:] if spec.startswith("hn:") else "off"
if spec == "pred":
    dp.STRENGTH_DISTANCE = "pred"
dp.reset_cache()
print("D8", spec, os.environ["ETY_COMBINER_HARD_NEG"], dp.STRENGTH_DISTANCE, flush=True)
from engine.evaluation.borrowing_eval import main
raise SystemExit(main())
