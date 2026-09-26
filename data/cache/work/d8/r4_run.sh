#!/bin/zsh
# D8: R4 yakalamaları + karşılaştırma (PREREG §5), sonra korumalar (§6.3). Sırayla, tek ağır süreç.
cd /Users/mshn/Documents/etimoloji
P=data/cache/work/d8/PREREG.md
H="caffeinate -i data/cache/work/heavy.sh .venv/bin/python"
export ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db
eval $H -m engine.evaluation.xborrowing_eval capture --split r4 --variant sca --tag d8 --prereg $P || exit 1
eval $H -m engine.evaluation.xborrowing_eval capture --split r4 --variant pred --tag d8pred --prereg $P || exit 1
eval $H -m engine.evaluation.xborrowing_eval compare --split r4 --prereg $P --base-tag d8 --tags "d8@hn2:ramp:thr" d8pred > data/cache/work/d8/r4_compare.log 2>&1 || exit 1
echo COMPARE_DONE
unset ETY_LEXICON_INDEX
for s in hn:2:ramp:thr pred; do
  eval $H data/cache/work/d8/run_beval_d8.py $s > data/cache/work/d8/beval_${s//:/_}.log 2>&1
  eval $H data/cache/work/d8/run_replay_d8.py $s > data/cache/work/d8/replay_${s//:/_}.log 2>&1
done
echo ALL_DONE
