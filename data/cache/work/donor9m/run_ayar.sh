#!/bin/zsh
# 9m seçilen adaylar (R1 = ik, R2 = d) ayarda + korumalar + K2 (rapordan ÖNCE).
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9m/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
for g in tr tdk_ayar tettl_ayar; do run gold $g; done
export NINEM_CONDS=prod,r1,r2,r3
run xt
run saha
run k2
echo DONE
