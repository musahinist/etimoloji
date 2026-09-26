#!/bin/zsh
# 9m R2 "güçlü" eşleşme varyantları (ayar: Türkçe train+dev, 9j TDK/TETTL ayar). R1 = ik.
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9m/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
export NINEM_CONDS=prod,r1,r2 NINEM_R1_DROP=ik
for v in b c d; do for g in tr tdk_ayar tettl_ayar; do NINEM_R2=$v NINEM_SUFFIX=_r2$v run gold $g; done; done
echo DONE
