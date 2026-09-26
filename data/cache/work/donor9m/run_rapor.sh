#!/bin/zsh
# 9m rapor (bir kez) + önceki altınlarda adaylar (görülmüş, bilgi).
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9m/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
export NINEM_CONDS=base,g1,g2,h1,prod,r1,r2,r3
run gold 9m
for g in 9l tdk_rapor tettl_rapor 9e_ayar 9e_rapor 9f 9g_ayar 9g_rapor; do NINEM_SUFFIX=_all run gold $g; done
echo DONE
