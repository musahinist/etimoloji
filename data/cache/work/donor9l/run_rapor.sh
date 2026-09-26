#!/bin/zsh
# 9l rapor (bir kez) + önceki raporlar (bilgi).
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9l/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
export NINEL_CONDS=off,s1,s2,s3b,s13
run gold 9l rapor
run gold9j
run gold9f
run gold9e
run gold9g
echo DONE
