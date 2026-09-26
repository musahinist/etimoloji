#!/bin/zsh
# 9n korumalar: Saha eval-donor (her DONOR_HONEST değeri) + xturkic ayar.
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9n/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
run xt
run saha
echo DONE
