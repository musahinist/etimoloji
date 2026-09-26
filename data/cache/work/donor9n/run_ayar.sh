#!/bin/zsh
# 9n ayar (Türkçe train+dev, 9j TDK/TETTL ayar) + K2 (rapordan ÖNCE).
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9n/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
for g in tr tdk_ayar tettl_ayar; do run gold $g; done
run k2
echo DONE
