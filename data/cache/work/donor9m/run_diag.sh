#!/bin/zsh
# 9m tanı + ayar (rapordan ÖNCE): Türkçe train+dev, 9j TDK/TETTL ayar, 9l rapor (görülmüş, bilgi).
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9m/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
for g in tr tdk_ayar tettl_ayar 9l; do run gold $g; done
echo DONE
