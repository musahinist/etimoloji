#!/bin/zsh
# 9l ayar + korumalar (rapordan ÖNCE). Tek ağır süreç, sırayla.
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9l/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
run gold tdk ayar
run tr
run xt
run saha
echo DONE
