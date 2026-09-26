#!/bin/zsh
# 9j ayar + korumalar (rapordan ÖNCE). Tek ağır süreç, sırayla.
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9j/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
run gold tdk ayar
run gold tettl ayar
run k2
run tr
run xt
run saha
run gold9f
run gold9e
run gold9g
echo DONE
