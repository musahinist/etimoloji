#!/bin/zsh
# 9o ayar: Türkçe train+dev, 9j TDK ayar, 9j TETTL ayar (dört koşul).
cd /Users/mshn/Documents/etimoloji
for g in tr tdk_ayar tettl_ayar; do
  echo "=== $g"
  caffeinate -i data/cache/work/heavy.sh .venv/bin/python data/cache/work/donor9o/harness.py gold $g 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$\|RuntimeWarning" || echo "FAIL $g"
done
echo DONE
