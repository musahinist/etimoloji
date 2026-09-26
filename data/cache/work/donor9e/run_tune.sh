#!/bin/zsh
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9e/harness.py
for c in "gold ayar" tr xt saha; do
  echo "=== $c"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H ${=c} || echo "FAIL $c"
done
