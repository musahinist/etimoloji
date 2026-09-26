#!/bin/zsh
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9g/harness.py
for c in tr xt saha gold9f; do
  echo "=== $c"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H $c 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $c"
done
echo DONE
