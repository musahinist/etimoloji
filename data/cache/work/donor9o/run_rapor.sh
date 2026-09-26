#!/bin/zsh
# 9o rapor — BİR KEZ (ön kayıt commit'inden sonra).
cd /Users/mshn/Documents/etimoloji
shasum -a 256 data/cache/work/donor9o/gold.json
caffeinate -i data/cache/work/heavy.sh .venv/bin/python data/cache/work/donor9o/harness.py gold 9o 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$\|RuntimeWarning"
echo DONE
