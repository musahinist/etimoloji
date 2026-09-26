#!/bin/zsh
# 9n rapor — BİR KEZ (ön kayıt commit'inden sonra).
cd /Users/mshn/Documents/etimoloji
shasum -a 256 data/cache/work/donor9n/gold.json
caffeinate -i data/cache/work/heavy.sh .venv/bin/python data/cache/work/donor9n/harness.py gold 9n 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$"
echo DONE
