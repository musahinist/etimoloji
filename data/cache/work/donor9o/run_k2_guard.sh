#!/bin/zsh
# 9o: K2 (sızıntı, yeni altın; doğruluk yok) + korumalar (Saha eval-donor her koşulda, xturkic ayar).
cd /Users/mshn/Documents/etimoloji
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$\|RuntimeWarning" || echo "FAIL $*"; }
run data/cache/work/donor9o/harness.py k2
run data/cache/work/donor9o/harness.py saha
run data/cache/work/donor9n/harness.py xt
echo DONE
