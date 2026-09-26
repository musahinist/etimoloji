#!/bin/zsh
# 9l ayar-2: birleşik S1+S3 (s13) korumaları + K2 (yeni altın, doğruluk yok). Rapordan ÖNCE.
cd /Users/mshn/Documents/etimoloji
H=data/cache/work/donor9l/harness.py
run() { echo "=== $*"; caffeinate -i data/cache/work/heavy.sh .venv/bin/python $H "$@" 2>&1 | grep -v "^[0-9]* [0-9]* [0-9]*s$" || echo "FAIL $*"; }
export NINEL_SUFFIX=_s13 NINEL_CONDS=off,s13
run gold tdk ayar
run tr
run xt
run saha
export NINEL_SUFFIX= NINEL_CONDS=off,s1,s2,s3b,s13
run k2
echo DONE
