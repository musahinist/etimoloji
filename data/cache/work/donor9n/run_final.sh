#!/bin/zsh
# 9n kabul sonrası: eval-borrowing (off / a2) + make eval-tr-donor (DONOR_HONEST=a2 varsayılan).
cd /Users/mshn/Documents/etimoloji
for m in off a2; do
  echo "=== eval-borrowing $m"
  caffeinate -i data/cache/work/heavy.sh .venv/bin/python data/cache/work/donor9n/eval_borrowing.py $m > data/cache/work/donor9n/eval_borrowing_$m.log 2>&1 || echo FAIL
done
echo "=== eval-tr-donor"
make eval-tr-donor > data/cache/work/donor9n/eval_tr_donor.log 2>&1 || echo FAIL
echo DONE
