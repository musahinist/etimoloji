#!/bin/zsh
# 9o kabul sonrası: eval-borrowing (off / c2; JSON donor9o/evalb/) + eval-tr-donor (DONOR_FORM_FIRST=c2 varsayılan).
# İndeksler KOPYA (donor9o/index_full.db, donor9o/index_blind.db): paralel iş canlı indeksleri yeniden kurdu.
cd /Users/mshn/Documents/etimoloji
D=data/cache/work/donor9o
FULL=$PWD/$D/index_full.db
BLIND=$PWD/$D/index_blind.db
for m in off c2; do
  echo "=== eval-borrowing $m"
  ETY_LEXICON_INDEX=$FULL caffeinate -i data/cache/work/heavy.sh .venv/bin/python $D/eval_borrowing.py $m > $D/eval_borrowing_$m.log 2>&1 || echo FAIL
done
echo "=== eval-tr-donor"
for a2 in off on; do v=$([ $a2 = on ] && echo 1 || echo 0)
  ETY_DONOR_CLEAN=$v ETY_DONOR_RAMP_CHANCE=$v ETY_LEXICON_INDEX=$FULL caffeinate -i data/cache/work/heavy.sh .venv/bin/python $D/trdonor.py capture --index full --a2 $a2 >> $D/eval_tr_donor.log 2>&1 || echo FAIL
  ETY_DONOR_CLEAN=$v ETY_DONOR_RAMP_CHANCE=$v ETY_LEXICON_INDEX=$BLIND caffeinate -i data/cache/work/heavy.sh .venv/bin/python $D/trdonor.py capture --index blind --a2 $a2 >> $D/eval_tr_donor.log 2>&1 || echo FAIL
done
.venv/bin/python $D/trdonor.py report >> $D/eval_tr_donor.log 2>&1 || echo FAIL
echo DONE
