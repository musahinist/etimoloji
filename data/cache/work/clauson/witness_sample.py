"""Elle kesinlik denetimi için 40 rastgele Clauson tanığı (başlık + kronoloji kümeleri)."""
import json
import random
import sys

from engine.fetchers import clauson_edt as ce

words = set()
h = json.load(open('data/eval/headline.json'))
for s in h['subsets'].values():
    words.update(i['word'] for i in s['items'])
c = json.load(open('data/eval/chronology.json'))
words.update(i['word'] for i in c['items'])
rows = []
for w in sorted(words):
    for form, meaning, rec in ce.matches(w):
        rows.append(dict(word=w, glosses=ce.query_glosses(w)[:2], head=rec['headword'], prefix=rec.get('prefix'),
                         clauson=ce.record_glosses(rec), form=round(form, 2), meaning=meaning,
                         base=ce.base_form(rec)))
print(len(words), 'kelime,', len({r['word'] for r in rows}), 'eşleşen,', len(rows), 'tanık', file=sys.stderr)
random.seed(20260926)
json.dump({"n_words": len(words), "n_matched": len({r['word'] for r in rows}), "n_witness": len(rows),
           "sample": random.sample(rows, 40)}, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
