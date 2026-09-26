"""Clauson eşleşme yoklaması: 45 kök-yok kelime (Starling kapalı başlık)."""
import json

from engine.evaluation.headline_eval import score_item, split_halves
from engine.fetchers import clauson_edt as ce

d = json.load(open('data/eval/headline.json'))
s = d['subsets']['starling/starling_yok']['items']
half = split_halves(i['word'] for i in s)
W23 = set('besim bürgü büz dala darga duma evin göbelek güce koçkar obuz savur sümek sümter tansık tatu toru tumağan tın uçarı yaprağı çaput öke'.split())
rows = []
for i in s:
    w, h = i['word'], i['headline']
    noroot = 'kök belirlenemedi' in i['provenance'] or h.lstrip('*') == w
    if not noroot:
        continue
    g = ce.query_glosses(w)
    fc = ce.form_candidates(w)
    m = ce.matches(w, g)
    r = dict(word=w, half=half[w], w23=w in W23, refs=i['reference'], glosses=g[:3],
             form=[(round(f, 2), x['headword'], x.get('gloss_tw') or x.get('gloss')) for f, x in fc[:4]],
             match=[(f, ms, x['headword'], x.get('base')) for f, ms, x in m])
    if m:
        for mode in ('head', 'base', 'chain'):
            b = ce.base_form(m[0][2], mode)
            sc = score_item(b, i['reference'])
            r[mode] = (b, sc['exact'], sc['tradition_equivalent'])
    rows.append(r)
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
print(len(rows), 'eşleşen', sum(1 for r in rows if r['match']), '23lük eşleşen', sum(1 for r in rows if r['match'] and r['w23']))
