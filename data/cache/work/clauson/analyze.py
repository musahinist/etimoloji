"""Clauson A/B çözümlemesi (ön kayıt PREREG.md). Kullanım: analyze.py <box_off> <box_on>

Kutular: aynı çalışma ağacı kodu, ETY_CLAUSON kapalı/açık; her biri
data/eval/{headline,chronology,homonym}.json yazar."""
import json
import sys
from math import comb
from pathlib import Path

from engine.evaluation.headline_eval import score_item, split_halves
from engine.fetchers import clauson_edt as ce

W23 = set('besim bürgü büz dala darga duma evin göbelek güce koçkar obuz savur sümek sümter tansık '
          'tatu toru tumağan tın uçarı yaprağı çaput öke'.split())


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(box, name):
    return json.load(open(Path(box) / 'data' / 'eval' / f'{name}.json'))


def cmp(old, new):
    """old/new: {kelime: bool}. (eski k, yeni k, kazanç, kayıp, p)."""
    win = sum(1 for w in old if not old[w] and new[w])
    loss = sum(1 for w in old if old[w] and not new[w])
    return sum(old.values()), sum(new.values()), win, loss, round(mcnemar(win, loss), 4)


off_box, on_box = sys.argv[1], sys.argv[2]
out = {}
H0, H1 = load(off_box, 'headline'), load(on_box, 'headline')
base_cache = {}


def clauson_base(word, mode):
    if word not in base_cache:
        m = ce.matches(word)
        base_cache[word] = m[0][2] if m else None
    rec = base_cache[word]
    return ce.base_form(rec, mode) if rec else ''


for name in H0['subsets']:
    it0 = {i['word']: i for i in H0['subsets'][name]['items']}
    it1 = {i['word']: i for i in H1['subsets'][name]['items']}
    half = split_halves(it0)
    for h in ('A', 'B', 'all'):
        ws = [w for w in it0 if w in it1 and (h == 'all' or half[w] == h)]
        res = {'n': len(ws)}
        for key in ('exact', 'tradition_equivalent'):
            res[f'motor_{key}'] = cmp({w: it0[w][key] for w in ws}, {w: it1[w][key] for w in ws})
        if 'starling_yok' in name:
            for mode in ('head', 'base', 'chain'):
                eff = {}
                changed = []
                for w in ws:
                    i = it1[w]
                    sc = {'exact': i['exact'], 'tradition_equivalent': i['tradition_equivalent']}
                    if 'kök belirlenemedi' in i['provenance']:
                        b = clauson_base(w, mode)
                        if b:
                            sc = score_item(b, i['reference'])
                            changed.append((w, b, sc['exact'], sc['tradition_equivalent']))
                    eff[w] = sc
                for key in ('exact', 'tradition_equivalent'):
                    res[f'etkin_{mode}_{key}'] = cmp({w: it0[w][key] for w in ws}, {w: eff[w][key] for w in ws})
                res[f'etkin_{mode}_changed'] = changed
        out[f'headline|{name}|{h}'] = res

# 45 kök-yok ve 23'lük sınıf (Starling kapalı, OFF koşusundaki kök-yok hatalar)
sy0 = {i['word']: i for i in H0['subsets']['starling/starling_yok']['items']}
sy1 = {i['word']: i for i in H1['subsets']['starling/starling_yok']['items']}
noroot = [w for w, i in sy0.items() if not i['exact']
          and ('kök belirlenemedi' in i['provenance'] or i['headline'].lstrip('*') == w)]
cls = []
for w in noroot:
    m = ce.matches(w)
    rec = m[0][2] if m else None
    row = {'word': w, 'w23': w in W23, 'match': rec['headword'] if rec else None,
           'on_headline': sy1[w]['headline'], 'on_trad': sy1[w]['tradition_equivalent'],
           'on_prov': sy1[w]['provenance'][:40]}
    if rec:
        for mode in ('head', 'base', 'chain'):
            b = ce.base_form(rec, mode)
            s = score_item(b, sy0[w]['reference'])
            row[mode] = [b, s['exact'], s['tradition_equivalent']]
    cls.append(row)
out['noroot'] = {'n': len(noroot), 'n23': sum(1 for w in noroot if w in W23),
                 'kapsam': sum(1 for r in cls if r['match']),
                 'kapsam23': sum(1 for r in cls if r['match'] and r['w23']),
                 'rows': cls}

# kronoloji
C0, C1 = load(off_box, 'chronology'), load(on_box, 'chronology')
c0 = {i['word']: i for i in C0['items']}
c1 = {i['word']: i for i in C1['items']}
half = split_halves(c0)
for cfg in ('yerel', 'starling_yok'):
    for h in ('A', 'B', 'all'):
        ws = [w for w in c0 if h == 'all' or half[w] == h]

        def within(i, k=cfg):
            y = i.get(f'{k}_year')
            return y is not None and abs(y - i['reference_year']) <= 100

        def point(i, k=cfg):
            return i.get(f'{k}_year') is not None and i.get(f'{k}_precision') == 'point'
        out[f'chron|{cfg}|{h}'] = {
            'n': len(ws),
            'yüzyıl_içi': cmp({w: within(c0[w]) for w in ws}, {w: within(c1[w]) for w in ws}),
            'kapsam': cmp({w: c0[w].get(f'{cfg}_year') is not None for w in ws},
                          {w: c1[w].get(f'{cfg}_year') is not None for w in ws}),
            'nokta_kapsam': cmp({w: point(c0[w]) for w in ws}, {w: point(c1[w]) for w in ws}),
            'aynı_eser': cmp({w: c0[w].get(f'{cfg}_year') is not None and abs(c0[w][f'{cfg}_year'] - c0[w]['reference_year']) <= 5 for w in ws},
                             {w: c1[w].get(f'{cfg}_year') is not None and abs(c1[w][f'{cfg}_year'] - c1[w]['reference_year']) <= 5 for w in ws}),
        }
out['chron_systems'] = {'off': C0['systems'], 'on': C1['systems']}

# eşsesli
M0, M1 = load(off_box, 'homonym'), load(on_box, 'homonym')
for cfg in M0['configs']:
    s0, s1 = M0['configs'][cfg]['summary'], M1['configs'][cfg]['summary']
    out[f'homonym|{cfg}'] = {'a_fark_ediyor': (s0['a']['a_fark_ediyor']['k'], s1['a']['a_fark_ediyor']['k']),
                             'b_E1_payi': (s0['b']['baslik_E1_payi'], s1['b']['baslik_E1_payi']),
                             'tutarsiz_baslik': (s0['b']['tutarsiz_baslik']['k'], s1['b']['tutarsiz_baslik']['k']),
                             'n': (s0['n'], s1['n'])}
json.dump(out, open(Path(on_box).parent / 'analysis.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items():
    if k in ('noroot', 'chron_systems'):
        continue
    print(k, json.dumps({a: b for a, b in v.items() if not a.endswith('changed')}, ensure_ascii=False))
print('noroot', out['noroot']['n'], 'n23', out['noroot']['n23'], 'kapsam', out['noroot']['kapsam'],
      'kapsam23', out['noroot']['kapsam23'])
for r in cls:
    if r['match']:
        print('  ', json.dumps(r, ensure_ascii=False))
