"""45 kök-yok maddesinin sınıflaması (Starling kapalı başlık hataları)."""
import json, sys
from engine.search_engine import _index_source_proto
from engine.db.lexicon_index import LexiconIndex
from engine.nlp import derivation
from engine.utils.proto_notation import same_root_across_traditions
from engine.evaluation.metrics import best_match

d = json.load(open('data/eval/headline.json'))
s = d['subsets']['starling/starling_yok']
loc = {i['word']: i for i in d['subsets']['starling/yerel/tümü']['items']}
idx = LexiconIndex()
out = []
for i in s['items']:
    if i['exact']:
        continue
    h, w = i['headline'], i['word']
    noroot = 'kök belirlenemedi' in i['provenance'] or h.lstrip('*') == w
    if not noroot and 'ALINTI' not in i['provenance']:
        continue
    refs = i['reference']
    rows = idx.lookup(w, languages=['tr', 'ota'], limit=20)
    recs = [(r.get('lang_code'), r.get('origin'), r.get('pos'), (r.get('etymology') or '')[:0]) for r in rows]
    own, _ = _index_source_proto(w)
    ana = derivation.analyze(w)
    roots = []
    for a in ana[:4]:
        p, _ = _index_source_proto(a.root)
        m = bool(p) and (best_match(p if p.startswith('*') else '*'+p, refs)[1] or any(same_root_across_traditions(p, c) for c in refs))
        roots.append((a.formula, p, m))
    L = loc.get(w, {})
    out.append(dict(word=w, headline=h, prov=i['provenance'][:60], refs=refs, recs=recs, own=own,
                    roots=roots, starling_on=(L.get('headline'), L.get('exact'), L.get('provenance', '')[:50])))
for o in out:
    print(json.dumps(o, ensure_ascii=False))
