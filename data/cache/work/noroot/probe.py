import json, sys
from engine.evaluation.headline_eval import build_engine
W = sys.argv[1].split(",")
eng = build_engine(ablate_starling=True)
for w in W:
    f = eng.search(w, save_to_db=False, use_qwen_agent=False, use_cache=False)
    r = (f.get("nlp_analysis") or {}).get("reconstruction") or f.get("reconstruction") or {}
    if not r:
        for k, v in f.items():
            if isinstance(v, dict) and "reconstruction" in v: r = v["reconstruction"]
    tl = f.get("turkic_languages") or []
    print(json.dumps(dict(word=w, n_wit=len(tl), langs=sorted({e.get("lang_code") for e in tl})[:12],
        wit=[(e.get("lang_code"), e.get("word")) for e in tl][:8],
        method=r.get("method"), recon=r.get("reconstructed_root"), ok=r.get("is_reconstructible"),
        ev=r.get("evidence_available"), withheld=r.get("withheld_reconstruction"), wc=r.get("witness_count"),
        reason=(r.get("reason") or r.get("abstain_reason") or "")[:100], blocked=r.get("borrowing_blocked"),
        prov=f["root"]["provenance"][:80]), ensure_ascii=False), flush=True)
