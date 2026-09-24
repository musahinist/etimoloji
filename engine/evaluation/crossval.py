"""
Rekonstrüksiyon için 5 katlı çapraz doğrulama — ``make eval-cv``.

Neden: dev bölümü 83 madde. O büyüklükte NED'de 0,035'lik bir fark anlamlı
çıkamıyor ve "motor taban çizgisini geçmiyor" hükmü aslında "ölçüm
ayırt edemiyor" demek olabiliyordu. train+dev (320 madde) kavram bazlı beş
kata bölünür; her katta öğrenilmiş örüntü tablosu (``proto_patterns``)
YALNIZ öbür dört kattan yeniden öğrenilir, böylece sınanan kat hiç
görülmez. Test bölümü okunmaz.

Ölçüldü (2026-09-24, n=320)::

    sistem                 tam      NED
    yalnız kural           0,2531   0,3679
    öğrenilmiş tablo       0,2719   0,3426
    majority_character     0,2531   0,3709
    tablo − taban çizgisi  NED −0,0283  GA[−0,047, −0,009]  ANLAMLI

Denenip kazanç vermeyen: Starling ``turcet`` kökleri tabloya ek eğitim
verisi (154 -> 1.301 küme, sınanan kat ve test kavramlarıyla aynı Türkçe
biçimi taşıyanlar elendi): tam +0,003 GA[−0,006, +0,013], NED +0,0006.
Bu tablo (dil, ses) başına oy dağılımı öğreniyor; dağılımlar 154 kümeyle
zaten oturmuş. Veriden yararlanabilecek bağlam duyarlı bir model ayrıca
denenmeli.
"""

from __future__ import annotations

import zlib
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

K = 5


def fold_of(concept: str, k: int = K) -> int:
    """Kavramın katı — kararlı (CRC32), koşudan koşuya değişmez."""
    return zlib.crc32(concept.encode("utf-8")) % k


def learn_table(examples: list[tuple[str, dict[str, str]]]) -> Any:
    """``(ata biçim, {dil: biçim})`` örneklerinden örüntü tablosu.

    ``proto_patterns.learn`` ile AYNI kural: bilgi taşıyan sütun sayısı ata
    biçim uzunluğuna eşit değilse küme atlanır.
    """
    from engine.nlp.multi_alignment import align_forms
    from engine.nlp.proto_patterns import ProtoPatternTable
    from engine.utils.orthography import to_comparison_form

    table = ProtoPatternTable(trained_on="crossval")
    for proto, raw_forms in examples:
        proto = to_comparison_form(proto)
        forms = {k: to_comparison_form(v) for k, v in raw_forms.items() if to_comparison_form(v)}
        if len(forms) < 2 or not proto:
            continue
        columns = [c for c in align_forms(forms) if c.gap_ratio <= 0.5]
        if len(columns) != len(proto):
            continue
        table.n_sets += 1
        for column, proto_sound in zip(columns, proto, strict=True):
            table.n_columns += 1
            for language, sound in column.present.items():
                if sound:
                    table.observe(language, sound, proto_sound)
    return table


def run(dataset: str = "savelyevturkic", k: int = K) -> dict[str, Any]:
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation import harness
    from engine.evaluation.baselines import majority_character
    from engine.evaluation.gold import GoldStandard
    from engine.evaluation.significance import bootstrap_metric_difference
    from engine.nlp import proto_phonology

    gold = GoldStandard.build(dataset)
    items = list(gold.split("train")) + list(gold.split("dev"))
    mapping = build_mapping(CldfWordlist.load(dataset))

    systems = ("rules", "learned_table", "majority_character")
    correct: dict[str, list[bool]] = {s: [] for s in systems}
    ned: dict[str, list[float]] = {s: [] for s in systems}
    try:
        for fold in range(k):
            held = [it for it in items if fold_of(it.concept, k) == fold]
            train = [it for it in items if fold_of(it.concept, k) != fold]
            table = learn_table([
                (it.gold_form, {mapping[lang]: f for lang, f in it.witnesses.items() if lang in mapping})
                for it in train
            ])
            for name, pattern_table in (("rules", None), ("learned_table", table)):
                proto_phonology._PATTERN_TABLE = pattern_table
                proto_phonology._PATTERN_TABLE_LOADED = True
                result = harness.run(harness.comparative_reconstructor(), held, mapping=mapping)
                correct[name] += result.item_correct
                ned[name] += result.item_ned
            base = harness.run(majority_character, held, mapping=mapping, system="majority")
            correct["majority_character"] += base.item_correct
            ned["majority_character"] += base.item_ned
    finally:
        # Diskteki (train'de öğrenilmiş) tablo bir sonraki kullanımda yeniden yüklensin.
        proto_phonology.reset_pattern_cache()

    n = len(correct["majority_character"])
    summary = {s: {"exact": round(sum(correct[s]) / n, 4), "ned": round(sum(ned[s]) / n, 4)} for s in systems}
    comparisons = {}
    for a, b in (("learned_table", "majority_character"), ("learned_table", "rules")):
        exact = bootstrap_metric_difference([float(x) for x in correct[a]], [float(x) for x in correct[b]])
        dist = bootstrap_metric_difference(ned[a], ned[b], lower_is_better=True)
        comparisons[f"{a}-{b}"] = {"exact": exact, "ned": dist}
    return {"dataset": dataset, "k": k, "n": n, "systems": summary, "comparisons": comparisons}


def main() -> int:
    import json

    from engine.evaluation.report import EVAL_DIR

    payload = run()
    print(f"\n=== rekonstrüksiyon · {payload['k']} katlı çapraz doğrulama · train+dev n={payload['n']} ===")
    for name, row in payload["systems"].items():
        print(f"  {name:20} tam {row['exact']:.4f}   NED {row['ned']:.4f}")
    for name, cmp in payload["comparisons"].items():
        e, d = cmp["exact"], cmp["ned"]
        print(f"  {name}: tam {e['difference']:+.4f} GA{e['ci95']} · NED {d['difference']:+.4f} "
              f"GA{d['ci95']} {'ANLAMLI' if d['significant'] else 'anlamlı değil'}")
    out = EVAL_DIR / "crossval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
