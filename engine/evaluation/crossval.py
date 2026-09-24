"""
Rekonstrüksiyon için 5 katlı çapraz doğrulama — ``make eval-cv``.

Neden: dev bölümü 83 madde. O büyüklükte NED'de 0,035'lik bir fark anlamlı
çıkamıyor ve "motor taban çizgisini geçmiyor" hükmü aslında "ölçüm
ayırt edemiyor" demek olabiliyordu. train+dev (320 madde) kavram bazlı beş
kata bölünür; her katta öğrenilmiş örüntü tablosu (``proto_patterns``)
YALNIZ öbür dört kattan yeniden öğrenilir, böylece sınanan kat hiç
görülmez. Test bölümü okunmaz.

Ölçüldü (2026-09-24, n=320)::

    sistem                 tam      NED      BCFS
    yalnız kural           0,2531   0,3679   0,5414
    öğrenilmiş tablo       0,2719   0,3426   0,5504
    majority_character     0,2531   0,3709   0,5192
    tablo − taban çizgisi  NED −0,0283  GA[−0,047, −0,009]  ANLAMLI
                           BCFS +0,0311 GA[+0,012, +0,047]  ANLAMLI
    tablo − yalnız kural   BCFS +0,0089 GA[+0,001, +0,017]  ANLAMLI

BCFS havuzlanmış 320 tahmin üzerinde veri kümesi düzeyinde hesaplanır;
fark GA'sı maddeler üzerinde eşleşmiş bootstrap'tir (her örneklemde iki
sistemin BCFS'si yeniden hesaplanır). Ön kayıttaki H2 (motor BCFS'de
``majority_character``ı geçer, GA sıfırı dışlar) bu koşuda **destekleniyor**
— ancak train+dev çapraz doğrulamasıdır, dondurulmuş test bölümü değil.

Denenip kazanç vermeyen: Starling ``turcet`` kökleri tabloya ek eğitim
verisi (154 -> 1.301 küme, sınanan kat ve test kavramlarıyla aynı Türkçe
biçimi taşıyanlar elendi): tam +0,003 GA[−0,006, +0,013], NED +0,0006.
Bu tablo (dil, ses) başına oy dağılımı öğreniyor; dağılımlar 154 kümeyle
zaten oturmuş.

Denenip kazanç vermeyen (2): konum duyarlı tablo (baş/orta/son, konumsuz
sayıma geri çekilmeli): tam +0,003; Starling ile birlikte +0,006. İkisi de
anlamlı değil.

Hata dökümü (öğrenilmiş tablo, n=320): doğru 87 · ses hatası 98 · söz başı
yanlış 48 · ek soyulmamış 38 · ünlü uzunluğu 22 · çapa kısa 16. Tahminlerin
%35'inde uzunluk bile tutmuyor; ses tablosu bunları düzeltemez.

- "Ek soyulmamış" çoğunlukla altın kökün tanıklardan KISA olmasıdır
  (tanık yaχšï "iyi", altın *jak): görev tanımı, düzeltilecek hata değil.
- "Söz başı" hatalarının yarısı ön ses ötümlülüğü (*t/*d 17, *k/*g 7) ve
  altının kendisi tutarsız: Oğuz tanıkları çoğunlukla d- iken altın 11 kez
  *d, 8 kez *t; g- iken 5 kez *g, 7 kez *k. Kural yazı-tura düzeyinde
  kalır (d için ~+3, g için ~−2 madde) ve 320 maddeye uydurulmuş olur.

Sütun modeli (``column_model``, 2026-09-24): her katta öbür katlar + Starling
turcet (katta 1.719–1.798 kök; sınanan/test Türkçe biçimi veya sınanan kökle
çakışanlar elenir) ile eğitilir. Tabloya karşı (tanıksız kök yasağı dahil):
tam 0,2969 vs 0,2719 GA[+0,006, +0,047] · NED 0,3303 vs 0,3445
GA[−0,026, −0,003] ANLAMLI · BCFS +0,013 GA[+0,001, +0,024].

Aday listesi tavanı (konformal kümeler için üst sınır): doğru cevabın N-best
listesinde bulunma oranı 5 adayda 0,309, 20'de 0,316, 100'de 0,334, 500'de
0,338 (top-1 0,272). Adaylar aynı iskelette sütun sütun ses değiştirerek
üretildiği için uzunluk hatalarını (%35) hiç kapsamaz; "%80 kapsama
garantili" aday kümesi bu üreteçle MÜMKÜN DEĞİLDİR. Darboğaz küme seçimi
değil, aday üretimidir.
"""

from __future__ import annotations

import random
import zlib
from collections import Counter
from collections.abc import Sequence
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


def item_columns(pair: tuple[str, str] | None) -> Counter[tuple[str, str]]:
    """Bir maddenin ``(tahmin_sesi, altın_sesi)`` sütun sayımı.

    ``metrics.reconstruction_bcubed`` ile AYNI hizalama (``trim=False``,
    boşluk sütunları dahil). Çekimser madde (``None``) sütun üretmez —
    veri kümesi düzeyindeki hesapta da hiçbir konum eklemez.
    """
    from engine.evaluation.metrics import normalize_proto
    from engine.nlp.multi_alignment import GAP, align_forms

    counts: Counter[tuple[str, str]] = Counter()
    if pair is None:
        return counts
    prediction, gold = normalize_proto(pair[0]), normalize_proto(pair[1])
    if not prediction or not gold:
        return counts
    for column in align_forms({"pred": prediction, "gold": gold}, trim=False):
        left = column.sounds.get("pred", GAP) or GAP
        right = column.sounds.get("gold", GAP) or GAP
        if left == GAP and right == GAP:
            continue
        counts[(left, right)] += 1
    return counts


def bcubed_from_counts(counts: Counter[tuple[str, str]]) -> float:
    """Sütun sayımından B-Cubed F — ``reconstruction_bcubed`` ile eşdeğer.

    Aynı ``(p, g)`` karşılığını taşıyan her konumun kesinliği ``c(p,g)/c(p)``,
    duyarlılığı ``c(p,g)/c(g)``; toplam ``Σ c(p,g)²/c(p)`` biçimine iner.
    Konumları tek tek gezmeden bootstrap'i hızlı kılar.
    """
    total = sum(counts.values())
    if not total:
        return 0.0
    by_pred: Counter[str] = Counter()
    by_gold: Counter[str] = Counter()
    for (p, g), c in counts.items():
        by_pred[p] += c
        by_gold[g] += c
    precision = sum(c * c / by_pred[p] for (p, _), c in counts.items()) / total
    recall = sum(c * c / by_gold[g] for (_, g), c in counts.items()) / total
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap_bcubed_difference(
    a: Sequence[Counter[tuple[str, str]]],
    b: Sequence[Counter[tuple[str, str]]],
    *,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Maddeler üzerinde **eşleşmiş** bootstrap: BCFS(a) − BCFS(b).

    B-Cubed veri kümesi düzeyinde bir ölçüdür, madde ortalaması değildir;
    bu yüzden her örneklemde iki sistemin BCFS'si AYNI yeniden örneklenmiş
    madde kümesi üzerinden baştan hesaplanır. Daha yüksek daha iyidir.
    """
    from engine.evaluation.significance import SIGNIFICANCE_SEED

    if len(a) != len(b) or not a:
        return {"difference": 0.0, "ci95": [0.0, 0.0], "significant": False, "n": len(a)}
    rng = random.Random(SIGNIFICANCE_SEED if seed is None else seed)
    size = len(a)
    observed = bcubed_from_counts(sum(a, Counter())) - bcubed_from_counts(sum(b, Counter()))
    samples: list[float] = []
    for _ in range(iterations):
        weights = Counter(rng.randrange(size) for _ in range(size))
        pooled_a: Counter[tuple[str, str]] = Counter()
        pooled_b: Counter[tuple[str, str]] = Counter()
        for index, weight in weights.items():
            for key, c in a[index].items():
                pooled_a[key] += c * weight
            for key, c in b[index].items():
                pooled_b[key] += c * weight
        samples.append(bcubed_from_counts(pooled_a) - bcubed_from_counts(pooled_b))
    samples.sort()
    low = samples[int(alpha / 2 * iterations)]
    high = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    excludes_zero = low > 0 or high < 0
    return {
        "difference": round(observed, 5),
        "ci95": [round(low, 5), round(high, 5)],
        "significant": excludes_zero,
        "a_is_better": bool(observed > 0 and excludes_zero),
        "n": size,
    }


def run(dataset: str = "savelyevturkic", k: int = K) -> dict[str, Any]:
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation import harness
    from engine.evaluation.baselines import majority_character
    from engine.evaluation.gold import GoldStandard
    from engine.evaluation.metrics import reconstruction_bcubed
    from engine.evaluation.significance import bootstrap_metric_difference
    from engine.nlp import proto_phonology

    gold = GoldStandard.build(dataset)
    items = list(gold.split("train")) + list(gold.split("dev"))
    mapping = build_mapping(CldfWordlist.load(dataset))

    from engine.nlp import column_model
    from engine.nlp.column_model import norm as proto_norm

    # Sütun modeli (``column_model``): her katta YALNIZ öbür katlardan + Starling'den
    # eğitilir. Starling hizalamaları ve altın sütunları kattan bağımsızdır, bir kez kurulur.
    starling_sets = column_model.prepare_starling()
    gold_sets = column_model.prepare_gold(items, mapping)
    # ⚠️ Test bölümünden YALNIZ Türkçe tanık biçimi okunur (Starling sızıntı süzgeci).
    test_turkish = set().union(
        *(column_model.gold_turkish(it, mapping) for it in gold.items if it.split == "test")
    )
    if not starling_sets:
        logger.warning("Starling turcet yok; column_model sistemi atlanıyor (`make starling`)")
    systems = ("rules", "learned_table", "majority_character") + (("column_model",) if starling_sets else ())
    starling_used: list[int] = []
    correct: dict[str, list[bool]] = {s: [] for s in systems}
    ned: dict[str, list[float]] = {s: [] for s in systems}
    # Madde başına (tahmin, altın) çifti — çekimserse None. B-Cubed veri
    # kümesi düzeyinde olduğu için bootstrap'te maddeyle birlikte taşınmalı.
    pairs: dict[str, list[tuple[str, str] | None]] = {s: [] for s in systems}
    held: list[Any] = []

    def _collect(name: str, reconstructor: Any, extra: dict[str, Any]) -> None:
        # Madde madde koşulur: harness ``pairs`` listesi çekimser maddeleri
        # atladığı için toplu koşuda çift -> madde eşlemesi kurulamaz.
        for item in held:
            result = harness.run(reconstructor, [item], mapping=mapping, **extra)
            correct[name] += result.item_correct
            ned[name] += result.item_ned
            pairs[name].append(result.pairs[0] if result.pairs else None)

    try:
        for fold in range(k):
            held = [it for it in items if fold_of(it.concept, k) == fold]
            train = [it for it in items if fold_of(it.concept, k) != fold]
            table = learn_table([
                (it.gold_form, {mapping[lang]: f for lang, f in it.witnesses.items() if lang in mapping})
                for it in train
            ])
            column_model.set_model(None)
            for name, pattern_table in (("rules", None), ("learned_table", table)):
                proto_phonology._PATTERN_TABLE = pattern_table
                proto_phonology._PATTERN_TABLE_LOADED = True
                _collect(name, harness.comparative_reconstructor(), {})
            _collect("majority_character", majority_character, {"system": "majority"})
            if starling_sets:
                allowed = column_model.starling_allowed(
                    starling_sets,
                    set().union(*(gold_sets[it.set_id].turkish for it in held)) | test_turkish,
                    {proto_norm(g) for it in held for g in it.gold_candidates},
                )
                model, used = column_model.train(
                    gold_sets, [it.set_id for it in train], allowed, table, fold_of=fold_of
                )
                starling_used.append(used)
                proto_phonology._PATTERN_TABLE = table
                proto_phonology._PATTERN_TABLE_LOADED = True
                column_model.set_model(model)
                _collect("column_model", harness.comparative_reconstructor(), {})
    finally:
        # Diskteki (train'de öğrenilmiş) tablo ve sütun modeli bir sonraki kullanımda yeniden yüklensin.
        proto_phonology.reset_pattern_cache()
        column_model.reset_model_cache()

    n = len(correct["majority_character"])
    columns = {s: [item_columns(p) for p in pairs[s]] for s in systems}
    summary = {
        s: {
            "exact": round(sum(correct[s]) / n, 4),
            "ned": round(sum(ned[s]) / n, 4),
            # Havuzlanmış 320 tahmin üzerinde, harness'in kullandığı ölçü.
            "bcfs": reconstruction_bcubed([p for p in pairs[s] if p is not None])["fscore"],
        }
        for s in systems
    }
    comparisons = {}
    pairs_to_compare = [("learned_table", "majority_character"), ("learned_table", "rules")]
    if "column_model" in systems:
        pairs_to_compare.append(("column_model", "learned_table"))
    for a, b in pairs_to_compare:
        exact = bootstrap_metric_difference([float(x) for x in correct[a]], [float(x) for x in correct[b]])
        dist = bootstrap_metric_difference(ned[a], ned[b], lower_is_better=True)
        bcfs = bootstrap_bcubed_difference(columns[a], columns[b])
        comparisons[f"{a}-{b}"] = {"exact": exact, "ned": dist, "bcfs": bcfs}
    h2 = comparisons["learned_table-majority_character"]["bcfs"]
    return {
        "dataset": dataset,
        "k": k,
        "n": n,
        "systems": summary,
        "comparisons": comparisons,
        "column_model_starling_per_fold": starling_used,
        # PREREGISTRATION H2: motor majority_character'ı B-Cubed F'de geçer —
        # eşleşmiş bootstrap %95 GA sıfırı dışlar, motor lehine.
        "h2": {
            "criterion": "BCFS(motor) − BCFS(majority_character), eşleşmiş bootstrap %95 GA sıfırı dışlar, motor lehine",
            "system": "learned_table",
            "supported": bool(h2["a_is_better"]),
        },
    }


def main() -> int:
    import json

    from engine.evaluation.report import EVAL_DIR

    payload = run()
    print(f"\n=== rekonstrüksiyon · {payload['k']} katlı çapraz doğrulama · train+dev n={payload['n']} ===")
    for name, row in payload["systems"].items():
        print(f"  {name:20} tam {row['exact']:.4f}   NED {row['ned']:.4f}   BCFS {row['bcfs']:.4f}")
    for name, cmp in payload["comparisons"].items():
        e, d, b = cmp["exact"], cmp["ned"], cmp["bcfs"]
        print(f"  {name}: tam {e['difference']:+.4f} GA{e['ci95']} · NED {d['difference']:+.4f} "
              f"GA{d['ci95']} {'ANLAMLI' if d['significant'] else 'anlamlı değil'} · "
              f"BCFS {b['difference']:+.4f} GA{b['ci95']} {'ANLAMLI' if b['significant'] else 'anlamlı değil'}")
    h2 = payload["h2"]
    print(f"\n  H2 (ön kayıt, BCFS'de majority_character'ı geçer): "
          f"{'EVET — destekleniyor' if h2['supported'] else 'HAYIR — desteklenmiyor'}")
    out = EVAL_DIR / "crossval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
