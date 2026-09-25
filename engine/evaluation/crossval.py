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

Sinir ağı ikinci üreteç (``neural_reconstruction``, ``make eval-cv-neural``,
2026-09-25, ön kayıt ``data/cache/work/neural/PREREG.md``): kaynak
belirteçli küçük karakter Transformer, TRAIN + Starling (aynı sızıntı
süzgeci). Sütun modeline karşı NED +0,011 GA[−0,014, +0,037], tam
−0,022, BCFS −0,036 GA[−0,061, −0,008] -> RED. Işın log-olasılığıyla
yeniden sıralama birleşimi sinir top-1'iyle aynı -> RED. Ancak ışın-5
kapsamı 0,472 (sütun N-best tavanı 500'de 0,338): üreteç olarak umut
verici, seçici olarak değil.

Sinir üretir, sütun seçer (ön kayıt 2, 3 aday, Holm): B2 ``nsel_ranker``
(TRAIN'de 3 iç katla öğrenilen lojistik sıralayıcı, ışın-10 ∪ sütun top-1)
NED 0,3100 vs 0,3282, fark −0,018 GA[−0,034, −0,003], tam +0,025
GA[+0,003, +0,050], BCFS +0,012 (anlamlı değil) — ham p=0,020, Holm
p=0,059 -> ön kayıtlı ölçüt sağlanmadı, RED. B1 (sütun log-olasılığı)
−0,015 anlamlı değil; B3 (sıra füzyonu) +0,001. Işın kapsamı 5'te 0,481,
10'da 0,553. Üretime bağlanmadı.
"""

from __future__ import annotations

import os
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


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], iterations: int = 10000) -> float:
    """Eşleşmiş bootstrap ile iki yönlü p: ortalama(a − b) örneklerinin sıfırın öbür yanındaki payı."""
    from engine.evaluation.significance import SIGNIFICANCE_SEED

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    if not diffs:
        return 1.0
    rng = random.Random(SIGNIFICANCE_SEED)
    size = len(diffs)
    means = [sum(diffs[rng.randrange(size)] for _ in range(size)) / size for _ in range(iterations)]
    below = sum(m <= 0 for m in means) / iterations
    above = sum(m >= 0 for m in means) / iterations
    return round(min(1.0, 2 * min(below, above)), 5)


def holm_adjust(p: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni düzeltmesi (tekdüze)."""
    ordered = sorted(p, key=p.get)
    out: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p[key]))
        out[key] = round(running, 5)
    return out


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
    # Sinir ağı ikinci üreteç (``neural_reconstruction``) — torch ve Starling
    # gerekir, kat başına ~10 dk; yalnız ``CV_NEURAL=1`` (``make eval-cv-neural``).
    neural_enabled = bool(starling_sets) and os.environ.get("CV_NEURAL") == "1"
    if neural_enabled:
        from engine.nlp import neural_reconstruction as neural

        neural_starling = neural.prepare_starling()
        systems += ("neural", "neural_rerank", "nsel_column", "nsel_ranker", "nsel_vote")
    neural_meta: dict[str, Any] = {"seconds_per_fold": [], "maxrss_mb": 0.0, "beam5": [], "beam10": []}
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

    def _neural_inputs(item: Any) -> tuple[str, list[Any]] | None:
        """Harness'in motora verdiği ``(çapa, tanıklar)`` — çapa dili çıkarılmış."""
        witnesses = harness._witnesses_for(item, mapping)
        anchor, anchor_lang = harness._anchor_for(witnesses)
        if not anchor:
            return None
        return anchor, [w for w in witnesses if w["lang_code"] != anchor_lang]

    def _neural_examples(train_items: list[Any], excluded_items: list[Any]) -> list[Any]:
        excluded_tr = set().union(*(gold_sets[it.set_id].turkish for it in excluded_items)) | test_turkish
        excluded_protos = {proto_norm(g) for it in excluded_items for g in it.gold_candidates}
        examples = [e for e in (neural.gold_example(it, mapping) for it in train_items) if e]
        return examples + neural.starling_allowed(neural_starling, excluded_tr, excluded_protos)

    def _ranker(fold: int, held_items: list[Any], train_items: list[Any]) -> tuple[dict[str, float], float]:
        """B2 sıralayıcı: YALNIZ TRAIN, 3 iç kat (ön kayıt 2)."""
        import gc

        from engine.evaluation.metrics import best_match

        rows: list[tuple[dict[str, float], int]] = []
        for g in range(3):
            inner_held = [it for it in train_items if fold_of(it.concept + "#rank", 3) == g]
            inner_train = [it for it in train_items if fold_of(it.concept + "#rank", 3) != g]
            excluded = held_items + inner_held
            inner_table = learn_table([
                (it.gold_form, {mapping[lang]: f for lang, f in it.witnesses.items() if lang in mapping})
                for it in inner_train
            ])
            allowed = column_model.starling_allowed(
                starling_sets,
                set().union(*(gold_sets[it.set_id].turkish for it in excluded)) | test_turkish,
                {proto_norm(x) for it in excluded for x in it.gold_candidates},
            )
            inner_col, _ = column_model.train(
                gold_sets, [it.set_id for it in inner_train], allowed, inner_table, fold_of=fold_of
            )
            inner_neural = neural.train(_neural_examples(inner_train, excluded), seed=100 + 3 * fold + g)
            proto_phonology._PATTERN_TABLE = inner_table
            proto_phonology._PATTERN_TABLE_LOADED = True
            column_model.set_model(inner_col)
            recon = harness.comparative_reconstructor()
            for item in inner_held:
                inputs = _neural_inputs(item)
                if inputs is None:
                    continue
                word, entries = inputs
                result = harness.run(recon, [item], mapping=mapping)
                column_pred = result.pairs[0][0] if result.pairs else None
                informative = gold_sets[item.set_id].informative
                scored = inner_col.score_columns(informative, inner_table)[1] if informative else []
                cands = neural.build_candidates(inner_neural, scored, informative, word, entries, column_pred)
                for c in cands:
                    feats = neural.candidate_features(c, len(informative), word, entries)
                    rows.append((feats, int(best_match(c.text, item.gold_candidates)[1])))
            del inner_neural
            gc.collect()
        weights, intercept = column_model.fit_logistic(rows)
        logger.info("sıralayıcı kat %d: %d satır, ağırlıklar %s", fold, len(rows), weights)
        return weights, intercept

    def _neural_fold(fold: int, held_items: list[Any], train_items: list[Any], col_model: Any, table: Any) -> None:
        import gc
        import resource
        import time

        from engine.evaluation.metrics import best_match

        start = time.time()
        # Sütun modelinin bu kattaki tahminleri (son len(held) madde).
        column_preds = pairs["column_model"][-len(held_items):]
        weights, intercept = _ranker(fold, held_items, train_items)
        proto_phonology._PATTERN_TABLE = table
        proto_phonology._PATTERN_TABLE_LOADED = True
        column_model.set_model(col_model)
        examples = _neural_examples(train_items, held_items)
        model = neural.train(examples, seed=fold)
        for item, column_pair in zip(held_items, column_preds, strict=True):
            column_pred = column_pair[0] if column_pair else None
            informative = gold_sets[item.set_id].informative
            scored = col_model.score_columns(informative, table)[1] if informative else []
            memo: dict[str, Any] = {}

            def candidates(word: str, entries: list[Any], memo: dict[str, Any] = memo,
                           informative: list[Any] = informative, scored: Any = scored,
                           column_pred: str | None = column_pred) -> list[Any]:
                if "c" not in memo:
                    memo["c"] = neural.build_candidates(model, scored, informative, word, entries, column_pred)
                return memo["c"]

            def wrap(select: Any, candidates: Any = candidates) -> Any:
                def chooser(word: str, entries: list[Any]) -> dict[str, Any]:
                    chosen = select(candidates(word, entries), word, entries)
                    return {"reconstructed_root": chosen or "", "is_reconstructible": bool(chosen)}
                return chooser

            def rerank(word: str, entries: list[Any], column_pred: str | None = column_pred) -> dict[str, Any]:
                scored_nb = [(lp, "*" + text) for lp, text in model.beam(word, entries)]
                if column_pred:
                    scored_nb.append((model.score(word, entries, column_pred), column_pred))
                if not scored_nb:
                    return {"reconstructed_root": "", "is_reconstructible": False}
                return {"reconstructed_root": max(scored_nb)[1], "is_reconstructible": True}

            n_cols = len(informative)
            for name, rec in (
                ("neural", model.reconstruct),
                ("neural_rerank", rerank),
                ("nsel_column", wrap(lambda c, w, e: neural.select_column(c))),
                ("nsel_ranker", wrap(lambda c, w, e, n_cols=n_cols: neural.select_ranker(c, weights, intercept, n_cols, w, e))),
                ("nsel_vote", wrap(lambda c, w, e: neural.select_vote(c))),
            ):
                result = harness.run(rec, [item], mapping=mapping)
                correct[name] += result.item_correct
                ned[name] += result.item_ned
                pairs[name].append(result.pairs[0] if result.pairs else None)
            beam = sorted((c for c in memo.get("c", []) if c.in_beam), key=lambda c: c.rank)
            for width in (5, 10):
                neural_meta[f"beam{width}"].append(
                    any(best_match(c.text, item.gold_candidates)[1] for c in beam[:width])
                )
        gc.collect()
        neural_meta["seconds_per_fold"].append(round(time.time() - start))
        neural_meta["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        logger.info("neural kat %d: %d örnek, %.0f sn", fold, len(examples), time.time() - start)

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
            if neural_enabled:
                _neural_fold(fold, held, train, model, table)
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
    if neural_enabled:
        pairs_to_compare += [(c, "column_model") for c in systems if c.startswith(("neural", "nsel_"))]
    for a, b in pairs_to_compare:
        exact = bootstrap_metric_difference([float(x) for x in correct[a]], [float(x) for x in correct[b]])
        dist = bootstrap_metric_difference(ned[a], ned[b], lower_is_better=True)
        bcfs = bootstrap_bcubed_difference(columns[a], columns[b])
        comparisons[f"{a}-{b}"] = {"exact": exact, "ned": dist, "bcfs": bcfs}
    neural_result = None
    if neural_enabled:
        # PREREG (data/cache/work/neural/PREREG.md): NED farkı GA üst ucu < 0
        # VE iki aday üzerinde Holm sonrası p < 0,05.
        def _prereg(cands: tuple[str, ...]) -> dict[str, Any]:
            raw = {c: paired_bootstrap_p(ned[c], ned["column_model"]) for c in cands}
            holm = holm_adjust(raw)
            out = {}
            for c in cands:
                cmp = comparisons[f"{c}-column_model"]
                out[c] = {
                    "ned_diff": cmp["ned"]["difference"],
                    "ci95": cmp["ned"]["ci95"],
                    "bcfs_diff": cmp["bcfs"]["difference"],
                    "bcfs_ci95": cmp["bcfs"]["ci95"],
                    "p": raw[c],
                    "p_holm": holm[c],
                    "accepted": bool(cmp["ned"]["ci95"][1] < 0 and holm[c] < 0.05 and cmp["bcfs"]["ci95"][1] >= 0),
                }
            return out

        neural_result = {
            **neural_meta,
            "beam5": round(sum(neural_meta["beam5"]) / n, 4),
            "beam10": round(sum(neural_meta["beam10"]) / n, 4),
            # Ön kayıt 1: sinir tek başına / sinir yeniden sıralama (Holm, 2 aday).
            "prereg": _prereg(("neural", "neural_rerank")),
            # Ön kayıt 2: sinir üretir, sütun seçer (Holm, 3 aday; BCFS anlamlı kötüleşmez).
            "prereg2": _prereg(("nsel_column", "nsel_ranker", "nsel_vote")),
        }
    h2 = comparisons["learned_table-majority_character"]["bcfs"]
    return {
        "dataset": dataset,
        "k": k,
        "n": n,
        "systems": summary,
        "comparisons": comparisons,
        "column_model_starling_per_fold": starling_used,
        "neural": neural_result,
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
    if payload.get("neural"):
        nr = payload["neural"]
        print(f"\n  neural: kat başına sn {nr['seconds_per_fold']} · maxrss {nr['maxrss_mb']} MB · "
              f"ışın-5 kapsamı {nr['beam5']:.4f} · ışın-10 {nr['beam10']:.4f}")
        for key in ("prereg", "prereg2"):
            for c, r in nr[key].items():
                print(f"  {key.upper()} {c}: NED {r['ned_diff']:+.4f} GA{r['ci95']} p={r['p']} Holm p={r['p_holm']} "
                      f"BCFS {r['bcfs_diff']:+.4f} GA{r['bcfs_ci95']} -> {'KABUL' if r['accepted'] else 'RED'}")
    out = EVAL_DIR / "crossval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
