"""
İstatistiksel anlamlılık — "daha iyi" demenin şartı.

Önceki taslakta hiç yoktu ve bu, hakem raporunun "kabul öncesi mutlaka
düzeltilmeli" listesindeydi. İki ayrı soruya cevap verir:

**1. Sistem A gerçekten B'den iyi mi?**
    ``%22,3`` ile ``%23,7`` arasındaki fark n=400'de gürültü olabilir.
    :func:`permutation_test` ve :func:`mcnemar_test` bunu ayırır.
    "10 koşu + varyans" yalnız koşu-içi gürültüyü ölçer; deterministik bir
    sistemde o sıfırdır ve hiçbir şey söylemez.

**2. Bulunan benzerlik şans eseri olabilir mi?**
    Motor 18 dilde, yüz binlerce kelimelik bir indekste öngörülen biçme
    benzeyen bir şey **arıyor**. Kısa bir CVC öngörüsü salt şansla eşleşme
    bulur. Bu, devasa bir çoklu karşılaştırma problemidir ve kontrolsüz
    bırakılırsa "yeni etimoloji bulduk" iddiası uzak-akrabalık amatörlüğüyle
    aynı kefeye konur.

    :func:`chance_resemblance_test` — Kessler (2001), *The Significance of
    Word Lists* yönteminin uyarlaması: anlam eşlemesi karıştırılarak bir
    **null model** kurulur ve gözlenen benzerlik ona karşı sınanır.
    :func:`benjamini_hochberg` çoklu karşılaştırma için FDR kontrolü uygular.

Tüm rastgelelik tohumlanır: aynı veri, aynı p değeri.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

SIGNIFICANCE_SEED = 20260827


@dataclass(frozen=True)
class TestResult:
    """Bir anlamlılık testinin sonucu."""

    name: str
    statistic: float
    p_value: float
    n: int
    note: str = ""

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "test": self.name,
            "statistic": round(self.statistic, 4),
            "p": round(self.p_value, 5),
            "n": self.n,
            "significant": self.significant,
            "note": self.note,
        }


def permutation_test(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    iterations: int = 10000,
    seed: int = SIGNIFICANCE_SEED,
) -> TestResult:
    """İki sistemin **eşleşmiş** doğruluk farkı anlamlı mı?

    Aynı maddeler üzerinde çalıştıkları için eşleşmiş test kullanılır: her
    maddede iki sistemin sonucu rastgele yer değiştirilir ve gözlenen fark
    bu null dağılıma karşı sınanır.
    """
    if len(a) != len(b) or not a:
        return TestResult("permutation", 0.0, 1.0, len(a), "veri yok veya eşleşmiyor")

    observed = (sum(a) - sum(b)) / len(a)
    rng = random.Random(seed)
    pairs = list(zip(a, b, strict=True))
    extreme = 0
    for _ in range(iterations):
        diff = 0
        for x, y in pairs:
            if rng.random() < 0.5:
                x, y = y, x
            diff += int(x) - int(y)
        if abs(diff / len(pairs)) >= abs(observed):
            extreme += 1
    p = (extreme + 1) / (iterations + 1)
    return TestResult(
        "permutation",
        observed,
        p,
        len(a),
        f"gözlenen fark {observed:+.4f}",
    )


def mcnemar_test(a: Sequence[bool], b: Sequence[bool]) -> TestResult:
    """Eşleşmiş ikili sonuçlar için McNemar (kesin binom sürümü).

    Yalnız **uyuşmayan** çiftlere bakar: ikisinin de doğru veya ikisinin de
    yanlış olduğu maddeler bilgi taşımaz.
    """
    if len(a) != len(b) or not a:
        return TestResult("mcnemar", 0.0, 1.0, len(a), "veri yok veya eşleşmiyor")

    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    discordant = only_a + only_b
    if discordant == 0:
        return TestResult("mcnemar", 0.0, 1.0, 0, "hiç uyuşmazlık yok")

    # İki yanlı kesin binom testi, p = 0,5
    from math import comb

    def tail(k: int) -> float:
        return sum(comb(discordant, i) for i in range(k + 1)) / (2**discordant)

    smaller = min(only_a, only_b)
    p = min(1.0, 2 * tail(smaller))
    return TestResult(
        "mcnemar",
        float(only_a - only_b),
        p,
        discordant,
        f"yalnız A doğru: {only_a}, yalnız B doğru: {only_b}",
    )


def bootstrap_difference(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int = SIGNIFICANCE_SEED,
) -> tuple[float, float, float]:
    """İki sistem farkının bootstrap güven aralığı.

    :returns: ``(fark, alt_sınır, üst_sınır)``. Aralık sıfırı içeriyorsa fark
        anlamlı değildir.
    """
    if len(a) != len(b) or not a:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    size = len(a)
    observed = (sum(a) - sum(b)) / size
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rng.randrange(size) for _ in range(size)]
        samples.append(sum(int(a[i]) - int(b[i]) for i in picked) / size)
    samples.sort()
    low = samples[int(alpha / 2 * iterations)]
    high = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (observed, low, high)


def benjamini_hochberg(p_values: Sequence[float], *, fdr: float = 0.05) -> list[bool]:
    """Çoklu karşılaştırma için FDR kontrolü.

    Yüzlerce öngörü aynı anda sınandığında, salt şansla bazıları ``p < 0,05``
    çıkar. Benjamini-Hochberg, yanlış keşif oranını ``fdr`` düzeyinde tutar.

    :returns: her p değeri için "anlamlı sayılsın mı" kararı, **girdi
        sırasıyla**.
    """
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    total = len(p_values)
    cutoff_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= fdr * rank / total:
            cutoff_rank = rank
    decisions = [False] * total
    for rank, (original_index, _) in enumerate(indexed, start=1):
        if rank <= cutoff_rank:
            decisions[original_index] = True
    return decisions


def chance_resemblance_test(
    observed_matches: int,
    query_forms: Sequence[str],
    candidate_pool: Sequence[str],
    similarity: Callable[[str, str], bool],
    *,
    iterations: int = 1000,
    seed: int = SIGNIFICANCE_SEED,
) -> TestResult:
    """Gözlenen benzerlik sayısı şans eseri olabilir mi? (Kessler 2001)

    Null model: sorgu biçimleri **aynı havuzdan rastgele** seçilseydi kaç
    eşleşme beklenirdi? Gözlenen sayı bu dağılımın üst kuyruğunda değilse,
    "eşleşme bulduk" bir bulgu değildir.

    :param observed_matches: gerçekte bulunan eşleşme sayısı
    :param query_forms: aranan biçimler
    :param candidate_pool: içinde arandığı havuz
    :param similarity: iki biçmin "eşleşti" sayılıp sayılmayacağı
    """
    if not query_forms or not candidate_pool:
        return TestResult("chance_resemblance", 0.0, 1.0, 0, "veri yok")

    rng = random.Random(seed)
    pool = list(candidate_pool)
    null_counts: list[int] = []
    for _ in range(iterations):
        # Sorgu sayısı kadar RASTGELE biçim çek ve aynı ölçütle eşleştir.
        decoys = [rng.choice(pool) for _ in query_forms]
        count = 0
        for decoy in decoys:
            target = rng.choice(pool)
            if similarity(decoy, target):
                count += 1
        null_counts.append(count)

    at_least_as_extreme = sum(1 for c in null_counts if c >= observed_matches)
    p = (at_least_as_extreme + 1) / (iterations + 1)
    expected = sum(null_counts) / len(null_counts)
    return TestResult(
        "chance_resemblance",
        float(observed_matches),
        p,
        len(query_forms),
        f"şans beklentisi {expected:.1f}, gözlenen {observed_matches}",
    )


def compare_systems(
    results: dict[str, Sequence[bool]],
    *,
    reference: str,
) -> list[dict[str, object]]:
    """Bütün sistemleri bir referansa karşı sınar ve FDR uygular.

    Raporlanan her karşılaştırma için hem p değeri hem bootstrap aralığı
    verilir; aralık sıfırı içeriyorsa fark anlamlı değildir.
    """
    if reference not in results:
        raise KeyError(f"referans sistem yok: {reference}")

    baseline = results[reference]
    rows: list[dict[str, object]] = []
    for name, outcome in results.items():
        if name == reference:
            continue
        perm = permutation_test(outcome, baseline)
        mcn = mcnemar_test(outcome, baseline)
        diff, low, high = bootstrap_difference(outcome, baseline)
        rows.append(
            {
                "system": name,
                "vs": reference,
                "difference": round(diff, 4),
                "ci95": [round(low, 4), round(high, 4)],
                "permutation_p": round(perm.p_value, 5),
                "mcnemar_p": round(mcn.p_value, 5),
                "ci_excludes_zero": low > 0 or high < 0,
            }
        )

    decisions = benjamini_hochberg([float(r["permutation_p"]) for r in rows])
    for row, significant in zip(rows, decisions, strict=True):
        row["significant_after_fdr"] = significant
    return rows


def bootstrap_metric_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int = SIGNIFICANCE_SEED,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """**Sürekli** metrikler (NED, ED, FER) için eşleşmiş bootstrap farkı.

    ⚠️ :func:`permutation_test` ve :func:`mcnemar_test` yalnız ikili
    doğru/yanlış bayrakları üzerinde çalışır. NED/ED/FER sürekli ölçülerdir
    ve alanın **birincil** metrikleridir (SIGTYP 2022); onlar için ayrı bir
    test gerekir. Bu eksik olduğu sürece birincil metriklerde "anlamlı fark"
    iddiası kurulamazdı.

    :param lower_is_better: ED/NED/FER için ``True`` — fark işareti buna
        göre yorumlanır ve "A daha iyi" doğru yönde raporlanır.
    """
    if len(a) != len(b) or not a:
        return {"difference": 0.0, "ci95": [0.0, 0.0], "significant": False, "n": len(a)}

    rng = random.Random(seed)
    size = len(a)
    observed = sum(a) / size - sum(b) / size
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rng.randrange(size) for _ in range(size)]
        samples.append(sum(a[i] - b[i] for i in picked) / size)
    samples.sort()
    low = samples[int(alpha / 2 * iterations)]
    high = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    excludes_zero = low > 0 or high < 0
    better = (observed < 0) if lower_is_better else (observed > 0)
    return {
        "difference": round(observed, 5),
        "ci95": [round(low, 5), round(high, 5)],
        "significant": excludes_zero,
        "a_is_better": bool(better and excludes_zero),
        "n": size,
        "lower_is_better": lower_is_better,
    }
