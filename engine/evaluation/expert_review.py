"""
Uzman değerlendirmesi protokolü ve kodlayıcılar arası uyum (Faz E2).

Motorun ürettiği iddialar altın standartla ölçülüyor, ama altın standart
yalnız **bilinen** kümeleri kapsıyor. Yeni iddiaların değeri ancak
Türkologlara sorularak ölçülebilir. Bu modül o değerlendirmenin
**ölçüm tarafını** kurar: derecelendirme biçimi, uyum katsayısı ve güven
aralığı.

## ⚠️ Metrik Cohen κ DEĞİLDİR

Cohen κ **iki** kodlayıcı ve **nominal** ölçek içindir. Bizim kurulumumuz
2-3 kodlayıcı ve **ordinal** (0-5) ölçek. Yanlış metrik iki ayrı hata yapar:

* κ, "0 vs 5" ile "3 vs 4" uyuşmazlığını **aynı** sayar; ordinal ölçekte
  ilki çok daha ağırdır.
* κ üçüncü kodlayıcıyı doğrudan alamaz; çiftler ortalanırsa güven aralığı
  bozulur.

Krippendorff α ikisini de çözer: kodlayıcı sayısından bağımsızdır, eksik
derecelendirmeye izin verir ve **ordinal fark fonksiyonu** kullanır
(Artstein & Poesio 2008, *Computational Linguistics*).

⚠️ Nokta tahmin tek başına yetmez: Zapf ve ark. (2016, *BMC Medical
Research Methodology*) α için **bootstrap güven aralığı** raporlanmasını
öneriyor; n≈100 maddede α oynaktır.

## Ölçek

===  =====================================================================
0    açıkça yanlış — dilbilimsel olarak imkânsız
1    çok zayıf — kanıt yok
2    zayıf — mümkün ama gerekçe yetersiz
3    makul — kabul edilebilir ama tartışmalı
4    güçlü — yerleşik yönteme uygun
5    kesin — literatürdeki yerleşik görüşle aynı
===  =====================================================================

⚠️ Bu modül **derecelendirme üretmez**. Uzman girdisi yoksa hiçbir sayı
üretilmez; uydurma derecelendirme, hiç değerlendirme yapmamaktan kötüdür.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

REVIEW_DIR = PROJECT_ROOT / "data" / "expert_review"

#: Ordinal ölçeğin sınırları.
SCALE_MIN, SCALE_MAX = 0, 5

#: Ölçek etiketleri — anket formunda birebir kullanılır.
SCALE_LABELS: dict[int, str] = {
    0: "açıkça yanlış — dilbilimsel olarak imkânsız",
    1: "çok zayıf — kanıt yok",
    2: "zayıf — mümkün ama gerekçe yetersiz",
    3: "makul — kabul edilebilir ama tartışmalı",
    4: "güçlü — yerleşik yönteme uygun",
    5: "kesin — literatürdeki yerleşik görüşle aynı",
}

BOOTSTRAP_SEED = 20260827


@dataclass
class RatingMatrix:
    """``madde -> {kodlayıcı: derece}``. Eksik derece **kabul edilir**."""

    ratings: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, item: str, coder: str, value: int) -> None:
        if not SCALE_MIN <= value <= SCALE_MAX:
            raise ValueError(f"derece ölçek dışı: {value}")
        self.ratings.setdefault(item, {})[coder] = value

    @property
    def coders(self) -> list[str]:
        return sorted({c for row in self.ratings.values() for c in row})

    @property
    def usable_items(self) -> list[str]:
        """En az iki kodlayıcının derecelendirdiği maddeler.

        ⚠️ Tek kodlayıcılı madde uyuma **hiçbir bilgi** taşımaz; α'nın
        paydasına girerse katsayı sulanır.
        """
        return sorted(item for item, row in self.ratings.items() if len(row) >= 2)

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> RatingMatrix:
        matrix = cls()
        for item, row in data.items():
            for coder, value in row.items():
                matrix.add(item, coder, int(value))
        return matrix


def _ordinal_distance(a: int, b: int, counts: dict[int, int]) -> float:
    """Krippendorff'un ordinal fark fonksiyonu.

    ⚠️ Nominal farktan (``a != b``) temel ayrımı budur: "0 vs 5"
    uyuşmazlığı "3 vs 4"ten **çok daha ağır** sayılır ve ağırlık, ölçeğin
    gözlenen dağılımından gelir.
    """
    low, high = (a, b) if a <= b else (b, a)
    total = sum(counts.get(v, 0) for v in range(low, high + 1))
    total -= (counts.get(low, 0) + counts.get(high, 0)) / 2
    return float(total**2)


def krippendorff_alpha(matrix: RatingMatrix) -> float | None:
    """Ordinal Krippendorff α. Hesaplanamıyorsa ``None``.

    ⚠️ ``None`` "uyum yok" demek DEĞİLDİR, "ölçülemedi" demektir. Sıfır
    döndürmek ikisini karıştırırdı.
    """
    items = matrix.usable_items
    if len(items) < 2:
        return None

    counts: dict[int, int] = {}
    for item in items:
        for value in matrix.ratings[item].values():
            counts[value] = counts.get(value, 0) + 1
    total_ratings = sum(counts.values())
    if total_ratings < 2 or len(counts) < 2:
        # Tek bir değer varsa uyuşmazlık sıfırdır ama α tanımsızdır:
        # beklenen uyuşmazlık da sıfır olur ve 0/0 çıkar.
        return None

    observed = 0.0
    pairable = 0
    for item in items:
        values = list(matrix.ratings[item].values())
        weight = len(values) - 1
        for i, a in enumerate(values):
            for b in values[i + 1 :]:
                observed += 2 * _ordinal_distance(a, b, counts) / weight
        pairable += len(values)

    expected = 0.0
    values_flat = [v for value, count in counts.items() for v in [value] * count]
    for i, a in enumerate(values_flat):
        for b in values_flat[i + 1 :]:
            expected += 2 * _ordinal_distance(a, b, counts)
    if pairable < 2:
        return None
    observed /= pairable
    expected /= total_ratings * (total_ratings - 1)
    if expected == 0:
        return None
    return 1.0 - observed / expected


def bootstrap_alpha(
    matrix: RatingMatrix,
    *,
    iterations: int = 2000,
    alpha_level: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """α için bootstrap güven aralığı (Zapf ve ark. 2016).

    ⚠️ Yeniden örnekleme **madde düzeyindedir**, derecelendirme düzeyinde
    değil: bir maddenin dereceleri birlikte gider. Ayrı ayrı örneklemek
    madde içi uyumu yapay olarak bozar ve aralığı geniş gösterir.
    """
    point = krippendorff_alpha(matrix)
    items = matrix.usable_items
    if point is None or len(items) < 3:
        return {"alpha": point, "ci95": None, "n_items": len(items)}

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        picked = [items[rng.randrange(len(items))] for _ in items]
        resampled = RatingMatrix()
        for index, item in enumerate(picked):
            for coder, value in matrix.ratings[item].items():
                resampled.add(f"{item}#{index}", coder, value)
        value = krippendorff_alpha(resampled)
        if value is not None:
            samples.append(value)
    if len(samples) < iterations // 4:
        return {"alpha": point, "ci95": None, "n_items": len(items)}
    samples.sort()
    low = samples[int(alpha_level / 2 * len(samples))]
    high = samples[min(len(samples) - 1, int((1 - alpha_level / 2) * len(samples)))]
    return {
        "alpha": round(point, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "n_items": len(items),
        "n_coders": len(matrix.coders),
    }


def load_ratings(path: Path | None = None) -> RatingMatrix | None:
    """Uzman derecelendirmelerini okur; dosya yoksa ``None``.

    ⚠️ Dosya yoksa **hiçbir sayı üretilmez**. Uydurma derecelendirme, hiç
    değerlendirme yapmamaktan kötüdür.
    """
    source = path or (REVIEW_DIR / "ratings.json")
    if not source.exists():
        logger.info("Uzman derecelendirmesi yok (%s); E2 ölçümü atlanıyor", source)
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Uzman derecelendirmesi okunamadı: %s", source)
        return None
    return RatingMatrix.from_dict(data.get("ratings") or {})


def build_review_sheet(claims: list[dict[str, Any]], path: Path | None = None) -> Path:
    """Uzmana gidecek anket formunu yazar.

    ⚠️ Form, iddiayı **gerekçesiyle** taşır. Yalnız ``göz -> *köŕ`` diye
    sormak, uzmanı motorun akıl yürütmesini görmeden karar vermeye zorlar;
    ölçülen şey iddia değil, uzmanın kendi bilgisi olurdu.
    """
    target = path or (REVIEW_DIR / "review_sheet.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-expert-review/v1",
                "scale": SCALE_LABELS,
                "instructions": (
                    "Her iddiayı 0-5 ölçeğinde derecelendirin. Gerekçeyi de "
                    "okuyun: değerlendirilen şey yalnız sonuç değil, ona "
                    "götüren kanıttır. Emin değilseniz boş bırakın — tahmin "
                    "yürütmek uyum katsayısını bozar."
                ),
                "claims": claims,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def main() -> int:
    matrix = load_ratings()
    if matrix is None:
        print(
            "Uzman derecelendirmesi yok.\n"
            f"Beklenen dosya: {REVIEW_DIR / 'ratings.json'}\n"
            '  {"ratings": {"madde_kimliği": {"kodlayıcı": 0-5}}}\n\n'
            "⚠️ Bu ölçüm uydurma veriyle koşturulmaz."
        )
        return 1
    result = bootstrap_alpha(matrix)
    print("=== UZMAN DEĞERLENDİRMESİ — KODLAYICILAR ARASI UYUM ===")
    print(f"madde {result['n_items']} · kodlayıcı {result.get('n_coders', 0)}")
    if result["alpha"] is None:
        print("α hesaplanamadı (yeterli örtüşen derecelendirme yok).")
        return 1
    interval = result["ci95"]
    print(f"Krippendorff ordinal α = {result['alpha']:.4f}")
    if interval:
        print(f"%95 bootstrap GA = [{interval[0]:.4f}, {interval[1]:.4f}]")
    print(
        "\n⚠️ Metrik Cohen κ DEĞİLDİR: κ iki kodlayıcı ve nominal ölçek "
        "içindir;\n   burada 2-3 kodlayıcı ve ordinal (0-5) ölçek var."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
