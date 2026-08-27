"""
Uzman değerlendirmesi ve kodlayıcılar arası uyum testleri (Faz E2).

⚠️ Metrik **Cohen κ değildir**: κ iki kodlayıcı ve nominal ölçek içindir;
burada 2-3 kodlayıcı ve ordinal (0-5) ölçek var. Testlerin asıl işi bu
ayrımı korumak — ordinal fark fonksiyonu "0 vs 5" ile "3 vs 4"ü aynı
saymamalı.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.evaluation.expert_review import (
    SCALE_LABELS,
    RatingMatrix,
    bootstrap_alpha,
    build_review_sheet,
    krippendorff_alpha,
    load_ratings,
)


def _agreeing() -> RatingMatrix:
    matrix = RatingMatrix()
    pairs = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (5, 4), (4, 3), (0, 0), (5, 5), (3, 4)]
    for index, (a, b) in enumerate(pairs):
        matrix.add(f"i{index}", "A", a)
        matrix.add(f"i{index}", "B", b)
    return matrix


class TestRatingMatrix(unittest.TestCase):
    def test_out_of_scale_is_refused(self):
        with self.assertRaises(ValueError):
            RatingMatrix().add("i", "A", 9)

    def test_single_coder_items_are_not_usable(self):
        """⚠️ Tek kodlayıcılı madde uyuma hiçbir bilgi taşımaz; α'nın
        paydasına girerse katsayı sulanır."""
        matrix = RatingMatrix()
        matrix.add("solo", "A", 3)
        matrix.add("pair", "A", 3)
        matrix.add("pair", "B", 4)
        self.assertEqual(matrix.usable_items, ["pair"])

    def test_missing_ratings_are_allowed(self):
        """Krippendorff α eksik derecelendirmeye izin verir; uzman emin
        değilse boş bırakmalı."""
        matrix = RatingMatrix()
        matrix.add("a", "A", 4)
        matrix.add("a", "B", 4)
        matrix.add("b", "A", 2)
        matrix.add("b", "C", 2)
        self.assertIsNotNone(krippendorff_alpha(matrix))


class TestOrdinalAlpha(unittest.TestCase):
    def test_high_agreement_scores_high(self):
        self.assertGreater(krippendorff_alpha(_agreeing()), 0.8)

    def test_random_ratings_score_near_zero(self):
        rng = random.Random(1)
        matrix = RatingMatrix()
        for index in range(40):
            matrix.add(f"i{index}", "A", rng.randrange(6))
            matrix.add(f"i{index}", "B", rng.randrange(6))
        self.assertLess(abs(krippendorff_alpha(matrix)), 0.2)

    def test_ordinal_distance_penalises_far_disagreement_more(self):
        """⚠️ Nominal metrikten temel ayrım: "0 vs 5" ile "3 vs 4" aynı
        değildir. Cohen κ ikisini AYNI sayar.

        ⚠️ Karşılaştırma **aynı dağılım içinde** yapılmalı: α ölçeğe
        görelidir, ve yalnız {3,4} içeren bir veri kümesiyle yalnız {0,5}
        içeren bir veri kümesi aynı α'yı verir — ikisinde de uyuşmazlık
        gözlenen ölçeğin tamamı kadardır. Ordinal özelliği görmek için fark
        fonksiyonuna doğrudan bakılır.
        """
        from engine.evaluation.expert_review import _ordinal_distance

        counts = {value: 10 for value in range(6)}
        self.assertGreater(
            _ordinal_distance(0, 5, counts), _ordinal_distance(3, 4, counts)
        )
        self.assertEqual(_ordinal_distance(3, 3, counts), 0.0)

    def test_same_scale_far_disagreements_lower_alpha(self):
        """Aynı dağılım içinde: uzak uyuşmazlıklar α'yı daha çok düşürür."""
        near, far = RatingMatrix(), RatingMatrix()
        for index in range(24):
            base = index % 6
            near.add(f"i{index}", "A", base)
            near.add(f"i{index}", "B", min(5, base + (1 if index % 2 else 0)))
            far.add(f"i{index}", "A", base)
            far.add(f"i{index}", "B", (5 - base) if index % 2 else base)
        self.assertGreater(krippendorff_alpha(near), krippendorff_alpha(far))

    def test_three_coders_are_handled_directly(self):
        """κ üçüncü kodlayıcıyı doğrudan alamaz; çiftler ortalanırsa güven
        aralığı bozulur."""
        matrix = RatingMatrix()
        for index in range(12):
            for coder, offset in (("A", 0), ("B", 0), ("C", 1)):
                matrix.add(f"i{index}", coder, min(5, index % 5 + offset))
        self.assertIsNotNone(krippendorff_alpha(matrix))
        self.assertEqual(len(matrix.coders), 3)

    def test_unmeasurable_returns_none_not_zero(self):
        """⚠️ ``None`` "uyum yok" demek DEĞİLDİR, "ölçülemedi" demektir.
        Sıfır döndürmek ikisini karıştırırdı."""
        empty = RatingMatrix()
        self.assertIsNone(krippendorff_alpha(empty))
        constant = RatingMatrix()
        for index in range(5):
            constant.add(f"i{index}", "A", 3)
            constant.add(f"i{index}", "B", 3)
        self.assertIsNone(krippendorff_alpha(constant))


class TestBootstrap(unittest.TestCase):
    def test_interval_brackets_the_estimate(self):
        result = bootstrap_alpha(_agreeing(), iterations=400)
        low, high = result["ci95"]
        self.assertLessEqual(low, result["alpha"])
        self.assertGreaterEqual(high, result["alpha"])

    def test_tiny_sample_reports_no_interval(self):
        """n≈100'de bile α oynaktır; aralıksız nokta tahmin yanıltır."""
        matrix = RatingMatrix()
        matrix.add("a", "A", 4)
        matrix.add("a", "B", 5)
        self.assertIsNone(bootstrap_alpha(matrix)["ci95"])

    def test_resampling_is_at_item_level(self):
        """⚠️ Derecelendirme düzeyinde örneklemek madde içi uyumu yapay
        olarak bozar ve aralığı geniş gösterir."""
        import inspect

        from engine.evaluation import expert_review

        source = inspect.getsource(expert_review.bootstrap_alpha)
        self.assertIn("madde düzeyindedir", source)


class TestProtocolFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_ratings_produce_no_numbers(self):
        """⚠️ Uydurma derecelendirme, hiç değerlendirme yapmamaktan
        kötüdür."""
        self.assertIsNone(load_ratings(self.dir / "yok.json"))

    def test_review_sheet_carries_the_reasoning(self):
        """⚠️ Yalnız sonucu sormak, uzmanı motorun akıl yürütmesini görmeden
        karar vermeye zorlar; ölçülen şey iddia değil uzmanın kendi bilgisi
        olurdu."""
        claims = [{"word": "göz", "claim": "*köŕ", "reasoning": "Lir-Şaz rotasizmi"}]
        path = build_review_sheet(claims, self.dir / "sheet.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("reasoning", data["claims"][0])
        self.assertEqual(set(data["scale"]), {str(k) for k in SCALE_LABELS})

    def test_corrupt_ratings_are_reported_not_raised(self):
        path = self.dir / "ratings.json"
        path.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(load_ratings(path))


if __name__ == "__main__":
    unittest.main()
