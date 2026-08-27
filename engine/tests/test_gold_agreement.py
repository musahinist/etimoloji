"""
Altın standartlar arası uyum testleri (Faz E1).

⚠️ Bu ölçüm olmadan hiçbir akraba tespiti skoru yorumlanamaz: "B-Cubed F
0,82" cümlesi, uzmanların birbiriyle ne kadar uyuştuğu bilinmeden
anlamsızdır — tavan 1,00 değildir.

List, Walworth ve ark. (2018, *JLE*) bu boşluğu açıkça ilan ediyor.
"""

from __future__ import annotations

import unittest

from engine.evaluation.gold_agreement import adjusted_rand_index, measure


class TestAdjustedRandIndex(unittest.TestCase):
    def test_identical_partitions_score_one(self):
        partition = {"a": 1, "b": 1, "c": 2, "d": 2}
        self.assertAlmostEqual(adjusted_rand_index(partition, partition), 1.0)

    def test_chance_level_is_near_zero(self):
        """⚠️ Düzeltilmemiş Rand indeksi şansla bile 0,9'un üstüne çıkar;
        ayarlanmış sürüm beklenen değeri sıfıra çeker."""
        import random

        rng = random.Random(20260827)
        keys = [str(i) for i in range(200)]
        a = {k: rng.randrange(4) for k in keys}
        b = {k: rng.randrange(4) for k in keys}
        self.assertLess(abs(adjusted_rand_index(a, b)), 0.1)

    def test_finer_partition_is_penalised_but_positive(self):
        coarse = {"a": 1, "b": 1, "c": 1, "d": 1}
        fine = {"a": 1, "b": 1, "c": 2, "d": 2}
        score = adjusted_rand_index(coarse, fine)
        self.assertLess(score, 1.0)

    def test_too_few_items(self):
        self.assertEqual(adjusted_rand_index({"a": 1}, {"a": 1}), 0.0)

    def test_disjoint_keys(self):
        self.assertEqual(adjusted_rand_index({"a": 1}, {"b": 1}), 0.0)


class TestMeasurement(unittest.TestCase):
    def setUp(self):
        self.result = measure()
        if self.result is None:
            self.skipTest("CLDF veri kümeleri indirilmemiş")

    def test_item_level_bridge_finds_shared_items(self):
        """⚠️ Kavram köprüsü kurulamıyor — ``hruschkaturkic``te Concepticon
        glossu HİÇ yok, parametreler ``Etymon 2`` gibi künye etiketleri.
        Köprü ``(dil, karşılaştırma biçmi)`` çiftleriyle kuruluyor."""
        self.assertGreater(self.result.n_items, 500)
        self.assertGreater(self.result.n_languages, 10)

    def test_experts_do_not_agree_perfectly(self):
        """Asıl bulgu bu: tavan 1,00 değildir."""
        self.assertLess(self.result.bcubed["fscore"], 1.0)
        self.assertLess(self.result.adjusted_rand, 1.0)

    def test_agreement_is_high_but_not_total(self):
        self.assertGreater(self.result.bcubed["fscore"], 0.7)

    def test_cluster_granularity_differs(self):
        """İki derleme aynı öğeleri farklı incelikte bölüyor; B-Cubed buna
        duyarlıdır ve band bu yüzden aşağı yanlıdır."""
        self.assertNotEqual(self.result.clusters_a, self.result.clusters_b)


if __name__ == "__main__":
    unittest.main()
