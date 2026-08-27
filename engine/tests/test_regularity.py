"""
Denklik düzenliliği teşhisi testleri (Faz D1, CoPaR).

⚠️ **Plan CoPaR'ı elle yazılmış denkliklerin YERİNE koymayı öngörüyordu.**
Ölçüm o planı değiştirdi: elle yazılmış denklikleri öğrenilmiş sayımların
önüne koymak dilbilimsel olarak yerleşik iki kararı bozuyordu
(``{tr: y, kk: z, otk: d}`` -> ``*j``, doğrusu ``*d̮``; ``*teŋiŕ`` ->
``*teniŕ``). CoPaR bu yüzden **teşhis** olarak kullanılıyor, karar katmanı
olarak değil.
"""

from __future__ import annotations

import unittest

from engine.evaluation.regularity import MIN_REFS, RegularityReport, _cv, measure


class TestCvSkeleton(unittest.TestCase):
    def test_cv_is_space_separated_and_aligned(self):
        """⚠️ İskelet karşılaştırma biçmi üzerinden kurulur; bölütlemeyle
        birebir hizalanmalı, yoksa CoPaR konumları kaydırır."""
        skeleton = _cv("göz")
        self.assertEqual(skeleton, "C V C")
        self.assertEqual(len(skeleton.split()), len("göz"))

    def test_long_word(self):
        self.assertEqual(len(_cv("kitapçı").split()), 7)

    def test_empty(self):
        self.assertEqual(_cv(""), "")


class TestReport(unittest.TestCase):
    def test_ratio_is_serialised(self):
        report = RegularityReport(10, 100, 40, 15, 0.71, 0.9)
        data = report.as_dict()
        self.assertEqual(data["regular_ratio"], 0.71)
        self.assertEqual(data["min_refs"], MIN_REFS)

    def test_absent_purity_is_null_not_zero(self):
        """Ölçülemedi ile sıfır aynı şey değildir."""
        self.assertIsNone(RegularityReport(1, 1, 1, 0, 0.5, None).as_dict()["purity"])


class TestMeasurement(unittest.TestCase):
    def setUp(self):
        self.report = measure()
        if self.report is None:
            self.skipTest("lingrex yok veya CLDF verisi indirilmemiş")

    def test_most_columns_are_regular_but_not_all(self):
        """⚠️ Bu oran rekonstrüksiyon doğruluğunun ÜST SINIRINI belirler:
        hiçbir örüntüye oturmayan sütunda kural tabanlı bir sistem ancak
        şansa kalır. Ölçüldü: 0,7125."""
        self.assertGreater(self.report.regular_ratio, 0.4)
        self.assertLess(self.report.regular_ratio, 1.0)

    def test_singleton_patterns_exist(self):
        """Tekil örüntü = o sütunun denkliği başka hiçbir yerde
        görülmüyor. Sayısı düzensizliğin doğrudan ölçüsüdür."""
        self.assertGreater(self.report.n_singleton_patterns, 0)

    def test_only_train_split_is_used(self):
        """dev/test kavramlarını görmek ölçümü geçersiz kılardı."""
        import inspect

        from engine.evaluation import regularity

        self.assertIn('split: str | None = "train"', inspect.getsource(regularity.measure))


class TestNotWiredIntoDecisions(unittest.TestCase):
    def test_hand_written_correspondences_are_still_in_place(self):
        """⚠️ Ölçüm planı değiştirdi: elle yazılmış denklikler dar ve
        küratörlüdür, öğrenilmiş sayım onları geçemedi."""
        from engine.nlp.proto_phonology import CORRESPONDENCES

        self.assertTrue(CORRESPONDENCES)

    def test_reconstruction_does_not_import_copar(self):
        import inspect

        from engine.nlp import comparative_reconstruction

        self.assertNotIn("copar", inspect.getsource(comparative_reconstruction).lower())


if __name__ == "__main__":
    unittest.main()
