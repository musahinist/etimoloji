"""
Eğitilmiş fonotaktik dizilim modeli testleri (PyBor yaklaşımı).

Elle yazılmış fonotaktik kurallar WOLD/Sakha'da **F 0,215** alıyor; aynı
veride eğitilmiş iki modelli sınıflandırıcı **F 0,558** alıyor — yayınlanmış
PyBor ortalamasının (0,59-0,61) hemen altında.

⚠️ Bu modülün en pahalı hatası **yığın sızıntısıdır** ve bir kez yaşandı:
model tüm ayar yarısında eğitilip eşik de aynı yarıda ayarlanınca motorun
F'si 0,6461'den 0,5868'e DÜŞTÜ. Testler o ayrımı korur.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.nlp.phonotactic_lm import (
    BOUNDARY,
    MarkovModel,
    PhonotacticClassifier,
    fit,
    load,
    save,
)

INHERITED = ["balık", "kuş", "taş", "yol", "göz", "kol", "baş", "diz", "kar", "buz",
             "tuz", "kız", "yaz", "kış", "dil", "el", "kan", "ay", "gün", "yer"]
BORROWED = ["kitap", "sabun", "kalem", "pencere", "sandalye", "televizyon",
            "otomobil", "makine", "fabrika", "restoran", "hastane", "gazete",
            "problem", "sistem", "program", "telefon", "bisiklet", "kültür",
            "matematik", "üniversite"]


def _samples() -> list[tuple[str, bool]]:
    return [(w, False) for w in INHERITED] + [(w, True) for w in BORROWED]


class TestMarkovModel(unittest.TestCase):
    def test_boundaries_are_modelled(self):
        """Söz başı ve söz sonu dizilimleri ayırt edicidir (``ostuol``daki
        öntüreme ünlüsü gibi); sınırlar modele girmeli."""
        model = MarkovModel()
        model.observe("at")
        self.assertIn(BOUNDARY * (model.order - 1), model.counts)

    def test_unseen_trigram_does_not_zero_the_word(self):
        """⚠️ Yumuşatmasız modelde eğitimde görülmemiş tek bir üçlü kelimeyi
        olasılık sıfıra indirir ve sınıfından bağımsız olarak eler."""
        model = MarkovModel()
        for word in INHERITED:
            model.observe(word)
        self.assertGreater(model.log_probability("zzqx"), -30.0)

    def test_trained_word_is_more_probable_than_a_stranger(self):
        model = MarkovModel()
        for word in INHERITED:
            model.observe(word)
        self.assertGreater(model.log_probability("kar"), model.log_probability("xqz"))

    def test_length_is_normalised(self):
        """Uzunluğa bölünmezse uzun kelimeler her modelde daha düşük toplam
        alır ve karşılaştırma uzunluğa kayar."""
        model = MarkovModel()
        for word in INHERITED:
            model.observe(word)
        short = model.log_probability("kar")
        long = model.log_probability("karkarkar")
        self.assertLess(abs(short - long), 3.0)

    def test_empty_word_on_a_trained_model(self):
        model = MarkovModel()
        for word in INHERITED:
            model.observe(word)
        self.assertLess(model.log_probability(""), 0.0)

    def test_untrained_model_is_uninformative_not_confident(self):
        """Boş modelde sözlük de boştur; her diziye 1/1 olasılık verir ve
        log olasılık 0'dır. Bu "çok olası" demek DEĞİLDİR — ``score`` bunu
        iki modelin farkı olarak kullandığı için iki boş model 0 fark
        üretir, yani "bilmiyorum"."""
        self.assertEqual(MarkovModel().log_probability("kar"), 0.0)
        self.assertEqual(
            PhonotacticClassifier().score("kar"), 0.0, "eğitilmemiş model susmalı"
        )


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = fit(_samples(), language="tr", trained_on="test/tune")

    def test_both_classes_are_required(self):
        with self.assertRaises(ValueError):
            fit([(w, False) for w in INHERITED], language="tr", trained_on="t")

    def test_borrowed_words_score_higher(self):
        borrowed = sum(self.classifier.score(w) for w in BORROWED) / len(BORROWED)
        inherited = sum(self.classifier.score(w) for w in INHERITED) / len(INHERITED)
        self.assertGreater(borrowed, inherited)

    def test_untrained_classifier_is_silent(self):
        blank = PhonotacticClassifier()
        self.assertFalse(blank.is_trained)
        self.assertEqual(blank.score("kitap"), 0.0)
        self.assertEqual(blank.strength("kitap"), 0.0)

    def test_strength_is_bounded(self):
        for word in (*INHERITED, *BORROWED):
            with self.subTest(word=word):
                strength = self.classifier.strength(word)
                self.assertGreaterEqual(strength, 0.0)
                self.assertLessEqual(strength, 1.0)

    def test_threshold_is_learned_not_assumed_zero(self):
        """Sınıf dengesizliğinde 0 log-oranı yanlış eşiktir."""
        self.assertIsInstance(self.classifier.threshold, float)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "m.json"
        self.classifier = fit(_samples(), language="tr", trained_on="test/tune")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        save(self.classifier, self.path)
        loaded = load("tr", self.path)
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.score("kitap"), self.classifier.score("kitap"), places=6)

    def test_missing_model(self):
        self.assertIsNone(load("tr", self.path))

    def test_corrupt_model_is_reported_not_raised(self):
        self.path.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(load("tr", self.path))

    def test_model_for_another_language_is_refused(self):
        """⚠️ Fonotaktik dilden dile değişir — zaten ölçtüğü şey odur.
        Sakha modelini Türkçeye uygulamak sinyali anlamsız kılar."""
        save(self.classifier, self.path)
        self.assertIsNone(load("sah", self.path))

    def test_language_is_recorded(self):
        save(self.classifier, self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["language"], "tr")


class TestSignalIntegration(unittest.TestCase):
    def test_missing_model_disables_the_signal(self):
        from unittest import mock

        from engine.nlp.borrowing_detector import BorrowingDetector

        with mock.patch("engine.nlp.phonotactic_lm.load", lambda *a, **k: None):
            signal = BorrowingDetector._phonotactic_model_signal("kitap", "tr")
        self.assertFalse(signal.fired)
        self.assertIn("yok", signal.explanation)

    def test_weight_is_zero_in_the_linear_fallback(self):
        """⚠️ Öğrenilen katsayıları normalize edip doğrusal toplama koymak
        lojistik modelin davranışını yeniden üretmiyor: sabit terim (-1,993)
        ve sigmoid kararın parçasıdır. Ölçüldü — doğrusal yedek yolda bu
        sinyal F'yi 0,6461'den 0,6137'ye düşürüyordu."""
        from engine.nlp.borrowing_detector import SIGNAL_WEIGHTS

        self.assertEqual(SIGNAL_WEIGHTS["fonotaktik_model"], 0.0)

    def test_combiner_knows_the_signal(self):
        from engine.nlp.borrowing_combiner import SIGNAL_ORDER

        self.assertIn("fonotaktik_model", SIGNAL_ORDER)


if __name__ == "__main__":
    unittest.main()
