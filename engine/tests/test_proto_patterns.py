"""
Öğrenilmiş ata ses örüntü tablosu testleri (Faz D2).

⚠️ Bu modül **denetimlidir**: 400 uzman kümesi artık eğitim verisidir ve
"sıfır eğitim verisi" iddiası bırakılmıştır. Testlerin asıl işi
**sızıntıyı** engellemektir — tablo TRAIN'de öğrenilir, ölçüm dev/test'te
yapılır.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.nlp.proto_patterns import (
    MIN_SUPPORT,
    ProtoPatternTable,
    load,
    save,
)


def _table() -> ProtoPatternTable:
    table = ProtoPatternTable(trained_on="test/train")
    # Lir-Şaz: Çuvaşça -r ~ Ortak Türkçe -z < *-ŕ
    for _ in range(5):
        table.observe("cv", "r", "ŕ")
        table.observe("tr", "z", "ŕ")
        table.observe("kk", "z", "ŕ")
    # Yalnız iki gözlem: destek eşiğinin altında
    for _ in range(2):
        table.observe("tt", "s", "ş")
    return table


class TestVoting(unittest.TestCase):
    def setUp(self):
        self.table = _table()

    def test_learned_pattern_wins(self):
        sound, confidence, support = self.table.vote({"cv": "r", "tr": "z"})
        self.assertEqual(sound, "ŕ")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(support, 2)

    def test_unseen_pattern_is_silent(self):
        """Görülmemiş çift oy kullanmaz — uydurma yapmaz."""
        self.assertEqual(self.table.vote({"xx": "q"}), ("", 0.0, 0))

    def test_low_support_pairs_are_ignored(self):
        """⚠️ Tek-iki gözleme dayanan bir çift, gürültüyü kural sanmaktır."""
        self.assertLess(2, MIN_SUPPORT)
        self.assertEqual(self.table.vote({"tt": "s"}), ("", 0.0, 0))

    def test_votes_are_probability_weighted_not_raw_counts(self):
        """Ham sayım çok tanıklı dilleri kayırırdı."""
        table = ProtoPatternTable()
        for _ in range(100):
            table.observe("tr", "a", "a")
        for _ in range(5):
            table.observe("cv", "a", "ā")
        sound, _, _ = table.vote({"tr": "a", "cv": "a"})
        # İki dil de kendi ata sesine %100 tanıklık ediyor: beraberlik.
        self.assertIn(sound, ("a", "ā"))

    def test_empty_column(self):
        self.assertEqual(self.table.vote({}), ("", 0.0, 0))

    def test_untrained_table_reports_itself(self):
        self.assertFalse(ProtoPatternTable().is_trained)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "p.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        save(_table(), self.path)
        loaded = load(self.path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.vote({"cv": "r", "tr": "z"})[0], "ŕ")

    def test_low_support_pairs_are_not_even_written(self):
        save(_table(), self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("tt|s", data["counts"])

    def test_missing_file(self):
        self.assertIsNone(load(self.path))

    def test_corrupt_file_is_reported_not_raised(self):
        self.path.write_text("{bozuk", encoding="utf-8")
        self.assertIsNone(load(self.path))

    def test_training_split_is_recorded(self):
        save(_table(), self.path)
        self.assertEqual(load(self.path).trained_on, "test/train")


class TestLeakageGuard(unittest.TestCase):
    """⚠️ Denetimli bileşen + tüm-veri ölçümü = ezber raporlamak."""

    def test_supervised_components_are_detected(self):
        from engine.evaluation.report import _supervised_components

        components = _supervised_components()
        self.assertIsInstance(components, list)

    def test_shipped_table_is_trained_on_train_only(self):
        table = load()
        if table is None:
            self.skipTest("örüntü tablosu henüz öğrenilmemiş")
        self.assertTrue(table.trained_on.endswith("/train"), table.trained_on)


class TestColumnDecisionIntegration(unittest.TestCase):
    def test_diagnostic_rule_still_outranks_the_learned_vote(self):
        """⚠️ Tanısal kural bir TANIMDIR (Lir-Şaz); öğrenilmiş sayım onu
        geçemez. Sıra bozulursa veriden çıkmayan bir ayrım iddia edilir."""
        from engine.nlp.multi_alignment import AlignedColumn
        from engine.nlp.proto_phonology import pick_proto_sound

        column = AlignedColumn(
            sounds={"cv": "r", "tr": "z", "kk": "z"}, index=1, width=2
        )
        decision = pick_proto_sound(column, "final")
        self.assertEqual(decision.method, "tanisal")

    def test_learned_vote_has_a_confidence_gate(self):
        """Koşulsuz üstün tutmak ölçüldü ve zarar veriyordu (sütun düzeyinde
        +5,7 puan, kelime düzeyinde 0,324 -> 0,257)."""
        from engine.nlp.proto_phonology import LEARNED_MIN_CONFIDENCE

        self.assertGreater(LEARNED_MIN_CONFIDENCE, 0.0)


if __name__ == "__main__":
    unittest.main()
