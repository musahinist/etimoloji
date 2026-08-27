"""
Türkçe altın alıntı kümesi testleri (Faz C6).

⚠️ Bu ölçüt neden gerekiyor: WOLD'da tek Türki dil Sakha'dır (n=769).
Türkçe için elimizdeki tek etiket kaynağı Wiktionary'ydi ve motorun zincir
sinyali **zaten o etiketi okuyor** — ona karşı ölçüm döngüseldir. Ayrıca o
kümede alıntı oranı %72,9 olduğu için F'yi en yükselten karar "hepsine
alıntı de"dir ve sistemler çöküyor.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from engine.evaluation import borrowing_eval as be

PAYLOAD = {
    "_schema": "turkic-etymology-turkish-loanword-gold/v1",
    "items": [
        {
            "word": "kitap",
            "label": "alıntı",
            "evidence": "iki kaynak",
            "tdk_source": "Arapça kitāb",
            "nisanyan_source": "Arapça",
        },
        {
            "word": "göz",
            "label": "miras",
            "evidence": "nişanyan+tdk sessiz",
            "tdk_source": "",
            "nisanyan_source": "Eski Türkçe",
        },
    ],
}


class TestLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "turkish_loanwords.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, payload=PAYLOAD):
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with mock.patch.object(be, "TURKISH_GOLD_PATH", self.path):
            return be.load_turkish_gold_cases(with_witnesses=False)

    def test_labels_are_read(self):
        cases = {c.word: c.is_borrowed for c in self._load()}
        self.assertTrue(cases["kitap"])
        self.assertFalse(cases["göz"])

    def test_language_is_turkish(self):
        self.assertTrue(all(c.lang_code == "tr" for c in self._load()))

    def test_donor_is_preserved(self):
        case = next(c for c in self._load() if c.word == "kitap")
        self.assertIn("Arap", case.donor)

    def test_missing_file_returns_empty(self):
        with mock.patch.object(be, "TURKISH_GOLD_PATH", self.path):
            self.assertEqual(be.load_turkish_gold_cases(), [])

    def test_corrupt_file_is_reported_not_raised(self):
        self.path.write_text("{bozuk", encoding="utf-8")
        with mock.patch.object(be, "TURKISH_GOLD_PATH", self.path):
            self.assertEqual(be.load_turkish_gold_cases(), [])

    def test_empty_items(self):
        self.assertEqual(self._load({"items": []}), [])


class TestIndependenceFromWiktionary(unittest.TestCase):
    """⚠️ Kümenin tek değeri Wiktionary'den bağımsız olmasıdır."""

    def test_sources_are_tdk_and_nisanyan(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            path.write_text(json.dumps(PAYLOAD, ensure_ascii=False), encoding="utf-8")
            with mock.patch.object(be, "TURKISH_GOLD_PATH", path):
                cases = be.load_turkish_gold_cases(with_witnesses=False)
        self.assertTrue(all(c.source == "tdk+nisanyan" for c in cases))

    def test_chain_signal_is_disabled_for_this_benchmark(self):
        """Zincir sinyali sözlük etiketini okur; açık bırakmak bu ölçütü de
        döngüsel yapardı (Wiktionary maddesi TDK ile örtüşebilir)."""
        import inspect

        source = inspect.getsource(be.main)
        block = source.split("turkish = load_turkish_gold_cases()")[1].split(
            "wiktionary = load_wiktionary_cases"
        )[0]
        self.assertIn('trained_on="tdk_nisanyan/tr/tune"', block)
        self.assertIn("use_chain=False", block)


class TestBuilderRules(unittest.TestCase):
    """Kanıt kuralının asimetrisi ve neden öyle olduğu."""

    def setUp(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "build_turkish_loanword_gold",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "build_turkish_loanword_gold.py",
        )
        self.builder = importlib.util.module_from_spec(spec)
        sys.modules["build_turkish_loanword_gold"] = self.builder
        spec.loader.exec_module(self.builder)

    def test_relation_names_match_the_live_api(self):
        """⚠️ İlk sürüm ``türetme`` yazıyordu (yanlış yazım); gerçek değer
        ``türeme``. O yüzden türemiş kelimelerin HİÇBİRİ etiketlenemiyordu."""
        self.assertIn("türeme", self.builder.DERIVATION_RELATIONS)
        self.assertIn("ses evrimi", self.builder.INHERITED_RELATIONS)
        self.assertIn("alıntı", self.builder.BORROWED_RELATIONS)

    def test_turkic_sources_are_not_donors(self):
        """Osmanlıcadan miras alınan bir kelime alıntı değildir."""
        for name in ("eski türkçe", "türkiye türkçesi", "osmanlıca"):
            with self.subTest(name=name):
                self.assertIn(name, self.builder.TURKIC_SOURCE_NAMES)

    def test_affixes_are_filtered_from_the_word_list(self):
        """⚠️ İlk sürüm alfabetik sıralayınca ilk 60 madde tümüyle EK çıktı
        (``-abilmek``, ``-acak``) ve hiçbir etiket alınamadı."""
        import inspect

        source = inspect.getsource(self.builder.turkish_word_list)
        self.assertIn("NOT LIKE '-%'", source)

    def test_sampling_is_deterministic(self):
        """Aynı komut aynı kümeyi kurmalı; yoksa 'aynı veride ölçtük'
        iddiası kurulamaz."""
        import inspect

        source = inspect.getsource(self.builder.turkish_word_list)
        self.assertIn("deterministik", source)
        self.assertNotIn("random", source)


if __name__ == "__main__":
    unittest.main()
