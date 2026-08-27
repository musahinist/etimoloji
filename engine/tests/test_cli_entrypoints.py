"""
Komut satırı giriş noktalarının uçtan uca testleri.

Her modülün ``main()``'i bir **sözleşmedir**: `make eval`, `make chains`,
`make dialect` bunları çağırır. Kırıldıklarında test paketi sessiz kalırsa,
kırıklık ancak elle koşturunca fark edilir.

Ağ ve büyük veri gerektirenler, veri yoksa atlanır.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from engine.config import CLDF_DIR, LEXICON_DIR

HAS_CLDF = (CLDF_DIR / "savelyevturkic" / "forms.csv").exists()
HAS_WOLD = (CLDF_DIR / "wold" / "forms.csv").exists()
HAS_LEXICON = any(LEXICON_DIR.glob("tr.jsonl*"))
HAS_INDEX = (LEXICON_DIR / "index.db").exists()


class CliCase(unittest.TestCase):
    """``main()`` çağrılarını argümanlarla koşturan yardımcı."""

    def run_main(self, module, argv: list[str], *, expect: int = 0):
        with mock.patch.object(sys, "argv", ["prog", *argv]):
            code = module.main()
        self.assertEqual(code, expect)
        return code


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş")
class TestDataEntrypoints(CliCase):
    def test_cldf_wordlist_summary(self):
        from engine.db import cldf_wordlist

        self.run_main(cldf_wordlist, ["savelyevturkic"])

    def test_gold_build_reports_no_leakage(self):
        from engine.evaluation import gold

        self.run_main(gold, [])

    def test_gold_freeze_writes_sealed_files(self):
        from engine.evaluation import gold

        with TemporaryDirectory() as tmp:
            with mock.patch.object(gold, "GOLD_DIR", Path(tmp)):
                self.run_main(gold, ["--freeze"])
                seal = json.loads((Path(tmp) / "SEAL.json").read_text(encoding="utf-8"))
        self.assertIn("checksums", seal)
        self.assertTrue((seal["counts"]["test"]) > 0)

    def test_cognate_eval_runs(self):
        from engine.evaluation import cognate_eval

        with TemporaryDirectory() as tmp:
            with mock.patch("engine.evaluation.report.EVAL_DIR", Path(tmp)):
                self.run_main(cognate_eval, ["--min-forms", "5"])

    def test_prediction_eval_runs(self):
        from engine.evaluation import prediction_eval

        with TemporaryDirectory() as tmp:
            with mock.patch("engine.evaluation.report.EVAL_DIR", Path(tmp)):
                self.run_main(prediction_eval, ["--split", "dev"])

    def test_harness_runs_on_dev(self):
        from engine.evaluation import harness

        self.run_main(harness, ["--split", "dev"])

    def test_harness_refuses_the_test_split_without_consent(self):
        """Dondurulmuş test seti kazara açılmamalı."""
        from engine.evaluation import harness

        with mock.patch.object(sys, "argv", ["prog", "--split", "test"]):
            with self.assertRaises(PermissionError):
                harness.main()

    def test_calibration_eval_runs(self):
        from engine.evaluation import calibration

        with TemporaryDirectory() as tmp:
            with mock.patch("engine.evaluation.report.EVAL_DIR", Path(tmp)):
                self.run_main(calibration, ["--split", "all"])

    def test_correspondence_learning_runs(self):
        from engine.nlp import cognate_prediction

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.json"
            with mock.patch.object(cognate_prediction, "CORRESPONDENCE_PATH", path):
                self.run_main(cognate_prediction, ["--split", "train"])
                data = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(str(data["trained_on"]).endswith("/train"))
        self.assertGreater(data["n_pairs"], 10)

    def test_semantic_bridge_build_runs(self):
        from engine.nlp import semantic_plausibility

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bridge.json"
            with mock.patch.object(semantic_plausibility, "TURKISH_CONCEPT_PATH", path):
                with mock.patch.object(semantic_plausibility, "SEMANTIC_DIR", Path(tmp)):
                    self.run_main(semantic_plausibility, ["--build-bridge"])
                    data = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreater(data["n"], 50)


class TestStandaloneEntrypoints(CliCase):
    def test_negative_controls_run(self):
        from engine.evaluation import negative_controls

        with TemporaryDirectory() as tmp:
            with mock.patch("engine.evaluation.report.EVAL_DIR", Path(tmp)):
                self.run_main(negative_controls, ["--verbose"])
                data = json.loads((Path(tmp) / "negative_controls.json").read_text("utf-8"))
        self.assertTrue(data["batteries"])
        for battery in data["batteries"]:
            self.assertEqual(battery["strong_claim_rate"], 0.0, battery["battery"])

    def test_borrowing_detector_controls_run(self):
        from engine.nlp import borrowing_detector

        self.run_main(borrowing_detector, ["--controls"])

    def test_borrowing_detector_explains_words(self):
        from engine.nlp import borrowing_detector

        self.run_main(borrowing_detector, ["göz", "deniz"])

    def test_hypothesis_ranking_runs(self):
        from engine.nlp import hypothesis_ranking

        self.run_main(hypothesis_ranking, ["göz"])

    def test_semantic_plausibility_pair(self):
        from engine.nlp import semantic_plausibility

        self.run_main(semantic_plausibility, ["--pair", "ağaç", "odun"])


@unittest.skipUnless(HAS_INDEX, "sözlük indeksi kurulmamış")
class TestLexiconEntrypoints(CliCase):
    def test_stats(self):
        from engine.db import lexicon_index

        self.run_main(lexicon_index, ["--stats"])

    def test_lookup_and_fuzzy(self):
        from engine.db import lexicon_index

        self.run_main(lexicon_index, ["--lookup", "göz", "--fuzzy", "köz", "--limit", "3"])

    def test_borrowings_and_donors(self):
        from engine.db import lexicon_index

        self.run_main(lexicon_index, ["--borrowings", "tr", "--donors", "tr", "--limit", "3"])


@unittest.skipUnless(HAS_LEXICON, "sözlük dökümü indirilmemiş")
class TestBorrowingChainEntrypoint(CliCase):
    def test_chain_extraction_runs(self):
        from engine.nlp import borrowing_chain

        self.run_main(borrowing_chain, ["--lang", "tr", "--show", "3"])


@unittest.skipUnless(HAS_CLDF and HAS_WOLD and HAS_INDEX, "veri eksik")
class TestBorrowingEvalEntrypoint(CliCase):
    def test_borrowing_eval_runs(self):
        from engine.evaluation import borrowing_eval

        with TemporaryDirectory() as tmp:
            with mock.patch("engine.evaluation.report.EVAL_DIR", Path(tmp)):
                self.run_main(borrowing_eval, ["--wiktionary-limit", "200"])
                data = json.loads((Path(tmp) / "borrowing.json").read_text("utf-8"))
        self.assertIn("wold", data)
        self.assertEqual(data["wiktionary_ablation"]["chain_signal"], "disabled")


@unittest.skipUnless(HAS_CLDF and HAS_INDEX, "veri eksik")
class TestPredictionTestEntrypoint(CliCase):
    def test_generate_then_verify(self):
        from engine.evaluation import prediction_test

        with TemporaryDirectory() as tmp:
            with mock.patch.object(prediction_test, "PREDICTIONS_DIR", Path(tmp)):
                self.run_main(prediction_test, ["generate", "--name", "t", "--limit", "20"])
                self.assertTrue((Path(tmp) / "t.locked.json").exists())
                self.run_main(prediction_test, ["verify", "--name", "t"])
                report = json.loads((Path(tmp) / "t.verified.json").read_text("utf-8"))
        self.assertFalse(report["preregistered"])
        self.assertIn("summary", report)


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş")
class TestReportEntrypoint(CliCase):
    def test_baseline_report_runs_and_includes_controls(self):
        from engine.evaluation import report

        with TemporaryDirectory() as tmp:
            self.run_main(report, ["--out", tmp])
            payload = json.loads((Path(tmp) / "baseline.json").read_text("utf-8"))
            markdown = (Path(tmp) / "BASELINE.md").read_text("utf-8")
        self.assertIn("negative_controls", payload)
        self.assertIn("İstatistiksel durum", markdown)
        self.assertEqual(payload["gold_summary"]["concept_leakage"], 0)


if __name__ == "__main__":
    unittest.main()
