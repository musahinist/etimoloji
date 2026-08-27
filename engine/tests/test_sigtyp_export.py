"""
SIGTYP 2022 biçiminde dışa aktarım testleri (Faz E4).

Amaç, sayılarımızın **bağımsız olarak yeniden üretilebilmesi**: başka
sistemler aynı veride, alanın yayınlanmış paylaşılan görev biçiminde
ölçülebilsin.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# ``scripts/`` bir paket değil; modül yolla yüklenir.
_SPEC = importlib.util.spec_from_file_location(
    "export_sigtyp",
    Path(__file__).resolve().parents[2] / "scripts" / "export_sigtyp.py",
)
sigtyp = importlib.util.module_from_spec(_SPEC)
sys.modules["export_sigtyp"] = sigtyp
_SPEC.loader.exec_module(sigtyp)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.languages = ["tr", "kk", "cv", "sah"]
        self.rows = [
            {
                "COGID": f"c{i}",
                sigtyp.PROTO_COLUMN: "k ö ŕ",
                "tr": "g ö z",
                "kk": "k ö z",
                "cv": "k u ś",
                "sah": "h a r a h",
            }
            for i in range(20)
        ]

    def test_proto_column_is_never_hidden(self):
        """⚠️ ST2022 görevi 'eksik REFLEKSİ tahmin et'tir, 'ata biçmi tahmin
        et' değil. Ata sütununu gizlemek iki görevi karıştırır."""
        _, test, _ = sigtyp._mask(self.rows, self.languages, 0.5, 1)
        self.assertTrue(all(r[sigtyp.PROTO_COLUMN] != sigtyp.MASK for r in test))

    def test_at_least_one_daughter_stays_visible(self):
        """Hepsi gizlenirse tahmin edilecek bağlam kalmaz."""
        _, test, _ = sigtyp._mask(self.rows, self.languages, 1.0, 1)
        for row in test:
            visible = [
                lang for lang in self.languages if row.get(lang) not in (sigtyp.MASK, "")
            ]
            self.assertTrue(visible, row)

    def test_solutions_carry_exactly_the_hidden_cells(self):
        _, test, solutions = sigtyp._mask(self.rows, self.languages, 0.5, 1)
        for test_row, solution_row in zip(test, solutions, strict=True):
            hidden = {
                lang for lang in self.languages if test_row.get(lang) == sigtyp.MASK
            }
            self.assertEqual(hidden, set(solution_row) - {"COGID"})

    def test_training_drops_the_hidden_cells(self):
        training, test, _ = sigtyp._mask(self.rows, self.languages, 0.5, 1)
        for train_row, test_row in zip(training, test, strict=True):
            for lang in self.languages:
                if test_row.get(lang) == sigtyp.MASK:
                    self.assertNotIn(lang, train_row)

    def test_masking_is_deterministic(self):
        """⚠️ Aynı komut aynı bölmeleri üretmeli; yoksa 'aynı veride ölçtük'
        iddiası kurulamaz."""
        first = sigtyp._mask(self.rows, self.languages, 0.3, sigtyp.SPLIT_SEED)
        second = sigtyp._mask(self.rows, self.languages, 0.3, sigtyp.SPLIT_SEED)
        self.assertEqual(first, second)

    def test_rows_with_too_few_witnesses_are_left_alone(self):
        sparse = [{"COGID": "x", sigtyp.PROTO_COLUMN: "a", "tr": "a"}]
        training, test, _ = sigtyp._mask(sparse, self.languages, 0.5, 1)
        self.assertEqual(len(training), 1)
        self.assertEqual(test, [])


class TestSegmentation(unittest.TestCase):
    def test_cells_are_space_separated(self):
        self.assertEqual(sigtyp._segments("göz"), "g ö z")

    def test_empty_form(self):
        self.assertEqual(sigtyp._segments(""), "")


class TestExport(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.out = Path(self._tmp.name)
        try:
            self.provenance = sigtyp.export("savelyevturkic", self.out)
        except (FileNotFoundError, RuntimeError):
            self.skipTest("CLDF veri kümesi indirilmemiş")

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_proportions_are_written(self):
        for proportion in sigtyp.PROPORTIONS:
            tag = f"{proportion:.2f}"
            for prefix in ("training", "test", "solutions"):
                with self.subTest(file=f"{prefix}-{tag}"):
                    self.assertTrue((self.out / f"{prefix}-{tag}.tsv").exists())

    def test_provenance_records_checksums(self):
        """Yeniden indirip aynı özetleri almak, verinin değişmediğinin
        kanıtıdır."""
        data = json.loads((self.out / "_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(all(info["sha256"] for info in data["files"].values()))

    def test_caveats_are_shipped_with_the_data(self):
        """⚠️ Bölütlemenin harf düzeyinde olduğu, veriyle birlikte
        gitmelidir; yoksa kullanan kişi IPA bölütü sanır."""
        data = json.loads((self.out / "_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(any("HARF" in c for c in data["caveats"]))

    def test_cognates_file_has_the_proto_column(self):
        rows = _rows(self.out / "cognates.tsv")
        self.assertTrue(rows)
        self.assertIn(sigtyp.PROTO_COLUMN, rows[0])


if __name__ == "__main__":
    unittest.main()
